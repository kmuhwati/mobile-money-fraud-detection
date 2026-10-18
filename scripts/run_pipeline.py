"""Leakage audit -> tiered models -> operating point -> drift report -> model artifact.

Run from the project root:  python scripts/run_pipeline.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import lightgbm as lgb  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import shap  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from fraud_detection import evaluation, features, plotstyle  # noqa: E402
from fraud_detection.config import (  # noqa: E402
    ALERT_BUDGET, BEHAVIOURAL, DEPLOY_TIER, FIGURES, MODEL_DIR, REPORTS, SEED, TABLES, TIERS, TRAIN_END, VALID_END,
)

LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  metric="average_precision", seed=SEED, verbose=-1)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    plotstyle.apply()
    for d in (FIGURES, TABLES, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)
    out: dict = {"alert_budget": ALERT_BUDGET}

    # ------------------------------------------------------------------ data
    log("building features")
    raw = features.load_raw()
    out["n_all_transactions"] = raw.height
    out["fraud_by_type"] = raw.group_by("type").agg(pl.col("isFraud").sum()).to_pandas().set_index("type")["isFraud"].to_dict()
    feat = features.build(raw).drop(["nameOrig", "nameDest"]).to_pandas()
    del raw
    feat["split"] = np.select([feat.step <= TRAIN_END, feat.step <= VALID_END], ["train", "valid"], "test")
    out["rows"] = feat.split.value_counts().to_dict()
    out["fraud_rate"] = feat.groupby("split").isFraud.mean().to_dict()
    log(f"rows {out['rows']}, fraud rate {out['fraud_rate']}")

    # ------------------------------------------------------------------ leakage audit
    audit = feat.groupby("isFraud").agg(
        amount_equals_full_balance=("drains_account", "mean"),
        orig_balance_arithmetic_holds=("error_balance_orig", lambda s: (s.abs() < 0.01).mean()),
        new_orig_balance_zero=("log_newbalance_orig", lambda s: (s == 0).mean()),
        dest_balance_arithmetic_holds=("error_balance_dest", lambda s: (s.abs() < 1).mean()),
        flagged_by_legacy_rule=("isFlaggedFraud", "mean"),
    ).T.rename(columns={0: "legitimate", 1: "fraud"})
    audit.to_csv(TABLES / "leakage_audit.csv")
    out["leakage_audit"] = audit.round(4).to_dict()
    log("\n" + audit.to_string())

    tr, va, te = (feat[feat.split == s] for s in ("train", "valid", "test"))
    y_tr, y_va, y_te = (d.isFraud.to_numpy() for d in (tr, va, te))
    amt_te = te.amount.to_numpy()

    # ------------------------------------------------------------------ models
    results, scores, models = [], {}, {}

    def record(name, tier, s):
        scores[name] = s
        r = {"model": name, "tier": tier, **evaluation.summary(y_te, s, amt_te, ALERT_BUDGET),
             **evaluation.bootstrap_ci(y_te, s, amt_te, ALERT_BUDGET, seed=SEED)}
        results.append(r)
        log(f"{name:45s} AUPRC={r['auprc']:.4f} recall@budget={r['recall@budget']:.3f} precision@budget={r['precision@budget']:.3f}")

    rng = np.random.default_rng(SEED)
    record("Rule: amount drains the account", "B_authorisation_time", te.drains_account.to_numpy() + 1e-6 * rng.random(len(te)))
    record("Rule: legacy isFlaggedFraud", "A_all_columns_leaky", te.isFlaggedFraud.to_numpy() + 1e-6 * rng.random(len(te)))

    lr_cols = TIERS["B_authorisation_time"]
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    lr.fit(tr[lr_cols].fillna(-1), y_tr)
    record("Logistic regression (tier B)", "B_authorisation_time", lr.predict_proba(te[lr_cols].fillna(-1))[:, 1])

    for tier, cols in TIERS.items():
        log(f"LightGBM {tier}")
        dtr = lgb.Dataset(tr[cols], y_tr)
        dva = lgb.Dataset(va[cols], y_va, reference=dtr)
        m = lgb.train(LGB_PARAMS, dtr, num_boost_round=2000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        models[tier] = (m, cols)
        record(f"LightGBM ({tier})", tier, m.predict(te[cols], num_iteration=m.best_iteration))

    # Deployed design: the drain rule as an explicit business rule, the behaviour-only model for the rest.
    # Model scores are < 1, so rule hits rank first.
    HYBRID = "Drain rule + LightGBM (C): deployed"
    record(HYBRID, "C_behavioural_only", te.drains_account.to_numpy() + scores["LightGBM (C_behavioural_only)"])

    res = pd.DataFrame(results)
    res.to_csv(TABLES / "model_comparison.csv", index=False)
    out["models"] = res.round(4).to_dict(orient="records")

    # ------------------------------------------------------------------ deployable model: threshold + calibration
    model, cols = models[DEPLOY_TIER]
    raw_va = model.predict(va[cols], num_iteration=model.best_iteration)
    raw_te = scores[f"LightGBM ({DEPLOY_TIER})"]
    # Threshold on the model so that rule hits + model alerts fill the review budget on validation.
    drain_va, drain_te = va.drains_account.to_numpy(), te.drains_account.to_numpy()
    room = ALERT_BUDGET * len(va) - drain_va.sum()
    non_drain = raw_va[drain_va == 0]
    threshold = float(np.quantile(non_drain, 1 - room / len(non_drain))) if room > 0 else float("inf")
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw_va, y_va)
    hyb_va, hyb_te = drain_va + raw_va, drain_te + raw_te
    op_valid = evaluation.at_threshold(y_va, hyb_va, va.amount.to_numpy(), threshold)
    op_test = evaluation.at_threshold(y_te, hyb_te, amt_te, threshold)
    out["operating_point"] = {"threshold": threshold, "valid": op_valid, "test": op_test}
    log(f"operating point valid {op_valid} test {op_test}")

    te_days = te.assign(score=hyb_te, alert=hyb_te >= threshold, day=te.step // 24)
    daily = te_days.groupby("day").agg(transactions=("isFraud", "size"), fraud=("isFraud", "sum"),
                                       alerts=("alert", "sum"),
                                       caught=("isFraud", lambda s: (s & te_days.loc[s.index, "alert"]).sum()))
    daily["fraud_rate"] = daily.fraud / daily.transactions
    daily["alert_rate"] = daily.alerts / daily.transactions
    daily["recall"] = daily.caught / daily.fraud.clip(lower=1)
    daily["precision"] = daily.caught / daily.alerts.clip(lower=1)
    daily.to_csv(TABLES / "test_daily_operations.csv")

    # Budget curve: how much money is stopped as review capacity grows
    budgets = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05]
    curve = []
    for name in ("Rule: amount drains the account", "LightGBM (C_behavioural_only)",
                 "LightGBM (B_authorisation_time)", HYBRID):
        for b in budgets:
            curve.append({"model": name, "budget": b, **evaluation.at_budget(y_te, scores[name], amt_te, b)})
    curve = pd.DataFrame(curve)
    curve.to_csv(TABLES / "budget_curve.csv", index=False)

    # ------------------------------------------------------------------ drift monitoring
    log("drift")
    all_feat = feat.assign(day=feat.step // 24)
    psi_rows = []
    ref = tr
    for c in cols:
        row = {"feature": c}
        for split, frame in (("valid", va), ("test", te)):
            row[f"psi_{split}"] = evaluation.psi(ref[c].to_numpy(float), frame[c].to_numpy(float))
        psi_rows.append(row)
    raw_tr = model.predict(tr[cols], num_iteration=model.best_iteration)
    psi_rows.append({"feature": "MODEL SCORE", "psi_valid": evaluation.psi(raw_tr, raw_va), "psi_test": evaluation.psi(raw_tr, raw_te)})
    psi_tbl = pd.DataFrame(psi_rows).sort_values("psi_test", ascending=False)
    psi_tbl.to_csv(TABLES / "psi.csv", index=False)
    out["psi_top"] = psi_tbl.head(6).round(3).to_dict(orient="records")
    vol = all_feat.groupby("day").agg(transactions=("isFraud", "size"), fraud=("isFraud", "sum"))
    vol["fraud_rate"] = vol.fraud / vol.transactions
    vol.to_csv(TABLES / "daily_volume.csv")

    # ------------------------------------------------------------------ explainability
    sample = te[cols].sample(20000, random_state=SEED)
    sv = shap.TreeExplainer(model).shap_values(sample)
    sv = sv[1] if isinstance(sv, list) else sv
    imp = pd.Series(np.abs(sv).mean(0), index=cols).sort_values(ascending=False)
    imp.to_csv(TABLES / "shap_importance.csv")

    # ------------------------------------------------------------------ artifact
    model.save_model(str(MODEL_DIR / "fraud_lgbm.txt"), num_iteration=model.best_iteration)
    meta = {
        "model": "LightGBM binary + drain rule", "feature_tier": DEPLOY_TIER, "features": cols,
        "rules": {"drains_account": "alert when amount equals the full pre-transaction balance"},
        "threshold_raw": threshold, "alert_budget": ALERT_BUDGET,
        "calibration": {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()},
        "train_steps": [1, TRAIN_END], "valid_steps": [TRAIN_END + 1, VALID_END],
        "test_metrics": {k: v for k, v in res.set_index("model").loc[HYBRID].items() if k != "tier"},
        "test_metrics_model_only": {k: v for k, v in res.set_index("model").loc[f"LightGBM ({DEPLOY_TIER})"].items() if k != "tier"},
        "reference_score_quantiles": np.quantile(raw_tr, np.linspace(0, 1, 11)).tolist(),
        "best_iteration": model.best_iteration,
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2, default=float))
    out["deploy_best_iteration"] = model.best_iteration
    (REPORTS / "metrics.json").write_text(json.dumps(out, indent=2, default=float))

    # ------------------------------------------------------------------ figures
    log("figures")
    order = res.sort_values("auprc")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    color = {"A_all_columns_leaky": plotstyle.CRITICAL, "B_authorisation_time": plotstyle.SERIES[0], "C_behavioural_only": plotstyle.SERIES[2]}
    for i, (_, r) in enumerate(order.iterrows()):
        c = plotstyle.SERIES[1] if r.model == HYBRID else (color[r.tier] if r.model.startswith("LightGBM") else plotstyle.NEUTRAL)
        ax.barh(i, r.auprc, color=c, height=0.6)
        ax.errorbar(r.auprc, i, xerr=[[r.auprc - r.auprc_lo], [r.auprc_hi - r.auprc]], color=plotstyle.INK_2, lw=1, capsize=2)
        ax.annotate(f"{r.auprc:.3f}", (r.auprc_hi, i), xytext=(5, 0), textcoords="offset points", va="center", fontsize=9)
    ax.set_yticks(range(len(order)), order.model)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Average precision on the held-out last 8 days (95% bootstrap CI)")
    ax.set_title("Near-perfect scores come from leaked or simulated-artefact columns; behaviour-only is the honest number")
    plotstyle.save(fig, FIGURES / "01_model_comparison.png")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for j, name in enumerate(["LightGBM (A_all_columns_leaky)", "LightGBM (B_authorisation_time)", "LightGBM (C_behavioural_only)",
                              HYBRID, "Rule: amount drains the account"]):
        p, r, _ = precision_recall_curve(y_te, scores[name])
        style = dict(color=[plotstyle.CRITICAL, plotstyle.SERIES[0], plotstyle.SERIES[2], plotstyle.SERIES[3], plotstyle.NEUTRAL][j])
        if "leaky" in name:
            style["ls"] = "--"
        ax.plot(r, p, label=name.replace("LightGBM ", "LGBM "), **style)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall on the test period")
    ax.legend(fontsize=8, loc="lower left")
    plotstyle.save(fig, FIGURES / "02_pr_curves.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for j, name in enumerate(curve.model.unique()):
        g = curve[curve.model == name]
        col = [plotstyle.NEUTRAL, plotstyle.SERIES[2], plotstyle.SERIES[0], plotstyle.SERIES[1]][j]
        axes[0].plot(100 * g.budget, 100 * g.recall, marker="o", color=col, label=name)
        axes[1].plot(100 * g.budget, 100 * g.value_recall, marker="o", color=col, label=name)
    for ax, t in zip(axes, ("Share of fraud cases caught", "Share of fraud value caught")):
        ax.axvline(100 * ALERT_BUDGET, color=plotstyle.AXIS, ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("Alert budget (% of transfers and cash-outs reviewed, log scale)")
        ax.set_ylabel("%")
        ax.set_title(t)
    axes[0].legend(fontsize=8)
    plotstyle.save(fig, FIGURES / "03_alert_budget.png")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].bar(vol.index, vol.transactions / 1e3, color=plotstyle.NEUTRAL, width=0.8)
    axes[0].set_ylabel("Scored txns / day (k)")
    axes[0].set_title("Legitimate volume collapses after day 17 while fraud stays flat: fraud prevalence rises 15x")
    axes[1].plot(vol.index, 100 * vol.fraud_rate, color=plotstyle.CRITICAL, marker="o", ms=4)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Fraud rate (%, log scale)")
    axes[1].set_xlabel("Simulation day")
    for ax in axes:
        ax.axvspan(-0.5, TRAIN_END / 24, color=plotstyle.SERIES[0], alpha=0.06, lw=0)
        ax.axvspan(VALID_END / 24, vol.index.max() + 0.5, color=plotstyle.SERIES[1], alpha=0.06, lw=0)
    axes[1].annotate("train", (TRAIN_END / 48, axes[1].get_ylim()[1] * 0.5), fontsize=9, color=plotstyle.INK_2, ha="center")
    axes[1].annotate("test", ((VALID_END / 24 + vol.index.max()) / 2, axes[1].get_ylim()[1] * 0.5), fontsize=9, color=plotstyle.INK_2, ha="center")
    plotstyle.save(fig, FIGURES / "04_volume_and_fraud_rate.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    p10 = psi_tbl.head(10)[::-1]
    cols_c = [plotstyle.CRITICAL if v > 0.25 else (plotstyle.SERIES[3] if v > 0.1 else plotstyle.SERIES[2]) for v in p10.psi_test]
    ax.barh(p10.feature, p10.psi_test, color=cols_c, height=0.6)
    for x, lab in ((0.1, "moderate"), (0.25, "significant")):
        ax.axvline(x, color=plotstyle.AXIS, ls=":", lw=1)
        ax.annotate(lab, (x, len(p10) - 0.5), xytext=(3, 0), textcoords="offset points", fontsize=8, color=plotstyle.INK_2)
    ax.set_xlabel("PSI, train period vs test period")
    ax.set_title("Drift report: which inputs moved")
    plotstyle.save(fig, FIGURES / "05_psi.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    t12 = imp.head(12)[::-1]
    ax.barh(t12.index, t12.values, color=plotstyle.SERIES[0], height=0.6)
    ax.set_xlabel("Mean |SHAP| (log-odds), test sample")
    ax.set_title("Deployed model: what drives a fraud score")
    plotstyle.save(fig, FIGURES / "06_shap.png")

    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(daily.index, 100 * daily.alert_rate, marker="o", color=plotstyle.SERIES[0], label="Alert rate")
    ax.plot(daily.index, 100 * daily.recall, marker="o", color=plotstyle.SERIES[2], label="Recall")
    ax.plot(daily.index, 100 * daily.precision, marker="o", color=plotstyle.SERIES[1], label="Precision")
    ax.axhline(100 * ALERT_BUDGET, color=plotstyle.AXIS, ls=":", lw=1)
    ax.set_xlabel("Simulation day (test period)")
    ax.set_ylabel("%")
    ax.set_title("Deployed rule + model at a fixed threshold, day by day in the test period")
    ax.legend(fontsize=8)
    plotstyle.save(fig, FIGURES / "07_daily_operations.png")

    md = lambda f, fmt=".4f": f.to_markdown(floatfmt=fmt)  # noqa: E731
    lines = [
        "# Results (auto-generated by scripts/run_pipeline.py)\n",
        f"Rows by split: {out['rows']}  \nFraud rate by split: { {k: round(v, 5) for k, v in out['fraud_rate'].items()} }\n",
        "## Leakage audit (share of transactions where the condition holds)\n", md(audit),
        "\n## Model comparison on the test period\n", res.to_markdown(index=False, floatfmt=".4f"),
        f"\n## Operating point (threshold set on validation at {ALERT_BUDGET:.1%} alert budget)\n",
        f"```json\n{json.dumps(out['operating_point'], indent=2)}\n```",
        "\n## Daily operations in the test period\n", md(daily),
        "\n## Alert-budget curve\n", curve.to_markdown(index=False, floatfmt=".4f"),
        "\n## PSI (train vs later periods)\n", psi_tbl.to_markdown(index=False, floatfmt=".3f"),
        "\n## SHAP importance\n", imp.to_frame("mean_abs_shap").to_markdown(floatfmt=".4f"),
    ]
    (REPORTS / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    log("done")


if __name__ == "__main__":
    main()
