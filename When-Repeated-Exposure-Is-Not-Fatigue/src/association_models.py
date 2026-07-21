"""Estimate skip and non-skip watch-intensity associations."""

import os
import time

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.generalized_estimating_equations import GEE

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.abspath(os.path.join(HERE, ".."))
OUT_DIR = os.environ.get("REVIEW_RESULTS_DIR", os.path.join(PACKAGE_DIR, "results"))
FEATURES = os.environ.get("KUAIREC_FEATURES", os.path.join(OUT_DIR, "features_timegated.parquet"))
COMPARISON_OUT = os.path.join(OUT_DIR, "association_results.csv")
REPORT_OUT = os.path.join(OUT_DIR, "association_results.md")

RANDOM_STATE = 42
N_USERS_SAMPLE = 1000
CONTROLS = "video_duration_z + popularity_z + session_position_z + delta_t_z"

DEFINITIONS = [
    ("(a) Category streak", "consec_count"),
    ("(b) Time-gated category, 20 s", "cc_timegated_T20"),
    ("(c) Temporal burst, 20 s", "cc_burst_T20"),
    ("(d) Jaccard overlap, 0.5", "cc_jac_th050"),
    ("(e) Combined, 0.5 and 20 s", "cc_combo_th050_T20"),
    ("(f-1) Session cumulative", "cumexp_session"),
    ("(f-2) Rolling cumulative, 30 min", "cumexp_W30m"),
    ("(f-2) Rolling cumulative, 60 min", "cumexp_W60m"),
    ("(f-2) Rolling cumulative, 180 min", "cumexp_W180m"),
    ("(g) Sliding density, N=20", "density_N20"),
    ("(g) Sliding density, N=50", "density_N50"),
    ("(g) Sliding density, N=100", "density_N100"),
    ("(h) Time-decayed exposure, 30 min", "gauge_hl1800"),
    ("(h) Time-decayed exposure, 2 h", "gauge_hl7200"),
    ("(h) Time-decayed exposure, 1 day", "gauge_hl86400"),
    ("(i) Event-memory, T=20 s, p=0.2", "ed_T20_p02"),
    ("(i) Event-memory, T=20 s, p=0.5", "ed_T20_p05"),
    ("(i) Event-memory, T=20 s, p=1.0", "ed_T20_p10"),
    ("(i) Event-memory, T=60 s, p=0.2", "ed_T60_p02"),
    ("(i) Event-memory, T=60 s, p=0.5", "ed_T60_p05"),
    ("(i) Event-memory, T=60 s, p=1.0", "ed_T60_p10"),
]


def zscore(values):
    sd = values.std()
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError(f"Cannot standardize a variable with SD={sd}")
    return (values - values.mean()) / sd


def build_sample():
    columns = [
        "user_id", "skip_flag", "watch_ratio", "video_duration", "popularity",
        "session_position", "delta_t",
    ] + [column for _, column in DEFINITIONS]
    data = pd.read_parquet(FEATURES, columns=list(dict.fromkeys(columns)))
    data = data.dropna(subset=["delta_t"]).copy()

    rng = np.random.default_rng(RANDOM_STATE)
    users = data["user_id"].unique()
    selected = rng.choice(users, size=min(N_USERS_SAMPLE, len(users)), replace=False)
    sample = data[data["user_id"].isin(selected)].copy()

    sample["popularity_log"] = np.log1p(sample["popularity"])
    sample["video_duration_z"] = zscore(sample["video_duration"])
    sample["popularity_z"] = zscore(sample["popularity_log"])
    sample["session_position_z"] = zscore(sample["session_position"])
    sample["delta_t_z"] = zscore(sample["delta_t"])
    sample["watch_ratio_clipped"] = sample["watch_ratio"].clip(0, 5)

    scales = []
    for label, column in DEFINITIONS:
        sample[column + "_z"] = zscore(sample[column])
        scales.append({
            "label": label,
            "col": column,
            "x_mean": sample[column].mean(),
            "x_std": sample[column].std(),
            "x_min": sample[column].min(),
            "x_max": sample[column].max(),
        })
    return sample, pd.DataFrame(scales).set_index("col")


def fit_skip_gee(sample, xcol):
    result = GEE.from_formula(
        f"skip_flag ~ {xcol} + {CONTROLS}",
        groups="user_id",
        data=sample,
        family=Binomial(),
        cov_struct=Exchangeable(),
    ).fit(maxiter=200)
    interval = result.conf_int().loc[xcol]
    return {
        "skip_OR": np.exp(result.params[xcol]),
        "skip_coef": result.params[xcol],
        "skip_p": result.pvalues[xcol],
        "skip_ci_lo": np.exp(interval.iloc[0]),
        "skip_ci_hi": np.exp(interval.iloc[1]),
        "skip_converged": bool(getattr(result, "converged", True)),
    }


def fit_watch_mixed_model(sample, xcol):
    continued = sample[sample["skip_flag"] == 0]
    model = smf.mixedlm(
        f"watch_ratio_clipped ~ {xcol} + {CONTROLS}",
        data=continued,
        groups=continued["user_id"],
    )
    with np.errstate(all="ignore"):
        result = model.fit(method="bfgs", maxiter=200)
    interval = result.conf_int().loc[xcol]
    group_variance = result.cov_re.iloc[0, 0]
    return {
        "wr_coef": result.params[xcol],
        "wr_p": result.pvalues[xcol],
        "wr_ci_lo": interval.iloc[0],
        "wr_ci_hi": interval.iloc[1],
        "wr_group_var": group_variance,
        "wr_singular": bool(np.isclose(group_variance, 0.0, atol=1e-6)),
        "wr_converged": bool(getattr(result, "converged", True)),
    }


def fit_skip_glm_and_auc(sample, xcol):
    result = smf.glm(
        f"skip_flag ~ {xcol} + {CONTROLS}", data=sample, family=Binomial()
    ).fit()
    features = [
        xcol, "video_duration_z", "popularity_z", "session_position_z", "delta_t_z"
    ]
    x = sample[features].to_numpy()
    y = sample["skip_flag"].to_numpy()
    groups = sample["user_id"].to_numpy()
    aucs = []
    for train, test in GroupKFold(n_splits=5).split(x, y, groups):
        model = LogisticRegression(max_iter=1000, C=1e6)
        model.fit(x[train], y[train])
        aucs.append(roc_auc_score(y[test], model.predict_proba(x[test])[:, 1]))
    return {
        "skip_aic": result.aic,
        "skip_bic": result.bic_llf,
        "skip_cv_auc": np.mean(aucs),
        "skip_cv_auc_std": np.std(aucs),
    }


def write_report(results, n_rows, n_continued):
    lines = ["# Skip and watch-intensity associations\n\n"]
    lines.append(
        f"The analysis sample contains {N_USERS_SAMPLE:,} users, {n_rows:,} interactions, "
        f"and {n_continued:,} non-skipped interactions. Estimates are reported per one "
        "standard-deviation increase in the exposure.\n\n"
    )
    lines.append(
        "| Exposure | Skip OR | 95% CI | p | Watch beta | 95% CI | p | AIC | BIC | "
        "Grouped AUC | GEE converged | Mixed model converged |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|\n"
    )
    for _, row in results.iterrows():
        lines.append(
            f"| {row['label']} | {row['skip_OR']:.4f} | "
            f"[{row['skip_ci_lo']:.4f}, {row['skip_ci_hi']:.4f}] | {row['skip_p']:.3g} | "
            f"{row['wr_coef']:.5f} | [{row['wr_ci_lo']:.5f}, {row['wr_ci_hi']:.5f}] | "
            f"{row['wr_p']:.3g} | {row['skip_aic']:.1f} | {row['skip_bic']:.1f} | "
            f"{row['skip_cv_auc']:.5f} | {row['skip_converged']} | "
            f"{row['wr_converged']} |\n"
        )
    with open(REPORT_OUT, "w", encoding="utf-8") as handle:
        handle.writelines(lines)


def main():
    started = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    sample, scales = build_sample()
    rows = []
    for label, column in DEFINITIONS:
        xcol = column + "_z"
        print(f"Fitting {label}", flush=True)
        row = scales.loc[column].to_dict()
        row.update({"label": label, "col": column})
        row.update(fit_skip_gee(sample, xcol))
        row.update(fit_watch_mixed_model(sample, xcol))
        row.update(fit_skip_glm_and_auc(sample, xcol))
        rows.append(row)
        pd.DataFrame(rows).to_csv(COMPARISON_OUT, index=False)

    results = pd.DataFrame(rows)
    results.to_csv(COMPARISON_OUT, index=False)
    write_report(results, len(sample), int((sample["skip_flag"] == 0).sum()))
    print(f"Association models completed in {time.time() - started:.1f} seconds")


if __name__ == "__main__":
    main()
