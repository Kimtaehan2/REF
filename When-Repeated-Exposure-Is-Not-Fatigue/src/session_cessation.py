"""Estimate associations with inferred session cessation.

Cessation is defined from the inactivity gap following the end of playback.
The primary boundary is the intersection of the two slowest components in the
BIC-selected Gaussian mixture fitted to positive log gaps. The outcome is an
inferred session boundary because KuaiRec does not contain application-exit
events.
"""
import ast
import gc
import os
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.generalized_estimating_equations import GEE

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.abspath(os.path.join(HERE, ".."))
DATA_DIR = os.environ.get("KUAIREC_DATA_DIR", os.path.join(PACKAGE_DIR, "data"))
OUT_DIR = os.environ.get("REVIEW_RESULTS_DIR", os.path.join(PACKAGE_DIR, "results"))
FEATURES = os.environ.get("KUAIREC_FEATURES", os.path.join(OUT_DIR, "features_timegated.parquet"))

GAP_MIXTURE_OUT = os.path.join(OUT_DIR, "session_gap_mixture.csv")
PRIMARY_OUT = os.path.join(OUT_DIR, "session_cessation_results.csv")
SENSITIVITY_OUT = os.path.join(OUT_DIR, "session_boundary_sensitivity.csv")
REPORT_OUT = os.path.join(OUT_DIR, "session_cessation_results.md")

RANDOM_STATE = 42
N_USERS_SAMPLE = 1000
GMM_SAMPLE_SIZE = 800_000
GMM_COMPONENT_GRID = [2, 3, 4]
FIXED_THRESHOLDS_SECONDS = [900.0, 1800.0, 3600.0]
N_CV_FOLDS = 5

# Measures evaluated at the primary session boundary.
PRIMARY_DEFINITIONS = [
    ("(a) category-only streak", "a_streak"),
    ("(b) time-gated T=20s", "b_time20"),
    ("(c) burst T=20s", "c_burst20"),
    ("(d) Jaccard >= 0.5", "d_jac050"),
    ("(e) Jaccard >= 0.5 & T=20s", "e_combo050_T20"),
    ("(f-1) inferred-session cumulative", "f1_session_cum"),
    ("(f-2) rolling cumulative W=30m", "cumexp_W30m"),
    ("(f-2) rolling cumulative W=60m", "cumexp_W60m"),
    ("(f-2) rolling cumulative W=180m", "cumexp_W180m"),
    ("(g) sliding density N=20", "density_N20"),
    ("(g) sliding density N=50", "density_N50"),
    ("(g) sliding density N=100", "density_N100"),
    ("(h) time-decay gauge tau=1800s", "gauge_hl1800"),
    ("(h) time-decay gauge tau=7200s", "gauge_hl7200"),
    ("(h) time-decay gauge tau=86400s", "gauge_hl86400"),
    ("(i) event-decay T=20s penalty=0.2", "ed_T20_p02"),
    ("(i) event-decay T=20s penalty=0.5", "ed_T20_p05"),
    ("(i) event-decay T=20s penalty=1.0", "ed_T20_p10"),
    ("(i) event-decay T=60s penalty=0.2", "ed_T60_p02"),
    ("(i) event-decay T=60s penalty=0.5", "ed_T60_p05"),
    ("(i) event-decay T=60s penalty=1.0", "ed_T60_p10"),
]

# One representative measure per family for boundary sensitivity.
SENSITIVITY_DEFINITIONS = [
    ("(a)", "a_streak"),
    ("(b)", "b_time20"),
    ("(c)", "c_burst20"),
    ("(d)", "d_jac050"),
    ("(e)", "e_combo050_T20"),
    ("(f-1)", "f1_session_cum"),
    ("(f-2)", "cumexp_W30m"),
    ("(g)", "density_N20"),
    ("(h)", "gauge_hl1800"),
]

STATIC_EXPOSURES = [
    "cumexp_W30m", "cumexp_W60m", "cumexp_W180m",
    "density_N20", "density_N50", "density_N100",
    "gauge_hl1800", "gauge_hl7200", "gauge_hl86400",
    "ed_T20_p02", "ed_T20_p05", "ed_T20_p10",
    "ed_T60_p02", "ed_T60_p05", "ed_T60_p10",
]

BASE_CONTROLS = [
    "video_duration_z", "popularity_z", "session_position_log_z",
    "prev_idle_log_z", "hour_sin", "hour_cos", "weekend",
]


def zscore(values):
    values = np.asarray(values, dtype=np.float64)
    sd = np.nanstd(values, ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError(f"Cannot standardize a variable with SD={sd}")
    return (values - np.nanmean(values)) / sd


def streak_from_keep(keep):
    """Return streak lengths, continuing where ``keep`` is true."""
    keep = np.asarray(keep, dtype=bool)
    idx = np.arange(len(keep), dtype=np.int64)
    last_reset = np.maximum.accumulate(np.where(~keep, idx, -1))
    return (idx - last_reset + 1).astype(np.int32)


def popcount_int64(arr):
    """Count set bits without depending on a NumPy-specific bit-count API."""
    arr = np.asarray(arr, dtype=np.uint64)
    byte_view = arr.view(np.uint8).reshape(-1, 8)
    lookup = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    return lookup[byte_view].sum(axis=1).astype(np.int16)


def posterior_intersection(m1, s1, w1, m2, s2, w2):
    """Return the weighted-density intersection between two component means."""
    a = 1.0 / (2 * s2 * s2) - 1.0 / (2 * s1 * s1)
    b = m1 / (s1 * s1) - m2 / (s2 * s2)
    c = (m2 * m2 / (2 * s2 * s2) - m1 * m1 / (2 * s1 * s1)
         + np.log((w1 / s1) / (w2 / s2)))
    if np.isclose(a, 0):
        roots = np.array([-c / b])
    else:
        roots = np.roots([a, b, c])
    roots = np.real(roots[np.isreal(roots)])
    between = roots[(roots > min(m1, m2)) & (roots < max(m1, m2))]
    if len(between):
        return float(between[0])
    return float((m1 + m2) / 2)


def estimate_gap_threshold(idle_gap):
    positive = idle_gap[np.isfinite(idle_gap) & (idle_gap > 0)]
    rng = np.random.default_rng(RANDOM_STATE)
    sample = rng.choice(positive, size=min(GMM_SAMPLE_SIZE, len(positive)), replace=False)
    log_gap = np.log10(sample).reshape(-1, 1)
    rows, models = [], {}
    for k in GMM_COMPONENT_GRID:
        model = GaussianMixture(
            n_components=k, covariance_type="full", random_state=RANDOM_STATE,
            n_init=3, max_iter=300,
        ).fit(log_gap)
        bic = model.bic(log_gap)
        order = np.argsort(model.means_.ravel())
        models[k] = model
        for rank, comp in enumerate(order, start=1):
            rows.append({
                "n_components": k,
                "bic": bic,
                "component_rank": rank,
                "mean_log10_seconds": model.means_.ravel()[comp],
                "geometric_mean_seconds": 10 ** model.means_.ravel()[comp],
                "sd_log10": np.sqrt(model.covariances_.ravel()[comp]),
                "weight": model.weights_[comp],
            })
    best_k = min(models, key=lambda k: models[k].bic(log_gap))
    best = models[best_k]
    order = np.argsort(best.means_.ravel())
    c1, c2 = order[-2], order[-1]
    boundary_log10 = posterior_intersection(
        best.means_.ravel()[c1], np.sqrt(best.covariances_.ravel()[c1]), best.weights_[c1],
        best.means_.ravel()[c2], np.sqrt(best.covariances_.ravel()[c2]), best.weights_[c2],
    )
    threshold = 10 ** boundary_log10
    result = pd.DataFrame(rows)
    result["selected_model"] = result["n_components"].eq(best_k)
    result["selected_boundary_seconds"] = threshold
    return threshold, best_k, result


def build_user_adaptive_thresholds(raw, global_threshold):
    """Construct shrinkage-based user-specific boundaries for sensitivity analysis."""
    active = raw.loc[
        np.isfinite(raw["idle_gap"]) & (raw["idle_gap"] > 0)
        & (raw["idle_gap"] < global_threshold), ["user_id", "idle_gap"]
    ].copy()
    active["log_gap"] = np.log(active["idle_gap"])
    global_center = active["log_gap"].median()
    stats = active.groupby("user_id", sort=False)["log_gap"].agg(["median", "count"])
    weight = stats["count"] / (stats["count"] + 100.0)
    log_t = np.log(global_threshold) + weight * (stats["median"] - global_center)
    stats["threshold_seconds"] = np.exp(log_t).clip(900.0, 7200.0)
    return stats["threshold_seconds"].to_dict(), stats.reset_index(), global_center


def load_raw_and_estimate_threshold(reuse_saved=False):
    print("Loading KuaiRec big_matrix.csv for end-adjusted idle gaps ...")
    raw = pd.read_csv(
        os.path.join(DATA_DIR, "big_matrix.csv"),
        usecols=["user_id", "video_id", "timestamp", "play_duration"],
        dtype={"user_id": "int32", "video_id": "int32", "timestamp": "float64",
               "play_duration": "float32"},
    )
    raw = raw.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    raw["next_timestamp"] = raw.groupby("user_id", sort=False)["timestamp"].shift(-1)
    raw["idle_gap"] = (raw["next_timestamp"] - raw["timestamp"]
                       - raw["play_duration"] / 1000.0)
    known = raw["next_timestamp"].notna()
    positive = known & (raw["idle_gap"] > 0)
    nonpositive = known & (raw["idle_gap"] <= 0)
    print(f"rows={len(raw):,}, known next={known.sum():,}, positive idle={positive.sum():,}, "
          f"nonpositive/overlap={nonpositive.sum():,}, right-censored={(~known).sum():,}")

    if reuse_saved and os.path.exists(GAP_MIXTURE_OUT):
        mixture = pd.read_csv(GAP_MIXTURE_OUT)
        threshold = float(mixture["selected_boundary_seconds"].iloc[0])
        selected = mixture.loc[mixture["selected_model"], "n_components"]
        best_k = int(selected.iloc[0])
        print(f"Reusing saved GMM k={best_k}, data-derived boundary={threshold:.2f}s "
              f"({threshold/60:.2f}min); existing (a)~(h) comparison boundary preserved")
    else:
        threshold, best_k, mixture = estimate_gap_threshold(raw["idle_gap"].to_numpy())
        mixture.to_csv(GAP_MIXTURE_OUT, index=False)
        print(f"Selected GMM k={best_k}, data-derived boundary={threshold:.2f}s "
              f"({threshold/60:.2f}min)")
    adaptive_map, adaptive_stats, active_center = build_user_adaptive_thresholds(raw, threshold)
    print("Adaptive threshold minutes quantiles:",
          (adaptive_stats["threshold_seconds"] / 60).quantile([0, .1, .5, .9, 1]).to_dict())
    return raw, threshold, best_k, mixture, adaptive_map, adaptive_stats, active_center


def load_analysis_sample(raw):
    rng = np.random.default_rng(RANDOM_STATE)
    users = raw["user_id"].unique()
    sampled_users = np.sort(rng.choice(users, size=min(N_USERS_SAMPLE, len(users)), replace=False))
    raw_sample = raw[raw["user_id"].isin(sampled_users)].copy()
    raw_sample = raw_sample.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(drop=True)

    cols = [
        "user_id", "video_id", "timestamp", "skip_flag", "watch_ratio",
        "primary_category", "video_duration", "popularity",
    ] + STATIC_EXPOSURES
    print("Loading sampled past-only exposure features ...")
    try:
        feat = pd.read_parquet(FEATURES, columns=cols,
                               filters=[("user_id", "in", sampled_users.tolist())])
    except Exception as exc:
        print(f"Parquet filter fallback ({exc})")
        feat = pd.read_parquet(FEATURES, columns=cols)
        feat = feat[feat["user_id"].isin(sampled_users)]
    feat = feat.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(drop=True)

    if len(feat) != len(raw_sample):
        raise ValueError(f"Raw/feature row mismatch: {len(raw_sample):,} vs {len(feat):,}")
    keys_ok = (
        np.array_equal(feat["user_id"].to_numpy(), raw_sample["user_id"].to_numpy())
        and np.array_equal(feat["video_id"].to_numpy(), raw_sample["video_id"].to_numpy())
        and np.allclose(feat["timestamp"].to_numpy(), raw_sample["timestamp"].to_numpy(),
                        rtol=0, atol=1e-6)
    )
    if not keys_ok:
        raise ValueError("Raw and feature sort keys do not match")
    feat["play_duration"] = raw_sample["play_duration"].to_numpy()
    feat["next_timestamp"] = raw_sample["next_timestamp"].to_numpy()
    feat["idle_gap"] = raw_sample["idle_gap"].to_numpy()

    # Prior inactivity is available at the current interaction; current inactivity defines the outcome.
    feat["prev_idle_gap"] = feat.groupby("user_id", sort=False)["idle_gap"].shift(1)
    feat["start_gap"] = feat.groupby("user_id", sort=False)["timestamp"].diff()
    # Convert Unix timestamps to China Standard Time for time-of-day and weekend controls.
    ts = pd.to_datetime(feat["timestamp"], unit="s") + pd.Timedelta(hours=8)
    hour = ts.dt.hour.to_numpy() + ts.dt.minute.to_numpy() / 60.0
    feat["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    feat["weekend"] = (ts.dt.dayofweek.to_numpy() >= 5).astype(np.int8)

    # Exclude observations without a subsequent interaction or prior inactivity gap.
    feat = feat[feat["next_timestamp"].notna() & feat["prev_idle_gap"].notna()].copy()
    feat["popularity_log"] = np.log1p(feat["popularity"].clip(lower=0))
    feat["video_duration_z"] = zscore(feat["video_duration"])
    feat["popularity_z"] = zscore(feat["popularity_log"])
    feat["prev_idle_log_z"] = zscore(np.log1p(feat["prev_idle_gap"].clip(lower=0)))
    print(f"analysis sample rows={len(feat):,}, users={feat['user_id'].nunique():,}")
    return feat.reset_index(drop=True), sampled_users


def load_video_masks():
    ic = pd.read_csv(os.path.join(DATA_DIR, "item_categories.csv"))
    ic["mask"] = ic["feat"].map(
        lambda x: int(sum(1 << int(tag) for tag in ast.literal_eval(x)))
    )
    return dict(zip(ic["video_id"], ic["mask"]))


def dynamic_exposures(df, thresholds, vid2mask):
    """Recompute session-dependent exposures using past information only."""
    n = len(df)
    user = df["user_id"].to_numpy()
    cat = df["primary_category"].to_numpy()
    prev_idle = df["prev_idle_gap"].to_numpy()
    start_gap = df["start_gap"].to_numpy()
    if np.ndim(thresholds) == 0:
        threshold_arr = np.full(n, float(thresholds))
    else:
        threshold_arr = np.asarray(thresholds, dtype=np.float64)
    first_user = np.r_[True, user[1:] != user[:-1]]
    new_session = first_user | (prev_idle >= threshold_arr)
    session_id = np.cumsum(new_session).astype(np.int64)
    session_position = (pd.Series(session_id).groupby(session_id, sort=False).cumcount()
                        .to_numpy(dtype=np.int32) + 1)

    same_cat = np.r_[False, (user[1:] == user[:-1]) & (cat[1:] == cat[:-1])]
    cur_mask = df["video_id"].map(vid2mask).fillna(0).to_numpy(dtype=np.uint64)
    prev_mask = np.r_[np.uint64(0), cur_mask[:-1]]
    inter = popcount_int64(cur_mask & prev_mask)
    union = popcount_int64(cur_mask | prev_mask)
    jaccard = np.divide(inter, union, out=np.zeros(n, dtype=np.float64), where=union > 0)
    valid_prev = ~first_user & ~new_session

    out = {
        "a_streak": streak_from_keep(valid_prev & same_cat),
        "b_time20": streak_from_keep(valid_prev & same_cat & (start_gap <= 20)),
        "c_burst20": streak_from_keep(valid_prev & (start_gap <= 20)),
        "d_jac050": streak_from_keep(valid_prev & (jaccard >= 0.5)),
        "e_combo050_T20": streak_from_keep(valid_prev & (jaccard >= 0.5)
                                             & (start_gap <= 20)),
    }
    tmp = pd.DataFrame({"session_id": session_id, "category": cat})
    out["f1_session_cum"] = (
        tmp.groupby(["session_id", "category"], sort=False).cumcount().to_numpy(dtype=np.int32) + 1
    )
    return out, session_position, int(session_id[-1] - session_id[0] + 1)


def run_gee(df, outcome, xcol, add_skip=False):
    controls = BASE_CONTROLS + (["skip_flag"] if add_skip else [])
    formula = f"{outcome} ~ {xcol} + " + " + ".join(controls)
    res = GEE.from_formula(
        formula, groups="user_id", data=df, family=Binomial(),
        cov_struct=Exchangeable(),
    ).fit(maxiter=200)
    ci = res.conf_int().loc[xcol]
    return {
        "coef": res.params[xcol], "OR": np.exp(res.params[xcol]),
        "p": res.pvalues[xcol], "ci_lo": np.exp(ci.iloc[0]),
        "ci_hi": np.exp(ci.iloc[1]),
        "converged": bool(getattr(res, "converged", True)),
    }


def cv_auc_increment(df, xcol):
    """Compare controls-only and exposure models on identical user folds."""
    y = df["session_exit"].to_numpy(dtype=np.int8)
    groups = df["user_id"].to_numpy()
    base_cols = BASE_CONTROLS
    X0 = df[base_cols].to_numpy(dtype=np.float64)
    X1 = df[base_cols + [xcol]].to_numpy(dtype=np.float64)
    folds = GroupKFold(n_splits=N_CV_FOLDS)
    auc0, auc1 = [], []
    for train, test in folds.split(X0, y, groups):
        m0 = LogisticRegression(max_iter=500, C=1e6, solver="lbfgs")
        m1 = LogisticRegression(max_iter=500, C=1e6, solver="lbfgs")
        m0.fit(X0[train], y[train])
        m1.fit(X1[train], y[train])
        auc0.append(roc_auc_score(y[test], m0.predict_proba(X0[test])[:, 1]))
        auc1.append(roc_auc_score(y[test], m1.predict_proba(X1[test])[:, 1]))
    return np.mean(auc0), np.mean(auc1), np.std(auc1), np.mean(np.array(auc1) - np.array(auc0))


def pooled_glm_aic(df, xcol):
    X = sm.add_constant(df[BASE_CONTROLS + [xcol]], has_constant="add")
    return sm.GLM(df["session_exit"], X, family=sm.families.Binomial()).fit().aic


def prepare_scheme(df, threshold_values, vid2mask):
    dynamic, session_position, n_sessions = dynamic_exposures(df, threshold_values, vid2mask)
    for col, values in dynamic.items():
        df[col] = values
    df["session_position_log_z"] = zscore(np.log1p(session_position))
    threshold_arr = (np.full(len(df), float(threshold_values)) if np.ndim(threshold_values) == 0
                     else np.asarray(threshold_values, dtype=np.float64))
    df["session_exit"] = (df["idle_gap"].to_numpy() >= threshold_arr).astype(np.int8)
    return n_sessions


def main():
    started = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    raw, threshold, best_k, mixture, adaptive_map, adaptive_stats, active_center = (
        load_raw_and_estimate_threshold(reuse_saved=False)
    )
    df, sampled_users = load_analysis_sample(raw)
    del raw
    gc.collect()
    vid2mask = load_video_masks()

    user_threshold = df["user_id"].map(adaptive_map).fillna(threshold).to_numpy()
    schemes = [
        ("15m", 900.0),
        ("30m", 1800.0),
        (f"data-derived {threshold/60:.2f}m", float(threshold)),
        ("60m", 3600.0),
        ("user-adaptive", user_threshold),
    ]
    primary_rows, sensitivity_rows = [], []
    primary_scheme = f"data-derived {threshold/60:.2f}m"
    for scheme_label, threshold_values in schemes:
        print(f"\n{'='*80}\nSession boundary: {scheme_label}\n{'='*80}")
        n_sessions = prepare_scheme(df, threshold_values, vid2mask)
        exit_rate = df["session_exit"].mean()
        print(f"sessions={n_sessions:,}, exit rate={exit_rate:.4%}")

        # Reuse primary-boundary fits when the same measure appears below.
        sensitivity_cache = {}
        for family, col in SENSITIVITY_DEFINITIONS:
            xcol = col + "_z"
            df[xcol] = zscore(df[col])
            print(f"  sensitivity {family} ({col}) ...", end="", flush=True)
            gee = run_gee(df, "session_exit", xcol, add_skip=False)
            print(f" OR={gee['OR']:.4f}, p={gee['p']:.3g}, conv={gee['converged']}")
            row = {
                "boundary": scheme_label,
                "threshold_seconds": (float(threshold_values)
                                      if np.ndim(threshold_values) == 0 else np.nan),
                "n_sessions": n_sessions, "exit_rate": exit_rate,
                "family": family, "exposure": col,
                "x_mean": df[col].mean(), "x_std": df[col].std(),
                "OR": gee["OR"], "p": gee["p"],
                "ci_lo": gee["ci_lo"], "ci_hi": gee["ci_hi"],
                "converged": gee["converged"],
            }
            sensitivity_rows.append(row)
            sensitivity_cache[col] = gee
            pd.DataFrame(sensitivity_rows).to_csv(SENSITIVITY_OUT, index=False)

        if scheme_label != primary_scheme:
            continue

        # Primary boundary: full measure grid, skip adjustment, AIC, and grouped AUC.
        controls_only_X = sm.add_constant(df[BASE_CONTROLS], has_constant="add")
        controls_aic = sm.GLM(
            df["session_exit"], controls_only_X, family=sm.families.Binomial()
        ).fit().aic
        for label, col in PRIMARY_DEFINITIONS:
            print(f"  primary {label} ...", flush=True)
            xcol = col + "_z"
            if xcol not in df:
                df[xcol] = zscore(df[col])
            gee = sensitivity_cache.get(col) or run_gee(df, "session_exit", xcol, add_skip=False)
            gee_skip = run_gee(df, "session_exit", xcol, add_skip=True)
            aic = pooled_glm_aic(df, xcol)
            auc0, auc1, auc_sd, delta_auc = cv_auc_increment(df, xcol)
            primary_rows.append({
                "label": label, "exposure": col,
                "x_mean": df[col].mean(), "x_std": df[col].std(),
                "exit_OR": gee["OR"], "exit_p": gee["p"],
                "exit_ci_lo": gee["ci_lo"], "exit_ci_hi": gee["ci_hi"],
                "exit_converged": gee["converged"],
                "skip_adjusted_OR": gee_skip["OR"], "skip_adjusted_p": gee_skip["p"],
                "skip_adjusted_ci_lo": gee_skip["ci_lo"],
                "skip_adjusted_ci_hi": gee_skip["ci_hi"],
                "skip_adjusted_converged": gee_skip["converged"],
                "controls_aic": controls_aic, "exposure_aic": aic,
                "delta_aic": aic - controls_aic,
                "controls_cv_auc": auc0, "exposure_cv_auc": auc1,
                "exposure_cv_auc_sd": auc_sd, "delta_cv_auc": delta_auc,
            })
            pd.DataFrame(primary_rows).to_csv(PRIMARY_OUT, index=False)

    primary = pd.DataFrame(primary_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)
    primary.to_csv(PRIMARY_OUT, index=False)
    sensitivity.to_csv(SENSITIVITY_OUT, index=False)

    report = ["# Inferred session cessation\n\n"]
    report.append(
        "The outcome is based on `next_timestamp - timestamp - play_duration/1000`. "
        "The final interaction for each user is excluded because the next interaction "
        "is unobserved. Current inactivity gaps are used only to define the outcome.\n\n"
    )
    report.append(f"The sample contains {len(sampled_users):,} users and {len(df):,} "
                  f"observations (seed {RANDOM_STATE}). The BIC-selected mixture has "
                  f"{best_k} components and implies a boundary of {threshold:.1f} seconds "
                  f"({threshold/60:.2f} minutes).\n\n")
    report.append("| Component rank | Geometric mean (s) | log10 SD | Weight |\n"
                  "|---:|---:|---:|---:|\n")
    chosen = mixture[mixture["selected_model"]].sort_values("component_rank")
    for _, r in chosen.iterrows():
        report.append(f"| {int(r['component_rank'])} | {r['geometric_mean_seconds']:.2f} | "
                      f"{r['sd_log10']:.3f} | {r['weight']:.4f} |\n")
    report.append("\n## Primary models\n\n")
    report.append("Effects are reported per one standard deviation of exposure. Models "
                  "control for video duration, popularity, log session position, prior "
                  "inactivity, time of day, and weekend status.\n\n")
    report.append("| Exposure | Cessation OR | p | Skip-adjusted OR | p | Delta AIC | "
                  "Grouped AUC | Delta AUC |\n"
                  "|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for _, r in primary.iterrows():
        report.append(
            f"| {r['label']} | {r['exit_OR']:.4f} | {r['exit_p']:.3g} | "
            f"{r['skip_adjusted_OR']:.4f} | {r['skip_adjusted_p']:.3g} | "
            f"{r['delta_aic']:+.1f} | {r['exposure_cv_auc']:.5f} | "
            f"{r['delta_cv_auc']:+.6f} |\n"
        )

    report.append("\n## Session-boundary sensitivity\n\n")
    pivot_or = sensitivity.pivot(index="family", columns="boundary", values="OR")
    report.append("| Exposure family | " + " | ".join(pivot_or.columns) + " |\n")
    report.append("|---|" + "---:|" * len(pivot_or.columns) + "\n")
    for family, r in pivot_or.iterrows():
        report.append(f"| {family} | " + " | ".join(f"{v:.4f}" for v in r) + " |\n")

    report.append("\nThese are observational associations. Long inactivity may also reflect "
                  "sleep, daily schedules, or activity outside the platform.\n")
    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        f.writelines(report)
    print(f"\nSaved session-exit outputs (total {time.time()-started:.1f}s)")


if __name__ == "__main__":
    main()
