"""Estimate fixed-effects threshold models for the reset-based exposures.

Threshold locations are selected by profile SSR on the full user population.
Uncertainty is evaluated by a 2,000-draw user cases bootstrap and a 200-draw
wild bootstrap. Influence is assessed by refitting after excluding the fifty
users with the largest leave-one-user-out change in the threshold statistic.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.abspath(os.path.join(HERE, ".."))
OUT_DIR = os.environ.get("REVIEW_RESULTS_DIR", os.path.join(PACKAGE_DIR, "results"))
FEATURES = os.environ.get("KUAIREC_FEATURES", os.path.join(OUT_DIR, "features_timegated.parquet"))
COMPARISON_OUT = os.path.join(OUT_DIR, "threshold_results.csv")
REPORT_OUT = os.path.join(OUT_DIR, "threshold_results.md")

sys.path.insert(0, HERE)
import estimators as est  # noqa: E402

TRIM_FRAC = 0.001
RANDOM_STATE = 42
N_CASES_BOOT = 2000
N_WILD_BOOT = 200
CONTINUOUS_QUANTILES = np.array(
    [0.01, 0.025, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
     0.60, 0.70, 0.80, 0.90, 0.95, 0.975, 0.99]
)

SKIP_DEFS = [
    ("(a) category-only", "consec_count"),
    ("(b) time-gated T=20s", "cc_timegated_T20"),
    ("(b) time-gated T=60s", "cc_timegated_T60"),
    ("(d) Jaccard theta=0.34", "cc_jac_th034"),
    ("(d) Jaccard theta=0.5", "cc_jac_th050"),
    ("(d) Jaccard theta=0.67", "cc_jac_th067"),
    ("(d) Jaccard theta=1.0", "cc_jac_th100"),
    ("(e) combined theta=0.5 & T=20s", "cc_combo_th050_T20"),
    ("(e) combined theta=0.34 & T=60s", "cc_combo_th034_T60"),
]

RESET_COLUMNS = {col for _, col in SKIP_DEFS}


def prepare_x(series, col):
    """Apply the pre-specified upper cap used for threshold estimation."""
    x = series.to_numpy(dtype=float)
    if col in RESET_COLUMNS:
        x = np.clip(x, 1, 10)
    return x


def threshold_candidates(x, col):
    """Return all admissible discrete threshold candidates."""
    if col in RESET_COLUMNS:
        return est._threshold_candidates(x, trim_frac=TRIM_FRAC)
    candidates = np.unique(np.quantile(x, CONTINUOUS_QUANTILES))
    # Retain candidates with enough observations on both sides.
    xs = np.sort(x)
    counts = np.searchsorted(xs, candidates, side="right")
    valid = ((counts >= TRIM_FRAC * len(x)) &
             (counts <= (1 - TRIM_FRAC) * len(x)) &
             (candidates > xs[0]) & (candidates < xs[-1]))
    return candidates[valid]


def solve_threshold(a11, a22, a12, b1, b2, yty):
    det = a11 * a22 - a12 ** 2
    valid = np.abs(det) > 1e-12
    beta1 = np.divide(b1 * a22 - b2 * a12, det,
                      out=np.full_like(det, np.nan, dtype=float), where=valid)
    beta2 = np.divide(a11 * b2 - a12 * b1, det,
                      out=np.full_like(det, np.nan, dtype=float), where=valid)
    ssr = np.where(valid, yty - beta1 * b1 - beta2 * b2, np.inf)
    return beta1, beta2, np.where(np.isfinite(ssr), np.maximum(ssr, 1e-12), np.inf)


def analyze_definition(x, y, codes, n_groups, user_ids, candidates):
    """Fit the threshold, cases bootstrap, and influence specification."""
    x_dm = est.group_demean(x, codes, n_groups)
    y_dm = est.group_demean(y, codes, n_groups)
    n_cand = len(candidates)
    if n_cand == 0:
        return None

    # User-level sufficient statistics for each candidate.
    user_a11 = np.zeros((n_cand, n_groups)); user_a22 = np.zeros((n_cand, n_groups))
    user_a12 = np.zeros((n_cand, n_groups)); user_b1 = np.zeros((n_cand, n_groups))
    user_b2 = np.zeros((n_cand, n_groups))
    for ci, g in enumerate(candidates):
        below = x <= g
        r1 = np.where(below, x, 0.0); r2 = np.where(~below, x, 0.0)
        r1_dm = est.group_demean(r1, codes, n_groups)
        r2_dm = est.group_demean(r2, codes, n_groups)
        user_a11[ci] = np.bincount(codes, weights=r1_dm ** 2, minlength=n_groups)
        user_a22[ci] = np.bincount(codes, weights=r2_dm ** 2, minlength=n_groups)
        user_a12[ci] = np.bincount(codes, weights=r1_dm * r2_dm, minlength=n_groups)
        user_b1[ci] = np.bincount(codes, weights=r1_dm * y_dm, minlength=n_groups)
        user_b2[ci] = np.bincount(codes, weights=r2_dm * y_dm, minlength=n_groups)
    user_yty = np.bincount(codes, weights=y_dm ** 2, minlength=n_groups)
    user_denom = np.bincount(codes, weights=x_dm ** 2, minlength=n_groups)
    user_num = np.bincount(codes, weights=x_dm * y_dm, minlength=n_groups)
    user_n = np.bincount(codes, minlength=n_groups)

    def fit_w(w):
        a11 = user_a11 @ w; a22 = user_a22 @ w; a12 = user_a12 @ w
        b1 = user_b1 @ w; b2 = user_b2 @ w; yty = float(user_yty @ w)
        beta1, beta2, ssr = solve_threshold(a11, a22, a12, b1, b2, yty)
        best = int(np.argmin(ssr))
        denom = float(user_denom @ w); num = float(user_num @ w)
        beta_lin = num / denom; ssr_lin = yty - beta_lin * num
        n_w = float(user_n @ w)
        F = n_w * (ssr_lin - ssr[best]) / ssr[best]
        return candidates[best], beta1[best], beta2[best], F

    w1 = np.ones(n_groups)
    gamma_hat, b_below, b_above, F_full = fit_w(w1)

    # Draw all user-level cases-bootstrap weights at once for matrix multiplication.
    rng = np.random.default_rng(RANDOM_STATE)
    probs = np.full(n_groups, 1.0 / n_groups)
    weights = rng.multinomial(n_groups, probs, size=N_CASES_BOOT).T.astype(float)
    a11_b = user_a11 @ weights
    a22_b = user_a22 @ weights
    a12_b = user_a12 @ weights
    b1_b = user_b1 @ weights
    b2_b = user_b2 @ weights
    yty_b = user_yty @ weights
    _, _, ssr_b = solve_threshold(a11_b, a22_b, a12_b, b1_b, b2_b, yty_b)
    best_b = np.argmin(ssr_b, axis=0)
    gboot = candidates[best_b]
    del weights, a11_b, a22_b, a12_b, b1_b, b2_b, yty_b, ssr_b
    g_lo, g_hi = np.percentile(gboot, [2.5, 97.5])

    # Rank users by the leave-one-user-out change in the threshold statistic.
    gi = list(candidates).index(gamma_hat)
    a11_g = user_a11[gi] @ w1; a22_g = user_a22[gi] @ w1; a12_g = user_a12[gi] @ w1
    b1_g = user_b1[gi] @ w1; b2_g = user_b2[gi] @ w1; yty_all = user_yty @ w1
    a11_l = a11_g - user_a11[gi]; a22_l = a22_g - user_a22[gi]; a12_l = a12_g - user_a12[gi]
    b1_l = b1_g - user_b1[gi]; b2_l = b2_g - user_b2[gi]; yty_l = yty_all - user_yty
    det_l = a11_l * a22_l - a12_l ** 2
    beta1_l = (b1_l * a22_l - b2_l * a12_l) / det_l
    beta2_l = (a11_l * b2_l - a12_l * b1_l) / det_l
    ssr_thr_l = np.maximum(yty_l - beta1_l * b1_l - beta2_l * b2_l, 1e-12)
    denom_l = (user_denom @ w1) - user_denom; num_l = (user_num @ w1) - user_num
    ssr_lin_l = yty_l - (num_l / denom_l) * num_l
    n_l = (user_n @ w1) - user_n
    F_loo = n_l * (ssr_lin_l - ssr_thr_l) / ssr_thr_l
    delta_F = F_full - F_loo
    order = np.argsort(-delta_F)
    top50 = order[:min(50, max(n_groups - 1, 0))]

    w_excl = w1.copy(); w_excl[top50] = 0.0
    gamma_excl, b_below_excl, b_above_excl, F_excl = fit_w(w_excl)

    return {"candidates": list(candidates), "gamma_hat": gamma_hat, "beta_below": b_below,
            "beta_above": b_above, "F_full": F_full, "gamma_ci_lo": g_lo, "gamma_ci_hi": g_hi,
            "gamma_boot": gboot, "gamma_excl50": gamma_excl, "beta_below_excl50": b_below_excl,
            "beta_above_excl50": b_above_excl, "F_excl50": F_excl,
            "top50_users": user_ids[top50].tolist()}


if __name__ == "__main__":
    t_start = time.time()
    print("Loading features_timegated.parquet ...")
    need = ["user_id", "skip_flag", "delta_t", "watch_ratio"] + [c for _, c in SKIP_DEFS]
    df = pd.read_parquet(FEATURES, columns=list(dict.fromkeys(need)))
    df = df.dropna(subset=["delta_t"]).copy()
    print(f"rows: {len(df):,}, users: {df['user_id'].nunique():,}")

    codes, uniques = pd.factorize(df["user_id"])
    n_groups = len(uniques)
    y_skip = df["skip_flag"].to_numpy(dtype=float)

    rows = []
    for label, col in SKIP_DEFS:
        print(f"\n{'='*70}\n{label} ({col}) — skip_flag\n{'='*70}")
        t0 = time.time()
        x = prepare_x(df[col], col)
        candidates = threshold_candidates(x, col)
        res = analyze_definition(x, y_skip, codes, n_groups, uniques, candidates)
        if res is None:
            print("  no admissible threshold candidates")
            continue
        gamma_pct = 100.0 * np.mean(x <= res["gamma_hat"])
        gamma_tail = 100.0 - gamma_pct
        ci_pct_lo = 100.0 * np.mean(x <= res["gamma_ci_lo"])
        ci_pct_hi = 100.0 * np.mean(x <= res["gamma_ci_hi"])
        print(f"  candidates={np.round(res['candidates'], 6).tolist()}")
        print(f"  gamma_hat={res['gamma_hat']:.6g} (CDF {gamma_pct:.2f}%ile, "
              f"upper tail {gamma_tail:.2f}%), beta_below={res['beta_below']:.5f}, "
              f"beta_above={res['beta_above']:.5f}, F={res['F_full']:.1f}")
        print(f"  gamma 95% CI ({N_CASES_BOOT} cases-bootstrap draws): "
              f"[{res['gamma_ci_lo']:.1f}, {res['gamma_ci_hi']:.1f}]")
        gvc = pd.Series(res["gamma_boot"]).value_counts(normalize=True).sort_index()
        print(f"  gamma bootstrap distribution: "
              f"{dict((round(float(k), 6), round(v, 3)) for k, v in gvc.items())}")
        gamma_excl_pct = 100.0 * np.mean(x <= res["gamma_excl50"])
        print(f"  [excluding 50 influential users] gamma_hat={res['gamma_excl50']:.6g} "
              f"(CDF {gamma_excl_pct:.2f}%ile), "
              f"beta_below={res['beta_below_excl50']:.5f}, beta_above={res['beta_above_excl50']:.5f}, "
              f"F={res['F_excl50']:.1f}")
        model_df = pd.DataFrame({"user_id": df["user_id"].to_numpy(), "x": x, "y": y_skip})
        wild = est.hansen_bootstrap_test(
            model_df, n_bootstrap=N_WILD_BOOT, seed=RANDOM_STATE, trim_frac=TRIM_FRAC
        )
        print(f"  wild bootstrap ({N_WILD_BOOT} draws): p={wild['p_value']:.4f}")
        print(f"  (took {time.time()-t0:.1f}s)")

        rows.append({"label": label, "col": col,
                     "candidates": str(np.round(res["candidates"], 8).tolist()),
                     "gamma_hat": res["gamma_hat"], "beta_below": res["beta_below"],
                     "beta_above": res["beta_above"], "F_full": res["F_full"],
                     "gamma_ci_lo": res["gamma_ci_lo"], "gamma_ci_hi": res["gamma_ci_hi"],
                     "wild_p": wild["p_value"],
                     "gamma_percentile": gamma_pct, "gamma_upper_tail_pct": gamma_tail,
                     "gamma_ci_percentile_lo": ci_pct_lo, "gamma_ci_percentile_hi": ci_pct_hi,
                     "gamma_excl50": res["gamma_excl50"],
                     "gamma_excl50_percentile": gamma_excl_pct,
                     "beta_below_excl50": res["beta_below_excl50"],
                     "beta_above_excl50": res["beta_above_excl50"], "F_excl50": res["F_excl50"]})
        pd.DataFrame(rows).to_csv(COMPARISON_OUT, index=False)

    results = pd.DataFrame(rows)
    results.to_csv(COMPARISON_OUT, index=False)

    rep = ["# Threshold analysis\n\n"]
    rep.append(f"Full population: {n_groups:,} users and {len(df):,} observations. "
               f"The trimming fraction was {TRIM_FRAC}.\n\n")
    rep.append(f"Threshold confidence intervals use {N_CASES_BOOT:,} user-level cases "
               f"bootstrap draws. P-values use {N_WILD_BOOT} wild-bootstrap draws.\n\n")
    rep.append("| Exposure | Threshold | 95% CI | Slope below | Slope above | F | Wild p | "
               "Threshold after excluding 50 users |\n"
               "|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in rows:
        rep.append(f"| {r['label']} | {r['gamma_hat']:.6g} | "
                   f"[{r['gamma_ci_lo']:.6g}, {r['gamma_ci_hi']:.6g}] | "
                   f"{r['beta_below']:.5f} | {r['beta_above']:.5f} | "
                   f"{r['F_full']:.1f} | {r['wild_p']:.4f} | {r['gamma_excl50']:.6g} |\n")

    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        f.writelines(rep)
    print(f"\nSaved threshold results ({time.time()-t_start:.1f}s)")
