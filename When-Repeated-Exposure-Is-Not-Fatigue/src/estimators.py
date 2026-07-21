"""Fixed-effects threshold estimation utilities."""

import numpy as np
import pandas as pd


def factorize_groups(df, groupcol="user_id"):
    codes, groups = pd.factorize(df[groupcol])
    return codes.astype(np.int64), len(groups)


def group_demean(values, codes, n_groups):
    values = np.asarray(values, dtype=float)
    sums = np.bincount(codes, weights=values, minlength=n_groups)
    counts = np.bincount(codes, minlength=n_groups)
    return values - (sums / np.maximum(counts, 1))[codes]


def _threshold_candidates(x, trim_frac=0.05):
    """Return unique split values with adequate observations on both sides."""
    x_sorted = np.sort(np.asarray(x, dtype=float))
    unique = np.unique(x_sorted)
    if len(unique) < 2:
        return np.array([])
    counts = np.searchsorted(x_sorted, unique, side="right")
    lower = trim_frac * len(x_sorted)
    upper = (1 - trim_frac) * len(x_sorted)
    valid = (counts >= lower) & (counts <= upper) & (unique != unique[-1])
    return unique[valid]


def _threshold_fit_at_gamma(x, y_dm, gamma, codes, n_groups):
    below = x <= gamma
    x1 = group_demean(np.where(below, x, 0.0), codes, n_groups)
    x2 = group_demean(np.where(~below, x, 0.0), codes, n_groups)

    a11 = float(x1 @ x1)
    a22 = float(x2 @ x2)
    a12 = float(x1 @ x2)
    b1 = float(x1 @ y_dm)
    b2 = float(x2 @ y_dm)
    determinant = a11 * a22 - a12 * a12
    if abs(determinant) < 1e-10:
        return None

    beta1 = (b1 * a22 - b2 * a12) / determinant
    beta2 = (a11 * b2 - a12 * b1) / determinant
    ssr = float(y_dm @ y_dm) - beta1 * b1 - beta2 * b2
    return beta1, beta2, max(ssr, 0.0)


def _threshold_search(x, y, codes, n_groups, trim_frac=0.05):
    y_dm = group_demean(y, codes, n_groups)
    best = None
    for gamma in _threshold_candidates(x, trim_frac):
        fit = _threshold_fit_at_gamma(x, y_dm, gamma, codes, n_groups)
        if fit is None:
            continue
        beta1, beta2, ssr = fit
        if best is None or ssr < best[3]:
            best = (gamma, beta1, beta2, ssr)
    return best


def hansen_bootstrap_test(
    df,
    xcol="x",
    ycol="y",
    groupcol="user_id",
    n_bootstrap=500,
    seed=None,
    trim_frac=0.05,
):
    """Test a fixed-effects linear model against a two-slope threshold model."""
    rng = np.random.default_rng(seed)
    codes, n_groups = factorize_groups(df, groupcol)
    x = df[xcol].to_numpy(dtype=float)
    y = df[ycol].to_numpy(dtype=float)
    x_dm = group_demean(x, codes, n_groups)
    y_dm = group_demean(y, codes, n_groups)

    denominator = float(x_dm @ x_dm)
    beta_linear = float(x_dm @ y_dm) / denominator if denominator > 1e-12 else 0.0
    fitted = beta_linear * x_dm
    residuals = y_dm - fitted
    ssr_linear = float(residuals @ residuals)

    best = _threshold_search(x, y, codes, n_groups, trim_frac)
    if best is None:
        return None
    gamma, beta_below, beta_above, ssr_threshold = best
    observed = len(y) * (ssr_linear - ssr_threshold) / max(ssr_threshold, 1e-12)

    candidates = _threshold_candidates(x, trim_frac)
    bootstrap_f = np.empty(n_bootstrap)
    for draw in range(n_bootstrap):
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(y))
        y_star = fitted + residuals * signs
        beta_star = float(x_dm @ y_star) / denominator if denominator > 1e-12 else 0.0
        linear_residuals = y_star - beta_star * x_dm
        ssr_linear_star = float(linear_residuals @ linear_residuals)
        best_ssr = np.inf
        for candidate in candidates:
            fit = _threshold_fit_at_gamma(x, y_star, candidate, codes, n_groups)
            if fit is not None:
                best_ssr = min(best_ssr, fit[2])
        if not np.isfinite(best_ssr) or best_ssr <= 0:
            bootstrap_f[draw] = 0.0
        else:
            bootstrap_f[draw] = len(y) * (ssr_linear_star - best_ssr) / best_ssr

    return {
        "gamma_hat": float(gamma),
        "beta_below": beta_below,
        "beta_above": beta_above,
        "F_stat": float(observed),
        "p_value": float(np.mean(bootstrap_f >= observed)),
        "ssr_linear": ssr_linear,
        "ssr_threshold": ssr_threshold,
        "n": len(y),
        "n_bootstrap": n_bootstrap,
        "boot_F": bootstrap_f,
    }
