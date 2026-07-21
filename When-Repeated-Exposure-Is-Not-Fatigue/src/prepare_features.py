"""Construct the repeated-exposure measures used in the paper.

The script reads the three KuaiRec source tables, orders interactions by user
and timestamp, derives the behavioral outcomes and controls, and writes one
Parquet file shared by the downstream analyses.
"""
import ast
import os
import time

import numpy as np
import pandas as pd

np.random.seed(42)

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.abspath(os.path.join(HERE, ".."))
DATA_DIR = os.environ.get("KUAIREC_DATA_DIR", os.path.join(PACKAGE_DIR, "data"))
OUT_DIR = os.environ.get("REVIEW_RESULTS_DIR", os.path.join(PACKAGE_DIR, "results"))
BASE_FEATURES = os.environ.get("KUAIREC_BASE_FEATURES")
os.makedirs(OUT_DIR, exist_ok=True)

T_GRID = [10, 20, 30, 60, 120]
THETA_GRID = [0.2, 0.34, 0.5, 0.67, 1.0]  # 0.34≈1/3, 0.67≈2/3
THETA_TAGS = {0.2: "020", 0.34: "034", 0.5: "050", 0.67: "067", 1.0: "100"}
COMBO_GRID = [(0.5, 20), (0.34, 60), (0.5, 60), (0.34, 20)]  # (e): theta x T
F_WINDOW_MINUTES = [30, 60, 180]
G_WINDOW_N = [20, 50, 100]
H_DECAY_SECONDS = [1800, 7200, 86400]
EVENT_DECAY_T = [20, 60]
EVENT_DECAY_PENALTIES = [0.2, 0.5, 1.0]


def popcount(arr):
    """Count set bits in each integer."""
    x = arr.astype(np.int64).copy()
    c = np.zeros(len(x), dtype=np.int64)
    while x.any():
        c += (x & 1)
        x >>= 1
    return c


def streak_from_keep(keep):
    """Return streak lengths, continuing where ``keep`` is true."""
    streak_id = np.cumsum(~keep)
    cc = pd.Series(np.zeros(len(streak_id), dtype=np.int64)).groupby(streak_id).cumcount().to_numpy()
    return (cc + 1).astype(np.int32)


def load_or_build_base_features():
    """Load an optional feature cache or reconstruct it from the source CSV files."""
    feature_path = BASE_FEATURES
    keep_cols = ["user_id", "video_id", "timestamp", "date", "watch_ratio", "delta_t",
                 "session_id", "session_position", "skip_flag", "strong_positive_flag",
                 "primary_category", "consec_count", "video_duration", "popularity"]
    if feature_path and os.path.exists(feature_path):
        print(f"Loading base feature cache: {feature_path}")
        return pd.read_parquet(feature_path, columns=keep_cols)

    print("Building base features from the KuaiRec source tables ...")
    raw_cols = ["user_id", "video_id", "timestamp", "date", "watch_ratio", "video_duration"]
    df = pd.read_csv(os.path.join(DATA_DIR, "big_matrix.csv"), usecols=raw_cols)
    df = df.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    df["delta_t"] = df.groupby("user_id", sort=False)["timestamp"].diff()
    new_session = df["delta_t"].isna() | (df["delta_t"] >= 1800)
    df["session_id"] = (new_session.cumsum() - 1).astype(np.int32)
    df["session_position"] = (df.groupby("session_id", sort=False).cumcount() + 1).astype(np.int32)
    df["skip_flag"] = (df["watch_ratio"] < 0.2).astype(np.int8)
    df["strong_positive_flag"] = (df["watch_ratio"] > 2.0).astype(np.int8)

    item_cat = pd.read_csv(os.path.join(DATA_DIR, "item_categories.csv"))
    item_cat["primary_category"] = item_cat["feat"].map(ast.literal_eval).map(lambda x: x[0])
    df = df.merge(item_cat[["video_id", "primary_category"]], on="video_id", how="left", sort=False)
    prev_cat = df.groupby("session_id", sort=False)["primary_category"].shift(1)
    same_as_prev = df["primary_category"].eq(prev_cat).to_numpy()
    df["consec_count"] = streak_from_keep(same_as_prev)

    daily = pd.read_csv(os.path.join(DATA_DIR, "item_daily_features.csv"),
                        usecols=["video_id", "date", "play_cnt"])
    daily = daily.rename(columns={"play_cnt": "popularity"})
    df = df.merge(daily, on=["video_id", "date"], how="left", sort=False)
    if df["primary_category"].isna().any() or df["popularity"].isna().any():
        raise ValueError("Missing category or popularity values after merging source tables")
    return df[keep_cols]


def reset_free_exposure_features(df):
    """Compute cumulative, density, and time-decayed exposure measures.

    Rolling counts and densities use grouped binary searches. The decay score
    uses the log-domain equivalent of
      gauge_i = exp(logaddexp.accumulate(log(b_k)+T_k/tau)_i - T_i/tau).
    """
    n = len(df)
    users = df["user_id"].to_numpy(dtype=np.int64)
    cats = df["primary_category"].to_numpy(dtype=np.int16)
    ts = df["timestamp"].to_numpy(dtype=np.float64)
    orig = np.arange(n, dtype=np.int64)

    out = {}
    out["cumexp_session"] = (df.groupby(["session_id", "primary_category"], sort=False)
                               .cumcount().to_numpy(dtype=np.int32) + 1)

    user_start = np.r_[0, np.flatnonzero(users[1:] != users[:-1]) + 1]
    user_end = np.r_[user_start[1:], n]
    local_pos = np.empty(n, dtype=np.int32)
    for s, e in zip(user_start, user_end):
        local_pos[s:e] = np.arange(e - s, dtype=np.int32)

    group_order = np.lexsort((orig, ts, cats, users))
    uo, co = users[group_order], cats[group_order]
    group_start = np.r_[0, np.flatnonzero((uo[1:] != uo[:-1]) | (co[1:] != co[:-1])) + 1]
    group_end = np.r_[group_start[1:], n]

    f_arrays = {w: np.empty(n, dtype=np.int32) for w in F_WINDOW_MINUTES}
    g_arrays = {window_n: np.zeros(n, dtype=np.float32) for window_n in G_WINDOW_N}
    print(f"(f-2)/(g): processing {len(group_start):,} user-category groups ...")
    for s, e in zip(group_start, group_end):
        idx = group_order[s:e]
        t = ts[idx]
        p = local_pos[idx].astype(np.int64, copy=False)
        seq = np.arange(e - s, dtype=np.int64)
        for w in F_WINDOW_MINUTES:
            left = np.searchsorted(t, t - w * 60.0, side="left")
            f_arrays[w][idx] = (seq - left + 1).astype(np.int32)
        for window_n in G_WINDOW_N:
            left = np.searchsorted(p, p - window_n, side="left")
            previous_same = seq - left
            denom = np.minimum(p, window_n)
            density = np.divide(previous_same, denom, out=np.zeros_like(p, dtype=np.float64),
                                where=denom > 0)
            g_arrays[window_n][idx] = density.astype(np.float32)
    for w, arr in f_arrays.items():
        out[f"cumexp_W{w}m"] = arr
    for window_n, arr in g_arrays.items():
        out[f"density_N{window_n}"] = arr

    same_adjacent = np.zeros(n, dtype=bool)
    same_adjacent[1:] = (users[1:] == users[:-1]) & (cats[1:] == cats[:-1])
    print("(h): computing time-decayed exposure ...")
    for tau in H_DECAY_SECONDS:
        gauge = np.empty(n, dtype=np.float32)
        for s, e in zip(user_start, user_end):
            rel_t = (ts[s:e] - ts[s]) / float(tau)
            log_terms = np.where(same_adjacent[s:e], rel_t, -np.inf)
            log_sum = np.logaddexp.accumulate(log_terms)
            gauge[s:e] = np.exp(log_sum - rel_t).astype(np.float32)
        out[f"gauge_hl{tau}"] = gauge
    return out


def event_decay_features(df, delta_t_ok=None):
    """Compute the event-memory score on each user's complete timeline.

    A rapid same-category event adds one, a rapid category switch subtracts
    the configured penalty, and the state is bounded below at zero.
    """
    users = df["user_id"].to_numpy(dtype=np.int64)
    cats = df["primary_category"].to_numpy()
    delta_t = df["delta_t"].to_numpy(dtype=np.float64)
    n = len(df)
    if delta_t_ok is None:
        delta_t_ok = {T: delta_t <= T for T in EVENT_DECAY_T}

    first_user = np.r_[True, users[1:] != users[:-1]]
    same_category = np.r_[False, (users[1:] == users[:-1]) & (cats[1:] == cats[:-1])]
    user_starts = np.flatnonzero(first_user)
    user_ends = np.r_[user_starts[1:], n]
    out = {}

    print("(i): computing event-memory scores ...")
    for T in EVENT_DECAY_T:
        fast = np.asarray(delta_t_ok[T], dtype=bool) & ~first_user
        for penalty in EVENT_DECAY_PENALTIES:
            penalty_tenths = int(round(penalty * 10))
            increment = np.zeros(n, dtype=np.int16)
            increment[fast & same_category] = 10
            increment[fast & ~same_category] = -penalty_tenths
            score_tenths = np.empty(n, dtype=np.int32)
            for start, end in zip(user_starts, user_ends):
                cumulative = np.cumsum(increment[start:end], dtype=np.int32)
                running_floor = np.minimum.accumulate(np.r_[np.int32(0), cumulative])[:-1]
                score_tenths[start:end] = cumulative - np.minimum(running_floor, cumulative)
            col = f"ed_T{T}_p{penalty_tenths:02d}"
            out[col] = (score_tenths / 10.0).astype(np.float32)
            print(f"  {col}: mean={out[col].mean():.4f} median={np.median(out[col]):.4f} "
                  f"p95={np.quantile(out[col], 0.95):.4f} max={out[col].max():.4f} "
                  f"nonzero={(out[col] > 0).mean():.4%}")
    return out


if __name__ == "__main__":
    t_start = time.time()
    df = load_or_build_base_features()
    print(f"rows: {len(df):,}, users: {df['user_id'].nunique():,}, cols: {list(df.columns)}")
    print("\nLoading item categories ...")
    ic = pd.read_csv(os.path.join(DATA_DIR, "item_categories.csv"))
    ic["feat_list"] = ic["feat"].map(ast.literal_eval)
    feat_len = ic["feat_list"].map(len)
    print("Number of tags per item:")
    print(feat_len.describe())
    len_vc = feat_len.value_counts().sort_index()
    print("Item counts by number of tags:")
    print(len_vc)
    single_tag_pct = (feat_len == 1).mean()
    print(f"Single-tag item share: {single_tag_pct:.4%}")

    max_tag = max(max(f) for f in ic["feat_list"])
    print(f"max tag id: {max_tag}")
    ic["mask"] = ic["feat_list"].map(lambda fs: int(sum(1 << t for t in fs)))
    vid2mask = dict(zip(ic["video_id"], ic["mask"]))

    df["_mask"] = df["video_id"].map(vid2mask).fillna(0).astype(np.int64)
    n_missing_mask = (df["_mask"] == 0).sum()
    print(f"Interactions without a category mask: {n_missing_mask:,} "
          f"({n_missing_mask/len(df):.4%})")

    print("\nComputing adjacent-item similarities ...")
    prev_cat = df.groupby("session_id")["primary_category"].shift(1)
    same_cat = (df["primary_category"] == prev_cat).to_numpy()

    prev_mask = df.groupby("session_id")["_mask"].shift(1)
    prev_valid = prev_mask.notna().to_numpy()
    prev_mask_i = prev_mask.fillna(0).astype(np.int64).to_numpy()
    cur_mask = df["_mask"].to_numpy()

    inter = popcount(cur_mask & prev_mask_i)
    union = popcount(cur_mask | prev_mask_i)
    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    jaccard = np.where(prev_valid, jaccard, 0.0)

    delta_t = df["delta_t"].to_numpy()
    delta_t_ok = {T: (delta_t <= T) for T in T_GRID}

    print("\n(a) checking the category-streak implementation ...")
    cc_a_recomputed = streak_from_keep(same_cat)
    match_a = np.array_equal(cc_a_recomputed, df["consec_count"].to_numpy().astype(np.int32))
    print(f"  baseline check passed: {match_a}")
    if not match_a:
        n_mismatch = (cc_a_recomputed != df["consec_count"].to_numpy()).sum()
        raise ValueError(f"Category-streak check failed for {n_mismatch:,} interactions")

    variants = {}  # colname -> array
    variants["consec_count"] = df["consec_count"].to_numpy().astype(np.int32)

    # ------------------------------------------------------------------
    # (b) time-gated: same_cat AND delta_t<=T
    # ------------------------------------------------------------------
    print("\n(b) time-gated category streaks ...")
    for T in T_GRID:
        keep = same_cat & delta_t_ok[T]
        col = f"cc_timegated_T{T}"
        variants[col] = streak_from_keep(keep)
        print(f"  {col}: mean={variants[col].mean():.4f} median={np.median(variants[col]):.1f}")

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    print("\n(c) temporal burst streaks ...")
    for T in T_GRID:
        keep = delta_t_ok[T]
        col = f"cc_burst_T{T}"
        variants[col] = streak_from_keep(keep)
        print(f"  {col}: mean={variants[col].mean():.4f} median={np.median(variants[col]):.1f}")

    # ------------------------------------------------------------------
    # (d) Jaccard overlap
    # ------------------------------------------------------------------
    print("\n(d) Jaccard-overlap streaks ...")
    for theta in THETA_GRID:
        keep = prev_valid & (jaccard >= theta)
        col = f"cc_jac_th{THETA_TAGS[theta]}"
        variants[col] = streak_from_keep(keep)
        agree_a = (variants[col] == variants["consec_count"]).mean()
        print(f"  {col} (theta={theta}): mean={variants[col].mean():.4f} "
              f"median={np.median(variants[col]):.1f}, agreement with (a)={agree_a:.4%}")

    # ------------------------------------------------------------------
    # (e) combined: jaccard>=theta AND delta_t<=T
    # ------------------------------------------------------------------
    print("\n(e) similarity-and-time-gated streaks ...")
    for theta, T in COMBO_GRID:
        keep = prev_valid & (jaccard >= theta) & delta_t_ok[T]
        col = f"cc_combo_th{THETA_TAGS[theta]}_T{T}"
        variants[col] = streak_from_keep(keep)
        print(f"  {col} (theta={theta}, T={T}): mean={variants[col].mean():.4f} "
              f"median={np.median(variants[col]):.1f}")

    # ------------------------------------------------------------------
    # (f)-(h) cumulative and decayed measures
    # ------------------------------------------------------------------
    print("\n(f)-(h) cumulative and decayed measures ...")
    reset_free = reset_free_exposure_features(df)
    variants.update(reset_free)
    for col, arr in reset_free.items():
        print(f"  {col}: mean={np.mean(arr):.4f} median={np.median(arr):.4f} "
              f"p95={np.quantile(arr, 0.95):.4f} max={np.max(arr):.4f}")

    # ------------------------------------------------------------------
    # (i) event-memory measures
    # ------------------------------------------------------------------
    print("\n(i) event-memory measures ...")
    event_decay = event_decay_features(df, delta_t_ok)
    variants.update(event_decay)


    # ------------------------------------------------------------------
    # Save the shared analysis table.
    # ------------------------------------------------------------------
    out = df[["user_id", "video_id", "timestamp", "date", "watch_ratio", "delta_t",
              "session_id", "session_position", "skip_flag", "strong_positive_flag",
              "primary_category", "video_duration", "popularity"]].copy()
    for col, arr in variants.items():
        out[col] = arr

    out_path = os.path.join(OUT_DIR, "features_timegated.parquet")
    out.to_parquet(out_path, index=False)
    print(f"Saved {out_path} (shape={out.shape})")
    print(f"\nFeature construction completed in {time.time()-t_start:.1f} seconds")
    print("Exposure columns:", [c for c in variants if c != "consec_count"])
