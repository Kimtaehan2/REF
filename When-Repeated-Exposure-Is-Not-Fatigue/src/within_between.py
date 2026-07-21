"""Separate within-user changes from between-user exposure differences.

Each exposure is decomposed into the user's mean and the observation-level
deviation from that mean. Both terms enter the same cessation GEE.
"""
import gc
import importlib.util
import os
import time

import numpy as np
import pandas as pd
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.generalized_estimating_equations import GEE

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.abspath(os.path.join(HERE, ".."))
OUT_DIR = os.environ.get("REVIEW_RESULTS_DIR", os.path.join(PACKAGE_DIR, "results"))
OUT_CSV = os.path.join(OUT_DIR, "within_between_results.csv")
OUT_MD = os.path.join(OUT_DIR, "within_between_results.md")

spec = importlib.util.spec_from_file_location(
    "session_exit", os.path.join(HERE, "session_cessation.py")
)
sx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sx)


def fit_within_between(df, exposure):
    user_mean = df.groupby("user_id", sort=False)[exposure].transform("mean").to_numpy()
    within = df[exposure].to_numpy(dtype=np.float64) - user_mean
    df["exposure_within_z"] = sx.zscore(within)
    df["exposure_between_z"] = sx.zscore(user_mean)
    formula = (
        "session_exit ~ exposure_within_z + exposure_between_z + "
        + " + ".join(sx.BASE_CONTROLS)
    )
    res = GEE.from_formula(
        formula, groups="user_id", data=df, family=Binomial(),
        cov_struct=Exchangeable(),
    ).fit(maxiter=200)
    ci = res.conf_int()
    row = {
        "within_OR": np.exp(res.params["exposure_within_z"]),
        "within_p": res.pvalues["exposure_within_z"],
        "within_ci_lo": np.exp(ci.loc["exposure_within_z", 0]),
        "within_ci_hi": np.exp(ci.loc["exposure_within_z", 1]),
        "between_OR": np.exp(res.params["exposure_between_z"]),
        "between_p": res.pvalues["exposure_between_z"],
        "between_ci_lo": np.exp(ci.loc["exposure_between_z", 0]),
        "between_ci_hi": np.exp(ci.loc["exposure_between_z", 1]),
        "within_raw_sd": np.std(within, ddof=1),
        "between_repeated_sd": np.std(user_mean, ddof=1),
        "converged": bool(getattr(res, "converged", True)),
    }
    return row


def main():
    started = time.time()
    raw, threshold, _, _, _, _, _ = sx.load_raw_and_estimate_threshold(reuse_saved=True)
    df, sampled_users = sx.load_analysis_sample(raw)
    del raw
    gc.collect()
    vid2mask = sx.load_video_masks()
    sx.prepare_scheme(df, threshold, vid2mask)

    pooled = pd.read_csv(sx.PRIMARY_OUT).set_index("exposure")
    rows = []
    for label, exposure in sx.PRIMARY_DEFINITIONS:
        print(f"{label} ...", end="", flush=True)
        result = fit_within_between(df, exposure)
        result.update({
            "label": label, "exposure": exposure,
            "pooled_OR": pooled.loc[exposure, "exit_OR"],
            "pooled_p": pooled.loc[exposure, "exit_p"],
        })
        rows.append(result)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
        print(f" within OR={result['within_OR']:.4f}, "
              f"between OR={result['between_OR']:.4f}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    report = ["# Within- and between-user cessation associations\n\n"]
    report.append(f"The data-derived boundary was {threshold:.1f} seconds "
                  f"({threshold/60:.2f} minutes). The sample contains "
                  f"{len(sampled_users):,} users and {len(df):,} observations.\n\n")
    report.append("Both the user mean and the observation-level deviation from that mean "
                  "were included in the same GEE.\n\n")
    report.append("| Exposure | Pooled OR | Within-user OR | p | Between-user OR | p | Converged |\n"
                  "|---|---:|---:|---:|---:|---:|---|\n")
    for _, r in out.iterrows():
        report.append(f"| {r['label']} | {r['pooled_OR']:.4f} | {r['within_OR']:.4f} | "
                      f"{r['within_p']:.3g} | {r['between_OR']:.4f} | "
                      f"{r['between_p']:.3g} | {bool(r['converged'])} |\n")
    report.append("\nThe within-user estimate is the primary quantity for distinguishing "
                  "state changes from stable differences between users. Estimates are "
                  "associational and should not be interpreted as causal effects.\n")
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.writelines(report)
    print(f"Saved within-between audit ({time.time()-started:.1f}s)")


if __name__ == "__main__":
    main()
