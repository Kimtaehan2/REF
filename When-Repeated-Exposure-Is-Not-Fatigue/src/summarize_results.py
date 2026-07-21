"""Assemble the tables reported in the manuscript."""

import argparse
import os

import pandas as pd


REPRESENTATIVE_MEASURES = [
    {
        "label": "(a) Category streak",
        "association": "consec_count",
        "cessation": "a_streak",
        "interpretation": "Mixed",
    },
    {
        "label": "(b) Time-gated category, 20 s",
        "association": "cc_timegated_T20",
        "cessation": "b_time20",
        "interpretation": "Fatigue-consistent",
    },
    {
        "label": "(c) Temporal burst, 20 s",
        "association": "cc_burst_T20",
        "cessation": "c_burst20",
        "interpretation": "Item rejection only",
    },
    {
        "label": "(f-2) Rolling cumulative, 30 min",
        "association": "cumexp_W30m",
        "cessation": "cumexp_W30m",
        "interpretation": "Sustained engagement",
    },
    {
        "label": "(h) Time-decayed exposure, 30 min",
        "association": "gauge_hl1800",
        "cessation": "gauge_hl1800",
        "interpretation": "Sustained engagement",
    },
]


def indexed_csv(directory, name, key):
    path = os.path.join(directory, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing result file: {path}")
    frame = pd.read_csv(path)
    if frame[key].duplicated().any():
        raise ValueError(f"Duplicate {key} values in {path}")
    return frame.set_index(key)


def assemble(source_dir):
    associations = indexed_csv(source_dir, "association_results.csv", "col")
    cessation = indexed_csv(source_dir, "session_cessation_results.csv", "exposure")
    within = indexed_csv(source_dir, "within_between_results.csv", "exposure")
    rows = []
    for measure in REPRESENTATIVE_MEASURES:
        association = associations.loc[measure["association"]]
        cessation_row = cessation.loc[measure["cessation"]]
        within_row = within.loc[measure["cessation"]]
        rows.append({
            "Exposure measure": measure["label"],
            "Skip OR": association["skip_OR"],
            "Skip CI lower": association["skip_ci_lo"],
            "Skip CI upper": association["skip_ci_hi"],
            "Skip p": association["skip_p"],
            "Watch beta": association["wr_coef"],
            "Watch CI lower": association["wr_ci_lo"],
            "Watch CI upper": association["wr_ci_hi"],
            "Watch p": association["wr_p"],
            "Cessation OR": cessation_row["exit_OR"],
            "Cessation p": cessation_row["exit_p"],
            "Within-user OR": within_row["within_OR"],
            "Within-user p": within_row["within_p"],
            "Interpretation": measure["interpretation"],
        })
    return pd.DataFrame(rows)


def format_p(value):
    if value < 0.001:
        return "<.001"
    return f"{value:.3f}".lstrip("0")


def write_markdown(table, source_dir, output_path):
    lines = ["# Results reported in the manuscript\n\n"]
    lines.append(
        "All exposure coefficients are reported per one standard deviation. Skip and "
        "cessation effects are odds ratios; watch-intensity effects are linear mixed-model "
        "coefficients among non-skipped interactions.\n\n"
    )
    lines.append("## Representative behavioral associations\n\n")
    lines.append(
        "| Exposure measure | Skip OR (95% CI) | p | Watch beta (95% CI) | p | "
        "Cessation OR | Within-user OR | Interpretation |\n"
        "|---|---:|---:|---:|---:|---:|---:|---|\n"
    )
    for _, row in table.iterrows():
        lines.append(
            f"| {row['Exposure measure']} | {row['Skip OR']:.3f} "
            f"({row['Skip CI lower']:.3f}, {row['Skip CI upper']:.3f}) | "
            f"{format_p(row['Skip p'])} | {row['Watch beta']:.3f} "
            f"({row['Watch CI lower']:.3f}, {row['Watch CI upper']:.3f}) | "
            f"{format_p(row['Watch p'])} | {row['Cessation OR']:.3f} | "
            f"{row['Within-user OR']:.3f} | {row['Interpretation']} |\n"
        )

    associations = pd.read_csv(os.path.join(source_dir, "association_results.csv"))
    lines.append("\n## Complete skip and watch-intensity results\n\n")
    lines.append(
        "| Exposure | Skip OR | p | Watch beta | p | AIC | BIC | Grouped AUC | "
        "GEE converged | Mixed model converged |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|\n"
    )
    for _, row in associations.iterrows():
        lines.append(
            f"| {row['label']} | {row['skip_OR']:.4f} | {format_p(row['skip_p'])} | "
            f"{row['wr_coef']:.5f} | {format_p(row['wr_p'])} | "
            f"{row['skip_aic']:.1f} | {row['skip_bic']:.1f} | "
            f"{row['skip_cv_auc']:.5f} | {row['skip_converged']} | "
            f"{row['wr_converged']} |\n"
        )

    cessation = pd.read_csv(os.path.join(source_dir, "session_cessation_results.csv"))
    within = pd.read_csv(os.path.join(source_dir, "within_between_results.csv"))
    cessation = cessation.merge(
        within[["exposure", "within_OR", "within_p", "converged"]],
        on="exposure",
        how="left",
        suffixes=("", "_within"),
    )
    lines.append("\n## Complete cessation results\n\n")
    lines.append(
        "| Exposure | Cessation OR | p | Within-user OR | p | Grouped AUC | "
        "Cessation converged | Within model converged |\n"
        "|---|---:|---:|---:|---:|---:|---|---|\n"
    )
    for _, row in cessation.iterrows():
        lines.append(
            f"| {row['label']} | {row['exit_OR']:.4f} | {format_p(row['exit_p'])} | "
            f"{row['within_OR']:.4f} | {format_p(row['within_p'])} | "
            f"{row['exposure_cv_auc']:.5f} | {row['exit_converged']} | "
            f"{row['converged']} |\n"
        )

    sensitivity_path = os.path.join(source_dir, "session_boundary_sensitivity.csv")
    if os.path.exists(sensitivity_path):
        sensitivity = pd.read_csv(sensitivity_path)
        pivot = sensitivity.pivot(index="family", columns="boundary", values="OR")
        lines.append("\n## Session-boundary sensitivity\n\n")
        lines.append("| Exposure | " + " | ".join(pivot.columns) + " |\n")
        lines.append("|---|" + "---:|" * len(pivot.columns) + "\n")
        for exposure, row in pivot.iterrows():
            lines.append(
                f"| {exposure} | " + " | ".join(f"{value:.4f}" for value in row) + " |\n"
            )

    threshold_path = os.path.join(source_dir, "threshold_results.csv")
    if os.path.exists(threshold_path):
        thresholds = pd.read_csv(threshold_path)
        lines.append("\n## Threshold estimates\n\n")
        lines.append(
            "| Exposure | Threshold | 95% CI | F | Wild-bootstrap p | "
            "Threshold after excluding 50 users |\n"
            "|---|---:|---:|---:|---:|---:|\n"
        )
        for _, row in thresholds.iterrows():
            lines.append(
                f"| {row['label']} | {row['gamma_hat']:.0f} | "
                f"[{row['gamma_ci_lo']:.0f}, {row['gamma_ci_hi']:.0f}] | "
                f"{row['F_full']:.1f} | {row['wild_p']:.4f} | "
                f"{row['gamma_excl50']:.0f} |\n"
            )

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.writelines(lines)


def write_latex_table(table, output_path):
    lines = [
        "% Generated by src/summarize_results.py\n",
        "\\begin{table*}\n",
        "  \\caption{Associations Between Repeated Exposure and Behavioral Outcomes}\n",
        "  \\label{tab:associations}\n",
        "  \\begin{tabular}{lccccl}\n",
        "    \\toprule\n",
        "    Exposure measure & Skip OR & Watch $\\beta$ & Cessation OR & "
        "Within-user OR & Interpretation \\\\\n",
        "    \\midrule\n",
    ]
    for _, row in table.iterrows():
        label = row["Exposure measure"].replace("(f-2)", "(f-2)")
        lines.append(
            f"    {label} & {row['Skip OR']:.3f} & {row['Watch beta']:.3f} & "
            f"{row['Cessation OR']:.3f} & {row['Within-user OR']:.3f} & "
            f"{row['Interpretation']} \\\\\n"
        )
    lines.extend([
        "    \\bottomrule\n",
        "  \\end{tabular}\n",
        "  \\begin{flushleft}\n",
        "  \\footnotesize\n",
        "  Note. Estimates are reported per 1 SD increase in exposure. Watch "
        "$\\beta$ is estimated among non-skipped interactions. The within-user OR "
        "is the coefficient for the observation-level deviation from the user mean.\n",
        "  \\end{flushleft}\n",
        "\\end{table*}\n",
    ])
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.writelines(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    table = assemble(args.source_dir)
    table.to_csv(os.path.join(args.output_dir, "paper_table.csv"), index=False)
    write_markdown(table, args.source_dir, os.path.join(args.output_dir, "paper_results.md"))
    write_latex_table(table, os.path.join(args.output_dir, "table_associations.tex"))


if __name__ == "__main__":
    main()
