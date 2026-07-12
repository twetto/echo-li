"""Generate the all-session EuRoC position-ATE comparison figure."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SEQUENCES = [
    "V1_01", "V1_02", "V1_03",
    "V2_01", "V2_02", "V2_03",
    "MH_01", "MH_02", "MH_03", "MH_04", "MH_05",
]
BASELINE_M = np.array([
    0.172, 0.203, 0.246,
    0.140, 0.145, np.nan,
    np.nan, 0.304, 0.299, 0.599, 0.606,
])
OUTPUT_DIRS = {
    "V1_01": "eqvio_output_V1_01_easy",
    "V1_02": "eqvio_output_V1_02_medium",
    "V1_03": "eqvio_output_V1_03_difficult",
    "V2_01": "eqvio_output_V2_01_easy",
    "V2_02": "eqvio_output_V2_02_medium",
    "V2_03": "eqvio_output_V2_03_difficult",
    "MH_01": "eqvio_output_MH_01_easy",
    "MH_02": "eqvio_output_MH_02_easy",
    "MH_03": "eqvio_output_MH_03_medium",
    "MH_04": "eqvio_output_MH_04_difficult",
    "MH_05": "eqvio_output_MH_05_difficult",
}


def load_current_ate(repo_root):
    values = []
    for sequence in SEQUENCES:
        metrics_path = repo_root / OUTPUT_DIRS[sequence] / "trajectory_metrics.txt"
        metrics = {}
        for line in metrics_path.read_text().splitlines():
            key, value = line.split()
            metrics[key] = value
        values.append(float(metrics["ate_position_rmse_m"]))
    return np.array(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="fig_ate_9of9.png")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    current_m = load_current_ate(repo_root)
    baseline_cm = 100.0 * BASELINE_M
    current_cm = 100.0 * current_m
    improvement = 100.0 * (BASELINE_M - current_m) / BASELINE_M
    y = np.arange(len(SEQUENCES))
    bar_height = 0.34

    paper_rc = {
        "font.size": 13,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "axes.linewidth": 1.0,
        "savefig.dpi": 300,
    }
    with plt.style.context("seaborn-v0_8-whitegrid"), plt.rc_context(paper_rc):
        fig, ax = plt.subplots(figsize=(8.2, 6.2))
        ax.barh(y - bar_height / 2, baseline_cm, bar_height,
                color="#9AA0A6", label="Previous pipeline")
        ax.barh(y + bar_height / 2, current_cm, bar_height,
                color="#2878B5", label="Gate + robust initialization")

        for yi, before, delta in zip(y, baseline_cm, improvement):
            if np.isfinite(before):
                ax.text(before + 1.2, yi, f"\N{DOWNWARDS ARROW}{delta:.0f}%",
                        va="center", ha="left", color="#176B3A",
                        fontsize=12, fontweight="bold")
            else:
                ax.text(1.0, yi - bar_height / 2, "diverged",
                        va="center", ha="left", color="#A33A2B",
                        fontsize=11, fontweight="bold")

        ax.set_yticks(y, SEQUENCES)
        ax.invert_yaxis()
        ax.set_xlabel("Position ATE RMSE (cm)")
        ax.set_xlim(0, 72)
        ax.xaxis.grid(True, color="0.86", linewidth=0.8)
        ax.yaxis.grid(False)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.legend(loc="upper right", frameon=True)
        fig.tight_layout()
        fig.savefig(args.out, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    print(
        f"saved {args.out}; {len(SEQUENCES)} finite current sessions, "
        f"{np.isfinite(BASELINE_M).sum()} with finite historical baselines"
    )


if __name__ == "__main__":
    main()
