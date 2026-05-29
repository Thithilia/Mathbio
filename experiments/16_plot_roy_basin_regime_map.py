"""Plot Step 13 PDE-evolution basin-regime outcomes.

This script is intentionally read-only with respect to simulation data. It
loads the basin initial-condition scan produced by Step 13, aggregates basin
labels by stress, and writes one summary figure for the synthesis note.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "results" / "roy_pde_evo_basin_initial_condition_scan.csv"
OUTPUT_PATH = ROOT / "figures" / "roy_evo_spatial" / "13_basin_regime_map.png"

BASIN_LABELS = (
    "persistent_basin",
    "extinct_basin",
    "transient_basin",
    "unresolved_basin",
    "nonphysical_initial_condition",
)


def load_basin_counts(input_csv: Path) -> tuple[list[float], dict[str, list[int]]]:
    """Return basin-label counts by stress from the Step 13 basin scan."""
    counts_by_stress: dict[float, Counter[str]] = defaultdict(Counter)
    with input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"stress", "basin_label"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"{input_csv} is missing required columns: {missing_text}")

        for row in reader:
            stress = float(row["stress"])
            basin_label = row["basin_label"]
            counts_by_stress[stress][basin_label] += 1

    stresses = sorted(counts_by_stress)
    counts = {
        basin_label: [counts_by_stress[stress].get(basin_label, 0) for stress in stresses]
        for basin_label in BASIN_LABELS
    }
    return stresses, counts


def plot_basin_counts(stresses: list[float], counts: dict[str, list[int]], output_path: Path) -> None:
    """Write a stacked bar chart of basin outcomes by stress."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
    x_positions = list(range(len(stresses)))
    bottoms = [0] * len(stresses)

    for basin_label in BASIN_LABELS:
        values = counts[basin_label]
        ax.bar(x_positions, values, bottom=bottoms, label=basin_label.replace("_", " "))
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    stress_labels = [f"{stress:.9g}" for stress in stresses]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(stress_labels, rotation=25, ha="right")
    ax.set_xlabel("stress")
    ax.set_ylabel("number of initial-condition families")
    ax.set_title(
        "PDE-evolution basin outcomes by stress\n"
        "Bistability appears where persistent and extinct basins coexist."
    )
    ax.set_ylim(0, max(bottoms) + 0.8)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    ax.grid(axis="y", alpha=0.3)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    stresses, counts = load_basin_counts(INPUT_CSV)
    plot_basin_counts(stresses, counts, OUTPUT_PATH)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
