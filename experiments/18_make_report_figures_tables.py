"""Generate report figures and LaTeX table fragments from existing outputs.

This script is intentionally read-only with respect to simulation data. It
loads the completed Roy eco-evolutionary result CSVs and writes manuscript
figures plus LaTeX tabular fragments for the technical report.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures" / "roy_evo_spatial" / "report"
TABLE_DIR = ROOT / "tables" / "roy_evo_spatial"

ODE_THRESHOLD_CSV = RESULTS_DIR / "roy_evo_ode_threshold_scan.csv"
BASIN_REGIME_CSV = RESULTS_DIR / "roy_pde_evo_basin_initial_condition_scan.csv"
BASIN_BOUNDARY_CSV = RESULTS_DIR / "roy_pde_evo_basin_boundary_scan.csv"
BASIN_BOUNDARY_SUMMARY_CSV = RESULTS_DIR / "roy_pde_evo_basin_boundary_summary.csv"

BASIN_LABELS = (
    "persistent_basin",
    "extinct_basin",
    "transient_basin",
    "unresolved_basin",
    "nonphysical_initial_condition",
)

BASIN_COLORS = {
    "persistent_basin": "#2f6fbb",
    "extinct_basin": "#c23b3b",
    "transient_basin": "#d9a441",
    "unresolved_basin": "#7f7f7f",
    "nonphysical_initial_condition": "#5b3f92",
}

BASIN_BOUNDARY_STRESSES = (0.1584375, 0.16486816)


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file as a list of dictionaries."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def bool_text(value: str) -> bool:
    """Parse bool-like CSV text."""
    return value.strip().lower() in {"true", "1", "yes"}


def format_float(value: float, digits: int = 9) -> str:
    """Format floats compactly while retaining report-scale precision."""
    return f"{value:.{digits}g}"


def latex_escape_text(value: str) -> str:
    """Escape plain text for use inside simple LaTeX table cells."""
    return value.replace("\\", r"\textbackslash{}").replace("_", r"\_")


def load_ode_thresholds() -> tuple[float, float, float]:
    """Load no-evolution and evolution ODE thresholds from Step 09A output."""
    rows = read_csv(ODE_THRESHOLD_CSV)
    thresholds: dict[bool, float] = {}
    for row in rows:
        thresholds[bool_text(row["evolve"])] = float(row["threshold"])

    if False not in thresholds or True not in thresholds:
        raise ValueError(f"{ODE_THRESHOLD_CSV} does not contain both evolve states")

    no_evo = thresholds[False]
    evo = thresholds[True]
    return no_evo, evo, evo - no_evo


def load_basin_counts(path: Path) -> tuple[list[float], dict[str, list[int]]]:
    """Aggregate basin labels by stress."""
    counts_by_stress: dict[float, Counter[str]] = defaultdict(Counter)
    for row in read_csv(path):
        counts_by_stress[float(row["stress"])][row["basin_label"]] += 1

    stresses = sorted(counts_by_stress)
    counts = {
        basin_label: [counts_by_stress[stress].get(basin_label, 0) for stress in stresses]
        for basin_label in BASIN_LABELS
    }
    return stresses, counts


def load_boundary_summary() -> list[dict[str, str]]:
    """Load the Step 15 basin-boundary summary, or aggregate it if absent."""
    if BASIN_BOUNDARY_SUMMARY_CSV.exists():
        return read_csv(BASIN_BOUNDARY_SUMMARY_CSV)

    counts_by_stress: dict[float, Counter[str]] = defaultdict(Counter)
    for row in read_csv(BASIN_BOUNDARY_CSV):
        counts_by_stress[float(row["stress"])][row["basin_label"]] += 1

    summary_rows: list[dict[str, str]] = []
    for stress in sorted(counts_by_stress):
        counts = counts_by_stress[stress]
        persistent = counts.get("persistent_basin", 0)
        extinct = counts.get("extinct_basin", 0)
        transient = counts.get("transient_basin", 0)
        unresolved = counts.get("unresolved_basin", 0)
        nonphysical = counts.get("nonphysical_initial_condition", 0)
        if persistent > 0 and extinct > 0:
            regime = "bistable_persistent_extinct"
        elif persistent > 0 and extinct == 0 and transient + unresolved == 0:
            regime = "persistent_only"
        elif extinct > 0 and persistent == 0 and transient + unresolved == 0:
            regime = "extinct_only"
        elif persistent > 0 and extinct == 0:
            regime = "persistent_transient_mixed"
        elif extinct > 0 and persistent == 0:
            regime = "extinct_transient_mixed"
        elif transient + unresolved > persistent + extinct + nonphysical:
            regime = "mostly_transient_or_unresolved"
        else:
            regime = "unresolved"

        summary_rows.append(
            {
                "stress": format_float(stress),
                "persistent_count": str(persistent),
                "extinct_count": str(extinct),
                "transient_count": str(transient),
                "unresolved_count": str(unresolved),
                "nonphysical_count": str(nonphysical),
                "regime_label": regime,
            }
        )
    return summary_rows


def label_for_basin(basin_label: str) -> str:
    """Human-readable basin label."""
    return basin_label.replace("_", " ")


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    """Save a figure using report-friendly defaults."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_box(ax: plt.Axes, xy: tuple[float, float], text: str, width: float = 2.0) -> None:
    """Draw a rounded annotation box."""
    x, y = xy
    box = mpatches.FancyBboxPatch(
        (x - width / 2, y - 0.35),
        width,
        0.7,
        boxstyle="round,pad=0.08",
        linewidth=1.2,
        edgecolor="#333333",
        facecolor="#f4f7fb",
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=10)


def draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], text: str = "") -> None:
    """Draw an arrow with an optional label."""
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#333333"},
    )
    if text:
        ax.text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 + 0.12,
            text,
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333333",
        )


def make_model_schematic() -> Path:
    """Create the conceptual model schematic."""
    output_path = FIGURE_DIR / "fig01_model_schematic.png"
    fig, ax = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    draw_box(ax, (2.2, 4.7), "$n$\ntotal prey density", width=2.25)
    draw_box(ax, (5.0, 4.7), "$w$\npredator density", width=2.25)
    draw_box(ax, (2.2, 2.0), "$q$\nprey defense frequency", width=2.45)
    draw_box(ax, (5.0, 2.0), "$z=\\kappa^{-1}-n-w$\nfree space", width=2.75)
    draw_box(ax, (8.0, 4.7), "stress $s$\nincreases predator mortality", width=2.75)
    draw_box(ax, (8.0, 2.0), "$r(q), a(q), b(q)$\ntrade-off functions", width=2.55)

    draw_arrow(ax, (3.35, 4.7), (3.9, 4.7), "predation")
    draw_arrow(ax, (2.2, 4.35), (4.0, 2.35), "uses $n,w$")
    draw_arrow(ax, (5.0, 4.35), (5.0, 2.45), "crowding")
    ax.annotate(
        "",
        xy=(6.7, 1.78),
        xytext=(3.45, 1.78),
        arrowprops={
            "arrowstyle": "->",
            "lw": 1.4,
            "color": "#333333",
            "connectionstyle": "arc3,rad=0.35",
        },
    )
    ax.text(5.0, 1.05, "$q$ sets trade-offs", ha="center", va="center", fontsize=9, color="#333333")
    draw_arrow(ax, (8.0, 4.35), (8.0, 2.45), "")
    draw_arrow(ax, (7.05, 4.7), (6.1, 4.7), "$m+s$")
    draw_arrow(ax, (6.85, 2.2), (5.95, 4.35), "local growth")

    ax.text(
        5.0,
        0.65,
        "ODE uses spatial means; PDE adds diffusion of $n$, $w$, and $q$ with zero-flux boundaries.",
        ha="center",
        fontsize=10,
    )
    ax.set_title("Roy eco-evolutionary rescue model variables", fontsize=14, pad=10)
    save_figure(fig, output_path)
    return output_path


def make_ode_threshold_plot(no_evo: float, evo: float, delta: float) -> Path:
    """Plot ODE no-evolution and evolution mortality thresholds."""
    output_path = FIGURE_DIR / "fig02_ode_thresholds.png"
    fig, ax = plt.subplots(figsize=(6.6, 4.8), constrained_layout=True)
    labels = ["ODE no evolution", "ODE evolution"]
    values = [no_evo, evo]
    bars = ax.bar(labels, values, color=["#8da0cb", "#66c2a5"], edgecolor="#333333")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.004,
            format_float(value, 9),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.annotate(
        f"$\\Delta_{{evo}}^{{ODE}}={format_float(delta, 9)}$",
        xy=(0.5, evo),
        xytext=(0.5, evo + 0.035),
        ha="center",
        arrowprops={"arrowstyle": "<->", "lw": 1.2},
        fontsize=11,
    )
    ax.set_ylim(0, max(values) + 0.07)
    ax.set_ylabel("predator mortality stress threshold")
    ax.set_title("Well-mixed ODE indirect evolutionary rescue")
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, output_path)
    return output_path


def make_diagnostic_sequence() -> Path:
    """Plot the diagnostic sequence that changed the threshold narrative."""
    output_path = FIGURE_DIR / "fig03_diagnostic_sequence.png"
    steps = [
        ("PR #4", "classifier-sensitive"),
        ("PR #5", "persistence unresolved"),
        ("PR #6", "hysteresis detected"),
        ("PR #7", "bistability mapped"),
        ("PR #9", "basin boundary mapped"),
    ]

    fig, ax = plt.subplots(figsize=(11.2, 3.2), constrained_layout=True)
    ax.set_xlim(-0.5, len(steps) - 0.5)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for index, (pr_label, result) in enumerate(steps):
        box = mpatches.FancyBboxPatch(
            (index - 0.42, 0.36),
            0.84,
            0.35,
            boxstyle="round,pad=0.05",
            linewidth=1.1,
            edgecolor="#333333",
            facecolor="#f4f7fb",
        )
        ax.add_patch(box)
        ax.text(index, 0.56, pr_label, ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(index, 0.42, result, ha="center", va="center", fontsize=9)
        if index < len(steps) - 1:
            ax.annotate(
                "",
                xy=(index + 0.54, 0.53),
                xytext=(index + 0.46, 0.53),
                arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#333333"},
            )

    ax.text(
        2,
        0.15,
        "The spatial PDE response is path-dependent, so scalar threshold language is insufficient.",
        ha="center",
        va="center",
        fontsize=11,
    )
    ax.set_title("Diagnostic sequence correcting the initial threshold narrative", fontsize=14)
    save_figure(fig, output_path)
    return output_path


def make_basin_regime_map() -> Path:
    """Create a stacked bar chart for the PR #7 basin-regime map."""
    output_path = FIGURE_DIR / "fig04_basin_regime_map.png"
    stresses, counts = load_basin_counts(BASIN_REGIME_CSV)
    fig, ax = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    x_positions = np.arange(len(stresses))
    bottoms = np.zeros(len(stresses))

    for basin_label in BASIN_LABELS:
        values = np.array(counts[basin_label])
        ax.bar(
            x_positions,
            values,
            bottom=bottoms,
            label=label_for_basin(basin_label),
            color=BASIN_COLORS[basin_label],
            edgecolor="white",
            linewidth=0.6,
        )
        bottoms += values

    ax.set_xticks(x_positions)
    ax.set_xticklabels([format_float(stress, 9) for stress in stresses], rotation=25, ha="right")
    ax.set_xlabel("stress")
    ax.set_ylabel("initial-condition families")
    ax.set_title("PDE-evolution basin outcomes by stress")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output_path)
    return output_path


def make_basin_boundary_heatmap() -> Path:
    """Create q0--w0 basin-boundary panels for the focused PR #9 stresses."""
    output_path = FIGURE_DIR / "fig05_basin_boundary_heatmap.png"
    rows = [
        row
        for row in read_csv(BASIN_BOUNDARY_CSV)
        if any(abs(float(row["stress"]) - stress) < 1e-12 for stress in BASIN_BOUNDARY_STRESSES)
    ]
    if not rows:
        raise ValueError(f"No focused stress rows found in {BASIN_BOUNDARY_CSV}")

    q_values = sorted({float(row["q0"]) for row in rows})
    w_scales = sorted({float(row["w0_scale"]) for row in rows})
    category_index = {label: index for index, label in enumerate(BASIN_LABELS)}
    cmap = mcolors.ListedColormap([BASIN_COLORS[label] for label in BASIN_LABELS])
    norm = mcolors.BoundaryNorm(np.arange(len(BASIN_LABELS) + 1) - 0.5, len(BASIN_LABELS))

    fig, axes = plt.subplots(1, len(BASIN_BOUNDARY_STRESSES), figsize=(12.4, 5.2), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.76, bottom=0.18, top=0.82, wspace=0.22)
    if len(BASIN_BOUNDARY_STRESSES) == 1:
        axes = [axes]

    for ax, stress in zip(axes, BASIN_BOUNDARY_STRESSES):
        matrix = np.full((len(w_scales), len(q_values)), category_index["unresolved_basin"])
        for row in rows:
            if abs(float(row["stress"]) - stress) >= 1e-12:
                continue
            q_index = q_values.index(float(row["q0"]))
            w_index = w_scales.index(float(row["w0_scale"]))
            matrix[w_index, q_index] = category_index[row["basin_label"]]

        ax.imshow(matrix, origin="lower", aspect="auto", cmap=cmap, norm=norm)
        ax.set_title(f"stress = {format_float(stress, 9)}")
        ax.set_xticks(np.arange(len(q_values)))
        ax.set_xticklabels([format_float(value, 3) for value in q_values], rotation=45)
        ax.set_yticks(np.arange(len(w_scales)))
        ax.set_yticklabels([format_float(value, 3) for value in w_scales])
        ax.set_xlabel("initial defense frequency $q_0$")
        ax.grid(color="white", linewidth=0.7)

    axes[0].set_ylabel("initial predator scale $w_0/w_{baseline}$")
    handles = [
        mpatches.Patch(color=BASIN_COLORS[label], label=label_for_basin(label))
        for label in BASIN_LABELS
    ]
    fig.legend(handles=handles, loc="center left", frameon=False, bbox_to_anchor=(0.79, 0.50))
    fig.suptitle("PDE-evolution basin boundary scan", fontsize=14)
    save_figure(fig, output_path)
    return output_path


def make_basin_boundary_counts() -> Path:
    """Create stacked basin-count bars for the PR #9 focused stresses."""
    output_path = FIGURE_DIR / "fig06_basin_boundary_counts.png"
    summary_rows = load_boundary_summary()
    stresses = [float(row["stress"]) for row in summary_rows]
    count_columns = [
        ("persistent_basin", "persistent_count"),
        ("extinct_basin", "extinct_count"),
        ("transient_basin", "transient_count"),
        ("unresolved_basin", "unresolved_count"),
        ("nonphysical_initial_condition", "nonphysical_count"),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    x_positions = np.arange(len(stresses))
    bottoms = np.zeros(len(stresses))
    for basin_label, column in count_columns:
        values = np.array([int(row[column]) for row in summary_rows])
        ax.bar(
            x_positions,
            values,
            bottom=bottoms,
            label=label_for_basin(basin_label),
            color=BASIN_COLORS[basin_label],
            edgecolor="white",
            linewidth=0.6,
        )
        bottoms += values

    ax.set_xticks(x_positions)
    ax.set_xticklabels([format_float(stress, 9) for stress in stresses])
    ax.set_xlabel("stress")
    ax.set_ylabel("q0--w0 grid points")
    ax.set_title("Basin-boundary counts by focused stress")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output_path)
    return output_path


def write_table(path: Path, lines: list[str]) -> Path:
    """Write a LaTeX tabular fragment."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def make_table_parameters() -> Path:
    """Write the parameter table fragment."""
    rows = [
        (r"$\kappa$", "0.15", "free-space scale"),
        (r"$\xi$", "0.55", "prey loss term"),
        (r"$r_u$", "1", "undefended prey growth"),
        (r"$r_v$", "0.65", "defended prey growth"),
        (r"$a_u$", "1", "undefended predation coefficient"),
        (r"$a_v$", "0.35", "defended predation coefficient"),
        (r"$b_u$", "0.08", "undefended conversion coefficient"),
        (r"$b_v$", "0.02", "defended conversion coefficient"),
        (r"$m$", "0.1", "baseline predator mortality"),
        (r"$\mu$", "0.2", "predator density dependence"),
        (r"$\nu$", "0.05", "evolutionary rate"),
        (r"$D_n$", "0.01", "prey diffusion"),
        (r"$D_w$", "0.01", "predator diffusion"),
        (r"$D_q$", "0.005", "defense-frequency diffusion"),
    ]
    lines = [
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Symbol & Value & Description \\",
        r"\midrule",
    ]
    lines.extend([f"{symbol} & {value} & {description} \\\\" for symbol, value, description in rows])
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return write_table(TABLE_DIR / "table01_parameters.tex", lines)


def make_table_thresholds(no_evo: float, evo: float, delta: float) -> Path:
    """Write the threshold table fragment."""
    rows = [
        (r"$m_c^{ODE,\mathrm{no\ evo}}$", format_float(no_evo, 9), "ODE reference"),
        (r"$m_c^{ODE,\mathrm{evo}}$", format_float(evo, 9), "ODE reference"),
        (r"$\Delta_{evo}^{ODE}$", format_float(delta, 9), "ODE evolutionary rescue effect"),
        (r"representative $m_c^{PDE,\mathrm{no\ evo}}$", "0.06921875", "historical screening quantity"),
        (r"representative $m_c^{PDE,\mathrm{evo}}$", "0.11765625", "historical screening quantity"),
    ]
    lines = [
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Quantity & Value & Interpretation \\",
        r"\midrule",
    ]
    lines.extend([f"{quantity} & {value} & {interpretation} \\\\" for quantity, value, interpretation in rows])
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return write_table(TABLE_DIR / "table02_thresholds.tex", lines)


def make_table_bistability_regimes() -> Path:
    """Write the Step 13 stress-regime table fragment."""
    rows = [
        ("0.141262205", "persistent_transient_mixed"),
        ("0.15", "persistent_transient_mixed"),
        ("0.1584375", "bistable_persistent_extinct"),
        ("0.16486816", "bistable_persistent_extinct"),
        ("0.175", "bistable_persistent_extinct"),
    ]
    lines = [
        r"\begin{tabular}{ll}",
        r"\toprule",
        r"Stress & Regime \\",
        r"\midrule",
    ]
    lines.extend([f"{stress} & \\texttt{{{latex_escape_text(regime)}}} \\\\" for stress, regime in rows])
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return write_table(TABLE_DIR / "table03_bistability_regimes.tex", lines)


def make_table_basin_boundary_counts() -> Path:
    """Write the Step 15 basin-boundary count table fragment."""
    summary_rows = load_boundary_summary()
    lines = [
        r"\begin{tabular}{rrrrrrl}",
        r"\toprule",
        r"Stress & Persistent & Extinct & Transient & Unresolved & Nonphysical & Regime \\",
        r"\midrule",
    ]
    for row in summary_rows:
        lines.append(
            f"{row['stress']} & {row['persistent_count']} & {row['extinct_count']} & "
            f"{row['transient_count']} & {row['unresolved_count']} & {row['nonphysical_count']} & "
            f"\\texttt{{{latex_escape_text(row['regime_label'])}}} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return write_table(TABLE_DIR / "table04_basin_boundary_counts.tex", lines)


def main() -> None:
    """Generate all report figures and table fragments."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    no_evo, evo, delta = load_ode_thresholds()
    outputs = [
        make_model_schematic(),
        make_ode_threshold_plot(no_evo, evo, delta),
        make_diagnostic_sequence(),
        make_basin_regime_map(),
        make_basin_boundary_heatmap(),
        make_basin_boundary_counts(),
        make_table_parameters(),
        make_table_thresholds(no_evo, evo, delta),
        make_table_bistability_regimes(),
        make_table_basin_boundary_counts(),
    ]

    for output_path in outputs:
        print(output_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
