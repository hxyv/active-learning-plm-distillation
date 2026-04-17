"""Plot active learning results.

Usage
-----
Single strategy::

    python scripts/plot_al_results.py \\
        --results outputs/al/random_12345/al_results.json \\
        --label random

Multiple strategies (learning curve comparison)::

    python scripts/plot_al_results.py \\
        --results /ocean/projects/cis250233p/xhu15/outputs/al/random_12345/al_results.json \\
                  /ocean/projects/cis250233p/xhu15/outputs/al/mc_dropout_67890/al_results.json \\
        --labels random mc_dropout \\
        --output-dir /ocean/projects/cis250233p/xhu15/outputs/plots

SS8 composition of acquired proteins (one strategy)::

    python scripts/plot_al_results.py \\
        --results /ocean/projects/cis250233p/xhu15/outputs/al/mc_dropout_67890/al_results.json \\
        --labels mc_dropout \\
        --composition

If --output-dir is omitted, plots are saved next to the first results file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from data.constants import SS8_CLASSES

# Colours for SS8 classes: G H I T E B S C
_SS8_COLORS = [
    "#e41a1c",  # G  3-10 helix     red
    "#ff7f00",  # H  α-helix        orange
    "#ffdd44",  # I  π-helix        yellow
    "#4daf4a",  # T  turn           green
    "#377eb8",  # E  β-strand       blue
    "#984ea3",  # B  β-bridge       purple
    "#a65628",  # S  bend           brown
    "#999999",  # C  coil           grey
]

_SS8_LABELS = {
    "G": "G (3₁₀-helix)",
    "H": "H (α-helix)",
    "I": "I (π-helix)",
    "T": "T (turn)",
    "E": "E (β-strand)",
    "B": "B (β-bridge)",
    "S": "S (bend)",
    "C": "C (coil)",
}


def load_results(path: Path) -> List[Dict]:
    return json.loads(path.read_text())


def extract_curve(results: List[Dict]) -> tuple:
    """Return (num_labeled, num_residues, test_acc, test_ce) arrays."""
    num_labeled = [r["num_train"] for r in results]
    num_residues = [r.get("num_train_residues", float("nan")) for r in results]
    test_acc = [r["metrics"].get("test_teacher_top1_acc", float("nan")) for r in results]
    test_ce = [r["metrics"].get("test_teacher_ce", float("nan")) for r in results]
    return (
        np.array(num_labeled),
        np.array(num_residues, dtype=float),
        np.array(test_acc),
        np.array(test_ce),
    )


def plot_learning_curves(
    results_list: List[List[Dict]],
    labels: List[str],
    output_dir: Path,
) -> None:
    # Two rows: top = x by protein count, bottom = x by residue count.
    # Residue row is drawn only if all results carry num_train_residues.
    curves = [extract_curve(r) for r in results_list]
    have_residues = all(not np.isnan(c[1]).any() for c in curves)
    n_rows = 2 if have_residues else 1
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 5 * n_rows), squeeze=False)

    for (num_labeled, num_residues, test_acc, test_ce), label in zip(curves, labels):
        axes[0, 0].plot(num_labeled, test_acc, marker="o", markersize=4, label=label)
        axes[0, 1].plot(num_labeled, test_ce, marker="o", markersize=4, label=label)
        if have_residues:
            axes[1, 0].plot(num_residues, test_acc, marker="o", markersize=4, label=label)
            axes[1, 1].plot(num_residues, test_ce, marker="o", markersize=4, label=label)

    axes[0, 0].set_xlabel("Labeled proteins")
    axes[0, 0].set_ylabel("Test teacher top-1 accuracy")
    axes[0, 0].set_title("AL Learning Curve — Accuracy (by proteins)")
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel("Labeled proteins")
    axes[0, 1].set_ylabel("Test teacher CE loss")
    axes[0, 1].set_title("AL Learning Curve — CE (by proteins)")
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    if have_residues:
        axes[1, 0].set_xlabel("Labeled residues")
        axes[1, 0].set_ylabel("Test teacher top-1 accuracy")
        axes[1, 0].set_title("AL Learning Curve — Accuracy (by residues)")
        axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].set_xlabel("Labeled residues")
        axes[1, 1].set_ylabel("Test teacher CE loss")
        axes[1, 1].set_title("AL Learning Curve — CE (by residues)")
        axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / "al_learning_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_pool_uncertainty(
    results_list: List[List[Dict]],
    labels: List[str],
    output_dir: Path,
) -> None:
    """Per-round pool-wide uncertainty distribution (mean ± p10/p90 band).

    Drawn only for runs whose rounds contain acquisition.pool_stats — i.e.
    model-based strategies. Random runs silently skipped.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plotted_any = False
    for results, label in zip(results_list, labels):
        rounds, var_mean, var_p10, var_p90 = [], [], [], []
        bald_mean, bald_p10, bald_p90 = [], [], []
        for r in results:
            ps = (r.get("acquisition") or {}).get("pool_stats")
            if not ps:
                continue
            rounds.append(r["round"])
            var_mean.append(ps["variance"]["mean"])
            var_p10.append(ps["variance"]["p10"])
            var_p90.append(ps["variance"]["p90"])
            bald_mean.append(ps["bald"]["mean"])
            bald_p10.append(ps["bald"]["p10"])
            bald_p90.append(ps["bald"]["p90"])
        if not rounds:
            continue
        plotted_any = True
        axes[0].plot(rounds, var_mean, marker="o", label=label)
        axes[0].fill_between(rounds, var_p10, var_p90, alpha=0.2)
        axes[1].plot(rounds, bald_mean, marker="o", label=label)
        axes[1].fill_between(rounds, bald_p10, bald_p90, alpha=0.2)

    if not plotted_any:
        plt.close(fig)
        return

    axes[0].set_xlabel("Round"); axes[0].set_ylabel("Pool variance score")
    axes[0].set_title("Pool variance (selection score) — mean, p10–p90 band")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Round"); axes[1].set_ylabel("Pool BALD (mutual info)")
    axes[1].set_title("Pool BALD — mean, p10–p90 band")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / "al_pool_uncertainty.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_per_class_accuracy(
    results: List[Dict],
    label: str,
    output_dir: Path,
) -> None:
    """Per-round per-SS8-class test accuracy as a heatmap."""
    rows = [r for r in results if "test_per_class_acc" in r.get("metrics", {})]
    if not rows:
        return
    round_nums = [r["round"] for r in rows]
    matrix = np.array([
        [r["metrics"]["test_per_class_acc"].get(cls, float("nan")) for cls in SS8_CLASSES]
        for r in rows
    ])  # [n_rounds, 8]

    fig, ax = plt.subplots(figsize=(max(6, len(round_nums) * 0.4), 4))
    im = ax.imshow(matrix.T, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(len(SS8_CLASSES)))
    ax.set_yticklabels(SS8_CLASSES)
    ax.set_xticks(range(len(round_nums)))
    ax.set_xticklabels(round_nums, fontsize=8)
    ax.set_xlabel("Round")
    ax.set_ylabel("SS8 class")
    ax.set_title(f"Per-class test accuracy — {label}")
    fig.colorbar(im, ax=ax, label="accuracy")
    fig.tight_layout()
    out_path = output_dir / f"al_per_class_acc_{label}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_ss8_composition(
    results: List[Dict],
    label: str,
    output_dir: Path,
) -> None:
    """Stacked bar chart: per-round SS8 composition of acquired proteins."""
    rounds_with_comp = [
        r for r in results
        if "selected_ss8_composition" in r and r["selected_ss8_composition"]
    ]

    if not rounds_with_comp:
        print("No SS8 composition data found in results (requires al_loop >= this version).")
        return

    round_nums = [r["round"] for r in rounds_with_comp]
    num_labeled = [r["num_train"] for r in rounds_with_comp]
    x_labels = [f"R{r}\n({n})" for r, n in zip(round_nums, num_labeled)]

    fracs = np.array([
        [r["selected_ss8_composition"].get(cls, 0.0) for cls in SS8_CLASSES]
        for r in rounds_with_comp
    ])  # [n_rounds, 8]

    fig, ax = plt.subplots(figsize=(max(8, len(round_nums) * 0.7), 5))

    bottoms = np.zeros(len(rounds_with_comp))
    x = np.arange(len(rounds_with_comp))
    for i, (cls, color) in enumerate(zip(SS8_CLASSES, _SS8_COLORS)):
        ax.bar(x, fracs[:, i], bottom=bottoms, color=color,
               label=_SS8_LABELS[cls], width=0.7)
        bottoms += fracs[:, i]

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_xlabel("Round (labeled set size)")
    ax.set_ylabel("Mean SS8 fraction of acquired proteins")
    ax.set_title(f"SS8 Composition of Acquired Proteins — {label}")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / f"al_ss8_composition_{label}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def print_round_table(results: List[Dict], label: str) -> None:
    print(f"\n{'='*70}")
    print(f"Strategy: {label}")
    print(f"{'Round':>6}  {'Labeled':>8}  {'test_acc':>9}  {'test_ce':>8}  SS8 top-2")
    print("-" * 70)
    for r in results:
        acc = r["metrics"].get("test_teacher_top1_acc", float("nan"))
        ce = r["metrics"].get("test_teacher_ce", float("nan"))
        comp = r.get("selected_ss8_composition", {})
        if comp:
            top2 = sorted(comp.items(), key=lambda x: -x[1])[:2]
            top2_str = " ".join(f"{cls}:{v:.2f}" for cls, v in top2)
        else:
            top2_str = "—"
        print(f"{r['round']:>6}  {r['num_train']:>8}  {acc:>9.4f}  {ce:>8.4f}  {top2_str}")
    print("=" * 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot AL results")
    parser.add_argument(
        "--results", type=Path, nargs="+", required=True,
        help="Path(s) to al_results.json files",
    )
    parser.add_argument(
        "--labels", type=str, nargs="+",
        help="Strategy labels (same order as --results). Defaults to file parent dir name.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory to write plots. Defaults to the parent directory of the first results file.",
    )
    parser.add_argument(
        "--composition", action="store_true",
        help="Also plot per-round SS8 composition for each strategy",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    labels = args.labels
    if labels is None:
        labels = [p.parent.name for p in args.results]
    if len(labels) != len(args.results):
        raise ValueError("--labels must have the same number of entries as --results")

    if args.output_dir is None:
        args.output_dir = args.results[0].parent.parent / "plots"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to: {args.output_dir}")

    results_list = [load_results(p) for p in args.results]

    plot_learning_curves(results_list, labels, args.output_dir)
    plot_pool_uncertainty(results_list, labels, args.output_dir)

    for results, label in zip(results_list, labels):
        print_round_table(results, label)
        plot_per_class_accuracy(results, label, args.output_dir)
        if args.composition:
            plot_ss8_composition(results, label, args.output_dir)


if __name__ == "__main__":
    main()
