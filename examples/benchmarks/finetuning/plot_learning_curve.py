"""Plot the learning curve of MACE fine-tuned from MACE-OFF23 against MACE trained
from random weights, from the CSV ``collect_mace_rmd17.py`` writes.

The figure shows the test MAE of energy and forces over the number of training
structures, as the mean over the replicates with their standard deviation as error
bars. MACE-OFF23 zero shot, from the CSV ``eval_zero_shot_rmd17.py`` writes, is a
dashed reference line: its energies depend on the training structures through the
fitted E0 correction, its forces do not.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

# categorical slots in fixed order, so MACE-OFF keeps the color of the teacher in
# the distillation figure
COLORS = {"finetune": "#2a78d6", "scratch": "#eb6834"}
MARKERS = {"finetune": "o", "scratch": "s"}
INK = "#000000"
GRID = "#e4e3df"
# the zero-shot model is a reference, not a third trained model: neutral ink
REFERENCE = "#52514e"
DEFAULT_CSV = (
    "/home/jonas/Documents/cluster/schnetpack_debugging/finetuning/runs/"
    "learning_curve.csv"
)
NAMES = {
    "finetune": "fine-tuned MACE-OFF23",
    "scratch": "from scratch",
    "zeroshot": "MACE-OFF23 zero shot",
}
PANELS = (
    ("test_energy_mae", "energy MAE [meV]"),
    ("test_forces_mae", "force MAE [meV/Å]"),
)

# "screen" to inspect a run, "paper" for a single column of a two-column journal: no
# title (the caption carries it), print font sizes
LAYOUTS = {
    "screen": dict(
        figsize=(6, 6.5),
        font=10,
        tick_font=10,
        line=2,
        marker=7,
        title=True,
        extensions=("png", "pdf"),
    ),
    "paper": dict(
        figsize=(3.37, 3.6),
        font=8,
        tick_font=7,
        line=1.3,
        marker=4,
        title=False,
        extensions=("pdf",),
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "csv",
        nargs="?",
        default=DEFAULT_CSV,
        help="the learning_curve.csv of collect_mace_rmd17.py",
    )
    parser.add_argument(
        "--zero-shot",
        default=None,
        help="the zero_shot.csv of eval_zero_shot_rmd17.py. Defaults to zero_shot.csv "
        "next to the learning curve, if there is one",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="the figure, without extension; a png and a pdf are written, only a pdf "
        "with --paper. Defaults to learning_curve next to the CSV",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="single-column figure without title, for the manuscript",
    )
    return parser.parse_args()


def style_axis(ax, sizes, style):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(n) for n in sizes])
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    # decades alone leave a panel spanning less than one with a single label
    ax.yaxis.set_major_locator(ticker.LogLocator(subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.grid(True, which="major", color=GRID, linewidth=0.4 * style["line"])
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
    ax.tick_params(colors=INK, labelsize=style["tick_font"])


def mean_std(runs, arm, metric):
    """Mean and standard deviation over the replicates per training size, in meV."""
    return (
        runs[runs["arm"] == arm].groupby("num_train")[metric].agg(["mean", "std"])
    ).mul(1e3)


def plot(runs, output, zero_shot=None, layout="screen"):
    style = LAYOUTS[layout]
    sizes = sorted(runs["num_train"].unique())
    fig, axes = plt.subplots(len(PANELS), 1, sharex=True, figsize=style["figsize"])

    for ax, (metric, ylabel), letter in zip(axes, PANELS, "ab"):
        for arm in ("finetune", "scratch"):
            stats = mean_std(runs, arm, metric)
            ax.errorbar(
                stats.index,
                stats["mean"],
                yerr=stats["std"].fillna(0),
                color=COLORS[arm],
                marker=MARKERS[arm],
                markersize=style["marker"],
                linewidth=style["line"],
                capsize=style["marker"] / 2,
                label=NAMES[arm],
            )
        if zero_shot is not None:
            stats = mean_std(zero_shot, "zeroshot", metric)
            ax.errorbar(
                stats.index,
                stats["mean"],
                yerr=stats["std"].fillna(0),
                color=REFERENCE,
                linestyle="--",
                linewidth=0.8 * style["line"],
                capsize=style["marker"] / 2,
                label=NAMES["zeroshot"],
            )
        style_axis(ax, sizes, style)
        ax.set_ylabel(ylabel, color=INK, fontsize=style["font"])
        ax.text(
            -0.2,
            1.0,
            letter,
            transform=ax.transAxes,
            fontsize=style["font"],
            fontweight="bold",
            va="top",
        )

    axes[-1].set_xlabel(
        "training set size [structures]", color=INK, fontsize=style["font"]
    )
    # below the zero-shot line, above the trained models, which fall to the right
    axes[-1].legend(
        frameon=False,
        fontsize=style["tick_font"],
        loc="upper right",
        bbox_to_anchor=(1.0, 0.9),
    )
    if style["title"]:
        reps = runs.groupby(["arm", "num_train"]).size()
        axes[0].set_title(
            f"rMD17 ethanol, test MAE, {reps.min()}-{reps.max()} replicates per point",
            color=INK,
            fontsize=style["font"],
        )
    fig.tight_layout(pad=0.3 if layout == "paper" else 1.08)
    fig.align_ylabels(axes)
    paths = [f"{output}.{extension}" for extension in style["extensions"]]
    for path in paths:
        fig.savefig(path, dpi=300 if layout == "paper" else 200)
    plt.close(fig)
    return paths


def main():
    args = parse_args()
    runs = pd.read_csv(args.csv)
    zero_shot_csv = args.zero_shot or os.path.join(
        os.path.dirname(args.csv), "zero_shot.csv"
    )
    zero_shot = pd.read_csv(zero_shot_csv) if os.path.exists(zero_shot_csv) else None
    output = args.output or os.path.join(os.path.dirname(args.csv), "learning_curve")
    paths = plot(runs, output, zero_shot, layout="paper" if args.paper else "screen")
    print(f"figure written to {' and '.join(paths)}")


if __name__ == "__main__":
    main()
