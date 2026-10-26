"""Plot the RMSD of the relaxed test structures to the closest DFT conformer per model,
from the CSV ``relax_test_structures.py`` writes.

Panel a shows the RMSD of every structure after relaxation with the fine-tuned MACE,
the MACE trained from scratch and MACE-OFF23 zero shot, with the median as a bar; the
starting MD snapshots are a faint reference. Panel b shows the steps the batch-wise
LBFGS took per structure.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# the slots of the learning-curve figure, so each model keeps its color
COLORS = {"finetune": "#2a78d6", "scratch": "#eb6834", "zeroshot": "#52514e"}
MARKERS = {"finetune": "o", "scratch": "s", "zeroshot": "D"}
INK = "#000000"
GRID = "#e4e3df"
# the starting structures are a reference, not a model: faint neutral ink
START = "#b5b4b0"
DEFAULT_CSV = (
    "/home/jonas/Documents/cluster/schnetpack_debugging/finetuning/runs/"
    "relaxation_n100_r0/relaxation.csv"
)
NAMES = {
    "start": "start",
    "finetune": "fine-tuned\nMACE-OFF23",
    "scratch": "from\nscratch",
    "zeroshot": "MACE-OFF23\nzero shot",
}

# "screen" to inspect a run, "paper" for a single column of a two-column journal: no
# title (the caption carries it), print font sizes
LAYOUTS = {
    "screen": dict(
        figsize=(6, 6.5),
        font=10,
        tick_font=10,
        line=2,
        marker=6,
        title=True,
        extensions=("png", "pdf"),
    ),
    "paper": dict(
        figsize=(3.37, 3.6),
        font=8,
        tick_font=7,
        line=1.3,
        marker=3.5,
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
        help="the relaxation.csv of relax_test_structures.py",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="the figure, without extension; a png and a pdf are written, only a pdf "
        "with --paper. Defaults to relaxation_rmsd next to the CSV",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="single-column figure without title, for the manuscript",
    )
    return parser.parse_args()


def style_axis(ax, style):
    ax.grid(True, axis="y", which="major", color=GRID, linewidth=0.4 * style["line"])
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
    ax.tick_params(colors=INK, labelsize=style["tick_font"])


def strip(ax, x, values, color, marker, style, jitter):
    """The points of one model at ``x``, spread sideways, with the median as a bar."""
    offsets = np.linspace(-jitter, jitter, len(values)) if len(values) > 1 else [0]
    ax.scatter(
        x + np.asarray(offsets),
        np.sort(values),
        color=color,
        marker=marker,
        s=style["marker"] ** 2,
        linewidths=0,
        alpha=0.8,
        zorder=3,
    )
    ax.hlines(
        np.median(values),
        x - 1.6 * jitter,
        x + 1.6 * jitter,
        color=INK,
        linewidth=style["line"],
        zorder=4,
    )


def plot(rows, output, layout="screen"):
    style = LAYOUTS[layout]
    models = [m for m in ("finetune", "scratch", "zeroshot") if m in set(rows["model"])]
    fig, (ax_rmsd, ax_steps) = plt.subplots(
        2, 1, figsize=style["figsize"], height_ratios=(3, 2)
    )

    # a: RMSD, the starting structures first as the reference
    columns = ["start", *models]
    for x, model in enumerate(columns):
        values = rows.loc[rows["model"] == model, "rmsd"].to_numpy()
        color = START if model == "start" else COLORS[model]
        marker = "o" if model == "start" else MARKERS[model]
        strip(ax_rmsd, x, values, color, marker, style, jitter=0.18)
    ax_rmsd.set_yscale("log")
    ax_rmsd.yaxis.set_major_locator(ticker.LogLocator(subs=(1, 2, 5)))
    ax_rmsd.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    ax_rmsd.yaxis.set_minor_locator(ticker.NullLocator())
    ax_rmsd.set_xticks(range(len(columns)))
    ax_rmsd.set_xticklabels([NAMES[m] for m in columns])
    ax_rmsd.set_ylabel(
        "RMSD to closest DFT\nconformer [Å]", color=INK, fontsize=style["font"]
    )

    # b: steps of the batch-wise LBFGS, models only
    for x, model in enumerate(models):
        values = rows.loc[rows["model"] == model, "steps"].to_numpy()
        strip(ax_steps, x, values, COLORS[model], MARKERS[model], style, jitter=0.18)
    ax_steps.set_xticks(range(len(models)))
    ax_steps.set_xticklabels([NAMES[m] for m in models])
    ax_steps.set_xlim(ax_rmsd.get_xlim()[0] - 1, ax_rmsd.get_xlim()[1] - 1)
    ax_steps.set_ylim(bottom=0)
    ax_steps.set_ylabel("LBFGS steps", color=INK, fontsize=style["font"])

    for ax, letter in zip((ax_rmsd, ax_steps), "ab"):
        style_axis(ax, style)
        ax.text(
            -0.2,
            1.0,
            letter,
            transform=ax.transAxes,
            fontsize=style["font"],
            fontweight="bold",
            va="top",
        )
    if style["title"]:
        n = rows.loc[rows["model"] == "start"].shape[0]
        ax_rmsd.set_title(
            f"rMD17 ethanol, {n} test structures relaxed with batch-wise LBFGS",
            color=INK,
            fontsize=style["font"],
        )
    fig.tight_layout(pad=0.3 if layout == "paper" else 1.08)
    fig.align_ylabels((ax_rmsd, ax_steps))
    paths = [f"{output}.{extension}" for extension in style["extensions"]]
    for path in paths:
        fig.savefig(path, dpi=300 if layout == "paper" else 200)
    plt.close(fig)
    return paths


def main():
    args = parse_args()
    rows = pd.read_csv(args.csv)
    unconverged = rows[(rows["model"] != "start") & ~rows["converged"]]
    if len(unconverged):
        print(f"not converged, plotted at their last step:\n{unconverged}")
    output = args.output or os.path.join(os.path.dirname(args.csv), "relaxation_rmsd")
    paths = plot(rows, output, layout="paper" if args.paper else "screen")
    print(f"figure written to {' and '.join(paths)}")


if __name__ == "__main__":
    main()
