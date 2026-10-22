"""Plot the relaxation time against the batch size, from the output of ``run_benchmark.py``.

Prints the mean wall time of both optimizers per batch size with its confidence
interval and the speedup, and plots both curves on log-log axes. With only a handful of
repeats the interval is a Student-t interval, which is noticeably wider than the
standard error: increase ``--repeats`` of the benchmark run to tighten it.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

METHODS = ("batch-wise", "sequential")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = {"batch-wise": "#2a78d6", "sequential": "#eb6834"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--input",
        default="./batchwise_relaxation_benchmark",
        help="output directory of run_benchmark.py",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="figure path, <input>/relaxation_time.png if unset",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--show", action="store_true", help="also open the figure")
    return parser.parse_args()


def load_timings(input_dir):
    """The run settings and the wall times as ``(n_batch_sizes, n_repeats)`` arrays."""
    with open(os.path.join(input_dir, "timings.json")) as f:
        results = json.load(f)
    meta = results["meta"]
    batch_sizes = meta["batch_sizes"]

    timings = {
        method: np.full((len(batch_sizes), meta["repeats"]), np.nan)
        for method in METHODS
    }
    for record in results["records"]:
        size_idx = batch_sizes.index(record["batch_size"])
        timings[record["method"]][size_idx, record["repeat"]] = record["time_s"]
    for method, samples in timings.items():
        assert not np.isnan(samples).any(), f"{method}: missing timings"
    return meta, timings


def mean_ci(samples, confidence):
    """Mean and half-width of the Student-t confidence interval, along the repeats."""
    mean = samples.mean(axis=1)
    half_width = stats.sem(samples, axis=1) * stats.t.ppf(
        0.5 + confidence / 2, samples.shape[1] - 1
    )
    return mean, half_width


def print_table(batch_sizes, summary):
    (mean_bw, ci_bw), (mean_seq, ci_seq) = summary["batch-wise"], summary["sequential"]
    header = f"{'batch size':>10} {'batch-wise [s]':>22} {'sequential [s]':>22} {'speedup':>9}"
    print(header)
    print("-" * len(header))
    for i, n_structures in enumerate(batch_sizes):
        print(
            f"{n_structures:>10} "
            f"{mean_bw[i]:>13.2f} +- {ci_bw[i]:<6.2f} "
            f"{mean_seq[i]:>13.2f} +- {ci_seq[i]:<6.2f} "
            f"{mean_seq[i] / mean_bw[i]:>8.2f}x"
        )


def plot(meta, summary, confidence):
    batch_sizes = meta["batch_sizes"]

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for label, (mean, half_width) in summary.items():
        color = SERIES[label]
        # confidence interval as a band, the mean as the line on top of it
        ax.fill_between(
            batch_sizes,
            mean - half_width,
            mean + half_width,
            color=color,
            alpha=0.18,
            linewidth=0,
            zorder=2,
        )
        ax.plot(
            batch_sizes,
            mean,
            color=color,
            linewidth=2,
            marker="o",
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            label=label,
            zorder=3,
        )
        # label the series at its end, so the curves can be read without the legend
        ax.annotate(
            label,
            (batch_sizes[-1], mean[-1]),
            textcoords="offset points",
            xytext=(10, 0),
            va="center",
            color=color,
            fontsize=10,
        )

    # both scalings are power laws, which read as straight lines on log-log axes
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(batch_sizes)
    ax.set_xticklabels(batch_sizes)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.minorticks_off()
    ax.set_xlim(batch_sizes[0] / 1.3, batch_sizes[-1] * 2.6)

    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)

    ax.set_xlabel(
        "batch size (number of structures)",
        color=INK_SECONDARY,
        fontsize=10,
        labelpad=8,
    )
    ax.set_ylabel("relaxation time [s]", color=INK_SECONDARY, fontsize=10, labelpad=8)
    ax.set_title(
        "Batch-wise relaxation pulls ahead as the batch grows",
        color=INK,
        fontsize=13,
        loc="left",
        pad=34,
    )
    ax.text(
        0.0,
        1.015,
        f"wall time to relax the whole batch to fmax = {meta['fmax']} eV/A "
        f"on {meta.get('gpu') or meta['device']},\n"
        f"mean of {meta['repeats']} runs, band is the {confidence:.0%} "
        "confidence interval",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=9,
        va="bottom",
        linespacing=1.6,
    )
    legend = ax.legend(frameon=False, loc="upper left", fontsize=10)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    return fig


def main():
    args = parse_args()
    meta, timings = load_timings(args.input)
    summary = {
        method: mean_ci(samples, args.confidence) for method, samples in timings.items()
    }

    print_table(meta["batch_sizes"], summary)

    fig = plot(meta, summary, args.confidence)
    output = args.output or os.path.join(args.input, "relaxation_time.png")
    fig.savefig(output, facecolor=fig.get_facecolor())
    print(f"figure written to {output}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
