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

# "screen" to inspect a run, "paper" for a single column of a two-column journal: no
# title (the caption carries it), print font sizes, every other batch size labelled
LAYOUTS = {
    "screen": dict(
        figsize=(7.2, 4.8),
        dpi=130,
        surface=SURFACE,
        font=10,
        tick_font=9,
        line=2,
        marker=6,
        pad=8,
        tick_every=1,
        title=True,
    ),
    "paper": dict(
        figsize=(3.37, 2.5),
        dpi=300,
        surface="white",
        font=8,
        tick_font=7,
        line=1.3,
        marker=3.5,
        pad=4,
        tick_every=2,
        title=False,
    ),
}


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
    parser.add_argument(
        "--extrapolate-sequential",
        type=int,
        default=None,
        metavar="BATCH_SIZE",
        help="extend the sequential curve as a dashed linear fit up to this batch size",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="single-column figure without title, for the manuscript",
    )
    parser.add_argument("--show", action="store_true", help="also open the figure")
    return parser.parse_args()


def load_timings(input_dir):
    """The run settings and, per method, its batch sizes and the wall times as a
    ``(n_batch_sizes, n_repeats)`` array.

    A method only covers the batch sizes it was timed at: the sequential baseline may
    stop early (``--sequential-max-batch-size``), and a run that died keeps the batch
    sizes it finished.
    """
    with open(os.path.join(input_dir, "timings.json")) as f:
        results = json.load(f)
    meta = results["meta"]

    timings = {}
    for method in METHODS:
        records = [r for r in results["records"] if r["method"] == method]
        batch_sizes = sorted({r["batch_size"] for r in records})
        samples = np.full((len(batch_sizes), meta["repeats"]), np.nan)
        for record in records:
            size_idx = batch_sizes.index(record["batch_size"])
            samples[size_idx, record["repeat"]] = record["time_s"]
        assert not np.isnan(samples).any(), f"{method}: missing timings"
        timings[method] = (batch_sizes, samples)
    return meta, timings


def mean_ci(samples, confidence):
    """Mean and half-width of the Student-t confidence interval, along the repeats."""
    mean = samples.mean(axis=1)
    half_width = stats.sem(samples, axis=1) * stats.t.ppf(
        0.5 + confidence / 2, samples.shape[1] - 1
    )
    return mean, half_width


def linear_fit(batch_sizes, samples):
    """Intercept and slope of ``time = a + b * batch_size``, fitted to every repeat."""
    sizes = np.repeat(batch_sizes, samples.shape[1])
    slope, intercept = np.polyfit(sizes, samples.ravel(), 1)
    return intercept, slope


def extrapolate(batch_sizes, samples, all_batch_sizes, max_batch_size):
    """The linear fit evaluated from the last timed batch size up to ``max_batch_size``,
    at the batch sizes of the run and at ``max_batch_size`` itself."""
    intercept, slope = linear_fit(batch_sizes, samples)
    sizes = [n for n in all_batch_sizes if batch_sizes[-1] <= n < max_batch_size]
    sizes = np.array([batch_sizes[-1], *sizes[1:], max_batch_size])
    return sizes, intercept + slope * sizes


def print_table(summary):
    # (mean, half-width) per method and batch size, absent where it was not timed
    rows = {
        method: {n: (m, ci) for n, m, ci in zip(sizes, mean, half_width)}
        for method, (sizes, mean, half_width) in summary.items()
    }
    header = f"{'batch size':>10} {'batch-wise [s]':>22} {'sequential [s]':>22} {'speedup':>9}"
    print(header)
    print("-" * len(header))
    for n_structures in sorted(set().union(*rows.values())):
        line = f"{n_structures:>10}"
        for method in METHODS:
            if n_structures in rows[method]:
                mean, half_width = rows[method][n_structures]
                line += f" {mean:>13.2f} +- {half_width:<6.2f}"
            else:
                line += f" {'-':>22}"
        if all(n_structures in rows[method] for method in METHODS):
            speedup = (
                rows["sequential"][n_structures][0]
                / rows["batch-wise"][n_structures][0]
            )
            line += f" {speedup:>8.2f}x"
        print(line)


def plot(meta, summary, confidence, extrapolation=None, layout="screen"):
    """``extrapolation`` is an optional ``(sizes, times)`` continuation of the
    sequential curve, drawn dashed. ``layout`` is a key of ``LAYOUTS``."""
    style = LAYOUTS[layout]
    batch_sizes = sorted(set().union(*(sizes for sizes, *_ in summary.values())))
    if extrapolation is not None:
        batch_sizes = sorted({*batch_sizes, *extrapolation[0].tolist()})

    fig, ax = plt.subplots(figsize=style["figsize"], dpi=style["dpi"])
    fig.patch.set_facecolor(style["surface"])
    ax.set_facecolor(style["surface"])

    for label, (sizes, mean, half_width) in summary.items():
        color = SERIES[label]
        # confidence interval as a band, the mean as the line on top of it
        ax.fill_between(
            sizes,
            mean - half_width,
            mean + half_width,
            color=color,
            alpha=0.18,
            linewidth=0,
            zorder=2,
        )
        ax.plot(
            sizes,
            mean,
            color=color,
            linewidth=style["line"],
            marker="o",
            markersize=style["marker"],
            markeredgecolor=style["surface"],
            markeredgewidth=style["marker"] / 3,
            label=label,
            zorder=3,
        )
        end = (sizes[-1], mean[-1])
        if label == "sequential" and extrapolation is not None:
            ax.plot(
                *extrapolation,
                color=color,
                linewidth=0.75 * style["line"],
                linestyle=(0, (4, 3)),
                label="linear fit",
                zorder=3,
            )
            end = (extrapolation[0][-1], extrapolation[1][-1])
        # label the series at its end, so the curves can be read without the legend;
        # a single column has no room for it to the right, there the legend does it
        if style["title"]:
            ax.annotate(
                label,
                end,
                textcoords="offset points",
                xytext=(10, 0),
                va="center",
                color=color,
                fontsize=style["font"],
            )

    if extrapolation is not None:
        # the gap between both curves at the end of the extrapolation, as a speedup
        n_end, sequential_end = extrapolation[0][-1], extrapolation[1][-1]
        sizes, mean, _ = summary["batch-wise"]
        if n_end in sizes:
            batchwise_end = mean[sizes.index(n_end)]
            ax.annotate(
                "",
                (n_end, batchwise_end),
                xytext=(n_end, sequential_end),
                arrowprops=dict(
                    arrowstyle="<->",
                    color=INK_SECONDARY,
                    linewidth=style["line"] / 2,
                    shrinkA=style["marker"],
                    shrinkB=style["marker"],
                ),
                zorder=4,
            )
            ax.annotate(
                f"~{sequential_end / batchwise_end:.0f}\u00d7"
                + ("\n" if layout == "paper" else " ")
                + "faster",
                (n_end, np.sqrt(sequential_end * batchwise_end)),  # middle on log axes
                # left of the arrow, where neither curve passes
                textcoords="offset points",
                xytext=(-style["pad"], 0),
                ha="right",
                va="center",
                color=INK,
                fontsize=style["font"],
                fontweight="bold",
            )

    # both scalings are power laws, which read as straight lines on log-log axes
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(batch_sizes)
    ax.set_xticklabels(
        [n if i % style["tick_every"] == 0 else "" for i, n in enumerate(batch_sizes)]
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.minorticks_off()
    ax.set_xlim(
        batch_sizes[0] / 1.3, batch_sizes[-1] * (2.6 if style["title"] else 1.3)
    )

    ax.grid(True, which="major", color=GRID, linewidth=0.4 * style["line"])
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=style["tick_font"], length=0)

    ax.set_xlabel(
        "batch size (number of structures)",
        color=INK_SECONDARY,
        fontsize=style["font"],
        labelpad=style["pad"],
    )
    ax.set_ylabel(
        "relaxation time [s]",
        color=INK_SECONDARY,
        fontsize=style["font"],
        labelpad=style["pad"],
    )
    legend = ax.legend(frameon=False, loc="upper left", fontsize=style["font"])
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    if not style["title"]:
        fig.tight_layout(pad=0.3)
        return fig

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
    fig.tight_layout()
    return fig


def main():
    args = parse_args()
    meta, timings = load_timings(args.input)
    summary = {
        method: (sizes, *mean_ci(samples, args.confidence))
        for method, (sizes, samples) in timings.items()
    }

    print_table(summary)

    extrapolation = None
    if args.extrapolate_sequential is not None:
        sizes, samples = timings["sequential"]
        intercept, slope = linear_fit(sizes, samples)
        print(f"sequential linear fit: {intercept:.3f} s + {slope:.4f} s * batch size")
        extrapolation = extrapolate(
            sizes, samples, meta["batch_sizes"], args.extrapolate_sequential
        )

    fig = plot(
        meta,
        summary,
        args.confidence,
        extrapolation,
        layout="paper" if args.paper else "screen",
    )
    output = args.output or os.path.join(args.input, "relaxation_time.png")
    fig.savefig(output, facecolor=fig.get_facecolor())
    print(f"figure written to {output}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
