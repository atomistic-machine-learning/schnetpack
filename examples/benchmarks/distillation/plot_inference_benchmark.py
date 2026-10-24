"""Plot the inference times written by ``run_inference_benchmark.py`` over batch size.

The figure shows the wall time of one energy and force call, as the median over the
repeats with the interquartile range as error bars. A model that ran out of GPU memory
has no point at that batch size.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# categorical slots in fixed order, so a model keeps its color across figures
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
MARKERS = ("o", "s", "D", "^", "v", "P")
INK = "#000000"
GRID = "#e4e3df"
DEFAULT_TIMINGS = (
    "/home/jonas/Documents/cluster/schnetpack_debugging/distillation/"
    "results_5124647/timings.json"
)
# the names of the models in the manuscript, as in its table of the distillation
PAPER_NAMES = {
    "teacher": "teacher (MACE-OFF23)",
    "student_f": "student, E/F",
    "student_hvp": "student, E/F/HVP",
}

# "screen" to inspect a run, "paper" for a single column of a two-column journal: no
# title (the caption carries it), print font sizes
LAYOUTS = {
    "screen": dict(
        figsize=(6, 4.5),
        font=10,
        tick_font=10,
        line=2,
        marker=7,
        title=True,
        extensions=("png", "pdf"),
    ),
    "paper": dict(
        figsize=(3.37, 2.5),
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
        "timings",
        nargs="?",
        default=DEFAULT_TIMINGS,
        help="the timings.json of a benchmark run",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="the figure, without extension; a png and a pdf are written, only a pdf "
        "with --paper. Defaults to inference_vs_batch_size next to the timings",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="single-column figure without title, for the manuscript",
    )
    return parser.parse_args()


def summarize(records):
    """``{model: {batch_size: record}}`` with the median and quartiles of the times."""
    models = {}
    for record in records:
        record = dict(record)
        if not record["oom"]:
            times = np.asarray(record["times_s"])
            record["median_s"] = np.median(times)
            record["q25_s"], record["q75_s"] = np.percentile(times, [25, 75])
        models.setdefault(record["model"], {})[record["batch_size"]] = record
    return models


def style_axis(ax, batch_sizes, style):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(batch_sizes)
    ax.set_xticklabels([str(n) for n in batch_sizes])
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    # decades alone leave a panel spanning less than one with a single label
    ax.yaxis.set_major_locator(ticker.LogLocator(subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlabel("batch size [structures]", color=INK, fontsize=style["font"])
    ax.grid(True, which="major", color=GRID, linewidth=0.4 * style["line"])
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
    ax.tick_params(colors=INK, labelsize=style["tick_font"])


def plot(models, meta, output, layout="screen"):
    style = LAYOUTS[layout]
    batch_sizes = meta["batch_sizes"]
    fig, ax = plt.subplots(figsize=style["figsize"])

    for idx, (label, by_size) in enumerate(models.items()):
        color, marker = COLORS[idx % len(COLORS)], MARKERS[idx % len(MARKERS)]
        done = [n for n in batch_sizes if n in by_size and not by_size[n]["oom"]]
        median = np.array([by_size[n]["median_s"] for n in done])
        q25 = np.array([by_size[n]["q25_s"] for n in done])
        q75 = np.array([by_size[n]["q75_s"] for n in done])
        if layout == "paper":
            name = PAPER_NAMES.get(label, label)
        else:
            n_parameters = meta["models"][label]["n_parameters"]
            name = f"{label} ({n_parameters / 1e6:.2f}M params)"

        ax.errorbar(
            done,
            1e3 * median,
            yerr=1e3 * np.stack([median - q25, q75 - median]),
            color=color,
            marker=marker,
            markersize=style["marker"],
            linewidth=style["line"],
            capsize=style["marker"] / 2,
            label=name,
        )

    style_axis(ax, batch_sizes, style)
    ax.set_ylabel(
        "time per energy + force call [ms]", color=INK, fontsize=style["font"]
    )
    ax.legend(frameon=False, fontsize=style["tick_font"], loc="upper left")
    if style["title"]:
        ax.set_title(
            f"Inference on {meta['gpu'] or meta['device']}, {meta['dtype']}, "
            f"{meta['repeats']} repeats (job {meta['slurm_job_id']})",
            color=INK,
            fontsize=style["font"],
        )
    fig.tight_layout(pad=0.3 if layout == "paper" else 1.08)
    paths = [f"{output}.{extension}" for extension in style["extensions"]]
    for path in paths:
        fig.savefig(path, dpi=300 if layout == "paper" else 200)
    plt.close(fig)
    return paths


def main():
    args = parse_args()
    with open(args.timings) as f:
        timings = json.load(f)
    output = args.output or os.path.join(
        os.path.dirname(os.path.abspath(args.timings)), "inference_vs_batch_size"
    )
    output = os.path.splitext(output)[0] if output.endswith(".pdf") else output

    models = summarize(timings["records"])
    for label, by_size in models.items():
        line = ", ".join(
            f"{n}: OOM" if record["oom"] else f"{n}: {1e3 * record['median_s']:.2f} ms"
            for n, record in sorted(by_size.items())
        )
        print(f"{label}: {line}")

    paths = plot(
        models, timings["meta"], output, layout="paper" if args.paper else "screen"
    )
    print(f"figure written to {' and '.join(paths)}")


if __name__ == "__main__":
    main()
