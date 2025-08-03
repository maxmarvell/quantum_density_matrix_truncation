import argparse
import json
import re

import matplotlib.pyplot as plt

# Reuse plotting defaults if available
try:
    from .utils.graph import *  # noqa: F401,F403
except Exception:
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot scaling from pytest-benchmark JSON results.")
    parser.add_argument(
        "result_file",
        type=str,
        help="Path to the benchmark JSON file produced by pytest-benchmark",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Optional filepath to save the plot instead of displaying it",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.result_file) as fh:
        data = json.load(fh)

    records = data.get("benchmarks", [])
    pattern = re.compile(r"(?P<name>.+)\[(?P<param>.+)\]")

    series = {}
    for rec in records:
        match = pattern.match(rec.get("name", ""))
        if not match:
            # skip benchmarks without parameterization information
            continue
        name = match.group("name")
        param = match.group("param")
        try:
            x_val = float(param)
        except ValueError:
            x_val = param
        y_val = rec.get("stats", {}).get("mean")
        if y_val is None:
            continue
        series.setdefault(name, []).append((x_val, y_val))

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, values in series.items():
        values.sort(key=lambda t: t[0])
        xs = [v[0] for v in values]
        ys = [v[1] for v in values]
        ax.plot(xs, ys, marker="o", label=name)

    ax.set_xlabel("Parameter")
    ax.set_yscale("log")
    ax.set_ylabel("Mean time (s)")
    ax.legend()
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()


if __name__ == "__main__":
    main()