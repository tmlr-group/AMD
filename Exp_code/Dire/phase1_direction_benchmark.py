import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.append(str(REPO_DIR))

from dataloader import load_data
from Exp_code.amd_core import select_rst_direction_grid


def parse_csv_values(values, cast):
    parsed = []
    for value in values:
        parsed.extend(value.split(","))
    return [cast(value) for value in parsed if value != ""]


def expected_direction(per):
    if per < 0.5:
        return 1
    if per > 0.5:
        return -1
    return 0


def to_tensor(array, device):
    return torch.as_tensor(array, device=device, dtype=torch.float)


def run_setting(args, dataset, sample_size, per):
    expected = expected_direction(per)
    directions = []
    agreements = []
    scores = []
    selected_kernels = []
    pilot_sample_size = max(int(sample_size), int(args.pilot_sample_size))

    for rep in range(args.reps):
        seed = args.seed + 10000 * rep + int(sample_size * 13) + int(per * 1000)
        X, Y, Z = load_data(dataset, pilot_sample_size, seed, per)
        X = to_tensor(X, args.device)
        Y = to_tensor(Y, args.device)
        Z = to_tensor(Z, args.device)
        direction, diagnostics = select_rst_direction_grid(
            X,
            Y,
            Z,
            kernel=args.kernel,
            seed=seed,
            max_samples=args.max_samples,
            num_bandwidths=args.num_bandwidths,
            num_bags=args.num_bags,
            bag_fraction=args.bag_fraction,
            top_k=args.top_k,
        )
        directions.append(direction)
        agreements.append(diagnostics["bag_agreement"])
        scores.append(diagnostics["selected_score"])
        selected_kernels.append(diagnostics["selected_kernel"])

    directions = np.asarray(directions)
    if expected == 0:
        accuracy = np.nan
    else:
        accuracy = float(np.mean(directions == expected))

    kernel_counts = Counter(selected_kernels)
    return {
        "dataset": dataset,
        "sample_size": sample_size,
        "pilot_sample_size": pilot_sample_size,
        "per": per,
        "expected_direction": expected,
        "direction_accuracy": accuracy,
        "positive_direction_rate": float(np.mean(directions == 1)),
        "negative_direction_rate": float(np.mean(directions == -1)),
        "mean_bag_agreement": float(np.mean(agreements)),
        "mean_selected_score": float(np.mean(scores)),
        "selected_kernel_counts": dict(kernel_counts),
    }


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark AMD Phase-I direction selection.")
    parser.add_argument("--datasets", nargs="+", default=["BLOB", "HDGM"])
    parser.add_argument("--sample_sizes", nargs="+", default=["120", "200", "500"])
    parser.add_argument("--pilot_sample_size", type=int, default=2048)
    parser.add_argument("--pers", nargs="+", default=["0.3", "0.5", "0.7"])
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--kernel", default="Gaussian")
    parser.add_argument("--num_bandwidths", type=int, default=15)
    parser.add_argument("--num_bags", type=int, default=3)
    parser.add_argument("--bag_fraction", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=7)
    parser.add_argument("--max_samples", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../../Results/Dire/phase1_direction_benchmark.csv"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.sample_sizes = parse_csv_values(args.sample_sizes, int)
    args.pers = parse_csv_values(args.pers, float)
    args.device = torch.device(args.device)

    rows = []
    for dataset in args.datasets:
        for sample_size in args.sample_sizes:
            for per in args.pers:
                row = run_setting(args, dataset, sample_size, per)
                rows.append(row)
                accuracy = row["direction_accuracy"]
                accuracy_text = "NA" if np.isnan(accuracy) else f"{accuracy:.3f}"
                print(
                    f"{dataset:5s} n={sample_size:4d} per={per:.2f} "
                    f"pilot={row['pilot_sample_size']:4d} acc={accuracy_text} "
                    f"pos={row['positive_direction_rate']:.3f} "
                    f"bag={row['mean_bag_agreement']:.3f} "
                    f"kernels={row['selected_kernel_counts']}",
                    flush=True,
                )

    write_rows(args.output, rows)
    print(f"\nSaved benchmark to {args.output}")


if __name__ == "__main__":
    main()
