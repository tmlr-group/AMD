"""Small Gaussian demo for anchor-based relative similarity testing.

The demo has three distributions: an anchor U and two candidates P and Q.  The
statistic is MMD^2(U, P) - MMD^2(U, Q), so negative values mean P is closer to
the anchor and positive values mean Q is closer.  AMD first selects a bandwidth
and a direction from pilot samples, then uses an independent test sample with a
wild-bootstrap threshold.  The demo reports fixed-direction, random-direction,
and two-sided variants so the comparison does not depend on choosing a favorable
baseline direction.
"""

import argparse
import csv
from pathlib import Path

import numpy as np


def pairwise_sq_dists(x, y):
    return ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=2)


def rbf_kernel(x, y, bandwidth):
    return np.exp(-pairwise_sq_dists(x, y) / (2.0 * bandwidth**2))


def sample_triplet(rng, sample_size, anchor_location):
    anchor = rng.normal(loc=(anchor_location, 0.0), scale=1.0, size=(sample_size, 2))
    p = rng.normal(loc=(-1.0, 0.0), scale=1.0, size=(sample_size, 2))
    q = rng.normal(loc=(1.0, 0.0), scale=1.0, size=(sample_size, 2))
    return anchor, p, q


def rst_statistic(anchor, p, q, bandwidth):
    n = len(anchor)
    if len(p) != n or len(q) != n:
        raise ValueError("anchor, p, and q must have the same sample size.")

    k_pp = rbf_kernel(p, p, bandwidth)
    k_qq = rbf_kernel(q, q, bandwidth)
    k_up = rbf_kernel(anchor, p, bandwidth)
    k_uq = rbf_kernel(anchor, q, bandwidth)

    h_matrix = 0.5 * (-k_up - k_up.T + k_uq + k_uq.T + k_pp - k_qq)
    off_diag = ~np.eye(n, dtype=bool)
    statistic = h_matrix[off_diag].mean()
    row_means = h_matrix[off_diag].reshape(n, n - 1).mean(axis=1)
    se = 2.0 * row_means.std(ddof=1) / np.sqrt(n)
    return statistic, h_matrix, max(se, 1e-12)


def bandwidth_grid(anchor, p, q, num_bandwidths):
    pooled = np.vstack((anchor, p, q))
    dists = pairwise_sq_dists(pooled, pooled)
    positive = dists[dists > 0]
    median_bandwidth = np.sqrt(np.median(positive)) if len(positive) else 1.0
    return median_bandwidth * np.logspace(-1.0, 1.0, num_bandwidths)


def select_bandwidth_and_direction(rng, sample_size, anchor_location, num_bandwidths):
    anchor, p, q = sample_triplet(rng, sample_size, anchor_location)
    best = None
    rows = []
    for bandwidth in bandwidth_grid(anchor, p, q, num_bandwidths):
        statistic, _, se = rst_statistic(anchor, p, q, bandwidth)
        score = abs(statistic) / se
        row = {
            "bandwidth": bandwidth,
            "statistic": statistic,
            "se": se,
            "score": score,
        }
        rows.append(row)
        if best is None or score > best["score"]:
            best = row

    direction = 1 if best["statistic"] > 0 else -1
    return best["bandwidth"], direction, rows


def wild_bootstrap_centered(rng, h_matrix, statistic, num_bootstrap):
    n = len(h_matrix)
    bootstrap_stats = np.empty(num_bootstrap)
    for b in range(num_bootstrap):
        weights = rng.exponential(scale=1.0, size=n)
        weights = weights / weights.mean()
        weighted_h = np.outer(weights, weights) * h_matrix
        bootstrap_statistic = (weighted_h.sum() - np.trace(weighted_h)) / (n * (n - 1.0))
        bootstrap_stats[b] = bootstrap_statistic - statistic
    return bootstrap_stats


def quantile(values, alpha):
    values = np.sort(values)
    index = int(np.ceil((1.0 - alpha) * len(values))) - 1
    return values[index]


def one_sided_reject(statistic, bootstrap_centered, direction, alpha):
    threshold = quantile(direction * bootstrap_centered, alpha)
    return int(direction * statistic > threshold)


def two_sided_reject(statistic, bootstrap_centered, alpha):
    threshold = quantile(np.abs(bootstrap_centered), alpha)
    return int(abs(statistic) > threshold)


def one_test(rng, args, anchor_location):
    bandwidth, amd_direction, _ = select_bandwidth_and_direction(
        rng,
        args.selection_size,
        anchor_location,
        args.num_bandwidths,
    )
    anchor, p, q = sample_triplet(rng, args.test_size, anchor_location)
    statistic, h_matrix, _ = rst_statistic(anchor, p, q, bandwidth)

    bootstrap_centered = wild_bootstrap_centered(
        rng, h_matrix, statistic, args.num_bootstrap
    )
    fixed_p_direction = -1
    fixed_q_direction = 1
    random_direction = rng.choice([fixed_p_direction, fixed_q_direction])

    return {
        "amd_reject": one_sided_reject(
            statistic, bootstrap_centered, amd_direction, args.alpha
        ),
        "fixed_p_reject": one_sided_reject(
            statistic, bootstrap_centered, fixed_p_direction, args.alpha
        ),
        "fixed_q_reject": one_sided_reject(
            statistic, bootstrap_centered, fixed_q_direction, args.alpha
        ),
        "random_direction_reject": one_sided_reject(
            statistic, bootstrap_centered, random_direction, args.alpha
        ),
        "two_sided_reject": two_sided_reject(
            statistic, bootstrap_centered, args.alpha
        ),
        "amd_direction": amd_direction,
        "random_direction": random_direction,
        "bandwidth": bandwidth,
        "statistic": statistic,
    }


def run_scenario(args, name, anchor_location):
    rng = np.random.default_rng(args.seed + int(round((anchor_location + 1.0) * 1000)))
    trials = [one_test(rng, args, anchor_location) for _ in range(args.reps)]
    amd_rejections = np.array([trial["amd_reject"] for trial in trials])
    fixed_p_rejections = np.array([trial["fixed_p_reject"] for trial in trials])
    fixed_q_rejections = np.array([trial["fixed_q_reject"] for trial in trials])
    random_rejections = np.array(
        [trial["random_direction_reject"] for trial in trials]
    )
    two_sided_rejections = np.array([trial["two_sided_reject"] for trial in trials])
    amd_directions = np.array([trial["amd_direction"] for trial in trials])
    random_directions = np.array([trial["random_direction"] for trial in trials])
    bandwidths = np.array([trial["bandwidth"] for trial in trials])
    statistics = np.array([trial["statistic"] for trial in trials])

    return {
        "scenario": name,
        "anchor_location": anchor_location,
        "selection_size": args.selection_size,
        "test_size": args.test_size,
        "amd_rejection_rate": amd_rejections.mean(),
        "fixed_p_rejection_rate": fixed_p_rejections.mean(),
        "fixed_q_rejection_rate": fixed_q_rejections.mean(),
        "random_direction_rejection_rate": random_rejections.mean(),
        "two_sided_rejection_rate": two_sided_rejections.mean(),
        "amd_p_closer_rate": np.mean(amd_directions < 0),
        "random_p_closer_rate": np.mean(random_directions < 0),
        "median_bandwidth": np.median(bandwidths),
        "mean_statistic": statistics.mean(),
    }


def write_summary(output_dir, rows):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "gaussian_amd_summary.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Gaussian AMD relative similarity demo.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--selection_size", type=int, default=120)
    parser.add_argument("--test_size", type=int, default=120)
    parser.add_argument("--reps", type=int, default=150)
    parser.add_argument("--num_bootstrap", type=int, default=200)
    parser.add_argument("--num_bandwidths", type=int, default=21)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output_dir", type=Path, default=Path("Results/demos"))
    return parser.parse_args()


def main():
    args = parse_args()
    scenarios = [
        ("P closer", -0.30),
        ("No relative difference", 0.00),
        ("Q closer", 0.30),
    ]
    rows = [run_scenario(args, name, anchor_location) for name, anchor_location in scenarios]
    output_path = write_summary(args.output_dir, rows)

    for row in rows:
        print(
            f"{row['scenario']:22s} AMD rejection={row['amd_rejection_rate']:.3f}, "
            f"fixed-P rejection={row['fixed_p_rejection_rate']:.3f}, "
            f"fixed-Q rejection={row['fixed_q_rejection_rate']:.3f}, "
            f"random-F rejection={row['random_direction_rejection_rate']:.3f}, "
            f"two-sided rejection={row['two_sided_rejection_rate']:.3f}, "
            f"AMD P-closer direction={row['amd_p_closer_rate']:.3f}"
        )
    print(f"\nSaved CSV summary to {output_path}")


if __name__ == "__main__":
    main()
