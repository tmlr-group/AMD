import numpy as np
import torch


def _flatten_samples(x):
    if x.dim() > 2:
        x = x.reshape(x.shape[0], -1)
    return x.detach()


def _subsample_triplet(X, Y, Z, max_samples, seed):
    n = min(X.shape[0], Y.shape[0], Z.shape[0])
    if max_samples is not None:
        n = min(n, int(max_samples))
    rng = np.random.default_rng(int(seed))
    ind_x = torch.as_tensor(rng.choice(X.shape[0], n, replace=False), device=X.device)
    ind_y = torch.as_tensor(rng.choice(Y.shape[0], n, replace=False), device=Y.device)
    ind_z = torch.as_tensor(rng.choice(Z.shape[0], n, replace=False), device=Z.device)
    return X.index_select(0, ind_x), Y.index_select(0, ind_y), Z.index_select(0, ind_z)


def _pairwise_sq_dists(x, y):
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    dists = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
    dists[dists < 0] = 0
    return dists


def _pairwise_l1_dists(x, y):
    return torch.cdist(x, y, p=1) * np.sqrt(2.0)


def _pairwise_l2_dists(x, y):
    return torch.sqrt(_pairwise_sq_dists(x, y).clamp_min(1e-12))


def _distance_matrix(x, y, kernel):
    if kernel == "Laplace":
        return _pairwise_l1_dists(x, y)
    if kernel == "Energy":
        return _pairwise_l2_dists(x, y)
    return _pairwise_sq_dists(x, y)


def _kernel_matrix(x, y, scale, kernel):
    return torch.exp(-_distance_matrix(x, y, kernel) / scale.clamp_min(1e-12))


def _bandwidth_grid(X, Y, Z, kernel, num_bandwidths):
    pooled = torch.cat((X, Y, Z), dim=0)
    dists = _distance_matrix(pooled, pooled, kernel)
    positive = dists[dists > 0]
    if positive.numel() == 0:
        base = torch.tensor(1.0, device=X.device, dtype=X.dtype)
    else:
        base = torch.median(positive).clamp_min(1e-12)
    multipliers = torch.as_tensor(
        np.logspace(-1.5, 1.5, int(num_bandwidths)),
        device=X.device,
        dtype=X.dtype,
    )
    return (base * multipliers).clamp_min(1e-12)


def _rst_statistic_and_se(X, Y, Z, scale, kernel):
    n = X.shape[0]
    if kernel == "Energy":
        Dx = _pairwise_l2_dists(X, X)
        Dy = _pairwise_l2_dists(Y, Y)
        Dzx = _pairwise_l2_dists(Z, X)
        Dzy = _pairwise_l2_dists(Z, Y)
        h_matrix = Dzx + Dzx.t() - Dzy - Dzy.t() - Dx + Dy
        mask = ~torch.eye(n, dtype=torch.bool, device=X.device)
        off_diag = h_matrix[mask].reshape(n, n - 1)
        statistic = off_diag.mean()
        row_means = off_diag.mean(dim=1)
        se = 2.0 * row_means.std(unbiased=True).clamp_min(1e-12) / np.sqrt(n)
        return statistic, se

    Kx = _kernel_matrix(X, X, scale, kernel)
    Ky = _kernel_matrix(Y, Y, scale, kernel)
    Kzx = _kernel_matrix(Z, X, scale, kernel)
    Kzy = _kernel_matrix(Z, Y, scale, kernel)
    h_matrix = 0.5 * (-Kzx - Kzx.t() + Kzy + Kzy.t() + Kx - Ky)

    mask = ~torch.eye(n, dtype=torch.bool, device=X.device)
    off_diag = h_matrix[mask].reshape(n, n - 1)
    statistic = off_diag.mean()
    row_means = off_diag.mean(dim=1)
    se = 2.0 * row_means.std(unbiased=True).clamp_min(1e-12) / np.sqrt(n)
    return statistic, se


def _kernel_candidates(kernel):
    if kernel == "Laplace":
        return ["Laplace", "Gaussian", "Energy"]
    if kernel == "Gaussian":
        return ["Gaussian", "Laplace", "Energy"]
    return ["Gaussian", "Laplace", "Energy"]


def _best_grid_row(X, Y, Z, kernels, num_bandwidths):
    best = None
    rows = []
    for kernel in kernels:
        scales = (
            [torch.tensor(1.0, device=X.device, dtype=X.dtype)]
            if kernel == "Energy"
            else _bandwidth_grid(X, Y, Z, kernel, num_bandwidths)
        )
        for scale in scales:
            statistic, se = _rst_statistic_and_se(X, Y, Z, scale, kernel)
            score = torch.abs(statistic) / se
            row = {
                "kernel": kernel,
                "scale": scale.detach(),
                "statistic": statistic.detach(),
                "se": se.detach(),
                "score": score.detach(),
            }
            rows.append(row)
            if best is None or row["score"].item() > best["score"].item():
                best = row
    return best, rows


def select_rst_direction_grid(
    X,
    Y,
    Z,
    kernel="Gaussian",
    seed=1102,
    max_samples=2048,
    num_bandwidths=15,
    num_bags=3,
    bag_fraction=0.8,
    top_k=7,
):
    """Select the AMD direction with a stable, practical kernel-grid criterion.

    The returned direction follows the existing Exp_code convention:
    F=1 means the signed statistic is expected to be positive; F=-1 means it is
    expected to be negative.  This routine does not use labels or dataset
    identities, only the optimization samples X, Y, and anchor Z.
    """
    X = _flatten_samples(X)
    Y = _flatten_samples(Y)
    Z = _flatten_samples(Z)
    X, Y, Z = _subsample_triplet(X, Y, Z, max_samples, seed)
    kernels = _kernel_candidates(kernel)

    with torch.no_grad():
        best, rows = _best_grid_row(X, Y, Z, kernels, num_bandwidths)
        top_rows = sorted(rows, key=lambda row: row["score"].item(), reverse=True)[:top_k]
        top_vote = sum(
            torch.sign(row["statistic"]).item() * row["score"].item()
            for row in top_rows
        )

        rng = np.random.default_rng(int(seed) + 7919)
        bag_votes = []
        n = X.shape[0]
        bag_n = max(4, min(n, int(round(n * float(bag_fraction)))))
        for _ in range(int(num_bags)):
            ind_x = torch.as_tensor(rng.choice(n, bag_n, replace=False), device=X.device)
            ind_y = torch.as_tensor(rng.choice(n, bag_n, replace=False), device=Y.device)
            ind_z = torch.as_tensor(rng.choice(n, bag_n, replace=False), device=Z.device)
            bag_best, _ = _best_grid_row(
                X.index_select(0, ind_x),
                Y.index_select(0, ind_y),
                Z.index_select(0, ind_z),
                kernels,
                num_bandwidths,
            )
            bag_votes.append(
                torch.sign(bag_best["statistic"]).item() * bag_best["score"].item()
            )

        vote = float(np.sum(bag_votes) + top_vote)
        if vote > 0:
            direction = 1
        elif vote < 0:
            direction = -1
        else:
            direction = 1 if best["statistic"].item() >= 0 else -1

        direction_rows = [
            row for row in rows if torch.sign(row["statistic"]).item() == direction
        ]
        selected = (
            max(direction_rows, key=lambda row: row["score"].item())
            if direction_rows
            else best
        )
        bag_signs = np.sign(np.asarray(bag_votes))
        diagnostics = {
            "direction": direction,
            "selected_kernel": selected["kernel"],
            "selected_scale": float(selected["scale"].detach().cpu()),
            "selected_statistic": float(selected["statistic"].detach().cpu()),
            "selected_score": float(selected["score"].detach().cpu()),
            "bag_agreement": float(np.mean(bag_signs == direction)) if len(bag_signs) else 1.0,
            "num_samples": int(X.shape[0]),
        }
    return direction, diagnostics
