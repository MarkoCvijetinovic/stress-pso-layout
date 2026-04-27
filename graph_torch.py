import numpy as np
import torch

def make_stress_tensors(target_distances, weights="inverse_square", device="cuda"):
    d = torch.tensor(target_distances, dtype=torch.float32, device=device)

    mask = torch.triu(torch.isfinite(d) & (d > 0), diagonal=1)

    if weights == "constant":
        w = torch.ones_like(d)
    elif weights == "inverse_square":
        w = torch.zeros_like(d)
        w[mask] = 1.0 / (d[mask] ** 2)
    else:
        raise ValueError(f"Unknown weight mode: {weights}")

    return d, w, mask

def compute_stress_torch(
    positions_batch,
    target_distances_tensor,
    weights_tensor,
    mask,
    normalize_stress=True,
):
    device = target_distances_tensor.device

    x = torch.as_tensor(positions_batch, dtype=torch.float32, device=device)

    if x.ndim == 2:
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, 2)

    diff = x[:, :, None, :] - x[:, None, :, :]
    euclidean = torch.linalg.norm(diff, dim=3)

    terms = weights_tensor[mask] * (
        euclidean[:, mask] - target_distances_tensor[mask]
    ) ** 2

    if normalize_stress:
        return terms.sum(dim=1) / weights_tensor[mask].sum()

    return terms.sum(dim=1)
