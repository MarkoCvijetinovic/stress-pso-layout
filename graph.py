import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from typing import Literal
from visualization import draw_layout

WeightMode = Literal["constant", "inverse_square"]


def all_paths(G: nx.Graph, nodes: list | None = None, normalize_paths: bool = True) -> tuple[np.ndarray, list]:
    """
    Computes all-pairs shortest path distances.

    Returns:
        distances: n x n NumPy array
        nodes: node ordering used in the matrix
    """
    if nodes is None:
        nodes = list(G.nodes())

    distances = nx.floyd_warshall_numpy(G, nodelist=nodes)

    if(normalize_paths):
        finite = distances[np.isfinite(distances) & (distances > 0)]
        scale = np.mean(finite)
        distances = distances / scale

    return np.asarray(distances, dtype=float), nodes

def compute_stress(
    positions: np.ndarray,
    target_distances: np.ndarray,
    weights: WeightMode = "inverse_square",
    normalize_stress: bool = True,
) -> float:
    """
    Computes graph drawing stress:

        sum_{i < j} w_ij (||x_i - x_j|| - d_ij)^2

    Args:
        positions: n x 2 array of node coordinates, or flattened n x 1 array
        target_distances: n x n array of graph-theoretic distances
        weights: "constant" or "inverse_square"

    Returns:
        Stress value.
    """

    positions = np.asarray(positions, dtype=float)

    if positions.ndim == 1:
        positions = positions.reshape(-1, 2)

    d = np.asarray(target_distances, dtype=float)

    # Compute all pairwise differences:
    # positions[:, None, :] → (n, 1, 2)
    # positions[None, :, :] → (1, n, 2)
    # Result: diff[i, j] = x_i - x_j → shape (n, n, 2)
    diff = positions[:, None, :] - positions[None, :, :]

    # Compute Euclidean distances for all pairs → shape (n, n)
    euclidean = np.linalg.norm(diff, axis=2)

    # Build mask selecting valid pairs:
    # - finite distances (ignore disconnected nodes)
    # - d > 0 (ignore self-distances)
    # - upper triangle only (i < j) to avoid duplicates
    mask = np.triu(np.isfinite(d) & (d > 0), k=1)

    if weights == "constant":
        w = np.ones_like(d)
    elif weights == "inverse_square":
        w = np.zeros_like(d)
        w[mask] = 1.0 / (d[mask] ** 2)
    else:
        raise ValueError(f"Unknown weight mode: {weights}")

    # Compute stress contributions only for valid pairs
    # euclidean[mask] and d[mask] are flattened vectors of valid pairs
    stress_terms = w[mask] * (euclidean[mask] - d[mask]) ** 2

    if normalize_stress:
        return float(stress_terms.sum() / w[mask].sum())

    return float(stress_terms.sum())

def random_layout(G: nx.Graph, nodes: list, scale: float = 1.0) -> np.ndarray:
    """
    Creates random 2D positions matching the given node ordering.
    """
    return np.random.uniform(-scale, scale, size=(len(nodes), 2))


if __name__ == "__main__":
    G = nx.karate_club_graph()

    distances, nodes = all_paths(G)

    positions = random_layout(G, nodes)

    stress = compute_stress(
        positions=positions,
        target_distances=distances,
        weights="inverse_square",
    )

    print(f"Initial stress: {stress:.4f}")

    draw_layout(G, nodes, positions)