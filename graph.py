import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from typing import Literal


WeightMode = Literal["constant", "inverse_square"]


def all_paths(G: nx.Graph, nodes: list | None = None) -> tuple[np.ndarray, list]:
    """
    Computes all-pairs shortest path distances.

    Returns:
        distances: n x n NumPy array
        nodes: node ordering used in the matrix
    """
    if nodes is None:
        nodes = list(G.nodes())

    distances = nx.floyd_warshall_numpy(G, nodelist=nodes)
    return np.asarray(distances, dtype=float), nodes


def compute_stress(
    positions: np.ndarray,
    target_distances: np.ndarray,
    weights: WeightMode = "inverse_square",
) -> float:
    """
    Computes graph drawing stress:

        sum_{i < j} w_ij (||x_i - x_j|| - d_ij)^2

    Args:
        positions: n x 2 array of node coordinates
        target_distances: n x n array of graph-theoretic distances
        weights: "constant" or "inverse_square"

    Returns:
        Stress value.
    """
    positions = np.asarray(positions, dtype=float)
    target_distances = np.asarray(target_distances, dtype=float)

    n = positions.shape[0]

    if positions.shape != (n, 2):
        raise ValueError("positions must have shape (n, 2)")

    if target_distances.shape != (n, n):
        raise ValueError("target_distances must have shape (n, n)")

    stress = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            d_ij = target_distances[i, j]

            # Skip disconnected pairs
            if not np.isfinite(d_ij) or d_ij == 0:
                continue

            euclidean_distance = np.linalg.norm(positions[i] - positions[j])

            if weights == "constant":
                w_ij = 1.0
            elif weights == "inverse_square":
                w_ij = 1.0 / (d_ij ** 2)
            else:
                raise ValueError(f"Unknown weight mode: {weights}")

            stress += w_ij * (euclidean_distance - d_ij) ** 2

    return float(stress)


def random_layout(G: nx.Graph, nodes: list, scale: float = 1.0) -> np.ndarray:
    """
    Creates random 2D positions matching the given node ordering.
    """
    return np.random.uniform(-scale, scale, size=(len(nodes), 2))


def draw_layout(G: nx.Graph, nodes: list, positions: np.ndarray) -> None:
    """
    Draws graph using positions stored as an n x 2 array.
    """
    pos_dict = {
        node: positions[i]
        for i, node in enumerate(nodes)
    }

    nx.draw(G, pos=pos_dict, with_labels=True, node_color="lightblue", edge_color="gray")
    plt.axis("equal")
    plt.show()


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