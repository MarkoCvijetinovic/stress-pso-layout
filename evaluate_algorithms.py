import time
import numpy as np
import networkx as nx
import pandas as pd

from graph import all_paths, compute_stress
from stress_layout_pso import stress_layout_pso_functions
from batched_pso import BatchedPSO
from visualization import draw_layout


def layout_array_from_nx_pos(pos: dict, nodes: list) -> np.ndarray:
    return np.array([pos[node] for node in nodes], dtype=float)


def brute_force_optimizer(
    G,
    distances,
    nodes,
    samples=50_000,
    batch_size=1_000,
):
    fitness, initialize, _ = stress_layout_pso_functions(
        G,
        distances,
        nodes,
        batched=True,
    )

    best_position = None
    best_value = float("inf")
    checked = 0

    while checked < samples:
        current_batch_size = min(batch_size, samples - checked)

        positions = np.stack([
            initialize()
            for _ in range(current_batch_size)
        ])

        values = fitness(positions)

        # Supports both torch tensor and numpy array
        if hasattr(values, "detach"):
            values_np = values.detach().cpu().numpy()
        else:
            values_np = np.asarray(values)

        best_idx = int(np.argmin(values_np))
        batch_best_value = float(values_np[best_idx])

        if batch_best_value < best_value:
            best_value = batch_best_value
            best_position = positions[best_idx].copy()

        checked += current_batch_size

    return best_position, best_value


def spring_optimizer(
    G,
    distances,
    nodes,
    iterations=8000,
):
    pos = nx.spring_layout(G, iterations=iterations, seed=0)
    layout = layout_array_from_nx_pos(pos, nodes)
    value = compute_stress(layout, distances, weights="inverse_square")
    return layout.flatten(), value


def kamada_kawai_optimizer(
    G,
    distances,
    nodes,
):
    pos = nx.kamada_kawai_layout(G)
    layout = layout_array_from_nx_pos(pos, nodes)
    value = compute_stress(layout, distances, weights="inverse_square")
    return layout.flatten(), value


def batched_pso_optimizer(
    G,
    distances,
    nodes,
    particle_count=400,
    iterations=6000,
):
    fitness, initialize, repair = stress_layout_pso_functions(
        G,
        distances,
        nodes,
        batched=True,
    )

    best_layout, best_value = BatchedPSO(
        batched_fitness_function=fitness,
        initialize_function=initialize,
        particle_count=particle_count,
        iterations=iterations,
        repair_function=repair,
        c_inertia=0.8,
        c_social=1.4,
        c_cognitive=0.8,
    )

    return best_layout, best_value


def default_graphs():
    return {
        "Path 30": nx.path_graph(30),
        "Cycle 30": nx.cycle_graph(30),
        "Grid 8x8": nx.convert_node_labels_to_integers(nx.grid_2d_graph(8, 8)),
        "Caveman 8x8": nx.connected_caveman_graph(8, 8),
        "Karate Club": nx.karate_club_graph(),
        "Barbasi Albert 150x3": nx.barabasi_albert_graph(150, 3),
        "Balanced Tree 4x4": nx.balanced_tree(4, 4),
    }


def evaluate_optimizers(optimizers, graphs=None):
    if graphs is None:
        graphs = default_graphs()

    results = []

    for graph_name, G in graphs.items():
        distances, nodes = all_paths(G)

        for optimizer_name, optimizer in optimizers.items():
            print(f"Running {optimizer_name} on {graph_name}...")

            start_time = time.perf_counter()
            best_layout, best_value = optimizer(G, distances, nodes)
            end_time = time.perf_counter()

            results.append({
                "graph": graph_name,
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "algorithm": optimizer_name,
                "stress": best_value,
                "time_seconds": end_time - start_time,
            })

    return pd.DataFrame(results)


if __name__ == "__main__":
    optimizers = {
        "Brute force": brute_force_optimizer,
        "Spring layout": spring_optimizer,
        "Kamada-Kawai": kamada_kawai_optimizer,
        "Batched PSO": batched_pso_optimizer,
    }

    df = evaluate_optimizers(optimizers)

    print(df)

    df.to_csv("data/showcase/experiment_results.csv", index=False)