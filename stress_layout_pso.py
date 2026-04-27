import numpy as np
import networkx as nx
import torch
from functools import partial
from graph import compute_stress, all_paths, random_layout
from pso import PSO, FitnessFunction, InitializeFunction, RepairFunction
from batched_pso import BatchedPSO, BatchedFitnessFunction
from visualization import draw_layout
from batched_graph import make_stress_tensors, compute_stress_torch

def initialize_graph_layout(G: nx.Graph, nodes: list, scale=1.0):
    return random_layout(G, nodes, scale=scale).flatten()

def graph_layout_repair(
    position: np.ndarray,
    velocity: np.ndarray,
    stagnation_counter: int,
    bounds = None,
    max_velocity=None,
    mutation=0.0,
    stagnation_limit=30,
    normalize_coords: bool = True,
):
    # Position clipping
    if bounds is not None:
        lower = bounds[:, 0]
        upper = bounds[:, 1]
        position = np.clip(position, lower, upper)

    # Velocity clipping
    if max_velocity is not None:
        velocity = np.clip(velocity, -max_velocity, max_velocity)

    # Graph centering
    if normalize_coords:
        coords = position.reshape(-1, 2)
        coords -= coords.mean(axis=0)
        position = coords.flatten()

    # Mutation
    if mutation > 0 and stagnation_counter > stagnation_limit:
        position += mutation * np.random.normal(0, 1, size=position.shape)
        position = np.clip(position, lower, upper)
        velocity = mutation * np.random.uniform(-0.5, 0.5, size=velocity.shape)
        stagnation_counter = 0

    return position, velocity, stagnation_counter

def stress_layout_pso_functions(
    G: nx.Graph, 
    distances: np.ndarray, 
    nodes: list, 
    bound_size: float = 10.0, 
    batched=False
) -> tuple[callable, InitializeFunction, RepairFunction]:
    '''
    Builds fitness, initialize and repair functions required by PSO for Stress-Based Graph Layout problem
    It can build fitness functions for PSO or BatchedPSO depending on the argument "batched"

    Args:
        G: graph
        distances: Distances between all pairs in G
        nodes: List of nodes in G
        bound_size: Used for clipping positions outside of [-bound_size / 2, bound_size / 2]
        batched: If false returns FitnessFunction, otherwise BatchedFitnessFunction

    '''
    n = len(nodes)
    dim = 2 * n

    bounds = np.array([[-bound_size / 2, bound_size / 2]] * dim)

    initialize = partial(initialize_graph_layout, G, nodes, scale=1.0)
    repair = partial(graph_layout_repair, bounds=bounds)

    fitness = None
    
    if not batched: 
        fitness = partial(compute_stress, target_distances=distances, weights="inverse_square")

    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        d_torch, w_torch, mask = make_stress_tensors(
            distances,
            weights="inverse_square",
            device=device,
        )

        fitness = lambda position: compute_stress_torch(
            position,
            d_torch,
            w_torch,
            mask,
            normalize_stress=True,
        )

    return fitness, initialize, repair

if __name__ == "__main__":
    G = nx.balanced_tree(3, 4)
    distances, nodes = all_paths(G)

    fitness, initialize, repair = stress_layout_pso_functions(G, distances, nodes, batched=True)

    best_layout, best_value = BatchedPSO(
        batched_fitness_function=fitness,
        initialize_function=initialize,
        particle_count=100,
        iterations=5000,
        repair_function=repair,
        c_inertia=0.8,
        c_social=1.7,
        c_cognitive=0.8,
        callback_function=lambda iteration, best_position, best_value: print(f"Iteration {iteration + 1}: {best_value:.4f}")
    )

    best_layout_2d = best_layout.reshape(-1, 2)
    draw_layout(G, nodes, best_layout_2d)
    
    print("Best stress:", best_value) 
    print("Best layout shape:", best_layout_2d.shape)