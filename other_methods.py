import numpy as np
import networkx as nx
from graph import all_paths
from stress_layout_pso import stress_layout_pso_functions
from visualization import draw_layout
import matplotlib.pyplot as plt
from graph import compute_stress

def batched_brute_force_layout(
    batched_fitness_function,
    initialize_function,
    samples: int = 1000000,
    batch_size: int = 1000000,
    callback_function=None,
):
    best_position = None
    best_value = float("inf")

    checked = 0

    while checked < samples:
        current_batch_size = min(batch_size, samples - checked)

        positions = np.stack([
            initialize_function()
            for _ in range(current_batch_size)
        ])

        values = batched_fitness_function(positions)

        best_idx = int(np.argmin(values.cpu()))
        batch_best_value = float(values[best_idx])

        if batch_best_value < best_value:
            best_value = batch_best_value
            best_position = positions[best_idx].copy()

        checked += current_batch_size

        if callback_function is not None:
            callback_function(checked, best_position, best_value)

    return best_position, best_value

if __name__ == "__main__":
    G = nx.connected_caveman_graph(10, 10)
    distances, nodes = all_paths(G)

    '''
    fitness, initialize, _ = stress_layout_pso_functions(G, distances, nodes, batched=True)

    best_layout, best_value = batched_brute_force_layout(fitness, initialize)

    best_layout_2d = best_layout.reshape(-1, 2)
    draw_layout(G, nodes, best_layout_2d)
    
    print("Best stress:", best_value) 
    print("Best layout shape:", best_layout_2d.shape)
    '''

    #pos = nx.spring_layout(G, iterations=500)
    pos = nx.kamada_kawai_layout(G)
    best_layout_2d = np.array(list(pos.values()))

    print("Best stress:", compute_stress(target_distances=distances, positions=best_layout_2d)) 
    draw_layout(G, nodes, best_layout_2d)