import numpy as np
import networkx as nx
from functools import partial
from graph import compute_stress, all_paths, random_layout, draw_layout
from pso import PSO

def initialize_graph_layout(G, nodes, scale=1.0):
    return random_layout(G, nodes, scale=scale).flatten()

def graph_layout_repair(
    position,
    velocity,
    stagnation_counter,
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

if __name__ == "__main__":
    '''
    G1 = nx.complete_graph(5)
    G2 = nx.complete_graph(5)
    G = nx.disjoint_union(G1, G2)
    G.add_edge(2, 7)
    '''

    #G = nx.karate_club_graph()
    #G = nx.complete_graph(10)
    #G = nx.path_graph(30)
    #G = nx.cycle_graph(30)
    #G = nx.grid_2d_graph(8, 8)
    #G = nx.balanced_tree(4,3)
    G = nx.connected_caveman_graph(10, 10)

    distances, nodes = all_paths(G)

    n = len(nodes)
    dim = 2 * n

    bounds = np.array([[-5.0, 5.0]] * dim)

    fitness = partial(compute_stress, target_distances=distances, weights="inverse_square")
    initialize = partial(initialize_graph_layout, G, nodes, scale=1.0)
    repair = partial(graph_layout_repair, bounds=bounds)


    best_layout, best_value = PSO(fitness_function=fitness, initialize_function=initialize,
                                  particle_count=50, iterations=4000, repair_function=repair,
                                  c_inertia=0.8, c_social=1.7, c_cognitive=0.8)

    best_layout_2d = best_layout.reshape(-1, 2)
    draw_layout(G, nodes, best_layout_2d)

    print("Best stress:", best_value)
    print("Best layout shape:", best_layout_2d.shape)