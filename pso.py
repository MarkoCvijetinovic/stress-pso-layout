import numpy as np
import networkx as nx
from functools import partial
from graph import compute_stress, all_paths, random_layout, draw_layout


class Particle:
    swarm_best_position = None
    swarm_best_value = float("inf")

    def __init__(
        self,
        fitness_function: callable,
        initialize_function: callable,
        bounds: np.ndarray,
        c_inertia: float,
        c_social: float,
        c_cognitive: float,
    ):
        self.f = fitness_function
        self.bounds = bounds

        self.c_inertia = c_inertia
        self.c_social = c_social
        self.c_cognitive = c_cognitive

        self.position = initialize_function()
        self.velocity = np.random.uniform(-1, 1, size=self.position.shape)

        self.value = self.f(self.position)

        self.best_position = self.position.copy()
        self.best_value = self.value

        self.update_swarm_best()

    def update_swarm_best(self):
        if self.value < Particle.swarm_best_value:
            #print("=====================CHANGE======================") 
            Particle.swarm_best_value = self.value
            Particle.swarm_best_position = self.position.copy()

    def apply_bounds(self):
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        self.position = np.clip(self.position, lower, upper)

    def apply_velocity_bounds(self, max_velocity=0.2):
        self.velocity = np.clip(self.velocity, -max_velocity, max_velocity)

    def move(self):
        r_social = np.random.random(size=self.position.shape)
        r_cognitive = np.random.random(size=self.position.shape)
        # r_social = np.random.random()
        # r_cognitive = np.random.random()

        social_component = (
            self.c_social
            * r_social
            * (Particle.swarm_best_position - self.position)
        )

        cognitive_component = (
            self.c_cognitive
            * r_cognitive
            * (self.best_position - self.position)
        )

        inertia_component = self.c_inertia * self.velocity

        self.velocity = inertia_component + social_component + cognitive_component

        self.position = self.position + self.velocity
        self.apply_bounds()

        self.value = self.f(self.position)
        #print(self.value)

        if self.value < self.best_value:
            self.best_value = self.value
            self.best_position = self.position.copy()

        self.update_swarm_best()

def initialize_graph_layout(G, nodes, scale=1.0):
    return random_layout(G, nodes, scale=scale).flatten()

def PSO(
    fitness_function: callable,
    initialize_function: callable,
    bounds: np.ndarray,
    particle_count: int = 100,
    iterations: int = 100,
    c_inertia: float = 0.7,
    c_social: float = 0.4,
    c_cognitive: float = 1.8,
) -> tuple[np.ndarray, float]:

    Particle.swarm_best_position = None
    Particle.swarm_best_value = float("inf")

    particles = [
        Particle(fitness_function, initialize_function, bounds, c_inertia, c_social, c_cognitive)
        for _ in range(particle_count)
    ]

    for iteration in range(iterations):
        for particle in particles:
            particle.move()

        print(f"Iteration {iteration + 1}: {Particle.swarm_best_value:.4f}")

    return Particle.swarm_best_position, Particle.swarm_best_value


if __name__ == "__main__":
    #G = nx.cycle_graph(10)
    #G = nx.path_graph(10)
    #G = nx.karate_club_graph

    G1 = nx.complete_graph(5)
    G2 = nx.complete_graph(5)
    G = nx.disjoint_union(G1, G2)
    G.add_edge(2, 7)

    #G = nx.grid_2d_graph(5,5)

    distances, nodes = all_paths(G)

    n = len(nodes)
    dim = 2 * n

    bounds = np.array([[-5.0, 5.0]] * dim)

    fitness = partial(compute_stress, target_distances=distances, weights="inverse_square")

    initialize = partial(initialize_graph_layout, G, nodes, scale=1.0)

    best_layout, best_value = PSO(fitness_function=fitness, initialize_function=initialize, bounds=bounds,
                                  particle_count=50, iterations=300)

    best_layout_2d = best_layout.reshape(-1, 2)
    #draw_layout(G, nodes, random_layout(G, nodes))
    draw_layout(G, nodes, best_layout_2d)

    print("Best stress:", best_value)
    print("Best layout shape:", best_layout_2d.shape)