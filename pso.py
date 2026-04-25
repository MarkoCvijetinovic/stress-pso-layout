import numpy as np
import networkx as nx
from functools import partial
from graph import compute_stress, all_paths, random_layout


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
        self.velocity = np.zeros_like(self.position)

        self.value = self.f(self.position)

        self.best_position = self.position.copy()
        self.best_value = self.value

        self.update_swarm_best()

    def update_swarm_best(self):
        if self.value < Particle.swarm_best_value:
            Particle.swarm_best_value = self.value
            Particle.swarm_best_position = self.position.copy()

    def apply_bounds(self):
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        self.position = np.clip(self.position, lower, upper)

    def move(self):
        r_social = np.random.random(size=self.position.shape)
        r_cognitive = np.random.random(size=self.position.shape)

        social_component = self.c_social * r_social * (Particle.swarm_best_position - self.position) 

        cognitive_component = self.c_cognitive * r_cognitive * (self.best_position - self.position)

        inertia_component = self.c_inertia * self.velocity

        self.velocity = inertia_component + social_component + cognitive_component
        self.position = self.position + self.velocity

        self.apply_bounds()

        self.value = self.f(self.position)

        if self.value < self.best_value:
            self.best_value = self.value
            self.best_position = self.position.copy()

        self.update_swarm_best()


def PSO(
    fitness_function: callable,
    initialize_function: callable,
    bounds: np.ndarray,
    particle_count: int = 100,
    iterations: int = 100,
    c_inertia: float = 0.2,
    c_social: float = 0.1,
    c_cognitive: float = 1.0,
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


def initialize_graph_layout(G, nodes, scale=1.0):
    return random_layout(G, nodes, scale=scale).flatten()


if __name__ == "__main__":
    G = nx.karate_club_graph()

    distances, nodes = all_paths(G)

    n = len(nodes)
    dim = 2 * n

    bounds = np.array([[-5.0, 5.0]] * dim)

    fitness = partial(compute_stress, target_distances=distances, weights="inverse_square")

    initialize = partial(initialize_graph_layout, G, nodes, scale=1.0)

    best_layout, best_value = PSO(fitness_function=fitness, initialize_function=initialize, 
                                  bounds=bounds, particle_count=50, iterations=100)

    best_layout_2d = best_layout.reshape(-1, 2)

    print("Best stress:", best_value)
    print("Best layout shape:", best_layout_2d.shape)