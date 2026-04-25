import numpy as np
import networkx as nx
from graph import compute_stress, all_paths, random_layout
from functools import partial

class Particle:
    swarm_best_position = None
    swarm_best_value = float('inf')

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
        self.value = self.f(self.position)

        self.best_position = self.position.copy()
        self.best_value = self.value

        if self.value < Particle.swarm_best_value:
            Particle.swarm_best_value = self.value
            Particle.swarm_best_position = self.position.copy()

    def move(self):
        pass

def PSO( 
        fitness_function: callable, 
        initialize_function: callable,
        bounds: np.ndarray,
        particle_count: int = 100, 
        iterations: int = 100,
        c_inertia: float = 0.2,
        c_social: float = 0.1,
        c_cognitive: float = 1.0,  
    ) -> tuple[float, float]:

    Particles = [Particle(fitness_function, initialize_function, bounds, c_inertia, c_social, c_cognitive) for _ in particle_count]
    
    for iteration in range(iterations):
        for particle in Particles:
            particle.move()

    return Particle.swarm_best_position, Particle.swarm_best_value

if __name__ == "__main__":
    G = nx.karate_club_graph()

    distances, nodes = all_paths(G)

    best_layout, best_value = PSO(partial(compute_stress, target_distances=distances)), partial(random_layout, G, nodes)