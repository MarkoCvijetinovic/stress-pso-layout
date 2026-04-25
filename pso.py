import numpy as np
from graph import compute_stress, all_paths, random_layout

class Particle:
    swarm_best_position = None
    swarm_best_value = float('inf')

    def __init__(
            self, 
            fitness_function: callable, 
            bounds: np.ndarray,
            c_inertia: float,
            c_social: float,
            c_cognitive: float,
        ):
        pass 

    def move(self):
        pass

def PSO( 
        fitness_function: callable, 
        bounds: np.ndarray,
        particle_count: int = 100, 
        iterations: int = 100,
        c_inertia: float = 0.2,
        c_social: float = 0.1,
        c_cognitive: float = 1.0,  
    ) -> tuple[float, float]:

    Particles = [Particle(fitness_function, bounds, c_inertia, c_social, c_cognitive) for _ in particle_count]
    
    for iteration in range(iterations):
        for particle in Particles:
            particle.move()

    return Particle.swarm_best_position, Particle.swarm_best_value

if __name__ == "__main__":
    pass