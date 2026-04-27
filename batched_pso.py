import numpy as np
from typing import Callable
from pso import InitializeFunction, RepairFunction, CallbackFunction


BatchedFitnessFunction = Callable[[np.ndarray], np.ndarray]

class BatchedParticle:
    swarm_best_position = None
    swarm_best_value = float("inf")

    def __init__(
        self,
        initialize_function: InitializeFunction,
        repair_function: RepairFunction,
        c_inertia: float,
        c_social: float,
        c_cognitive: float,
    ):
        self.repair = repair_function

        self.c_inertia = c_inertia
        self.c_social = c_social
        self.c_cognitive = c_cognitive

        self.position = initialize_function()
        self.velocity = np.random.uniform(-1, 1, size=self.position.shape)

        self.value = float("inf")

        self.best_position = self.position.copy()
        self.best_value = self.value

        self.stagnation_counter = 0

    def update_swarm_best(self):
        if self.value < BatchedParticle.swarm_best_value:
            BatchedParticle.swarm_best_value = self.value
            BatchedParticle.swarm_best_position = self.position.copy()

    def update_personal_best(self):
        if self.value < self.best_value:
            self.best_value = self.value
            self.best_position = self.position.copy()
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

        self.update_swarm_best()


    def move(self):
        r_social = np.random.random(size=self.position.shape)
        r_cognitive = np.random.random(size=self.position.shape)

        social_component = (
            self.c_social
            * r_social
            * (BatchedParticle.swarm_best_position - self.position)
        )

        cognitive_component = (
            self.c_cognitive
            * r_cognitive
            * (self.best_position - self.position)
        )

        inertia_component = self.c_inertia * self.velocity

        self.velocity = inertia_component + social_component + cognitive_component

        self.position = self.position + self.velocity

        if(self.repair is not None):
            self.position, self.velocity, self.stagnation_counter = self.repair(self.position, self.velocity, self.stagnation_counter)

def BatchedPSO(
    batched_fitness_function: BatchedFitnessFunction,
    initialize_function: InitializeFunction,
    particle_count: int = 100,
    iterations: int = 4000,
    c_inertia: float = 0.7,
    c_social: float = 1.4,
    c_cognitive: float = 1.4,
    repair_function: RepairFunction = None,
    callback_function: CallbackFunction = None,
) -> tuple[np.ndarray, float]:
    """
    Fully abstract Particle Swarm Optimization (PSO) algorithm.

    Minimizes a given fitness function over a continuous search space.

    Args:
        batched_fitness_function: Function mapping a position vector to a scalar value, for every particle
        initialize_function: Function that returns an initial position vector for a particle.
        particle_count: Number of particles in the swarm.
        iterations: Number of optimization iterations.
        c_inertia: Inertia coefficient (controls momentum of particles).
        c_social: Social coefficient (attraction toward global best).
        c_cognitive: Cognitive coefficient (attraction toward personal best).
        repair_function: Optional function to enforce constraints or modify particles
                         (e.g. clipping, normalization, mutation).
        callback_function: Optional function called once per iteration with
                           (iteration, best_position, best_value).

    Returns:
        best_position: Best position found by the swarm.
        best_value: Fitness value at the best position.
    """

    BatchedParticle.swarm_best_position = None
    BatchedParticle.swarm_best_value = float("inf")

    particles = [
        BatchedParticle(initialize_function, repair_function,
                    c_inertia, c_social, c_cognitive)
        for _ in range(particle_count)
    ]

    for iteration in range(iterations):
        positions = np.stack([p.position for p in particles])
        values = batched_fitness_function(positions)

        for particle, value in zip(particles, values):
            particle.value = float(value)
            particle.update_personal_best()

        for particle in particles:
            particle.move()

        if callback_function is not None:
            callback_function(
                iteration=iteration + 1,
                best_position=BatchedParticle.swarm_best_position,
                best_value=BatchedParticle.swarm_best_value,
            )

    return BatchedParticle.swarm_best_position, BatchedParticle.swarm_best_value