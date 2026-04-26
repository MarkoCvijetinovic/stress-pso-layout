import numpy as np
from typing import Callable


FitnessFunction = Callable[[np.ndarray], float]
InitializeFunction = Callable[[], np.ndarray]
RepairFunction = Callable[
    [np.ndarray, np.ndarray, int],
    tuple[np.ndarray, np.ndarray, int]
]
CallbackFunction = Callable[[int, np.ndarray, np.ndarray], None]

class Particle:
    swarm_best_position = None
    swarm_best_value = float("inf")

    def __init__(
        self,
        fitness_function: FitnessFunction,
        initialize_function: InitializeFunction,
        repair_function: RepairFunction,
        c_inertia: float,
        c_social: float,
        c_cognitive: float,
    ):
        self.fitness = fitness_function
        self.repair = repair_function

        self.c_inertia = c_inertia
        self.c_social = c_social
        self.c_cognitive = c_cognitive

        self.position = initialize_function()
        self.velocity = np.random.uniform(-1, 1, size=self.position.shape)

        self.value = self.fitness(self.position)

        self.best_position = self.position.copy()
        self.best_value = self.value

        self.update_swarm_best()
        self.stagnation_counter = 0

    def update_swarm_best(self):
        if self.value < Particle.swarm_best_value:
            Particle.swarm_best_value = self.value
            Particle.swarm_best_position = self.position.copy()

    def move(self):
        r_social = np.random.random(size=self.position.shape)
        r_cognitive = np.random.random(size=self.position.shape)

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

        if(self.repair is not None):
            self.position, self.velocity, self.stagnation_counter = self.repair(self.position, self.velocity, self.stagnation_counter)

        self.value = self.fitness(self.position)

        if self.value < self.best_value:
            self.best_value = self.value
            self.best_position = self.position.copy()
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

        self.update_swarm_best()

def PSO(
    fitness_function: FitnessFunction,
    initialize_function: InitializeFunction,
    particle_count: int = 50,
    iterations: int = 300,
    c_inertia: float = 0.7,
    c_social: float = 1.4,
    c_cognitive: float = 1.4,
    repair_function: RepairFunction = None,
    callback_function: CallbackFunction = None,
) -> tuple[np.ndarray, float]:

    Particle.swarm_best_position = None
    Particle.swarm_best_value = float("inf")

    particles = [
        Particle(fitness_function, initialize_function, repair_function,
                    c_inertia, c_social, c_cognitive)
        for _ in range(particle_count)
    ]

    for iteration in range(iterations):
        for particle in particles:
            particle.move()

        if callback_function is not None:
            callback_function(
                iteration=iteration + 1,
                best_position=Particle.swarm_best_position,
                best_value=Particle.swarm_best_value,
            )

        print(f"Iteration {iteration + 1}: {Particle.swarm_best_value:.4f}")

    return Particle.swarm_best_position, Particle.swarm_best_value