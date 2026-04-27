# Stress-Based Graph Layout with Particle Swarm Optimization

This project implements stress-based graph drawing using Particle Swarm Optimization (PSO). The PSO implementation is abstract and problem-independent, while graph-specific logic is separated into its own module. It also features batched versions of algorithms that can be ran on GPU with torch for improved performance.

## Overview

The layout objective is classical graph stress:

$$\mathrm{Stress}(X) = \sum_{i < j} w_{ij}(\|x_i - x_j\| - d_{ij})^2$$
where:

- $$\(x_i, x_j\)$$ are 2D node positions,
- $$d_{ij}$$ is the shortest-path distance between nodes,
- $$w_{ij}$$ is usually $$1 / d_{ij}^2$$.

The optimizer represents a graph layout as a flat vector:

```text
[x1, y1, x2, y2, ..., xn, yn]

.
├── pso.py
├── graph.py
├── stress_layout_pso.py
├── visualization.py
├── generate_showcase.py
├── data/
│   └── showcase/
└── README.md
```

## Visual Results

<img width=50% alt="evolution" src="https://github.com/user-attachments/assets/1027553b-7f68-44b8-bd6a-9412846d44e5" /><img width=50% alt="evolution" src="https://github.com/user-attachments/assets/8d38d781-40e2-487d-b085-d542d73b57e1" />


## Installation

pip install numpy networkx matplotlib pillow

## Modules

### pso.py & batched_pso.py

`pso.py` contains a fully abstract PSO implementation. It only requires:
* fitness_function(position) -> float
* initialize_function() -> np.ndarray
* optional repair_function(position, velocity, stagnation_counter) -> for clipping, normalization, mutation, or other constraint handling
* optional callback_function(iteration, best_position, best_value)
This makes the optimizer reusable for other problems.

`batched_pso.py` implements a faster batched variant. Instead of evaluating particles one at a time, it stacks all particle positions into a single array and evaluates the whole swarm at once

### graph.py & batched_graph.py

Contains graph/layout utilities:
* all-pairs shortest paths using NetworkX Floyd-Warshall
* optional normalization of graph distances
* fast NumPy vectorized stress computation
* random layout initialization
* optional Torch-based batched stress computation
  
### stress_layout_pso.py
Connects the abstract PSO implementation to the stress-layout problem.

It provides:
* graph layout initialization
* layout repair/normalization
* position bounds
* optional velocity clipping and mutation
* stress_layout_pso_functions(...), a convenient starting point for experiments

The function stress_layout_pso_functions(...) can return either a normal scalar fitness function or a batched fitness function:

### visualization.py

Contains visualization utilities:
* drawing graph layouts
* saving layout plots
* saving convergence plots
* creating GIF animations from saved frames
### generate_showcase.py
Runs a full showcase experiment and saves results into data/showcase.

It can generate:
* layout evolution GIF
* final layout image
* convergence plot

## Example Usage

Run a basic stress-layout experiment:

python stress_layout_pso.py

Generate showcase outputs:

python generate_showcase.py

### Example Graphs

The project works well with NetworkX graphs such as:

* nx.path_graph(30)
* nx.cycle_graph(30)
* nx.grid_2d_graph(8, 8)
* nx.balanced_tree(4, 3)
* nx.karate_club_graph()
* nx.connected_caveman_graph(8, 8)

Simple graphs such as paths, cycles, and grids are especially useful for verifying that the stress function behaves correctly.

## Optimization Notes

The stress computation is vectorized with NumPy instead of using explicit Python double loops. This keeps the classical all-pairs stress objective while making it much faster in practice.

Graph distances are normalized before optimization. This significantly improves stability across different graph families and reduces the need for graph-specific PSO parameter tuning.

## Showcase Output

Generated showcase files are stored in:

data/showcase/

Frames necessary for building gifs and converge plots are stored in:

data/tmp/

## Current Features
* Abstract global-best PSO
* Stress-based graph layout
* Fast vectorized stress computation
* Shortest-path distance normalization
* Layout repair/centering
* Optional mutation and velocity clipping
* Convergence plots
* GIF generation for layout evolution
* Batch evaluation of particles

## Possible Extensions
* Local-best or multi-leader PSO
* Approximate/sampled stress for larger graphs
* Comparison with NetworkX spring layout
* Parameter studies across graph families

