import networkx as nx
import numpy as np
from graph import all_paths
from pso import PSO
from stress_layout_pso import stress_layout_pso_functions
from visualization import save_layout_plot, make_gif, save_convergence_plot

def make_layout_snapshot_callback(G: nx.Graph, nodes: list, frame_dir: str, history: list):
    def callback(iteration: int, best_position: np.ndarray, best_value: float):
        history.append((iteration, best_value))

        if iteration % 50 == 0 or iteration == 1:
            filepath = f"{frame_dir}/layout_iter_{iteration:05d}.png"

            save_layout_plot(
                G=G,
                nodes=nodes,
                positions=best_position,
                filepath=filepath,
                title=f"Iteration {iteration}, stress={best_value:.6f}",
                with_labels=False,
            )

    return callback

if __name__ == "__main__":
    G = nx.connected_caveman_graph(8, 8)
    graph_str = "caveman_8x8"

    distances, nodes = all_paths(G)

    fitness, initialize, repair = stress_layout_pso_functions(G, distances, nodes)

    history = []

    callback = make_layout_snapshot_callback(
        G,
        nodes,
        frame_dir="data/tmp/" + graph_str + "_frames",
        history=history,
    )

    best_layout, best_value = PSO(
        fitness_function=fitness,
        initialize_function=initialize,
        particle_count=50,
        iterations=4000,
        repair_function=repair,
        c_inertia=0.8,
        c_social=1.7,
        c_cognitive=0.8,
        callback_function=callback,
    )

    make_gif(
        frame_dir="data/tmp/" + graph_str + "_frames",
        output_path="data/showcase/" + graph_str + "/evolution.gif",
        duration=120,
    )

    save_convergence_plot(
        history,
        output_path="data/showcase/" + graph_str + "/convergence.png",
    )

    save_layout_plot(
        G,
        nodes,
        best_layout,
        filepath="data/showcase/" + graph_str + "/final_layout.png",
        title=f"Final layout, stress={best_value:.6f}",
    )