import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image

def draw_layout(G: nx.Graph, nodes: list, positions: np.ndarray) -> None:
    """
    Draws graph using positions stored as an n x 2 array.
    """
    pos_dict = {
        node: positions[i]
        for i, node in enumerate(nodes)
    }

    nx.draw(G, pos=pos_dict, with_labels=True, node_color="lightblue", edge_color="gray")
    plt.axis("equal")
    plt.show()

def save_layout_plot(
    G,
    nodes,
    positions,
    filepath,
    title=None,
    node_size=100,
    with_labels=False,
):
    positions = positions.reshape(-1, 2)

    pos_dict = {
        node: positions[i]
        for i, node in enumerate(nodes)
    }

    plt.figure(figsize=(7, 7))
    nx.draw(
        G,
        pos=pos_dict,
        with_labels=with_labels,
        node_size=node_size,
        node_color="lightblue",
        edge_color="gray",
    )

    if title is not None:
        plt.title(title)

    plt.axis("equal")
    plt.tight_layout()

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=200)
    plt.close()

def make_gif(frame_dir, output_path, duration=120):
    frame_files = sorted(
        f for f in os.listdir(frame_dir)
        if f.endswith(".png")
    )

    frames = [
        Image.open(os.path.join(frame_dir, f)).convert("P")
        for f in frame_files
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )

def save_convergence_plot(history, output_path):
    iterations = [x[0] for x in history]
    values = [x[1] for x in history]

    plt.figure(figsize=(8, 5))
    plt.plot(iterations, values)
    plt.xlabel("Iteration")
    plt.ylabel("Best stress")
    plt.title("PSO Convergence")
    plt.grid(True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()