import os
import matplotlib.pyplot as plt
import networkx as nx

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