#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MusRank Interaction Matrix Plot
-------------------------------
Plot the interaction matrix of the network based on MusRank-ordered I-V results:
- X-axis: passive nodes (passive / plant), sorted by V_P in ascending order
- Y-axis: active nodes (active / pollinator), sorted by I_A in descending order
- Blue squares: A-P connections exist (G.has_edge)
Can be used to reproduce the compact nested structure figure (Figure 5) from the paper.
"""

import os
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from musrank_strategy import musrank


# ============================================================
# Load JSON file and build bipartite network
# ============================================================
def load_graph_from_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    G = nx.Graph()
    names, groups = [], []

    for node in data["nodes"]:
        name = node.get("name") or node.get("id") or node.get("label") or str(len(names))
        names.append(name)
        g = (node.get("group") or node.get("type") or "").strip().lower()
        groups.append(g)

    for name, g in zip(names, groups):
        if g in ("pollinator", "herbivore", "active", "a", "seed disperser"):
            G.add_node(name, bipartite=0, group="active")
        elif g in ("plant", "passive", "p"):
            G.add_node(name, bipartite=1, group="passive")
        else:
            G.add_node(name, bipartite=1, group="unknown")

    links = data.get("links") or data.get("edges") or []
    for e in links:
        u, v = e.get("source"), e.get("target")
        if isinstance(u, int) and 0 <= u < len(names):
            u = names[u]
        if isinstance(v, int) and 0 <= v < len(names):
            v = names[v]
        if u and v:
            G.add_edge(u, v)

    print(f"[LOAD] {os.path.basename(json_path)}: |A|={sum(d['bipartite']==0 for _,d in G.nodes(data=True))}, "
          f"|P|={sum(d['bipartite']==1 for _,d in G.nodes(data=True))}, edges={G.number_of_edges()}")
    return G


# ============================================================
# Generate matrix scatter plot
# ============================================================
def plot_musrank_matrix(G, I_scores, V_scores, title="", save_path=None):
    """Plot interaction matrix based on MusRank ordering (compact figure)"""
    actives = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 0]
    passives = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 1]

    # Sort actives by I_A descending, passives by V_P ascending
    sorted_A = sorted(actives, key=lambda n: -I_scores.get(n, 0))
    sorted_P = sorted(passives, key=lambda n: V_scores.get(n, 0))

    # Build adjacency matrix (binary)
    M = np.zeros((len(sorted_A), len(sorted_P)))
    for i, a in enumerate(sorted_A):
        for j, p in enumerate(sorted_P):
            if G.has_edge(a, p):
                M[i, j] = 1

    # Plot scatter (similar to Figure 5)
    plt.figure(figsize=(6, 6))
    rows, cols = np.where(M > 0)
    plt.scatter(cols, len(sorted_A) - 1 - rows, s=10, color="#1f77b4", marker='s')  # flip y-axis order
    plt.xlim(-1, len(sorted_P))
    plt.ylim(-1, len(sorted_A))
    plt.xlabel("Passive nodes (Plants, increasing vulnerability)")
    plt.ylabel("Active nodes (Pollinators, decreasing importance)")
    plt.title(f"Interaction Matrix Ordered by MusRank ({title})")
    plt.grid(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[SAVED] {save_path}")
    plt.close()


# ============================================================
# Main entry point
# ============================================================
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tests_dir = os.path.join(base_dir, "..", "tests")
    out_dir = os.path.join(tests_dir, "musrank_matrix")
    os.makedirs(out_dir, exist_ok=True)

    datasets = {
        "Prunus": os.path.join(tests_dir, "network_prunus.json"),
        "Pollinator": os.path.join(tests_dir, "network_pollinator.json"),
        "M_PL_017": os.path.join(tests_dir, "M_PL_017.json"),
        "M_SD_018": os.path.join(tests_dir, "M_SD_018.json"),
    }

    for name, path in datasets.items():
        if not os.path.exists(path):
            print(f"[WARN] {name}: file does not exist, skipping.")
            continue

        G = load_graph_from_json(path)
        I_scores, V_scores = musrank(G, max_iter=500, tol=1e-8, verbose=False)
        save_png = os.path.join(out_dir, f"matrix_{name.lower()}_musrank.png")
        plot_musrank_matrix(G, I_scores, V_scores, title=name, save_path=save_png)
