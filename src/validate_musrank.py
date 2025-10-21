#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_musrank.py
-------------------
Vérification et analyse détaillée de l'algorithme MusRank :
1. Convergence et stabilité numérique
2. Distribution des scores (importance et vulnérabilité)
3. Comparaison entre initialisations
4. Relation Importance–Vulnérabilité
5. Test de robustesse simple (LCC vs fraction supprimée)
Toutes les figures sont automatiquement sauvegardées
dans results/musrank_analysis/ (avec écrasement).
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import json
from scipy.stats import spearmanr
from musrank_strategy import musrank


# =====================================================
# 1. Charger un graphe biparti JSON
# =====================================================
def load_bipartite_json(path):
    with open(path) as f:
        data = json.load(f)

    G = nx.Graph()
    for node in data["nodes"]:
        node_id = int(node["nodeid"])
        group = node.get("group", "").lower()
        if "herbivore" in group or "pollinator" in group:
            bip = 0  # actif
        else:
            bip = 1  # passif
        G.add_node(node_id, bipartite=bip, label=node["name"])
    for link in data["links"]:
        G.add_edge(int(link["source"]), int(link["target"]))
    return G


# =====================================================
# 2. Version instrumentée de MusRank pour suivre les itérations
# =====================================================
def musrank_with_history(G, max_iter=100, tol=1e-6, init_mode="ones"):
    """Renvoie l'historique complet des vecteurs I et V pour tracer la convergence."""
    from networkx.algorithms import bipartite
    active_nodes = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 0]
    passive_nodes = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 1]
    M = bipartite.biadjacency_matrix(G, row_order=active_nodes, column_order=passive_nodes).toarray()

    if init_mode == "random":
        I = np.random.rand(len(active_nodes))
        V = np.random.rand(len(passive_nodes))
    else:
        I = np.ones(len(active_nodes))
        V = np.ones(len(passive_nodes))

    I_hist, V_hist, diffs = [], [], []

    for _ in range(max_iter):
        I_old, V_old = I.copy(), V.copy()
        I = M.dot(V)
        I /= np.mean(I)
        denom = M.T.dot(1 / I)
        denom[denom == 0] = 1e-12
        V = 1 / denom
        V /= np.mean(V)
        diff = np.linalg.norm(I - I_old) + np.linalg.norm(V - V_old)
        I_hist.append(I.copy())
        V_hist.append(V.copy())
        diffs.append(diff)
        if diff < tol:
            break

    return np.array(I_hist), np.array(V_hist), np.array(diffs), active_nodes, passive_nodes


# =====================================================
# 3. Fonction pour sauvegarder les figures
# =====================================================
def save_current_figure(fig, name, dataset_name):
    """Sauvegarde la figure dans le dossier 'tests/results/musrank_analysis' (écrasement autorisé)."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, "..", "tests", "results", "musrank_analysis")
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"{name}_{dataset_name.lower()}.png")
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f" Figure sauvegardée : {filename}")


# =====================================================
# 4. Convergence et stabilité
# =====================================================
def plot_convergence(G, dataset_name="network"):
    I1, V1, diffs1, _, _ = musrank_with_history(G, init_mode="ones")
    I2, V2, diffs2, _, _ = musrank_with_history(G, init_mode="random")

    fig = plt.figure(figsize=(6, 4))
    plt.semilogy(diffs1, label="init=ones")
    plt.semilogy(diffs2, label="init=random")
    plt.xlabel("Iteration")
    plt.ylabel("‖ΔI‖ + ‖ΔV‖ (log scale)")
    plt.title(f"Convergence MusRank ({dataset_name})")
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    save_current_figure(fig, "convergence", dataset_name)
    plt.show()
    plt.close(fig)


# =====================================================
# 5. Distribution des scores + top-10
# =====================================================
def plot_scores_distribution(G, dataset_name="network"):
    I_A, V_P = musrank_with_history(G, init_mode="ones")[0][-1], musrank_with_history(G, init_mode="ones")[1][-1]
    active_nodes = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 0]
    passive_nodes = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 1]
    I_dict = dict(zip(active_nodes, I_A))
    V_dict = dict(zip(passive_nodes, V_P))

    print("\n=== Top 10 Importance (Active) ===")
    for n, s in sorted(I_dict.items(), key=lambda x: -x[1])[:10]:
        print(f"{n}: {s:.4f}")
    print("\n=== Top 10 Vulnerability (Passive) ===")
    for n, s in sorted(V_dict.items(), key=lambda x: -x[1])[:10]:
        print(f"{n}: {s:.4f}")

    fig = plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(I_A, bins=20, color="#1f77b4", alpha=0.8)
    plt.title(f"{dataset_name} - Importance (Active)")
    plt.xlabel("I_A")
    plt.ylabel("Fréquence")

    plt.subplot(1, 2, 2)
    plt.hist(V_P, bins=20, color="#d62728", alpha=0.8)
    plt.title(f"{dataset_name} - Vulnerabilité (Passive)")
    plt.xlabel("V_P")
    plt.ylabel("Fréquence")

    plt.tight_layout()
    save_current_figure(fig, "distribution", dataset_name)
    plt.show()
    plt.close(fig)


# =====================================================
# 6. Comparaison entre initialisations (stabilité)
# =====================================================
def compare_initializations(G, dataset_name="network"):
    I1, _, _, _, _ = musrank_with_history(G, init_mode="ones")
    I2, _, _, _, _ = musrank_with_history(G, init_mode="random")
    I1_final, I2_final = I1[-1], I2[-1]
    rho, _ = spearmanr(I1_final, I2_final)

    fig = plt.figure(figsize=(5, 5))
    plt.scatter(I1_final, I2_final, alpha=0.7)
    plt.plot([I1_final.min(), I1_final.max()],
             [I1_final.min(), I1_final.max()],
             'k--', lw=1)
    plt.xlabel("Importance (init=ones)")
    plt.ylabel("Importance (init=random)")
    plt.title(f"{dataset_name}: Spearman ρ={rho:.3f}")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    save_current_figure(fig, "spearman", dataset_name)
    plt.show()
    plt.close(fig)


# =====================================================
# 7. Relation Importance–Vulnérabilité
# =====================================================
def plot_IV_relationship(G, dataset_name="network"):
    I_A, V_P, _, active_nodes, passive_nodes = musrank_with_history(G, init_mode="ones")
    I_final, V_final = I_A[-1], np.maximum(V_P[-1], 0)
    I_dict = dict(zip(active_nodes, I_final))
    V_dict = dict(zip(passive_nodes, V_final))

    x, y = [], []
    threshold = 1e-12
    for a, p in G.edges():
        if a in I_dict and p in V_dict and I_dict[a] > threshold and V_dict[p] > threshold:
            x.append(I_dict[a])
            y.append(V_dict[p])
        elif p in I_dict and a in V_dict and I_dict[p] > threshold and V_dict[a] > threshold:
            x.append(I_dict[p])
            y.append(V_dict[a])

    fig = plt.figure(figsize=(6, 5))
    plt.scatter(x, y, alpha=0.6, color="#2ca02c")
    plt.xlabel("Importance des actifs (I_A)")
    plt.ylabel("Vulnérabilité des passifs (V_P)")
    plt.title(f"I–V Relationship ({dataset_name})")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    save_current_figure(fig, "iv_relationship", dataset_name)
    plt.show()
    plt.close(fig)


# =====================================================
# 8. Test de robustesse simple (LCC)
# =====================================================
def robustness_curve(G, dataset_name="network"):
    from simulate import evaluate_robustness
    scores = musrank(G)
    lcc = evaluate_robustness(G, scores)
    x = np.arange(len(lcc)) / len(lcc)

    fig = plt.figure(figsize=(6, 4))
    plt.plot(x, lcc / len(G), marker='o')
    plt.xlabel("Fraction de nœuds supprimés (actifs)")
    plt.ylabel("LCC relative")
    plt.title(f"Robustesse MusRank ({dataset_name})")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    save_current_figure(fig, "robustesse", dataset_name)
    plt.show()
    plt.close(fig)

    auc = np.trapz(lcc / len(G), x)
    print(f"AUC (robustesse) = {auc:.4f}")


# =====================================================
# 9. Point d’entrée principal
# =====================================================
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tests_dir = os.path.join(base_dir, "..", "tests")

    G_prunus = load_bipartite_json(os.path.join(tests_dir, "network_prunus.json"))
    G_pollinator = load_bipartite_json(os.path.join(tests_dir, "network_pollinator.json"))

    print("=== Analyse MusRank sur PRUNUS ===")
    plot_convergence(G_prunus, "Prunus")
    plot_scores_distribution(G_prunus, "Prunus")
    compare_initializations(G_prunus, "Prunus")
    plot_IV_relationship(G_prunus, "Prunus")
    robustness_curve(G_prunus, "Prunus")

    print("\n=== Analyse MusRank sur POLLINATOR ===")
    plot_convergence(G_pollinator, "Pollinator")
    plot_scores_distribution(G_pollinator, "Pollinator")
    compare_initializations(G_pollinator, "Pollinator")
    plot_IV_relationship(G_pollinator, "Pollinator")
    robustness_curve(G_pollinator, "Pollinator")
