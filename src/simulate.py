#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulate.py
------------
Programme principal pour évaluer la robustesse d’un réseau biparti
selon différentes stratégies de suppression de nœuds.

Ce fichier :
- définit une fonction pour chaque stratégie (random, degree, pagerank, hits, musrank)
- contient la fonction principale simulate_all() qui compare plusieurs stratégies
- enregistre et affiche les figures correspondantes
"""

import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import datetime
import os
import json
import pandas as pd

# --- Importation de l’algorithme MusRank ---
from musrank_strategy import musrank


# =====================================================
# 1. Création d’un petit graphe biparti pour les tests
# =====================================================
def create_test_graph():
    """Crée un petit graphe biparti pour tester les stratégies."""
    G = nx.Graph()
    G.add_nodes_from(["A1", "A2", "A3", "A4"], bipartite=0)  # Ensemble actif
    G.add_nodes_from(["P1", "P2", "P3"], bipartite=1)        # Ensemble passif
    G.add_edges_from([
        ("A1", "P1"), ("A1", "P2"),
        ("A2", "P1"), ("A2", "P3"),
        ("A3", "P2"), ("A4", "P3"),
    ])
    return G

def load_bipartite_json(path):
    """Charge un réseau biparti à partir d’un fichier JSON"""
    with open(path) as f:
        data = json.load(f)

    G = nx.Graph()
    for node in data["nodes"]:
        node_id = int(node["nodeid"])
        group = node.get("group", "").lower()
        if "herbivore" in group or "pollinator" in group:
            bip = 0  # ensemble actif
        else:
            bip = 1  # ensemble passif
        G.add_node(node_id, bipartite=bip, label=node["name"])
    for link in data["links"]:
        G.add_edge(int(link["source"]), int(link["target"]))
    return G


# =====================================================
# 2. Fonctions de stratégie (chaque algorithme renvoie un score)
# =====================================================

def random_strategy(G):
    """Attribue un score aléatoire aux nœuds actifs."""
    nodes = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 0]
    scores = {n: random.random() for n in nodes}
    return scores


def degree_strategy(G):
    """Attribue un score basé sur le degré de chaque nœud actif."""
    degs = dict(G.degree())
    return {n: degs[n] for n, d in G.nodes(data=True) if d.get("bipartite") == 0}


def pagerank_strategy(G):
    """Calcule les scores PageRank pour les nœuds actifs."""
    pr = nx.pagerank(G)
    return {n: pr[n] for n, d in G.nodes(data=True) if d.get("bipartite") == 0}


def hits_strategy(G):
    """Calcule les scores HITS (valeurs de hub) pour les nœuds actifs."""
    hubs, authorities = nx.hits(G, max_iter=1000, tol=1e-8)
    return {n: hubs[n] for n, d in G.nodes(data=True) if d.get("bipartite") == 0}


def musrank_strategy(G):
    """Calcule les scores MusRank à l’aide de la fonction importée."""
    return musrank(G)


# =====================================================
# 3. Évaluation de la robustesse (suppression progressive des nœuds)
# =====================================================

def evaluate_robustness(G, scores):
    """
    Supprime les nœuds actifs dans l’ordre des scores décroissants
    et mesure la taille de la plus grande composante connexe (LCC).
    """
    G_copy = G.copy()
    lcc_sizes = []

    # Ordonner les nœuds actifs selon leur score décroissant
    order = sorted(scores, key=scores.get, reverse=True)

    # Ne supprimer que les nœuds actifs
    active_nodes = [n for n, d in G.nodes(data=True) if d.get('bipartite') == 0]
    order = [n for n in order if n in active_nodes]

    # Boucle de suppression
    for node in order:
        if nx.is_connected(G_copy):
            lcc = len(G_copy)
        else:
            lcc = len(max(nx.connected_components(G_copy), key=len))
        lcc_sizes.append(lcc)

        if G_copy.has_node(node):
            G_copy.remove_node(node)

    return np.array(lcc_sizes)


# =====================================================
# 4. Fonction principale : simulation de toutes les stratégies
# =====================================================

def simulate_all(G, strategies_to_test=None, dataset_name="network"):
    """
    Exécute la simulation pour toutes les stratégies spécifiées.
    Si aucune stratégie n’est donnée, toutes les stratégies disponibles seront testées.
    Chaque figure est enregistrée avec le nom du dataset (pour éviter l’écrasement).
    """
    available_strategies = {
        "random": random_strategy,
        "degree": degree_strategy,
        "pagerank": pagerank_strategy,
        "hits": hits_strategy,
        "musrank": musrank_strategy,
    }

    if strategies_to_test is None:
        strategies_to_test = list(available_strategies.keys())

    results = {}

    for name in strategies_to_test:
        print(f"\n--- Simulation avec la stratégie : {name} ---")
        func = available_strategies[name]
        scores = func(G)
        lcc_sizes = evaluate_robustness(G, scores)
        results[name] = lcc_sizes

    # =====================================================
    # 5. Visualisation et sauvegarde de la figure
    # =====================================================

    plt.figure(figsize=(8, 5))
    for name, lcc_sizes in results.items():
        plt.plot(
            range(len(lcc_sizes)),
            lcc_sizes,
            marker='o',
            label=name.capitalize()
        )

    plt.xlabel("Nombre de nœuds supprimés")
    plt.ylabel("Taille de la plus grande composante connexe (LCC)")
    plt.title("Comparaison des stratégies de suppression")
    plt.legend()
    plt.grid(True)

    # Forcer les graduations de l’axe X à être lisibles (éviter le chevauchement)
    max_len = max(len(v) for v in results.values())
    step = max(1, max_len // 10)  # environ 10 graduations
    plt.xticks(range(0, max_len, step))

    # =====================================================
    # Sauvegarde automatique dans le dossier 'tests/results'
    # =====================================================

    base_dir = os.path.dirname(os.path.abspath(__file__))
    report_dir = os.path.join(base_dir, "..", "tests", "results")
    os.makedirs(report_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(report_dir, f"LCC_comparison_{dataset_name}.png")

    plt.savefig(filename, dpi=300)
    print(f"Figure enregistrée dans {filename}")

    plt.show()


# =====================================================
# 6. Simulation MusRank + Sauvegarde CSV
# =====================================================
def simulate_musrank_and_save(G, network_name, save_dir="tests/results/musrank_analysis"):
    """Simule la suppression selon MusRank et sauvegarde le CSV."""
    scores = musrank(G)
    G0 = G.copy()
    Gc = G.copy()
    os.makedirs(save_dir, exist_ok=True)
    active_nodes = [n for n, d in G.nodes(data=True) if d.get('bipartite') == 0]
    order = sorted(active_nodes, key=scores.get, reverse=True)
    records = []
    removed_primary = 0

    for step, node in enumerate(order, start=1):
        if node not in Gc:
            continue
        Gc.remove_node(node)
        removed_primary += 1
        # secondary extinctions
        zeros = [n for n in list(Gc.nodes()) if Gc.degree(n) == 0]
        removed_secondary = len(zeros)
        Gc.remove_nodes_from(zeros)
        # LCC
        if Gc.number_of_nodes() == 0:
            lcc_ratio = 0
        else:
            Gu = Gc.to_undirected()
            largest = max((len(c) for c in nx.connected_components(Gu)), default=0)
            lcc_ratio = largest / G0.number_of_nodes()
        frac_primary = removed_primary / len(active_nodes)
        records.append({
            "step": step,
            "removed_node": node,
            "removed_primary": removed_primary,
            "removed_secondary_count": removed_secondary,
            "removed_fraction_primary": frac_primary,
            "lcc_ratio": lcc_ratio,
            "strategy": "MUSRANK",
            "network": network_name
        })

    df = pd.DataFrame(records)
    csv_path = os.path.join(save_dir, f"{network_name}_MUSRANK_removal.csv")
    df.to_csv(csv_path, index=False)
    print(f" CSV sauvegardé : {csv_path}")
    return df

# =====================================================
# 7. Point d’entrée du programme
# =====================================================
if __name__ == "__main__":
    # Déterminer le chemin du dossier des tests
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tests_dir = os.path.join(base_dir, "..", "tests")

    # Charger les deux réseaux JSON
    G_prunus = load_bipartite_json(os.path.join(tests_dir, "network_prunus.json"))
    G_pollinator = load_bipartite_json(os.path.join(tests_dir, "network_pollinator.json"))

    # Exécuter les simulations (pour l’instant uniquement MusRank)
    print(">>> Simulation sur le réseau PRUNUS (petit réseau)")
    simulate_all(G_prunus, strategies_to_test=["musrank"], dataset_name="prunus")

    print("\n>>> Simulation sur le réseau POLLINATOR (grand réseau)")
    simulate_all(G_pollinator, strategies_to_test=["musrank"], dataset_name="pollinator")

    # =====================================================
    # Ajout : génération des CSV MusRank
    # =====================================================
    print("\n>>> Sauvegarde des résultats MusRank au format CSV")
    simulate_musrank_and_save(G_prunus, "Prunus")
    simulate_musrank_and_save(G_pollinator, "Pollinator")

