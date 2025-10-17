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

def simulate_all(G, strategies_to_test=None):
    """
    Exécute la simulation pour toutes les stratégies spécifiées.
    Si aucune stratégie n’est donnée, toutes les stratégies disponibles seront testées.
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

    # Forcer les graduations de l’axe X à être des entiers
    max_len = max(len(v) for v in results.values())
    plt.xticks(range(max_len))

    # =====================================================
    # Sauvegarde automatique dans le dossier 'report/'
    # (avec chemin absolu pour éviter les erreurs de dossier)
    # =====================================================

    # Déterminer le chemin absolu du dossier actuel (src)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Construire le chemin du dossier 'report' à la racine du projet
    report_dir = os.path.join(base_dir, "..", "tests", "results")

    # Créer le dossier s’il n’existe pas
    os.makedirs(report_dir, exist_ok=True)

    # Nom du fichier avec date et heure
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(report_dir, "LCC_comparison.png")

    # Sauvegarder la figure
    plt.savefig(filename, dpi=300)
    print(f"Figure enregistrée dans {filename}")

    plt.show()


# =====================================================
# 6. Point d’entrée du programme
# =====================================================

if __name__ == "__main__":
    G = create_test_graph()
    # Pour l’instant, tester uniquement MusRank
    simulate_all(G, strategies_to_test=["musrank"])