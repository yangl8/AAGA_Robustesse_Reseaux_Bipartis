#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MusRank Strategy
----------------
Implémentation de l’algorithme MusRank pour les réseaux bipartis.
Ce module calcule l’importance (I_A) des nœuds actifs (Active set)
et la vulnérabilité (V_P) des nœuds passifs (Passive set).
"""

import numpy as np
import networkx as nx
from networkx.algorithms import bipartite


def musrank(G, max_iter=100, tol=1e-6, verbose=False):
    """
    Calcule le score MusRank pour les nœuds actifs d’un graphe biparti.

    Paramètres
    ----------
    G : networkx.Graph
        Graphe biparti (avec attribut 'bipartite' = 0 ou 1 pour chaque nœud)
    max_iter : int
        Nombre maximum d’itérations
    tol : float
        Tolérance de convergence (norme de la différence entre itérations)
    verbose : bool
        Si True, affiche les valeurs intermédiaires d'I et V à chaque itération

    Retour
    ------
    scores : dict
        Dictionnaire {nom_du_nœud_actif: score_d’importance}
    """

    # --- Identifier les deux ensembles de nœuds
    active_nodes = [n for n, d in G.nodes(data=True) if d.get('bipartite') == 0]
    passive_nodes = [n for n, d in G.nodes(data=True) if d.get('bipartite') == 1]

    if not active_nodes or not passive_nodes:
        raise ValueError("Le graphe doit contenir deux ensembles bipartis distincts avec l’attribut 'bipartite'.")

    # --- Construire la matrice biadjacente M (Active × Passive)
    M = bipartite.biadjacency_matrix(
        G, row_order=active_nodes, column_order=passive_nodes
    ).toarray()

    # --- Initialiser les vecteurs I (importance) et V (vulnérabilité)
    I = np.ones(len(active_nodes))
    V = np.ones(len(passive_nodes))

    # --- Itérations principales
    for iteration in range(max_iter):
        # Sauvegarder les anciennes valeurs
        I_old, V_old = I.copy(), V.copy()

        # (a) Mettre à jour l’importance I_A
        I = M.dot(V)
        I /= np.mean(I)  # normalisation moyenne = 1

        # (b) Mettre à jour la vulnérabilité V_P
        denom = M.T.dot(1 / I)
        denom[denom == 0] = 1e-12  # éviter division par zéro
        V = 1 / denom
        V /= np.mean(V)  # normalisation moyenne = 1

        # (c) Vérifier convergence
        diff = np.linalg.norm(I - I_old) + np.linalg.norm(V - V_old)
        if verbose:
            print(f"[Itération {iteration+1}] Différence={diff:.6f}")
            print(f"  I: {np.round(I, 4)}")
            print(f"  V: {np.round(V, 4)}\n")

        if diff < tol:
            if verbose:
                print(f"Convergence atteinte après {iteration+1} itérations.")
            break

    # --- Retourner les scores sous forme de dictionnaire
    scores = {active_nodes[i]: float(I[i]) for i in range(len(active_nodes))}
    return scores


# --- Test simple quand on exécute ce fichier directement
if __name__ == "__main__":
    # Exemple : petit graphe biparti
    G = nx.Graph()
    G.add_nodes_from(["A1", "A2", "A3"], bipartite=0)  # actifs
    G.add_nodes_from(["P1", "P2"], bipartite=1)         # passifs
    G.add_edges_from([
        ("A1", "P1"),
        ("A2", "P1"),
        ("A2", "P2"),
        ("A3", "P2")
    ])

    print("Test MusRank sur un petit graphe :")
    result = musrank(G, verbose=True)
    print("\nRésultats finaux :", result)