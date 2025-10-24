#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
稳定版 MusRank 算法（防止 underflow）
----------------------------------
适用于二部网络 (Active ↔ Passive)。
"""

import numpy as np
from networkx.algorithms import bipartite


def musrank(G, max_iter=500, tol=1e-8, verbose=False):
    # === 识别两类节点 ===
    active_nodes = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 0]
    passive_nodes = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 1]
    if not active_nodes or not passive_nodes:
        raise ValueError("图中必须包含 bipartite=0 和 bipartite=1 的节点")

    # === 构建矩阵 ===
    M = bipartite.biadjacency_matrix(G, row_order=active_nodes, column_order=passive_nodes).toarray().astype(float)

    # === 去除孤立节点（防止除 0）===
    nonzero_rows = np.where(M.sum(axis=1) > 0)[0]
    nonzero_cols = np.where(M.sum(axis=0) > 0)[0]
    M = M[nonzero_rows][:, nonzero_cols]
    active_nodes = [active_nodes[i] for i in nonzero_rows]
    passive_nodes = [passive_nodes[j] for j in nonzero_cols]

    nA, nP = len(active_nodes), len(passive_nodes)
    if nA == 0 or nP == 0:
        raise ValueError("去除孤立节点后无有效节点")

    # === 初始化 ===
    I = np.ones(nA)
    V = np.ones(nP)
    eps = 1e-12

    for it in range(max_iter):
        I_old, V_old = I.copy(), V.copy()

        # 更新 I
        I = M.dot(V)
        I = np.maximum(I, eps)
        I /= np.mean(I)

        # 更新 V
        denom = M.T.dot(1.0 / np.maximum(I, eps))
        denom = np.maximum(denom, eps)
        V = 1.0 / denom
        V = np.maximum(V, eps)
        V /= np.mean(V)

        # 数值剪切（防止爆炸/坍塌）
        I = np.clip(I, eps, 1 / eps)
        V = np.clip(V, eps, 1 / eps)

        # 收敛检查
        diff = np.linalg.norm(I - I_old) + np.linalg.norm(V - V_old)
        if verbose and it % 10 == 0:
            print(f"[Iter {it:03d}] diff={diff:.3e}, mean(I)={I.mean():.3f}, mean(V)={V.mean():.3f}")
        if diff < tol:
            if verbose:
                print(f"[Converged after {it+1} iterations]")
            break

    # 最终归一化
    I /= np.mean(I)
    V /= np.mean(V)

    # === 构建输出 ===
    I_scores = {active_nodes[i]: float(I[i]) for i in range(nA)}
    V_scores = {passive_nodes[j]: float(V[j]) for j in range(nP)}
    return I_scores, V_scores


# --- 自测 ---
if __name__ == "__main__":
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(["A1", "A2", "A3"], bipartite=0)
    G.add_nodes_from(["P1", "P2"], bipartite=1)
    G.add_edges_from([("A1","P1"),("A2","P1"),("A2","P2"),("A3","P2")])
    I, V = musrank(G, verbose=True)
    print("I:", I)
    print("V:", V)
