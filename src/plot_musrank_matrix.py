#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MusRank Interaction Matrix Plot
-------------------------------
根据 MusRank 排序后的 I–V 结果，绘制网络的交互矩阵：
- 横轴：被动节点（passive / plant），按 V_P 递增排序
- 纵轴：主动节点（active / pollinator），按 I_A 递减排序
- 蓝色方块：存在 A–P 连接（G.has_edge）
可用于重现论文 Figure 5 的紧凑嵌套结构图。
"""

import os
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from musrank_strategy import musrank


# ============================================================
# 加载 JSON 文件并构建二部网络
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
# 生成矩阵点图
# ============================================================
def plot_musrank_matrix(G, I_scores, V_scores, title="", save_path=None):
    """根据 MusRank 排序结果绘制交互矩阵（紧凑图）"""
    actives = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 0]
    passives = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 1]

    # 按 I_A 从大到小排列 actives，按 V_P 从小到大排列 passives
    sorted_A = sorted(actives, key=lambda n: -I_scores.get(n, 0))
    sorted_P = sorted(passives, key=lambda n: V_scores.get(n, 0))

    # 构建邻接矩阵 (binary)
    M = np.zeros((len(sorted_A), len(sorted_P)))
    for i, a in enumerate(sorted_A):
        for j, p in enumerate(sorted_P):
            if G.has_edge(a, p):
                M[i, j] = 1

    # 绘制点图（类似 Figure 5）
    plt.figure(figsize=(6, 6))
    rows, cols = np.where(M > 0)
    plt.scatter(cols, len(sorted_A) - 1 - rows, s=10, color="#1f77b4", marker='s')  # 翻转 y 轴顺序
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
# 主函数入口
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
            print(f"[WARN] {name}: 文件不存在，跳过。")
            continue

        G = load_graph_from_json(path)
        I_scores, V_scores = musrank(G, max_iter=500, tol=1e-8, verbose=False)
        save_png = os.path.join(out_dir, f"matrix_{name.lower()}_musrank.png")
        plot_musrank_matrix(G, I_scores, V_scores, title=name, save_path=save_png)
