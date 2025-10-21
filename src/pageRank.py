from typing import Dict, List

import networkx as nx 
__all__ = ["pageRank"]  


def multiply_sparse_matrix_vector(T: Dict[str, Dict[str, float]], X: Dict[str, float]) -> Dict[str, float]:
    """
    Sparse matrix × vector product.
    - T[i][j] = 1/out_degree(j) if j→i, else not stored (0)
    - X[j] = current PageRank value of node j
    """
    U = {i: 0.0 for i in T}  # 初始化结果向量
    for i in T:              # 遍历所有行
        for j, value in T[i].items():  # 遍历所有非零元素
            U[i] += value * X[j]
    return U


def normalize(X: Dict[str, float]) -> Dict[str, float]:
    """
    normalize vector so that sum(X) = 1.
    Equivalent to X[i] = X[i] / sum(X.values()).
    """
    total = sum(X.values())
    if total == 0:
        return X
    return {i: X[i] / total for i in X}


def pageRank(graph: Dict[str, List[str]], t: int = 100, s: float = 0.15) -> Dict[str, float]:
    """
    PageRank power iteration algorithm
    graph: adjacency list (u → [v1, v2, ...])
    t: number of iterations
    s: damping factor (teleportation probability)
    """
    nodes = list(graph.keys())
    n = len(nodes)

    # 构建转移矩阵 T: T[v][u] = 1/out_degree(u) if u→v
    T = {v: {} for v in nodes}
    for u in graph:
        if len(graph[u]) == 0:
            # dead-end: evenly link to all nodes
            for v in nodes:
                T[v][u] = 1 / n
        else:
            for v in graph[u]:
                T[v][u] = 1 / len(graph[u])

    # 初始化 PageRank 向量
    X = {i: 1 / n for i in nodes}
    I = {i: 1 / n for i in nodes}

    # 幂迭代
    for _ in range(t):
        prod = multiply_sparse_matrix_vector(T, X)
        # 更新 + 蒸发项
        X = {i: (1 - s) * prod[i] + s * I[i] for i in nodes}
        # 归一化，防止误差累积
        X = normalize(X)

    return X

import json
import networkx as nx

def load_bipartite_graph(path: str) -> nx.Graph:
    """
    从 .json 文件加载双部网络数据（适配 Prunus / Pollinator 文件）

    参数
    ----
    path : str
        JSON 文件路径

    返回
    ----
    G : nx.Graph
        一个 networkx 的双部图对象
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    G = nx.Graph()  # 无向双部网络

    # 1️⃣ 添加节点
    for node in data["nodes"]:
        G.add_node(node["name"], group=node["group"])

    # 2️⃣ 添加边（source/target 对应 nodeid）
    for link in data["links"]:
        source = data["nodes"][link["source"]]["name"]
        target = data["nodes"][link["target"]]["name"]
        G.add_edge(source, target)

    return G


from random_score import score_random
def test():
    # ===== 1. 构建一个示例图 =====
    graph = {
        "a": ["b", "c", "d"],
        "b": ["e", "c"],
        "c": ["g", "h"],
        "d": ["c", "f"],
        "e": ["a"],
        "f": ["a"],
        "g": ["a"],
        "h": ["a"],
    }

    # 使用 networkx（仅为了展示）
    G = nx.DiGraph()
    for u, outs in graph.items():
        for v in outs:
            G.add_edge(u, v)

    alive_A = list(G.nodes())  # 所有活着的节点

    # ===== 2. 调用 Random 算法 =====
    print("🎲 Random algorithm results:")
    random_scores = score_random(G, alive_A)
    for node, score in random_scores.items():
        print(f"  {node}: {score:.4f}")

    # ===== 3. 调用 PageRank 算法 =====
    print("\n📈 PageRank algorithm results:")
    pagerank_scores = pageRank(graph, t=100, s=0.15)
    for node, score in pagerank_scores.items():
        print(f"  {node}: {score:.4f}")

    # ===== 4. 结果比较 =====
    print("\n✅ Sum of random scores:", round(sum(random_scores.values()), 4))
    print("✅ Sum of pagerank scores:", round(sum(pagerank_scores.values()), 4))



    # 读取网络
    G = load_bipartite_graph("tests/network_prunus.json")

    # 所有 Herbivore 节点
    alive_A = [n for n, d in G.nodes(data=True) if d["group"] == "Herbivore"]

    # 测试 Random
    random_scores = score_random(G, alive_A)
    print("Random:", list(random_scores.items())[:5])

    # 测试 PageRank
    #   注意 PageRank 是针对整张图（有向或无向）计算的
    graph_dict = {u: list(G.neighbors(u)) for u in G.nodes()}
    pagerank_scores = pageRank(graph_dict)
    print("PageRank:", list(pagerank_scores.items())[:5])


if __name__ == "__main__":
    test()