from typing import Dict, List

import networkx as nx 

def SPARSEMATVECTPROD(T: Dict[str, Dict[str, float]], X: Dict[str, float]) -> Dict[str, float]:
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


def NORMALIZE(X: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize vector so that sum(X) = 1.
    Equivalent to X[i] = X[i] / sum(X.values()).
    """
    total = sum(X.values())
    if total == 0:
        return X
    return {i: X[i] / total for i in X}


def pageRank(graph: Dict[str, List[str]], t: int = 100, s: float = 0.15,tol: float = 1e-7) -> Dict[str, float]:
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
        X_prev = X
        prod = SPARSEMATVECTPROD(T, X_prev)
        # 更新 + 蒸发项
        X = {i: (1 - s) * prod[i] + s * I[i] for i in nodes}
        # 归一化，防止误差累积
        X = NORMALIZE(X)
                # 提前停止条件
        diff = sum(abs(X[i] - X_prev[i]) for i in nodes)
        if diff <= tol:
            break

    return X
