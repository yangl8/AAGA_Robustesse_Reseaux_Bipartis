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


def biPageRank(
    graph_UV: Dict[str, List[str]],
    graph_VU: Dict[str, List[str]],
    t: int = 100,
    s_U: float = 0.1,   # 植物侧随机跳转概率(→ α_U = 0.9)
    s_V: float = 0.2,   # 传粉者侧随机跳转概率(→ α_V = 0.8)
    tol: float = 1e-7
) -> Dict[str, Dict[str, float]]:
    """
    BiPageRank algorithm for bipartite graphs (U ↔ V).

    Parameters
    ----------
    graph_UV : Dict[str, List[str]]
        Adjacency list for U → V edges
    graph_VU : Dict[str, List[str]]
        Adjacency list for V → U edges
    s_U : float
        Random jump probability for U (1 - damping factor)
    s_V : float
        Random jump probability for V (1 - damping factor)
    Returns
    -------
    Dict with 'U' and 'V' scores
    """

    U_nodes = list(graph_UV.keys())
    V_nodes = list(graph_VU.keys())

    # ---- 构建转移矩阵 P_{VU} 和 P_{UV} ----
    P_VU = {u: {} for u in U_nodes}  # 从 V→U
    for v in graph_VU:
        outdeg = len(graph_VU[v]) if graph_VU[v] else len(U_nodes)
        targets = graph_VU[v] if graph_VU[v] else U_nodes
        for u in targets:
            P_VU[u][v] = 1 / outdeg

    P_UV = {v: {} for v in V_nodes}  # 从 U→V
    for u in graph_UV:
        outdeg = len(graph_UV[u]) if graph_UV[u] else len(V_nodes)
        targets = graph_UV[u] if graph_UV[u] else V_nodes
        for v in targets:
            P_UV[v][u] = 1 / outdeg

    # ---- 初始化 ----
    PR_U = {u: 1 / len(U_nodes) for u in U_nodes}
    PR_V = {v: 1 / len(V_nodes) for v in V_nodes}
    I_U = PR_U.copy()
    I_V = PR_V.copy()

    # ---- 迭代 ----
    for _ in range(t):
        PR_U_prev, PR_V_prev = PR_U, PR_V

        # 更新两边
        prod_U = SPARSEMATVECTPROD(P_VU, PR_V_prev)
        prod_V = SPARSEMATVECTPROD(P_UV, PR_U_prev)

        PR_U = {u: (1 - s_U) * prod_U[u] + s_U * I_U[u] for u in U_nodes}
        PR_V = {v: (1 - s_V) * prod_V[v] + s_V * I_V[v] for v in V_nodes}

        # 归一化
        PR_U = NORMALIZE(PR_U)
        PR_V = NORMALIZE(PR_V)

        # 收敛检测
        diffU = sum(abs(PR_U[u] - PR_U_prev[u]) for u in U_nodes)
        diffV = sum(abs(PR_V[v] - PR_V_prev[v]) for v in V_nodes)
        if diffU + diffV <= tol:
            break

    return {"U": PR_U, "V": PR_V}