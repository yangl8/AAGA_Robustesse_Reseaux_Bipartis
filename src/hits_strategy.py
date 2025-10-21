import networkx as nx
import numpy as np

def _is_active(attrs: dict) -> bool:
    t = attrs.get("type")
    if t is not None:
        return t == "active"
    return attrs.get("bipartite") == 0

def _active_nodes(G):
    return [n for n, d in G.nodes(data=True) if _is_active(d)]

def _passive_nodes(G):
    # 另一侧：type=='passive' 或 bipartite==1
    res = []
    for n, d in G.nodes(data=True):
        t = d.get("type")
        if t is not None:
            if t == "passive":
                res.append(n)
        else:
            if d.get("bipartite") == 1:
                res.append(n)
    return res

# 1) 调用库版本（networkx.hits）
def hits_strategy(G) -> dict:
    """
    HITS hub scores for active nodes: {node: hub_score}.
    """
    try:
        hubs, _ = nx.hits(G, max_iter=500, normalized=True)
    except nx.PowerIterationFailedConvergence:
        hubs, _ = nx.hits(G.to_directed(), max_iter=500, normalized=True)
    return {n: hubs.get(n, 0.0) for n, d in G.nodes(data=True) if _is_active(d)}

# 2) 自实现版本（线性 HITS，基于二部 biadjacency 迭代）
def hits_strategy_scratch(G, max_iter: int = 1000, tol: float = 1e-9) -> dict:
    """
    标准线性 HITS 的教学实现：
      h <- B a; a <- B^T h，每步 L2 归一化；返回 active 节点的 hub 向量 h。
    要求图能区分 active/passive（通过 type 或 bipartite）。
    """
    active = _active_nodes(G)
    passive = _passive_nodes(G)
    if not active or not passive:
        # 若没有分侧信息，退回库版本
        return hits_strategy(G)

    # 构造 biadjacency 矩阵 B (active x passive)
    a_index = {n: i for i, n in enumerate(active)}
    p_index = {n: i for i, n in enumerate(passive)}
    B = np.zeros((len(active), len(passive)), dtype=float)
    for u, v in G.edges():
        if u in a_index and v in p_index:
            B[a_index[u], p_index[v]] = 1.0
        elif v in a_index and u in p_index:
            B[a_index[v], p_index[u]] = 1.0

    # 线性迭代
    h = np.ones(len(active))
    a = np.ones(len(passive))

    def n2(x):
        n = np.linalg.norm(x)
        return x / (n if n != 0 else 1.0)

    for _ in range(max_iter):
        h_old, a_old = h.copy(), a.copy()
        h = n2(B.dot(a))          # hubs
        a = n2(B.T.dot(h))        # authorities
        if np.linalg.norm(h - h_old, 1) + np.linalg.norm(a - a_old, 1) < tol:
            break

    return {active[i]: float(h[i]) for i in range(len(active))}

__all__ = ["hits_strategy", "hits_strategy_scratch"]