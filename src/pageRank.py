from typing import Dict, List, Iterable, Optional

__all__ = ["pageRank"]


def multiply_sparse_matrix_vector(
    T: Dict[str, Dict[str, float]],
    X: Dict[str, float]
) -> Dict[str, float]:
    """
    Sparse matrix × vector product.
    - T[i][j] = probability of j → i
    - X[j] = current PageRank value of node j
    returns U[i] = sum_j T[i][j] * X[j]
    """
    U = {i: 0.0 for i in T}
    for i, row in T.items():
        s = 0.0
        for j, val in row.items():
            s += val * X[j]
        U[i] = s
    return U


def normalize(X: Dict[str, float]) -> Dict[str, float]:
    """Scale X so that sum(X.values()) = 1 (if total>0)."""
    total = sum(X.values())
    if total <= 0:
        return X
    inv = 1.0 / total
    return {i: X[i] * inv for i in X}


def _collect_all_nodes(graph: Dict[str, List[str]]) -> List[str]:
    """Ensure nodes include keys and all neighbors appearing in adjacency lists."""
    nodes = set(graph.keys())
    for u, nbrs in graph.items():
        for v in nbrs:
            nodes.add(v)
    return list(nodes)


def pageRank(
    graph: Dict[str, List[str]],
    t: int = 100,                # 最大迭代次数
    s: float = 0.15,             # 蒸发/随机跳转概率（damping = 1-s）
    tol: float = 1e-10,          # 提前停止阈值（L1 差异）
    personalize: Optional[Dict[str, float]] = None  # 个性化/定向随机跳转分布
) -> Dict[str, float]:
    """
    PageRank（幂迭代，带提前收敛停止）
    graph:  有向邻接表： u → [v1, v2, ...] ；若是无向图，先把每条边转换成双向。
    t:      最大迭代次数（max iterations）
    s:      随机跳转概率，典型取 0.15；(1 - s) 是沿边走的概率
    tol:    若相邻两次迭代的 L1 差异 <= tol，则提前停止
    personalize: 可选的随机跳转分布（不提供时为均匀分布）。会自动归一化。

    返回：节点到 PageRank 分值的映射（总和为 1）。
    """
    # 1) 节点全集（含仅出现在邻居位置的节点）
    nodes = _collect_all_nodes(graph)
    n = len(nodes)
    if n == 0:
        return {}

    # 2) 个性化向量 I（随机跳转的目标分布）
    if personalize is None:
        I = {i: 1.0 / n for i in nodes}
    else:
        # 缺失节点权重默认 0，未出现但在图中的节点补齐
        I = {i: float(personalize.get(i, 0.0)) for i in nodes}
        I = normalize(I)
        # 若用户给的分布全为 0，退回均匀
        if sum(I.values()) == 0.0:
            I = {i: 1.0 / n for i in nodes}

    # 3) 构建稀疏转移矩阵 T（列随机：每列 j 的和 = 1）
    # T[v][u] = 1/out_deg(u) if u→v；若 u 无外出边（dangling），对所有 v 均匀分配 1/n
    T: Dict[str, Dict[str, float]] = {v: {} for v in nodes}
    # 先计算每个 u 的 out_degree
    out_deg = {u: len(graph.get(u, [])) for u in nodes}

    for u in nodes:
        if out_deg[u] == 0:
            # dangling：把 u 的概率均匀分给所有节点
            share = 1.0 / n
            for v in nodes:
                T[v][u] = share
        else:
            inv = 1.0 / out_deg[u]
            for v in graph.get(u, []):
                # v 一定在 nodes 内（已统一）
                T[v].setdefault(u, 0.0)
                T[v][u] += inv

    # 4) 初始化 PageRank 向量（均匀）
    X = {i: 1.0 / n for i in nodes}

    # 5) 幂迭代（带提前停止）
    #    X_new = (1 - s) * T @ X + s * I
    for _ in range(t):
        prod = multiply_sparse_matrix_vector(T, X)
        X_new = {i: (1.0 - s) * prod[i] + s * I[i] for i in nodes}
        X_new = normalize(X_new)  # 归一化抑制数值漂移

        # 收敛检测（L1 范数差异）
        delta = sum(abs(X_new[i] - X[i]) for i in nodes)
        X = X_new
        if delta <= tol:
            break

    return X
