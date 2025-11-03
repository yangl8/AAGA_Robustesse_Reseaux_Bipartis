from typing import Dict, List
import numpy as np
from scipy.sparse import csr_matrix


# ==========================================================
# =============== 工具函数 =================================
# ==========================================================

def build_sparse_matrix(graph: Dict[str, List[str]], all_nodes=None):
    """
    构建稀疏列随机转移矩阵 (csr_matrix)。

    参数
    ----------
    graph : Dict[str, List[str]]
        邻接表，graph[u] = [v1, v2, ...] 表示 u → v。
    all_nodes : List[str], 可选
        节点顺序；若为 None，则自动从 graph 中提取。

    返回
    ----------
    P : csr_matrix
        列随机转移矩阵，使得 P[v,u] = 1/out_degree(u)
    nodes : List[str]
        节点顺序
    """
    if all_nodes is None:
        all_nodes = list(graph.keys())
    idx = {n: i for i, n in enumerate(all_nodes)}

    rows, cols, data = [], [], []
    for u, vs in graph.items():
        if not vs:  # 悬挂节点（dead-end）
            for v in all_nodes:
                rows.append(idx[v])
                cols.append(idx[u])
                data.append(1 / len(all_nodes))
        else:
            for v in vs:
                rows.append(idx[v])
                cols.append(idx[u])
                data.append(1 / len(vs))

    P = csr_matrix((data, (rows, cols)), shape=(len(all_nodes), len(all_nodes)))
    return P, all_nodes


def normalize_columns(M: np.ndarray) -> np.ndarray:
    """
    按列归一化矩阵，使每列和为 1（列随机矩阵）。

    对出度为 0 的列（悬挂节点）平均分配到所有行。

    参数
    ----------
    M : np.ndarray
        任意矩阵

    返回
    ----------
    M_norm : np.ndarray
        列归一后的矩阵
    """
    col_sum = M.sum(axis=0, keepdims=True)
    zero_cols = (col_sum == 0)
    if np.any(zero_cols):
        M[:, zero_cols.flatten()] = 1.0 / M.shape[0]
        col_sum = M.sum(axis=0, keepdims=True)
    return M / col_sum


# ==========================================================
# =============== 单侧 PageRank ============================
# ==========================================================

def pageRank(graph: Dict[str, List[str]],
             alpha: float = 0.85,
             max_iter: int = 100,
             tol: float = 1e-6) -> Dict[str, float]:
    """
    标准 PageRank 算法（稀疏矩阵实现）。

    参数
    ----------
    graph : Dict[str, List[str]]
        邻接表，graph[u] = [v1, v2, ...] 表示 u → v。
    alpha : float
        阻尼系数（通常取 0.85）
    max_iter : int
        最大迭代次数
    tol : float
        收敛阈值（L1 范数）

    返回
    ----------
    ranks : Dict[str, float]
        各节点的 PageRank 值
    """
    P, nodes = build_sparse_matrix(graph)
    n = len(nodes)

    r = np.ones(n) / n                       # 初始化
    teleport = np.ones(n) / n                # 随机跳转向量

    for _ in range(max_iter):
        r_prev = r.copy()
        r = (1 - alpha) / n + alpha * (P @ r_prev)
        r /= r.sum()
        if np.linalg.norm(r - r_prev, 1) < tol:
            break

    return {nodes[i]: float(r[i]) for i in range(n)}


# ==========================================================
# =============== 二部图 BiPageRank ========================
# ==========================================================

def biPageRank(W_AP: np.ndarray,
               alpha_a: float = 0.9,
               alpha_p: float = 0.8,
               max_iter: int = 100,
               tol: float = 1e-7):
    """
    BiPageRank 算法（二部图 PageRank）。

    参数
    ----------
    W_AP : np.ndarray, shape (n_a, n_p)
        A → P 的邻接矩阵 (1 表示有边)
    alpha_a : float
        Active 侧阻尼系数
    alpha_p : float
        Passive 侧阻尼系数
    max_iter : int
        最大迭代次数
    tol : float
        收敛阈值（L1 范数）

    返回
    ----------
    R_A : np.ndarray
        Active 节点的 PageRank 值
    R_P : np.ndarray
        Passive 节点的 PageRank 值
    """
    n_a, n_p = W_AP.shape

    # 构建列随机转移矩阵
    P_PA = normalize_columns(W_AP.T)  # P <- A
    P_AP = normalize_columns(W_AP)    # A <- P

    # 初始化
    R_A = np.ones(n_a) / n_a
    R_P = np.ones(n_p) / n_p

    # 迭代更新
    for _ in range(max_iter):
        R_A_prev, R_P_prev = R_A.copy(), R_P.copy()

        # 交替传播：P→A, A→P
        R_A = (1 - alpha_a) / n_a + alpha_a * (P_AP @ R_P_prev)
        R_P = (1 - alpha_p) / n_p + alpha_p * (P_PA @ R_A_prev)

        # 归一化
        R_A /= R_A.sum()
        R_P /= R_P.sum()

        # 收敛检测
        diff = np.sum(np.abs(R_A - R_A_prev)) + np.sum(np.abs(R_P - R_P_prev))
        if diff < tol:
            break

    return R_A, R_P


# ==========================================================
# =============== 示例与测试 ===============================
# ==========================================================

if __name__ == "__main__":
    # ---------- 单侧 PageRank 示例 ----------
    graph = {
        'A': ['B', 'C'],
        'B': ['C'],
        'C': []
    }
    ranks = pageRank(graph)
    print("PageRank (single):")
    for k, v in ranks.items():
        print(f"  {k}: {v:.4f}")

    # ---------- 二部图 BiPageRank 示例 ----------
    # A1, A2 -> P1, P2, P3
    W_AP = np.array([
        [1, 1, 0],  # A1 链接到 P1, P2
        [0, 1, 1],  # A2 链接到 P2, P3
    ], dtype=float)

    R_A, R_P = biPageRank(W_AP, alpha_a=0.8, alpha_p=0.9)
    print("\nBiPageRank (bipartite):")
    print("  Active  (A):", np.round(R_A, 4))
    print("  Passive (P):", np.round(R_P, 4))
