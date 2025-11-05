from typing import Dict, List
import numpy as np
from scipy.sparse import csr_matrix


# ==========================================================
# =============== Helper Functions =========================
# ==========================================================

def build_sparse_matrix(graph: Dict[str, List[str]], all_nodes=None):
    """
    Build csr_matrix
    """
    if all_nodes is None:
        all_nodes = list(graph.keys())
    idx = {n: i for i, n in enumerate(all_nodes)}

    rows, cols, data = [], [], []
    for u, vs in graph.items():
        if not vs:  
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
    col_sum = M.sum(axis=0, keepdims=True)
    zero_cols = (col_sum == 0)
    if np.any(zero_cols):
        M[:, zero_cols.flatten()] = 1.0 / M.shape[0]
        col_sum = M.sum(axis=0, keepdims=True)
    return M / col_sum


# ==========================================================
# =============== Standard PageRank ========================
# ==========================================================

def pageRank(graph: Dict[str, List[str]],
             alpha: float = 0.85,
             max_iter: int = 100,
             tol: float = 1e-6) -> Dict[str, float]:

    P, nodes = build_sparse_matrix(graph)
    n = len(nodes)

    r = np.ones(n) / n 
    # teleport = np.ones(n) / n               

    for _ in range(max_iter):
        r_prev = r.copy()
        r = (1 - alpha) / n + alpha * (P @ r_prev)
        r /= r.sum()
        if np.linalg.norm(r - r_prev, 1) < tol:
            break

    return {nodes[i]: float(r[i]) for i in range(n)}


# ==========================================================
# =============== Bipartite BiPageRank =====================
# ==========================================================

def biPageRank(W_AP: np.ndarray,
               alpha_a: float = 0.9,
               alpha_p: float = 0.8,
               max_iter: int = 100,
               tol: float = 1e-7):
    """

    input:
    W_AP : shape (n_a, n_p)
    alpha_a :
        Damping factor on the active side
    alpha_p : 
        Damping factor on the passive side

    return:
    1. PageRank values for active nodes
    2. PageRank values for passive nodes
    """
    n_a, n_p = W_AP.shape

    P_PA = normalize_columns(W_AP.T)  # P <- A
    P_AP = normalize_columns(W_AP)    # A <- P

    # Initialise
    R_A = np.ones(n_a) / n_a
    R_P = np.ones(n_p) / n_p


    for _ in range(max_iter):
        R_A_prev, R_P_prev = R_A.copy(), R_P.copy()

        # Alternate propagation: P→A, A→P
        R_A = (1 - alpha_a) / n_a + alpha_a * (P_AP @ R_P_prev)
        R_P = (1 - alpha_p) / n_p + alpha_p * (P_PA @ R_A_prev)

        # Normalise
        R_A /= R_A.sum()
        R_P /= R_P.sum()

        diff = np.sum(np.abs(R_A - R_A_prev)) + np.sum(np.abs(R_P - R_P_prev))
        if diff < tol:
            break

    return R_A, R_P



