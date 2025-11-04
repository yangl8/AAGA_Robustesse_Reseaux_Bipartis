from typing import Dict, List
import numpy as np
from scipy.sparse import csr_matrix


# ==========================================================
# =============== Helper Functions =========================
# ==========================================================

def build_sparse_matrix(graph: Dict[str, List[str]], all_nodes=None):
    """
    Build a sparse column-stochastic transition matrix (csr_matrix).

    Parameters
    ----------
    graph : Dict[str, List[str]]
        Adjacency list, graph[u] = [v1, v2, ...] means u → v.
    all_nodes : List[str], optional
        Node ordering; if None, extract from graph automatically.

    Returns
    -------
    P : csr_matrix
        Column-stochastic transition matrix where P[v,u] = 1/out_degree(u)
    nodes : List[str]
        Node ordering
    """
    if all_nodes is None:
        all_nodes = list(graph.keys())
    idx = {n: i for i, n in enumerate(all_nodes)}

    rows, cols, data = [], [], []
    for u, vs in graph.items():
        if not vs:  # dangling node (dead-end)
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
    Normalize each column so that the matrix becomes column-stochastic.

    Columns with zero out-degree (dangling nodes) are distributed evenly.

    Parameters
    ----------
    M : np.ndarray
        Any matrix

    Returns
    -------
    M_norm : np.ndarray
        Column-normalized matrix
    """
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
    """
    Standard PageRank algorithm (sparse matrix implementation).

    Parameters
    ----------
    graph : Dict[str, List[str]]
        Adjacency list, graph[u] = [v1, v2, ...] means u → v.
    alpha : float
        Damping factor (commonly 0.85)
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence threshold (L1 norm)

    Returns
    -------
    ranks : Dict[str, float]
        PageRank value of each node
    """
    P, nodes = build_sparse_matrix(graph)
    n = len(nodes)

    r = np.ones(n) / n                       # initial vector
    teleport = np.ones(n) / n                # teleportation vector

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
    BiPageRank algorithm for bipartite graphs.

    Parameters
    ----------
    W_AP : np.ndarray, shape (n_a, n_p)
        Adjacency matrix from active to passive nodes (1 means an edge)
    alpha_a : float
        Damping factor on the active side
    alpha_p : float
        Damping factor on the passive side
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence threshold (L1 norm)

    Returns
    -------
    R_A : np.ndarray
        PageRank values for active nodes
    R_P : np.ndarray
        PageRank values for passive nodes
    """
    n_a, n_p = W_AP.shape

    # Build column-stochastic transition matrices
    P_PA = normalize_columns(W_AP.T)  # P <- A
    P_AP = normalize_columns(W_AP)    # A <- P

    # Initialise scores
    R_A = np.ones(n_a) / n_a
    R_P = np.ones(n_p) / n_p

    # Iterative updates
    for _ in range(max_iter):
        R_A_prev, R_P_prev = R_A.copy(), R_P.copy()

        # Alternate propagation: P→A, A→P
        R_A = (1 - alpha_a) / n_a + alpha_a * (P_AP @ R_P_prev)
        R_P = (1 - alpha_p) / n_p + alpha_p * (P_PA @ R_A_prev)

        # Normalise
        R_A /= R_A.sum()
        R_P /= R_P.sum()

        # Convergence check
        diff = np.sum(np.abs(R_A - R_A_prev)) + np.sum(np.abs(R_P - R_P_prev))
        if diff < tol:
            break

    return R_A, R_P


# ==========================================================
# =============== Examples & Tests =========================
# ==========================================================

if __name__ == "__main__":
    # ---------- Single-side PageRank example ----------
    graph = {
        'A': ['B', 'C'],
        'B': ['C'],
        'C': []
    }
    ranks = pageRank(graph)
    print("PageRank (single):")
    for k, v in ranks.items():
        print(f"  {k}: {v:.4f}")

    # ---------- Bipartite BiPageRank example ----------
    # A1, A2 -> P1, P2, P3
    W_AP = np.array([
        [1, 1, 0],  # A1 connects to P1, P2
        [0, 1, 1],  # A2 connects to P2, P3
    ], dtype=float)

    R_A, R_P = biPageRank(W_AP, alpha_a=0.8, alpha_p=0.9)
    print("\nBiPageRank (bipartite):")
    print("  Active  (A):", np.round(R_A, 4))
    print("  Passive (P):", np.round(R_P, 4))
