import numpy as np
import networkx as nx

def musrank_rev(G, max_iter=100, tol=1e-6, verbose=False):
    active = [n for n, d in G.nodes(data=True) if d.get('bipartite') == 0]
    passive = [n for n, d in G.nodes(data=True) if d.get('bipartite') == 1]
    if not active or not passive:
        raise ValueError("Graph must have bipartite attributes 0 (active) and 1 (passive)")

    A = nx.bipartite.biadjacency_matrix(G, row_order=active, column_order=passive).toarray()
    M = A.T  # passive in rows, active in columns

    I_P = np.ones(M.shape[0])
    V_A = np.ones(M.shape[1])

    for it in range(max_iter):
        I_prev = I_P.copy()

        # (1) Update importance of passive nodes
        I_P = M.dot(V_A)
        I_P /= I_P.max()

        # (2) Update vulnerability of active nodes: 1 / Σ_P (M_PA / I_P)
        denom = M.T.dot(1.0 / np.clip(I_P, 1e-12, None))
        V_A = 1.0 / np.clip(denom, 1e-12, None)
        V_A /= V_A.max()

        if np.linalg.norm(I_P - I_prev) < tol:
            break
        if verbose:
            print(f"[iter {it+1}] diff={np.linalg.norm(I_P - I_prev):.3e}")

    I_dict = {p: float(I_P[i]) for i, p in enumerate(passive)}
    V_dict = {a: float(V_A[j]) for j, a in enumerate(active)}

    return I_dict, V_dict
