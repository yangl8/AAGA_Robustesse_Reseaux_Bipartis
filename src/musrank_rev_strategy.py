import numpy as np
import networkx as nx

def musrank_rev(G, max_iter=100, tol=1e-6, verbose=False):
    """
    反向 MusRank（MusRank-rev）
    -------------------------------------
    用于计算 passive 端（如植物）的重要性 I_P
    和 active 端（如动物）的脆弱性 V_A。
    
    参数：
        G : networkx.Graph (bipartite)
            二部网络，节点属性中需包含 'bipartite' = 0/1
        max_iter : int
            最大迭代次数
        tol : float
            收敛阈值
        verbose : bool
            若为 True，则输出迭代日志
    
    返回：
        I_rev : dict
            passive 端的重要性分数
        V_rev : dict
            active 端的脆弱性分数
    """
    
    # === 1. 获取两类节点 ===
    active = [n for n, d in G.nodes(data=True) if d.get('bipartite') == 0]
    passive = [n for n, d in G.nodes(data=True) if d.get('bipartite') == 1]
    
    # 如果属性缺失，可自动检测
    if not active or not passive:
        raise ValueError("Graph must have bipartite attributes 0 (active) and 1 (passive)")
    
    # === 2. 构建邻接矩阵 ===
    A = nx.bipartite.biadjacency_matrix(G, row_order=active, column_order=passive).toarray()
    M = A.T  # 转置矩阵：P 在行，A 在列（反向）
    
    # === 3. 初始化 I 和 V ===
    I_P = np.ones(M.shape[0])
    V_A = np.ones(M.shape[1])
    
    # === 4. 迭代更新 ===
    for it in range(max_iter):
        I_prev = I_P.copy()
        
        # 更新 passive 重要性
        I_P = M.dot(V_A)
        if I_P.max() > 0:
            I_P /= I_P.max()
        
        # 更新 active 脆弱性（注意这里取倒数）
        inv_I = np.where(I_P > 1e-12, 1.0 / I_P, 0.0)
        V_A = M.T.dot(inv_I)
        if V_A.max() > 0:
            V_A /= V_A.max()
        
        # 判断收敛
        diff = np.linalg.norm(I_P - I_prev)
        if verbose:
            print(f"[iter={it+1}] diff={diff:.3e}")
        if diff < tol:
            break
    
    # === 5. 转为字典返回 ===
    I_dict = {p: float(I_P[i]) for i, p in enumerate(passive)}
    V_dict = {a: float(V_A[j]) for j, a in enumerate(active)}

    print(f"[DEBUG] musrank_rev() passive={len(I_dict)}, active={len(V_dict)}")
    print(f"[DEBUG] first 3 passive scores: {list(I_dict.items())[:3]}")

    
    return I_dict, V_dict
