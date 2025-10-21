import networkx as nx

def _is_active(attrs: dict) -> bool:
    t = attrs.get("type")
    if t is not None:
        return t == "active"
    return attrs.get("bipartite") == 0

def degree_strategy(G) -> dict:
    """
    Degree scores for active nodes: {node: degree}.
   node['type']=='active' or node['bipartite']==0
    """
    Gu = G.to_undirected()
    deg = dict(Gu.degree())
    return {n: deg.get(n, 0) for n, d in G.nodes(data=True) if _is_active(d)}