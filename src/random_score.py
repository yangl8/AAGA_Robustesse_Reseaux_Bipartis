import random 
from typing import Dict
import networkx as nx 

__all__ = ["score_random"]  

def score_random( G: nx.Graph, rng: random.Random | None = None,) -> Dict[str, float]:
    """
    Assign random scores to all nodes, compatible with other algorithms.

    Parameters:
    G : current graph
    rng : random parameter, can be None

    Returns:
    Dict[str, float]
        Dictionary of the form {node: random_score}.
    """
    r = rng if rng is not None else random   
    result = {}
    for node in G.nodes:
        key = str(node)
        value = r.random()
        result[key] = value
    return result



