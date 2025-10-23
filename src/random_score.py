import random 
from typing import Dict
import networkx as nx 

__all__ = ["score_random"]  

def score_random( G: nx.Graph, rng: random.Random | None = None,) -> Dict[str, float]:
    """
    为所有的点添加随机分数，适配其他算法

    参数：
    G : 当前图
    rng : random参数可以为空

    返回：
    Dict[str, float]
        形如 {节点: 随机分} 的字典。
    """
    r = rng if rng is not None else random   
    result = {}
    for node in G.nodes:
        key = str(node)
        value = r.random()
        result[key] = value
    return result



