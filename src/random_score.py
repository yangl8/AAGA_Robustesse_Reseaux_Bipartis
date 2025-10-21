import random 
from typing import Dict, Iterable
import networkx as nx 

__all__ = ["score_random"]  

def score_random( G: nx.Graph, alive_A: Iterable[str], rng: random.Random | None = None,) -> Dict[str, float]:
    """
    为所有的点添加随机分数，适配其他算法

    参数：
    G : 当前图（适配）
    alive_A : 存活的点
    rng : random参数可以为空

    返回：
    Dict[str, float]
        形如 {节点: 随机分} 的字典。
    """
    r = rng if rng is not None else random     # 选用随机源（优先用传入的 rng）
    result = {}
    for a in alive_A:
        key = str(a)
        value = r.random()
        result[key] = value
    return result


