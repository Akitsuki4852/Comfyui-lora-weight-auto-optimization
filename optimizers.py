"""优化算法注册表 —— 工厂方法 & 方法发现。"""
from .cma_es import CMAESOptimizer
from .tournament_ga import TournamentGAOptimizer

METHODS = {
    "CMA-ES": CMAESOptimizer,
    "Tournament GA": TournamentGAOptimizer,
}

METHOD_NAMES = list(METHODS.keys())


def create_optimizer(method, dimension, bounds, batch_size=3, sigma0=0.2):
    """根据方法名创建优化器实例。"""
    cls = METHODS.get(method)
    if cls is None:
        raise ValueError(f"Unknown method: '{method}'. Available: {METHOD_NAMES}")
    return cls(dimension, bounds, batch_size=batch_size, sigma0=sigma0)


def optimizer_from_state(state):
    """从 state 字典恢复优化器，自动识别方法。"""
    method = state.get("method", "CMA-ES")
    cls = METHODS.get(method)
    if cls is None:
        raise ValueError(f"Unknown method in log: '{method}'. Available: {METHOD_NAMES}")
    return cls.from_state(state)
