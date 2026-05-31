"""Tournament GA 优化器 —— 锦标赛选择 + 混合交叉 + 变异。"""
import numpy as np


class TournamentGAOptimizer:
    def __init__(self, dimension, bounds, batch_size=4, sigma0=0.2):
        self.N = dimension
        self.bounds = np.array(bounds, dtype=float)
        self.lam = batch_size               # 种群大小 λ
        self.tournament_size = max(2, self.lam // 3)  # 锦标赛 k
        self.crossover_rate = 0.9           # 交叉概率
        self.mutation_rate = 0.25           # 每基因变异概率
        self.mutation_strength = sigma0     # 变异强度 (bounds 比例)
        self.elite_count = max(1, self.lam // 4)  # 精英保留数
        self.reset()

    # ── 公共接口（与 CMA-ES 对齐） ──
    def reset(self):
        """随机初始化种群。"""
        self.population = np.array([
            [np.random.uniform(low, high) for low, high in self.bounds]
            for _ in range(self.lam)
        ], dtype=float)
        self.fitness = np.zeros(self.lam, dtype=float)
        self.mean = np.mean(self.population, axis=0)
        self.C = np.eye(self.N, dtype=float)     # 兼容字段
        self.sigma = self.mutation_strength
        self.p_s = np.zeros(self.N, dtype=float)  # 兼容
        self.p_c = np.zeros(self.N, dtype=float)  # 兼容
        self.current_z = []
        self.current_y = []
        self.gen = 0

    def clip(self, x):
        return np.clip(x, self.bounds[:, 0], self.bounds[:, 1])

    def sample_batch(self):
        """返回当前种群作为候选批次。"""
        self.current_y = self.population.tolist()
        self.current_z = []  # GA 无 z 概念
        return [self.population[i].copy() for i in range(self.lam)]

    def update(self, ranking):
        """
        锦标赛选择 + 混合交叉 (BLX-α) + 高斯变异。
        ranking[i] = 1 表示第 i 个候选最好。
        """
        # 1. 将排名转为 fitness（排名越靠前 fitness 越高）
        order = np.argsort(ranking)
        fitness = np.zeros(self.lam, dtype=float)
        for rank_idx, pop_idx in enumerate(order):
            fitness[pop_idx] = float(self.lam - rank_idx)

        # 2. 精英保留
        elite_indices = order[:self.elite_count]
        elites = self.population[elite_indices].copy()

        # 3. 锦标赛选择 + 交叉 + 变异 生成新种群
        new_pop = np.zeros_like(self.population)
        new_pop[:self.elite_count] = elites  # 精英直接保留

        for i in range(self.elite_count, self.lam):
            # 锦标赛选择两个父代
            p1 = self._tournament_select(fitness)
            p2 = self._tournament_select(fitness)

            # 混合交叉 (BLX-α)
            if np.random.random() < self.crossover_rate:
                child = self._blend_crossover(p1, p2, alpha=0.3)
            else:
                child = p1.copy()

            # 高斯变异
            child = self._mutate(child)
            new_pop[i] = self.clip(child)

        self.population = new_pop
        self.mean = np.mean(self.population, axis=0)

        # 更新多样性指标
        self.sigma = float(np.mean(np.std(self.population, axis=0)))

        # 兼容：更新 current_y 为新一代种群
        self.current_y = self.population.tolist()
        self.current_z = []
        self.gen += 1

    # ── 内部方法 ──
    def _tournament_select(self, fitness):
        """k-锦标赛选择：随机选 k 个，返回 fitness 最高的个体。"""
        k = min(self.tournament_size, self.lam)
        candidates = np.random.choice(self.lam, size=k, replace=False)
        best = candidates[np.argmax(fitness[candidates])]
        return self.population[best].copy()

    def _blend_crossover(self, p1, p2, alpha=0.3):
        """BLX-α 混合交叉。若父代相同则至少在一个维度上外扩探索。"""
        child = np.zeros(self.N, dtype=float)
        any_diff = False
        for d in range(self.N):
            lo = min(p1[d], p2[d])
            hi = max(p1[d], p2[d])
            rng = hi - lo
            lo_ext = lo - alpha * rng
            hi_ext = hi + alpha * rng
            # 若 lo_ext == hi_ext（父代在该维度完全相同），仍可能产生相同值
            if lo_ext >= hi_ext:
                lo_ext = max(self.bounds[d, 0], lo - 0.05)
                hi_ext = min(self.bounds[d, 1], hi + 0.05)
            else:
                any_diff = True
            child[d] = np.random.uniform(lo_ext, hi_ext)
        # 父代所有维度相同 → 强制在随机维度做一次大步探索
        if not any_diff:
            d = np.random.randint(self.N)
            bound_range = self.bounds[d, 1] - self.bounds[d, 0]
            child[d] = np.random.uniform(
                max(self.bounds[d, 0], child[d] - 0.3 * bound_range),
                min(self.bounds[d, 1], child[d] + 0.3 * bound_range),
            )
        return child

    def _mutate(self, individual):
        """高斯变异：每基因独立变异，保证至少一个维度被修改。"""
        mutant = individual.copy()
        mutated_any = False
        for d in range(self.N):
            if np.random.random() < self.mutation_rate:
                bound_range = self.bounds[d, 1] - self.bounds[d, 0]
                noise = np.random.normal(0, self.mutation_strength * bound_range)
                mutant[d] += noise
                mutated_any = True
        # 保证至少一个基因被变异，防止完全克隆
        if not mutated_any:
            d = np.random.randint(self.N)
            bound_range = self.bounds[d, 1] - self.bounds[d, 0]
            noise = np.random.normal(0, self.mutation_strength * bound_range)
            mutant[d] += noise
        return mutant

    # ── 序列化 ──
    def get_state(self):
        return {
            "method": "Tournament GA",
            "N": int(self.N),
            "bounds": self.bounds.tolist(),
            "batch_size": int(self.lam),
            "tournament_size": int(self.tournament_size),
            "crossover_rate": float(self.crossover_rate),
            "mutation_rate": float(self.mutation_rate),
            "mutation_strength": float(self.mutation_strength),
            "elite_count": int(self.elite_count),
            "population": self.population.tolist(),
            "mean": self.mean.tolist(),
            "C": self.C.tolist(),
            "sigma": float(self.sigma),
            "p_s": self.p_s.tolist(),
            "p_c": self.p_c.tolist(),
            "gen": int(self.gen),
        }

    @classmethod
    def from_state(cls, state):
        optimizer = cls(
            state["N"], state["bounds"],
            batch_size=int(state.get("batch_size", 4)),
        )
        optimizer.tournament_size = int(state.get("tournament_size", optimizer.tournament_size))
        optimizer.crossover_rate = float(state.get("crossover_rate", optimizer.crossover_rate))
        optimizer.mutation_rate = float(state.get("mutation_rate", optimizer.mutation_rate))
        optimizer.mutation_strength = float(state.get("mutation_strength", optimizer.mutation_strength))
        optimizer.elite_count = int(state.get("elite_count", optimizer.elite_count))
        optimizer.population = np.array(state["population"], dtype=float)
        optimizer.mean = np.array(state["mean"], dtype=float)
        optimizer.C = np.array(state["C"], dtype=float)
        optimizer.sigma = float(state["sigma"])
        optimizer.p_s = np.array(state["p_s"], dtype=float)
        optimizer.p_c = np.array(state["p_c"], dtype=float)
        optimizer.gen = int(state["gen"])
        return optimizer
