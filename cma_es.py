"""CMA-ES 优化器 —— 协方差矩阵自适应进化策略。"""
import numpy as np


class CMAESOptimizer:
    def __init__(self, dimension, bounds, batch_size=3, sigma0=0.2):
        self.N = dimension
        self.bounds = np.array(bounds, dtype=float)
        self.lam = batch_size
        self.mu = max(1, self.lam // 2)
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)
        self.c_s = (self.mu_eff + 2) / (self.N + self.mu_eff + 5)
        self.d_s = 1.0 + 2.0 * max(0, np.sqrt((self.mu_eff - 1) / (self.N + 1)) - 1) + self.c_s
        self.c_c = (4 + self.mu_eff / self.N) / (self.N + 4 + 2 * self.mu_eff / self.N)
        self.c_1 = 2.0 / ((self.N + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(1 - self.c_1,
                        2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.N + 2) ** 2 + self.mu_eff))
        self.chiN = np.sqrt(self.N) * (1 - 1.0 / (4 * self.N) + 1.0 / (21 * self.N ** 2))
        self.reset()

    def reset(self):
        self.mean = np.array([np.random.uniform(low + 0.25 * (high - low), low + 0.75 * (high - low))
                              for low, high in self.bounds], dtype=float)
        self.C = np.eye(self.N, dtype=float)
        self.sigma = 0.2
        self.p_s = np.zeros(self.N, dtype=float)
        self.p_c = np.zeros(self.N, dtype=float)
        self.gen = 0

    def clip(self, x):
        return np.clip(x, self.bounds[:, 0], self.bounds[:, 1])

    def sample_batch(self):
        self.current_z = []
        self.current_y = []
        batch = []
        for _ in range(self.lam):
            z = np.random.randn(self.N)
            y = self.C @ z
            x = self.mean + self.sigma * y
            x = self.clip(x)
            batch.append(x)
            self.current_z.append(z.tolist())
            self.current_y.append(y.tolist())
        return batch

    def update(self, ranking):
        order = np.argsort(ranking)
        selected_idx = order[:self.mu]
        z_w = np.sum([self.weights[i] * np.array(self.current_z[idx], dtype=float)
                      for i, idx in enumerate(selected_idx)], axis=0)
        y_w = np.sum([self.weights[i] * np.array(self.current_y[idx], dtype=float)
                      for i, idx in enumerate(selected_idx)], axis=0)

        new_mean = np.sum(
            [self.weights[i] * self.clip(self.mean + self.sigma * np.array(self.current_y[idx], dtype=float))
             for i, idx in enumerate(selected_idx)], axis=0)
        self.mean = self.clip(new_mean)
        self.p_s = (1 - self.c_s) * self.p_s + np.sqrt(self.c_s * (2 - self.c_s) * self.mu_eff) * z_w
        sigma_ratio = np.linalg.norm(self.p_s) / self.chiN
        self.sigma *= np.exp(self.c_s / self.d_s * (sigma_ratio - 1))
        h_s = 1 if (np.linalg.norm(self.p_s) / np.sqrt(1 - (1 - self.c_s) ** (2 * (self.gen + 1)))
                    < (1.4 + 2 / (self.N + 1)) * self.chiN) else 0
        delta_h_s = (1 - h_s) * self.c_1 * self.c_c * (2 - self.c_c)
        self.p_c = (1 - self.c_c) * self.p_c + h_s * np.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * y_w
        rank_mu_update = np.zeros((self.N, self.N), dtype=float)
        for i, idx in enumerate(selected_idx):
            y = np.array(self.current_y[idx], dtype=float)
            rank_mu_update += self.weights[i] * np.outer(y, y)

        self.C = ((1 - self.c_1 - self.c_mu) * self.C
                  + self.c_1 * (np.outer(self.p_c, self.p_c) + delta_h_s * self.C)
                  + self.c_mu * rank_mu_update)
        self.C = (self.C + self.C.T) / 2
        self.gen += 1

    def get_state(self):
        return {
            "method": "CMA-ES",
            "N": int(self.N),
            "bounds": self.bounds.tolist(),
            "batch_size": int(self.lam),
            "mu": int(self.mu),
            "weights": self.weights.tolist(),
            "mu_eff": float(self.mu_eff),
            "c_s": float(self.c_s),
            "d_s": float(self.d_s),
            "c_c": float(self.c_c),
            "c_1": float(self.c_1),
            "c_mu": float(self.c_mu),
            "chiN": float(self.chiN),
            "mean": self.mean.tolist(),
            "C": self.C.tolist(),
            "sigma": float(self.sigma),
            "p_s": self.p_s.tolist(),
            "p_c": self.p_c.tolist(),
            "gen": int(self.gen),
        }

    @classmethod
    def from_state(cls, state):
        optimizer = cls(state["N"], state["bounds"], batch_size=int(state.get("batch_size", 3)))
        optimizer.mu = int(state.get("mu", optimizer.mu))
        optimizer.weights = np.array(state["weights"], dtype=float)
        optimizer.mu_eff = float(state["mu_eff"])
        optimizer.c_s = float(state["c_s"])
        optimizer.d_s = float(state["d_s"])
        optimizer.c_c = float(state["c_c"])
        optimizer.c_1 = float(state["c_1"])
        optimizer.c_mu = float(state["c_mu"])
        optimizer.chiN = float(state["chiN"])
        optimizer.mean = np.array(state["mean"], dtype=float)
        optimizer.C = np.array(state["C"], dtype=float)
        optimizer.sigma = float(state["sigma"])
        optimizer.p_s = np.array(state["p_s"], dtype=float)
        optimizer.p_c = np.array(state["p_c"], dtype=float)
        optimizer.gen = int(state["gen"])
        return optimizer
