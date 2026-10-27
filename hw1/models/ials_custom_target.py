from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .ials import ImplicitALS

DAY_MS = 86_400_000


def custom_target(ratings: np.ndarray, timestamps_ms: np.ndarray, t_end_ms: int,
                  beta: float, delta: float, half_life_days: float) -> np.ndarray:
    s = beta + (1.0 - beta) * (ratings - 1.0) / 4.0
    age_days = np.maximum(t_end_ms - timestamps_ms, 0) / DAY_MS
    d = delta + (1.0 - delta) * np.power(2.0, -age_days / half_life_days)
    return (s * d).astype(np.float32)


class CustomTargetIALS(ImplicitALS):
    def __init__(self, factors: int = 64, reg: float = 1.0, alpha: float = 10.0,
                 iterations: int = 15, seed: int = 42, beta: float = 0.3,
                 delta: float = 0.3, half_life_days: float = 365.0, solver: str = "cg",
                 cg_steps: int = 3):
        super().__init__(factors=factors, reg=reg, alpha=alpha, iterations=iterations, seed=seed,
                         solver=solver, cg_steps=cg_steps)
        self.beta = beta
        self.delta = delta
        self.half_life_days = half_life_days

    def fit_interactions(self, users: np.ndarray, items: np.ndarray, ratings: np.ndarray,
                         timestamps_ms: np.ndarray, n_users: int, n_items: int, t_end_ms: int):
        """Fit from raw training interactions (one row per user-item pair)."""
        y = custom_target(
            ratings,
            timestamps_ms,
            t_end_ms,
            self.beta,
            self.delta,
            self.half_life_days)
        P = sp.csr_matrix((y, (users, items)), shape=(n_users, n_items))
        P.sum_duplicates()
        P.sort_indices()
        W = P.copy()
        # same confidence as the baseline
        W.data = np.full_like(P.data, self.alpha)
        self.train_matrix = P
        return self.fit(P, P=P, W=W)
