from __future__ import annotations

import numpy as np
import scipy.linalg
import scipy.sparse as sp


class EASE:
    def __init__(self, reg: float = 500.0):
        self.reg = reg

    def fit(self, train: sp.csr_matrix):
        X = train.astype(bool).astype(np.float32).tocsr()
        self.X = X
        G = (X.T @ X).toarray().astype(np.float64)
        G[np.diag_indices_from(G)] += self.reg
        # G is symmetric positive definite -> Cholesky-based inverse
        c, low = scipy.linalg.cho_factor(G, overwrite_a=True)
        P = scipy.linalg.cho_solve(
            (c, low), np.eye(
                G.shape[0]), overwrite_b=True)
        del G, c
        B = P / (-np.diag(P))[None, :]
        B[np.diag_indices_from(B)] = 0.0
        self.B = B.astype(np.float32)
        return self

    def score(self, user_ids: np.ndarray) -> np.ndarray:
        return np.asarray(self.X[user_ids] @ self.B)
