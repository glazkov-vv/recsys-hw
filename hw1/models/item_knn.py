from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class ItemKNN:
    def __init__(self, k: int = 100, shrink: float = 10.0, alpha: float = 0.5,
                 user_weighting: bool = False, block_size: int = 2048):
        self.k = k
        self.shrink = shrink
        self.alpha = alpha
        self.user_weighting = user_weighting
        self.block_size = block_size

    def fit(self, train: sp.csr_matrix):
        X = train.astype(bool).astype(np.float32).tocsr()
        self.X = X
        n_items = X.shape[1]
        Xw = X
        if self.user_weighting:
            # log-damped inverse user activity
            act = np.asarray(X.sum(axis=1)).ravel()
            w = 1.0 / np.log2(2.0 + act)
            Xw = sp.diags(w.astype(np.float32)) @ X
        Xc = Xw.tocsc()
        norms2 = np.asarray(Xc.multiply(Xc).sum(axis=0)).ravel() + 1e-12
        left = norms2 ** self.alpha            # ||x_i||^{2a}
        right = norms2 ** (1.0 - self.alpha)   # ||x_j||^{2(1-a)}
        XT = Xc.T.tocsr()

        rows, cols, vals = [], [], []
        k = min(self.k, n_items - 1)
        for start in range(0, n_items, self.block_size):
            stop = min(start + self.block_size, n_items)
            # (n_items, block): <x_i, x_j>
            co = (XT @ Xc[:, start:stop]).toarray()
            sim = co / (left[:, None] * right[None, start:stop] + self.shrink)
            sim[np.arange(start, stop), np.arange(
                stop - start)] = 0.0  # no self-similarity
            # k neighbours per column j
            top = np.argpartition(-sim, kth=k - 1, axis=0)[:k]
            v = np.take_along_axis(sim, top, axis=0)
            rows.append(top.ravel())
            cols.append(np.tile(np.arange(start, stop), k))
            vals.append(v.ravel())
        S = sp.csr_matrix((np.concatenate(vals).astype(np.float32),
                           (np.concatenate(rows), np.concatenate(cols))), shape=(n_items, n_items))
        S.eliminate_zeros()
        self.S = S
        return self

    def score(self, user_ids: np.ndarray) -> np.ndarray:
        return (self.X[user_ids] @ self.S).toarray()
