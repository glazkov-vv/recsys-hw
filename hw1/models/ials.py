from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _solve_side(M: sp.csr_matrix, Cm1: sp.csr_matrix, Y: np.ndarray, reg: float,
                max_chunk_bytes: int = 256 * 2**20) -> np.ndarray:
    """Solve all rows of one side.

    M   — CSR with target values y_ui on observed entries (rows = side being solved).
    Cm1 — CSR with the same sparsity pattern holding c_ui - 1.
    Y   — fixed factors of the other side, (n_other, f).
    """
    n, f = M.shape[0], Y.shape[1]
    YtY = Y.T @ Y + reg * np.eye(f, dtype=np.float64)
    X = np.zeros((n, f), dtype=np.float32)
    indptr, indices = M.indptr, M.indices
    y_vals, w_vals = M.data, Cm1.data
    max_nnz = max(1, max_chunk_bytes // (8 * f * f))

    row = 0
    while row < n:
        # take as many rows as fit into the memory budget (at least one)
        end = np.searchsorted(indptr, indptr[row] + max_nnz, side="right") - 1
        end = min(max(end, row + 1), n)
        lo, hi = indptr[row], indptr[end]
        Yi = Y[indices[lo:hi]].astype(np.float64)                   # (nnz, f)
        w = w_vals[lo:hi].astype(np.float64)                         # c - 1
        cy = (w + 1.0) * y_vals[lo:hi]                               # c * y
        counts = np.diff(indptr[row:end + 1])
        A = np.broadcast_to(YtY, (end - row, f, f)).copy()
        b = np.zeros((end - row, f))
        nz = counts > 0
        if hi > lo:
            outer = np.einsum("n,nf,ng->nfg", w, Yi, Yi, optimize=True)
            starts = (indptr[row:end] - lo)[nz]
            A[nz] += np.add.reduceat(outer, starts, axis=0)
            b[nz] = np.add.reduceat(cy[:, None] * Yi, starts, axis=0)
        X[row:end] = np.linalg.solve(A, b[..., None])[..., 0]
        row = end
    return X


def _solve_side_cg(M: sp.csr_matrix, Cm1: sp.csr_matrix, Y: np.ndarray, X0: np.ndarray,
                   reg: float, steps: int = 3) -> np.ndarray:
    """Conjugate-gradient update of all rows at once (same systems as `_solve_side`)."""
    f = Y.shape[1]
    YtY = (Y.T @ Y + reg * np.eye(f, dtype=Y.dtype)).astype(np.float32)
    coo_rows = np.repeat(np.arange(M.shape[0]), np.diff(M.indptr))
    # (nnz, f)
    Yi = Y[M.indices]
    w = Cm1.data.astype(np.float32)

    def matvec(P):
        s = np.einsum("nf,nf->n", P[coo_rows], Yi) * \
            w                 # (c-1) * y_i^T p_u
        corr = sp.csr_matrix((s, M.indices, M.indptr), shape=M.shape) @ Y
        return P @ YtY + corr

    b = sp.csr_matrix(
        ((w + 1.0) * M.data,
         M.indices,
         M.indptr),
        shape=M.shape) @ Y  # Y^T C_u y_u
    X = X0.astype(np.float32).copy()
    r = b - matvec(X)
    p = r.copy()
    rs = np.einsum("nf,nf->n", r, r)
    for _ in range(steps):
        Ap = matvec(p)
        denom = np.einsum("nf,nf->n", p, Ap)
        a = np.where(denom > 0, rs / np.maximum(denom, 1e-30), 0.0)[:, None]
        X += a * p
        r -= a * Ap
        rs_new = np.einsum("nf,nf->n", r, r)
        p = r + np.where(rs > 0, rs_new /
                         np.maximum(rs, 1e-30), 0.0)[:, None] * p
        rs = rs_new
    return X.astype(np.float32)


class ImplicitALS:
    def __init__(self, factors: int = 64, reg: float = 1.0, alpha: float = 10.0,
                 iterations: int = 15, seed: int = 42, init_std: float = 0.01,
                 solver: str = "cg", cg_steps: int = 3):
        self.factors = factors
        self.reg = reg
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed
        self.init_std = init_std
        self.solver = solver
        self.cg_steps = cg_steps

    def _targets(self, train: sp.csr_matrix):
        """Baseline: binary preference, confidence 1 + alpha on observed pairs."""
        P = train.astype(bool).astype(np.float32).tocsr()
        W = P.copy()
        W.data = np.full_like(P.data, self.alpha)
        return P, W

    def fit(self, train: sp.csr_matrix, P: sp.csr_matrix |
            None = None, W: sp.csr_matrix | None = None):
        """Fit on a users x items matrix. P (targets) and W (= c - 1) may be passed explicitly."""
        if P is None or W is None:
            P, W = self._targets(train)
        P = P.tocsr()
        P.sort_indices()
        W = W.tocsr()
        W.sort_indices()
        assert np.array_equal(
            P.indices, W.indices) and np.array_equal(
            P.indptr, W.indptr)
        PT, WT = P.T.tocsr(), W.T.tocsr()
        PT.sort_indices()
        WT.sort_indices()

        rng = np.random.default_rng(self.seed)
        n_users, n_items = P.shape
        self.user_factors = (
            rng.standard_normal(
                (n_users,
                 self.factors)) *
            self.init_std).astype(
            np.float32)
        self.item_factors = (
            rng.standard_normal(
                (n_items,
                 self.factors)) *
            self.init_std).astype(
            np.float32)
        for _ in range(self.iterations):
            if self.solver == "exact":
                self.user_factors = _solve_side(
                    P, W, self.item_factors, self.reg)
                self.item_factors = _solve_side(
                    PT, WT, self.user_factors, self.reg)
            else:
                self.user_factors = _solve_side_cg(P, W, self.item_factors, self.user_factors,
                                                   self.reg, self.cg_steps)
                self.item_factors = _solve_side_cg(PT, WT, self.user_factors, self.item_factors,
                                                   self.reg, self.cg_steps)
        return self

    def loss(self, P: sp.csr_matrix, W: sp.csr_matrix) -> float:
        """Full weighted objective (for monitoring convergence)."""
        Xu, Yi = self.user_factors.astype(
            np.float64), self.item_factors.astype(
            np.float64)
        coo_p, coo_w = P.tocoo(), W.tocoo()
        pred = np.einsum("nf,nf->n", Xu[coo_p.row], Yi[coo_p.col])
        # sum over all pairs of (0 - pred)^2 + correction on observed pairs
        full = np.sum((Xu.T @ Xu) * (Yi.T @ Yi))
        obs = np.sum((1 + coo_w.data) * (coo_p.data - pred) ** 2 - pred ** 2)
        return float(full + obs + self.reg *
                     (np.sum(Xu ** 2) + np.sum(Yi ** 2)))

    def score(self, user_ids: np.ndarray) -> np.ndarray:
        return self.user_factors[user_ids] @ self.item_factors.T
