from __future__ import annotations

import os
import random

import numpy as np
import scipy.sparse as sp


def set_seed(seed: int = 42) -> None:
    """Fix all random generators used in the project."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def build_csr(users: np.ndarray, items: np.ndarray, values: np.ndarray | None,
              n_users: int, n_items: int, dtype=np.float32) -> sp.csr_matrix:
    """users x items CSR; duplicate (u, i) pairs are summed."""
    if values is None:
        values = np.ones(len(users), dtype=dtype)
    m = sp.csr_matrix(
        (values.astype(dtype), (users, items)), shape=(
            n_users, n_items))
    m.sum_duplicates()
    m.sort_indices()
    return m


def topk_excluding(scores: np.ndarray, exclude: sp.csr_matrix | None, k: int,
                   allowed_mask: np.ndarray | None = None) -> np.ndarray:
    """Top-k item indices per row of `scores`, sorted by score.

    exclude       — CSR (rows aligned with `scores`) of items that must not be recommended
                    (already seen in train).
    allowed_mask  — boolean (n_items,), False for items outside the candidate set.
    """
    scores = np.array(scores, dtype=np.float32, copy=True)
    if allowed_mask is not None:
        scores[:, ~allowed_mask] = -np.inf
    if exclude is not None:
        rows = np.repeat(np.arange(exclude.shape[0]), np.diff(exclude.indptr))
        scores[rows, exclude.indices] = -np.inf
    part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    part_scores = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-part_scores, axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)


def recommend(model, user_ids: np.ndarray, seen: sp.csr_matrix, k: int = 10,
              allowed_mask: np.ndarray | None = None, batch_size: int = 2048) -> np.ndarray:
    """Top-k recommendations for `user_ids` (indices), excluding items in `seen` (users x items)."""
    out = np.empty((len(user_ids), k), dtype=np.int64)
    for start in range(0, len(user_ids), batch_size):
        batch = user_ids[start:start + batch_size]
        scores = model.score(batch)
        out[start:start + len(batch)] = topk_excluding(scores,
                                                       seen[batch], k, allowed_mask)
    return out


def ndcg_at_k(recs: np.ndarray,
              relevant: list[set], k: int = 10) -> np.ndarray:
    """Binary-relevance NDCG@k per user."""
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    res = np.zeros(len(recs))
    for u, (row, rel) in enumerate(zip(recs, relevant)):
        hits = np.fromiter(
            (it in rel for it in row[:k]), dtype=bool, count=min(k, len(row)))
        dcg = discounts[:len(hits)][hits].sum()
        idcg = discounts[:min(len(rel), k)].sum()
        res[u] = dcg / idcg if idcg > 0 else 0.0
    return res


def recall_at_k(recs: np.ndarray,
                relevant: list[set], k: int = 10) -> np.ndarray:
    """|rec@k ∩ rel| / min(|rel|, k) per user (normalised recall, max = 1)."""
    return np.array([len(set(row[:k]) & rel) / min(len(rel), k)
                    for row, rel in zip(recs, relevant)])


def hitrate_at_k(recs: np.ndarray,
                 relevant: list[set], k: int = 10) -> np.ndarray:
    return np.array([float(len(set(row[:k]) & rel) > 0)
                    for row, rel in zip(recs, relevant)])


def map_at_k(recs: np.ndarray, relevant: list[set], k: int = 10) -> np.ndarray:
    res = np.zeros(len(recs))
    for u, (row, rel) in enumerate(zip(recs, relevant)):
        hits, s = 0, 0.0
        for pos, it in enumerate(row[:k], start=1):
            if it in rel:
                hits += 1
                s += hits / pos
        res[u] = s / min(len(rel), k)
    return res


def evaluate(recs: np.ndarray, relevant: list[set], n_items: int, k: int = 10,
             item_popularity: np.ndarray | None = None) -> dict:
    """Accuracy metrics (mean over users) plus catalogue coverage and novelty."""
    res = {
        f"NDCG@{k}": ndcg_at_k(recs, relevant, k).mean(),
        f"Recall@{k}": recall_at_k(recs, relevant, k).mean(),
        f"HitRate@{k}": hitrate_at_k(recs, relevant, k).mean(),
        f"MAP@{k}": map_at_k(recs, relevant, k).mean(),
        f"Coverage@{k}": len(np.unique(recs[:, :k])) / n_items,
    }
    if item_popularity is not None:
        p = item_popularity / item_popularity.sum()
        res[f"Novelty@{k}"] = float(np.mean(-np.log2(p[recs[:, :k]] + 1e-12)))
    return res


def paired_bootstrap(a: np.ndarray, b: np.ndarray,
                     n_boot: int = 1000, seed: int = 42):
    """95% CI of mean(a - b) via resampling the same users for both methods."""
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


class PopularRecommender:
    """Non-personalised baseline: items ranked by train popularity."""

    def fit(self, train: sp.csr_matrix):
        self.pop = np.asarray(
            train.astype(bool).sum(
                axis=0)).ravel().astype(
            np.float32)
        return self

    def score(self, user_ids: np.ndarray) -> np.ndarray:
        return np.tile(self.pop, (len(user_ids), 1))
