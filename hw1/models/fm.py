from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class _FMNet(nn.Module):
    def __init__(self, n_features: int, factors: int, init_std: float):
        super().__init__()
        self.V = nn.Embedding(n_features, factors)
        self.w = nn.Embedding(n_features, 1)
        nn.init.normal_(self.V.weight, std=init_std)
        nn.init.zeros_(self.w.weight)


class FactorizationMachine:
    def __init__(self, n_users: int, n_hist: int, item_feat_idx: np.ndarray, item_feat_w: np.ndarray,
                 n_item_feats: int, item_first_day: np.ndarray | None = None,
                 age_edges_days: np.ndarray | None = None, use_hist: bool = True,
                 factors: int = 64, lr: float = 3e-3, reg: float = 1e-5, epochs: int = 10,
                 batch_size: int = 4096, n_neg: int = 4, init_std: float = 0.01, seed: int = 42,
                 verbose: bool = False):
        self.n_users, self.n_hist, self.use_hist = n_users, n_hist, use_hist
        self.factors, self.lr, self.reg, self.epochs = factors, lr, reg, epochs
        self.batch_size, self.n_neg, self.init_std, self.seed = batch_size, n_neg, init_std, seed
        self.verbose = verbose
        self.use_age = item_first_day is not None and age_edges_days is not None
        # +1 overflow bucket, +1 unknown
        n_age = (len(age_edges_days) + 2) if self.use_age else 0

        # feature index layout: [user ids | history | item features | age
        # buckets]
        self.off_hist = n_users
        self.off_item = self.off_hist + n_hist
        self.off_age = self.off_item + n_item_feats
        self.n_features = self.off_age + n_age

        self.item_idx = torch.as_tensor(
            item_feat_idx + self.off_item, dtype=torch.long)
        self.item_w = torch.as_tensor(item_feat_w, dtype=torch.float32)
        if self.use_age:
            self.first_day = torch.as_tensor(
                np.nan_to_num(
                    item_first_day,
                    nan=np.inf),
                dtype=torch.float32)
            self.age_edges = torch.as_tensor(
                age_edges_days, dtype=torch.float32)
            self.age_unknown = self.off_age + n_age - 1

    # ------------------------------------------------------------------ feature blocks
    def _age_index(self, items: torch.Tensor,
                   day: torch.Tensor) -> torch.Tensor:
        first = self.first_day[items]
        age = torch.clamp(day - first, min=0.0)
        idx = self.off_age + torch.bucketize(age, self.age_edges)
        return torch.where(torch.isinf(first), torch.full_like(
            idx, self.age_unknown), idx)

    def _item_block(self, net: _FMNet, items: torch.Tensor,
                    day: torch.Tensor | None):
        """Returns e_I (B, f), linear (B,), pairwise-within-item (B,)."""
        idx, w = self.item_idx[items], self.item_w[items]                 # (B, L)
        if self.use_age:
            age = self._age_index(items, day)[:, None]
            idx = torch.cat([idx, age], dim=1)
            w = torch.cat(
                [w, torch.ones_like(age, dtype=torch.float32)], dim=1)
        # (B, L, f)
        v = net.V(idx) * w[..., None]
        e = v.sum(1)
        lin = (net.w(idx)[..., 0] * w).sum(1)
        pair = 0.5 * (e.pow(2).sum(1) - v.pow(2).sum((1, 2)))
        return e, lin, pair, v

    def _user_block(self, net: _FMNet, users: torch.Tensor,
                    hist: torch.Tensor | None):
        e = net.V(users)
        if self.use_hist and hist is not None:
            e = e + hist @ net.V.weight[self.off_hist:self.off_item]
        return e

    # ------------------------------------------------------------------ training
    def fit(self, users: np.ndarray, items: np.ndarray, days: np.ndarray, hist: np.ndarray | None,
            n_items: int, callback=None):
        """users/items/days — positive training interactions; hist — (N, n_hist) user profile
        computed from interactions strictly *before* each one.

        callback(model, epoch) -> float, optional: called after every epoch (e.g. validation
        NDCG); the parameters of the best epoch are restored at the end (early stopping)."""
        g = torch.Generator().manual_seed(self.seed)
        torch.manual_seed(self.seed)
        self.net = _FMNet(self.n_features, self.factors, self.init_std)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        U = torch.tensor(np.asarray(users), dtype=torch.long)
        I = torch.tensor(np.asarray(items), dtype=torch.long)
        D = torch.tensor(np.asarray(days), dtype=torch.float32)
        H = torch.as_tensor(hist, dtype=torch.float32) if (
            self.use_hist and hist is not None) else None
        n = len(U)
        self.history_, self.val_history_ = [], []
        best_val, best_state = -np.inf, None
        for epoch in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            total = 0.0
            for start in range(0, n, self.batch_size):
                b = perm[start:start + self.batch_size]
                u, i, d = U[b], I[b], D[b]
                h = H[b] if H is not None else None
                B = len(b)
                neg = torch.randint(0, n_items, (B, self.n_neg), generator=g)
                eu = self._user_block(
                    self.net, u, h)                                   # (B, f)
                ep, lp, pp, vp = self._item_block(self.net, i, d)
                en, ln, pn, vn = self._item_block(
                    self.net, neg.reshape(-1), d.repeat_interleave(self.n_neg))
                # (B,)
                s_pos = lp + pp + (eu * ep).sum(1)
                s_neg = (ln + pn).view(B, -1) + \
                    torch.einsum("bf,bkf->bk", eu, en.view(B, self.n_neg, -1))
                loss = -nn.functional.logsigmoid(s_pos[:, None] - s_neg).mean()
                reg = (eu.pow(2).sum() + vp.pow(2).sum() +
                       vn.pow(2).sum() / self.n_neg) / B
                opt.zero_grad()
                (loss + self.reg * reg).backward()
                opt.step()
                total += loss.item() * B
            self.history_.append(total / n)
            msg = f"epoch {epoch + 1}: BPR loss {total / n:.4f}"
            if callback is not None:
                val = callback(self, epoch)
                self.val_history_.append(val)
                msg += f", val {val:.5f}"
                if val > best_val:
                    best_val, self.best_epoch_ = val, epoch + 1
                    best_state = {k: v.detach().clone()
                                  for k, v in self.net.state_dict().items()}
            if self.verbose:
                print(msg)
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def set_inference_context(self, user_hist: np.ndarray | None, day: float):
        """Precompute item and user representations at recommendation time `day`
        (end of the training period) given each user's full-train platform profile."""
        n_items = self.item_idx.shape[0]
        items = torch.arange(n_items)
        e, lin, pair, _ = self._item_block(
            self.net, items, torch.full(
                (n_items,), float(day)))
        self.E_items = e.numpy()
        self.b_items = (lin + pair).numpy()
        H = torch.as_tensor(
            user_hist,
            dtype=torch.float32) if user_hist is not None else None
        self.E_users = self._user_block(
            self.net, torch.arange(
                self.n_users), H).numpy()
        return self

    def score(self, user_ids: np.ndarray) -> np.ndarray:
        return self.E_users[user_ids] @ self.E_items.T + self.b_items[None, :]
