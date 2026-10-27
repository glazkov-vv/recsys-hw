from __future__ import annotations

import gzip
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023"
FILES = {
    "ratings": ("Video_Games.csv.gz", f"{BASE_URL}/benchmark/5core/rating_only/Video_Games.csv.gz"),
    "meta": ("meta_Video_Games.jsonl.gz", f"{BASE_URL}/raw/meta_categories/meta_Video_Games.jsonl.gz"),
}
DAY_MS = 86_400_000


def download(data_dir: str | Path = "data") -> dict[str, Path]:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, (name, url) in FILES.items():
        path = data_dir / name
        if not path.exists():
            print(f"Downloading {url} ...", flush=True)
            tmp = path.with_suffix(path.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(path)
        paths[key] = path
    return paths


def load_ratings(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"user_id": str, "parent_asin": str, "rating": np.float32,
                                  "timestamp": np.int64})
    return df.rename(columns={"parent_asin": "item_id"})


def _parse_date(s):
    if not s:
        return pd.NaT
    return pd.to_datetime(s, errors="coerce", format="mixed")


def load_meta(path: str | Path, keep_items: set | None = None) -> pd.DataFrame:
    """Parse metadata into flat columns. Rating aggregates (average_rating, rating_number)
    and "Best Sellers Rank" are crawled in 2023, i.e. after our test period -> kept only for
    EDA, never used as model features."""
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if keep_items is not None and d["parent_asin"] not in keep_items:
                continue
            cats = d.get("categories") or []
            det = d.get("details") or {}
            price = d.get("price")
            try:
                price = float(price) if price is not None else np.nan
            except (TypeError, ValueError):
                price = np.nan
            rows.append({
                "item_id": d["parent_asin"],
                "title": d.get("title"),
                "main_category": d.get("main_category"),
                "platform": cats[1] if len(cats) > 1 else None,
                "product_type": cats[2] if len(cats) > 2 else None,
                "subtype": cats[3] if len(cats) > 3 else None,
                "store": d.get("store") or None,
                "price": price,
                "rated": det.get("Rated"),
                "date_first_available": det.get("Date First Available") or det.get("Release date"),
                "n_description": len(" ".join(d.get("description") or [])),
                "n_features": len(d.get("features") or []),
                "average_rating": d.get("average_rating"),
                "rating_number": d.get("rating_number"),
            })
    meta = pd.DataFrame(rows)
    meta["date_first_available"] = pd.to_datetime(meta["date_first_available"], errors="coerce",
                                                  format="mixed")
    return meta


def kcore(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Iteratively drop users and items with fewer than k interactions."""
    while True:
        uc = df["user_id"].map(df["user_id"].value_counts())
        ic = df["item_id"].map(df["item_id"].value_counts())
        keep = (uc >= k) & (ic >= k)
        if keep.all():
            return df
        df = df[keep]


def subsample(df: pd.DataFrame, n_items: int = 10_000,
              k: int = 5) -> pd.DataFrame:
    """Deterministic subsample: the n_items most-reviewed items (ties broken by item id),
    then re-apply the k-core filter."""
    cnt = df.groupby("item_id").size().reset_index(name="n")
    cnt = cnt.sort_values(["n", "item_id"], ascending=[False, True])
    top = set(cnt["item_id"].iloc[:n_items])
    return kcore(df[df["item_id"].isin(top)], k).reset_index(drop=True)


@dataclass
class Split:
    """One train / holdout pair. Indices are global (same user/item encoding everywhere)."""
    train: pd.DataFrame          # interactions used for fitting
    holdout: pd.DataFrame        # interactions in the evaluation window
    train_csr: sp.csr_matrix     # users x items, binary
    eval_users: np.ndarray       # users with >= 1 train interaction and >= 1 relevant item
    relevant: list               # list[set[item]] aligned with eval_users
    candidate_mask: np.ndarray   # items that appear in train (recommendable)
    t_rec_ms: int                # end of the training period = recommendation time


def make_split(df: pd.DataFrame, t_start_ms: int, t_end_ms: int, n_users: int, n_items: int,
               min_rating: float = 4.0) -> Split:
    """Global temporal split: train = [.., t_start), holdout = [t_start, t_end).

    Relevance (identical for every model): item reviewed in the holdout window with
    rating >= min_rating, the item is recommendable (seen in train by someone) and is
    not already in the user's train history. Users without train history are cold and
    are not evaluated (they would get the popularity fallback)."""
    train = df[df["ts"] < t_start_ms]
    hold = df[(df["ts"] >= t_start_ms) & (df["ts"] < t_end_ms)]
    train_csr = sp.csr_matrix((np.ones(len(train), dtype=np.float32), (train["u"], train["i"])),
                              shape=(n_users, n_items))
    train_csr.sum_duplicates()
    train_csr.data[:] = 1.0
    cand = np.zeros(n_items, dtype=bool)
    cand[train["i"].unique()] = True
    warm_users = np.zeros(n_users, dtype=bool)
    warm_users[train["u"].unique()] = True

    rel = hold[(hold["rating"] >= min_rating) &
               cand[hold["i"].to_numpy()] & warm_users[hold["u"].to_numpy()]]
    seen = set(zip(train["u"].to_numpy(), train["i"].to_numpy()))
    rel = rel[[(u, i) not in seen for u, i in zip(
        rel["u"].to_numpy(), rel["i"].to_numpy())]]
    grouped = rel.groupby("u")["i"].apply(set)
    return Split(train=train, holdout=hold, train_csr=train_csr,
                 eval_users=grouped.index.to_numpy(), relevant=list(grouped.values),
                 candidate_mask=cand, t_rec_ms=int(t_start_ms))


# ----------------------------------------------------------------------------- FM features

AGE_EDGES_DAYS = np.array(
    [30, 90, 180, 365, 730, 1460, 2920], dtype=np.float32)


def item_feature_table(meta: pd.DataFrame, item_index: pd.Index, min_store_items: int = 10,
                       n_price_bins: int = 5):
    """Static item features for the FM, encoded as padded (index, weight) arrays.

    Fields: item id, platform (categories[1]), product type (categories[2]),
    main_category, store (stores with < min_store_items items -> "other"), price quantile
    bucket (+ "missing"), ESRB rating. Returns idx, w, n_features, vocab (list of names),
    first_day (days since epoch of "Date First Available", nan if unknown)."""
    m = meta.set_index("item_id").reindex(item_index)
    store_counts = m["store"].value_counts()
    big_stores = set(store_counts[store_counts >= min_store_items].index)
    price_q = m["price"].quantile(np.linspace(
        0, 1, n_price_bins + 1)[1:-1]).to_numpy()
    price_bin = np.where(m["price"].isna(), "missing",
                         "q" + pd.Series(np.searchsorted(price_q, m["price"].fillna(0))).astype(str).to_numpy())
    fields = {
        "platform": m["platform"].fillna("unknown").to_numpy(),
        "ptype": m["product_type"].fillna("unknown").to_numpy(),
        "maincat": m["main_category"].fillna("unknown").to_numpy(),
        "store": np.where(m["store"].isin(big_stores), m["store"], "other").astype(str),
        "price": price_bin,
        "rated": m["rated"].fillna("unknown").to_numpy(),
    }
    n_items = len(item_index)
    vocab = [f"item={x}" for x in item_index]
    cols = [np.arange(n_items)]
    for name, values in fields.items():
        codes, uniques = pd.factorize(pd.Series(values).astype(str), sort=True)
        cols.append(codes + len(vocab))
        vocab += [f"{name}={u}" for u in uniques]
    idx = np.stack(cols, axis=1).astype(np.int64)
    w = np.ones_like(idx, dtype=np.float32)
    first_day = (m["date_first_available"].astype(
        "datetime64[ms]").astype("int64") / DAY_MS).to_numpy()
    # missing or implausible dates (e.g. 1899-12-30 placeholders) -> unknown
    bad = m["date_first_available"].isna().to_numpy() | (
        m["date_first_available"].dt.year < 1990).fillna(True).to_numpy()
    first_day = np.where(bad, np.nan, first_day)
    return idx, w, len(vocab), vocab, first_day


def platform_history(train: pd.DataFrame,
                     item_platform: np.ndarray, n_platforms: int, n_users: int):
    """User platform profile.

    Returns (per-interaction profile built from the user's interactions strictly before
    that one — used for training, so the target item never leaks into its own context;
    full-train profile per user — used at recommendation time). Rows are normalised
    shares; a user without history has an all-zero profile."""
    order = np.lexsort((train["ts"].to_numpy(), train["u"].to_numpy()))
    t = train.iloc[order]
    u = t["u"].to_numpy()
    onehot = np.zeros((len(t), n_platforms), dtype=np.float32)
    onehot[np.arange(len(t)), item_platform[t["i"].to_numpy()]] = 1.0
    cum = np.cumsum(onehot, axis=0)
    starts = np.r_[0, np.flatnonzero(np.diff(u)) + 1]
    first_of_user = np.repeat(starts, np.diff(np.r_[starts, len(u)]))
    before = cum - onehot - \
        np.where(first_of_user[:, None] > 0, cum[first_of_user - 1], 0.0)
    before /= np.maximum(before.sum(1, keepdims=True), 1.0)
    per_inter = np.empty_like(before)
    per_inter[order] = before
    full = np.zeros((n_users, n_platforms), dtype=np.float32)
    np.add.at(full, u, onehot)
    full /= np.maximum(full.sum(1, keepdims=True), 1.0)
    return per_inter, full
