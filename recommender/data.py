from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import requests

# This module keeps the data pathing and dataset parsing in one place.
ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


@dataclass
class DataBundle:
    ratings: pd.DataFrame
    items: pd.DataFrame
    user_to_index: Dict[int, int]
    item_to_index: Dict[int, int]
    index_to_item: Dict[int, int]
    train_csr: sp.csr_matrix
    test_csr: sp.csr_matrix


def _download_ml100k(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "ml-100k.zip"
    if not zip_path.exists():
        resp = requests.get(ML100K_URL, timeout=60)
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)
    return zip_path


def _extract_zip(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    return out_dir / "ml-100k"


def load_movielens_100k(data_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_root = Path(data_root)
    zip_path = _download_ml100k(data_root / "cache")
    extracted = _extract_zip(zip_path, data_root)

    ratings_path = extracted / "u.data"
    items_path = extracted / "u.item"

    ratings = pd.read_csv(
        ratings_path,
        sep="\t",
        names=["user_id", "item_id", "rating", "ts"],
        engine="python",
    )
    # MovieLens 100k item file has pipe-separated fields with ISO-8859-1 encoding.
    items = pd.read_csv(
        items_path,
        sep="|",
        header=None,
        encoding="latin-1",
        names=[
            "item_id",
            "title",
            "release_date",
            "video_release_date",
            "imdb_url",
            "unknown",
            "Action",
            "Adventure",
            "Animation",
            "Children",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Fantasy",
            "Film-Noir",
            "Horror",
            "Musical",
            "Mystery",
            "Romance",
            "Sci-Fi",
            "Thriller",
            "War",
            "Western",
        ],
    )

    return ratings, items


def build_sparse(ratings: pd.DataFrame) -> Tuple[sp.csr_matrix, Dict[int, int], Dict[int, int]]:
    # Use categorical codes to map external ids to consecutive indices for CSR.
    user_codes = ratings.user_id.astype("category").cat.codes
    item_codes = ratings.item_id.astype("category").cat.codes
    num_users = int(user_codes.max()) + 1
    num_items = int(item_codes.max()) + 1

    data = ratings.rating.astype(np.float32).to_numpy()
    mat = sp.coo_matrix((data, (user_codes, item_codes)), shape=(num_users, num_items)).tocsr()

    user_to_index = dict(zip(ratings.user_id, user_codes))
    item_to_index = dict(zip(ratings.item_id, item_codes))
    return mat, user_to_index, item_to_index


def temporal_train_test_split(ratings: pd.DataFrame, test_frac: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Global time split: keeps later interactions in test. Simple and robust.
    ratings_sorted = ratings.sort_values("ts")
    cutoff = int((1.0 - test_frac) * len(ratings_sorted))
    train = ratings_sorted.iloc[:cutoff]
    test = ratings_sorted.iloc[cutoff:]
    return train, test


def prepare_data_bundle(data_root: Path, alpha: float = 40.0) -> DataBundle:
    ratings, items = load_movielens_100k(data_root)
    train_df, test_df = temporal_train_test_split(ratings)

    train_csr, user_to_index, item_to_index = build_sparse(train_df)
    index_to_item = {idx: it for it, idx in item_to_index.items()}

    # Align test to the same shape; drop unknown ids if any (rare in this split).
    test_known = test_df[test_df.user_id.isin(user_to_index) & test_df.item_id.isin(item_to_index)].copy()
    if not test_known.empty:
        u = test_known.user_id.map(user_to_index).astype(int)
        v = test_known.item_id.map(item_to_index).astype(int)
        data = test_known.rating.astype(np.float32).to_numpy()
        test_csr = sp.coo_matrix((data, (u, v)), shape=train_csr.shape).tocsr()
    else:
        test_csr = sp.csr_matrix(train_csr.shape, dtype=np.float32)

    return DataBundle(
        ratings=ratings,
        items=items,
        user_to_index=user_to_index,
        item_to_index=item_to_index,
        index_to_item=index_to_item,
        train_csr=train_csr.astype(np.float32),
        test_csr=test_csr.astype(np.float32),
    )
