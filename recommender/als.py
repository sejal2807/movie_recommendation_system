from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import scipy.sparse as sp

# This is my custom ALS implementation - I built it from scratch to understand
# how collaborative filtering really works under the hood. It's based on the
# classic paper by Hu, Koren, and Volinsky, but I kept it simple and fast
# for small datasets like MovieLens-100k.


@dataclass
class ALSConfig:
    factors: int = 64
    regularization: float = 0.01
    iterations: int = 15
    alpha: float = 40.0  # confidence scale: C = 1 + alpha * R


class ImplicitALS:
    def __init__(self, config: ALSConfig):
        self.config = config
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None

    def fit(self, R: sp.csr_matrix) -> "ImplicitALS":
        # Start with random embeddings - the magic happens in the iterations
        num_users, num_items = R.shape
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        X = rng.normal(scale=0.01, size=(num_users, self.config.factors)).astype(np.float32)
        Y = rng.normal(scale=0.01, size=(num_items, self.config.factors)).astype(np.float32)

        # Regularization helps prevent overfitting - learned this the hard way!
        regI = np.eye(self.config.factors, dtype=np.float32) * self.config.regularization
        alpha = np.float32(self.config.alpha)

        R_csr = R.tocsr().astype(np.float32)
        RT_csr = R_csr.T.tocsr()

        # The alternating part - fix one set of factors, optimize the other
        for _ in range(self.config.iterations):
            # Step 1: Fix item factors, learn user preferences
            YtY = Y.T @ Y + regI
            X = self._least_squares(R_csr, Y, YtY, alpha)
            # Step 2: Fix user factors, learn item characteristics  
            XtX = X.T @ X + regI
            Y = self._least_squares(RT_csr, X, XtX, alpha)

        self.user_factors = X
        self.item_factors = Y
        return self

    @staticmethod
    def _least_squares(CSR: sp.csr_matrix, F: np.ndarray, FtF: np.ndarray, alpha: np.float32) -> np.ndarray:
        # Solves (FtF + sum((c-1)*f f^T)) * x = sum(c * p * f) per row, where c = 1 + alpha * r, p = 1 if r>0
        num_rows = CSR.shape[0]
        factors = F.shape[1]
        X = np.zeros((num_rows, factors), dtype=np.float32)

        for i in range(num_rows):
            start, end = CSR.indptr[i], CSR.indptr[i + 1]
            cols = CSR.indices[start:end]
            vals = CSR.data[start:end]

            if len(cols) == 0:
                X[i] = 0.0
                continue

            # Confidence and preferences for non-zeros
            c = 1.0 + alpha * vals
            p = np.ones_like(vals, dtype=np.float32)

            Fi = F[cols]  # shape: (nnz, factors)
            CuI = np.diag(c - 1.0)
            A = FtF + Fi.T @ CuI @ Fi
            b = (Fi.T * c) @ p
            # Solve A x = b
            X[i] = np.linalg.solve(A, b)

        return X

    def recommend(self, user_index: int, R: sp.csr_matrix, N: int = 10, filter_seen: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        # The fun part - actually getting recommendations!
        assert self.user_factors is not None and self.item_factors is not None
        user_vec = self.user_factors[user_index]
        scores = self.item_factors @ user_vec  # Dot product gives us similarity scores

        # Don't recommend movies they've already seen (unless debugging)
        if filter_seen:
            start, end = R.indptr[user_index], R.indptr[user_index + 1]
            seen = set(R.indices[start:end])
        else:
            seen = set()

        # Get top candidates efficiently - argpartition is faster than full sort
        top_idx = np.argpartition(-scores, range(min(N + len(seen), len(scores))))[: N + len(seen)]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        # Filter out already seen items and return top N
        filtered = [i for i in top_idx if i not in seen]
        topN = np.array(filtered[:N], dtype=int)
        return topN, scores[topN]

    def similar_items(self, item_index: int, N: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        assert self.item_factors is not None
        v = self.item_factors[item_index]
        norms = np.linalg.norm(self.item_factors, axis=1) * (np.linalg.norm(v) + 1e-8)
        sims = (self.item_factors @ v) / (norms + 1e-8)
        top = np.argpartition(-sims, range(N + 1))[: N + 1]
        top = top[np.argsort(-sims[top])]
        # Exclude the item itself
        top = top[top != item_index][:N]
        return top, sims[top]
