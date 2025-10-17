from __future__ import annotations

import numpy as np
import scipy.sparse as sp

# Simple top-K metrics for implicit feedback evaluation on a held-out set.


def recall_at_k(pred_indices: np.ndarray, test_csr: sp.csr_matrix, k: int) -> float:
    # pred_indices: (num_users, k) item indices
    num_users = test_csr.shape[0]
    hits = 0
    total = 0
    for u in range(num_users):
        start, end = test_csr.indptr[u], test_csr.indptr[u + 1]
        if start == end:
            continue
        true_items = set(test_csr.indices[start:end])
        recs = pred_indices[u, :k]
        hits += sum(1 for i in recs if i in true_items)
        total += len(true_items)
    if total == 0:
        return 0.0
    return hits / total


def ndcg_at_k(pred_indices: np.ndarray, test_csr: sp.csr_matrix, k: int) -> float:
    num_users = test_csr.shape[0]
    ndcgs = []
    for u in range(num_users):
        start, end = test_csr.indptr[u], test_csr.indptr[u + 1]
        if start == end:
            continue
        true_items = set(test_csr.indices[start:end])
        recs = pred_indices[u, :k]
        dcg = 0.0
        for rank, i in enumerate(recs, start=1):
            if i in true_items:
                dcg += 1.0 / np.log2(rank + 1)
        # Ideal DCG if all true items ranked first
        ideal_hits = min(k, len(true_items))
        idcg = sum(1.0 / np.log2(r + 1) for r in range(1, ideal_hits + 1))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    if not ndcgs:
        return 0.0
    return float(np.mean(ndcgs))
