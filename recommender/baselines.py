from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Dict, Tuple, List
from abc import ABC, abstractmethod

# Baseline models for comparison with ALS


class BaseRecommender(ABC):
    """Abstract base class for recommendation models."""
    
    @abstractmethod
    def fit(self, R: sp.csr_matrix) -> "BaseRecommender":
        """Train the model on the rating matrix."""
        pass
    
    @abstractmethod
    def recommend(self, user_index: int, R: sp.csr_matrix, N: int = 10, filter_seen: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Get top-N recommendations for a user."""
        pass


class PopularityRecommender(BaseRecommender):
    """Popularity-based baseline: recommend most popular items."""
    
    def __init__(self):
        self.item_popularity: np.ndarray | None = None
    
    def fit(self, R: sp.csr_matrix) -> "PopularityRecommender":
        """Calculate item popularity scores."""
        # Count ratings per item
        item_counts = np.array(R.sum(axis=0)).flatten()
        # Normalize by total possible ratings
        self.item_popularity = item_counts / R.shape[0]
        return self
    
    def recommend(self, user_index: int, R: sp.csr_matrix, N: int = 10, filter_seen: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Recommend most popular items."""
        if self.item_popularity is None:
            raise ValueError("Model not fitted")
        
        if filter_seen:
            start, end = R.indptr[user_index], R.indptr[user_index + 1]
            seen = set(R.indices[start:end])
        else:
            seen = set()
        
        # Get top items by popularity
        top_items = np.argsort(-self.item_popularity)
        top_items = [i for i in top_items if i not in seen][:N]
        scores = self.item_popularity[top_items]
        
        return np.array(top_items), scores


class RandomRecommender(BaseRecommender):
    """Random baseline: recommend random items."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.num_items: int | None = None
    
    def fit(self, R: sp.csr_matrix) -> "RandomRecommender":
        """Store number of items."""
        self.num_items = R.shape[1]
        return self
    
    def recommend(self, user_index: int, R: sp.csr_matrix, N: int = 10, filter_seen: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Recommend random items."""
        if self.num_items is None:
            raise ValueError("Model not fitted")
        
        rng = np.random.RandomState(self.random_state + user_index)
        
        if filter_seen:
            start, end = R.indptr[user_index], R.indptr[user_index + 1]
            seen = set(R.indices[start:end])
        else:
            seen = set()
        
        # Generate random items
        all_items = set(range(self.num_items)) - seen
        if len(all_items) < N:
            selected = list(all_items)
        else:
            selected = rng.choice(list(all_items), size=N, replace=False)
        
        scores = rng.random(len(selected))
        return np.array(selected), scores


class ContentBasedRecommender(BaseRecommender):
    """Content-based baseline using movie genres."""
    
    def __init__(self):
        self.genre_weights: np.ndarray | None = None
        self.item_features: np.ndarray | None = None
    
    def fit(self, R: sp.csr_matrix, items_df: pd.DataFrame) -> "ContentBasedRecommender":
        """Learn user preferences from genres."""
        # Get genre columns
        genre_cols = [col for col in items_df.columns 
                     if col not in ["item_id", "title", "release_date", "video_release_date", "imdb_url"]]
        
        # Create item-feature matrix (items x genres)
        self.item_features = items_df[genre_cols].values.astype(np.float32)
        
        # Learn user preferences by aggregating ratings
        num_users, num_items = R.shape
        num_genres = len(genre_cols)
        user_preferences = np.zeros((num_users, num_genres), dtype=np.float32)
        
        for u in range(num_users):
            start, end = R.indptr[u], R.indptr[u + 1]
            if start == end:
                continue
            
            user_items = R.indices[start:end]
            user_ratings = R.data[start:end]
            
            # Weight genre preferences by ratings
            for i, item_idx in enumerate(user_items):
                if item_idx < len(self.item_features):
                    user_preferences[u] += user_ratings[i] * self.item_features[item_idx]
            
            # Normalize
            if user_preferences[u].sum() > 0:
                user_preferences[u] /= user_preferences[u].sum()
        
        self.genre_weights = user_preferences
        return self
    
    def recommend(self, user_index: int, R: sp.csr_matrix, N: int = 10, filter_seen: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Recommend based on content similarity."""
        if self.genre_weights is None or self.item_features is None:
            raise ValueError("Model not fitted")
        
        if filter_seen:
            start, end = R.indptr[user_index], R.indptr[user_index + 1]
            seen = set(R.indices[start:end])
        else:
            seen = set()
        
        # Calculate content similarity
        user_prefs = self.genre_weights[user_index]
        scores = self.item_features @ user_prefs
        
        # Filter seen items and get top N
        valid_items = [i for i in range(len(scores)) if i not in seen]
        if not valid_items:
            return np.array([]), np.array([])
        
        valid_scores = scores[valid_items]
        top_indices = np.argsort(-valid_scores)[:N]
        
        selected_items = np.array(valid_items)[top_indices]
        selected_scores = valid_scores[top_indices]
        
        return selected_items, selected_scores


class ModelComparator:
    """Compare multiple recommendation models."""
    
    def __init__(self):
        self.models: Dict[str, BaseRecommender] = {}
        self.results: Dict[str, Dict] = {}
    
    def add_model(self, name: str, model: BaseRecommender):
        """Add a model to compare."""
        self.models[name] = model
    
    def fit_all(self, R: sp.csr_matrix, items_df: pd.DataFrame = None):
        """Train all models."""
        for name, model in self.models.items():
            if isinstance(model, ContentBasedRecommender) and items_df is not None:
                model.fit(R, items_df)
            else:
                model.fit(R)
    
    def evaluate_all(self, R: sp.csr_matrix, test_R: sp.csr_matrix, k: int = 10, 
                    sample_users: int = 100) -> Dict[str, Dict]:
        """Evaluate all models on test set."""
        from .metrics import recall_at_k, ndcg_at_k
        
        num_users = min(sample_users, R.shape[0])
        user_indices = np.random.choice(R.shape[0], size=num_users, replace=False)
        
        results = {}
        
        for name, model in self.models.items():
            # Generate recommendations for sample users
            predictions = np.zeros((num_users, k), dtype=int)
            
            for i, user_idx in enumerate(user_indices):
                try:
                    recs, _ = model.recommend(user_idx, R, N=k, filter_seen=True)
                    if len(recs) < k:
                        # Pad with -1 if not enough recommendations
                        padded = np.full(k, -1, dtype=int)
                        padded[:len(recs)] = recs
                        predictions[i] = padded
                    else:
                        predictions[i] = recs[:k]
                except:
                    predictions[i] = np.full(k, -1, dtype=int)
            
            # Evaluate on test set
            test_subset = test_R[user_indices]
            recall = recall_at_k(predictions, test_subset, k)
            ndcg = ndcg_at_k(predictions, test_subset, k)
            
            results[name] = {
                "recall_at_k": recall,
                "ndcg_at_k": ndcg,
                "model_type": type(model).__name__
            }
        
        self.results = results
        return results
    
    def get_comparison_table(self) -> pd.DataFrame:
        """Get comparison results as DataFrame."""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for name, metrics in self.results.items():
            data.append({
                "Model": name,
                "Recall@K": f"{metrics['recall_at_k']:.3f}",
                "NDCG@K": f"{metrics['ndcg_at_k']:.3f}",
                "Type": metrics["model_type"]
            })
        
        return pd.DataFrame(data).sort_values("Recall@K", ascending=False)
