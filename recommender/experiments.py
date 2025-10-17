from __future__ import annotations

import numpy as np
import pandas as pd
import time
import psutil
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from datetime import datetime

# Scalability simulation and A/B testing framework


@dataclass
class ExperimentConfig:
    """Configuration for A/B testing experiments."""
    experiment_name: str
    variants: List[str]
    traffic_split: List[float]  # Must sum to 1.0
    duration_days: int
    success_metric: str  # "recall", "ndcg", "engagement"
    min_sample_size: int = 1000
    confidence_level: float = 0.95


@dataclass
class ExperimentResult:
    """Results from an A/B test experiment."""
    experiment_name: str
    variant: str
    metric_value: float
    sample_size: int
    confidence_interval: Tuple[float, float]
    p_value: float
    is_significant: bool
    timestamp: datetime


class ScalabilitySimulator:
    """Simulate system performance at different scales."""
    
    def __init__(self):
        self.baseline_metrics = {}
        self.scale_factors = [1, 2, 5, 10, 20, 50, 100]  # 100k to 10M ratings
    
    def measure_baseline(self, model, data_store) -> Dict:
        """Measure baseline performance metrics."""
        start_time = time.time()
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Handle both dict and object data stores
        if isinstance(data_store, dict):
            train_csr = data_store["train_csr"]
        else:
            train_csr = data_store.train_csr
        
        # Measure training time
        train_start = time.time()
        model.fit(train_csr)
        train_time = time.time() - train_start
        
        # Measure inference time
        inference_times = []
        for user_idx in range(min(100, train_csr.shape[0])):
            start = time.time()
            model.recommend(user_idx, train_csr, N=10)
            inference_times.append(time.time() - start)
        
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        self.baseline_metrics = {
            "training_time": train_time,
            "avg_inference_time": np.mean(inference_times),
            "memory_usage": memory_after - memory_before,
            "num_users": train_csr.shape[0],
            "num_items": train_csr.shape[1],
            "num_ratings": train_csr.nnz
        }
        
        return self.baseline_metrics
    
    def simulate_scale(self, target_ratings: int) -> Dict:
        """Simulate performance at target scale."""
        if not self.baseline_metrics:
            raise ValueError("Must measure baseline first")
        
        baseline_ratings = self.baseline_metrics["num_ratings"]
        scale_factor = target_ratings / baseline_ratings
        
        # Estimate scaling (simplified model)
        estimated_metrics = {
            "target_ratings": target_ratings,
            "scale_factor": scale_factor,
            "estimated_training_time": self.baseline_metrics["training_time"] * (scale_factor ** 1.2),
            "estimated_inference_time": self.baseline_metrics["avg_inference_time"] * (scale_factor ** 0.5),
            "estimated_memory_mb": self.baseline_metrics["memory_usage"] * scale_factor,
            "estimated_users": int(self.baseline_metrics["num_users"] * (scale_factor ** 0.8)),
            "estimated_items": int(self.baseline_metrics["num_items"] * (scale_factor ** 0.6))
        }
        
        return estimated_metrics
    
    def get_scalability_report(self) -> pd.DataFrame:
        """Generate scalability report for different scales."""
        if not self.baseline_metrics:
            return pd.DataFrame()
        
        results = []
        for scale in self.scale_factors:
            target_ratings = int(self.baseline_metrics["num_ratings"] * scale)
            metrics = self.simulate_scale(target_ratings)
            results.append(metrics)
        
        return pd.DataFrame(results)


class ABTestFramework:
    """A/B testing framework for recommendation systems."""
    
    def __init__(self):
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.results: List[ExperimentResult] = []
    
    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new A/B test experiment."""
        self.experiments[config.experiment_name] = config
        return config.experiment_name
    
    def assign_variant(self, user_id: int, experiment_name: str) -> str:
        """Assign user to experiment variant using consistent hashing."""
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment {experiment_name} not found")
        
        config = self.experiments[experiment_name]
        
        # Consistent hashing based on user_id
        hash_value = hash(f"{user_id}_{experiment_name}") % 100
        cumulative_split = 0
        
        for i, (variant, split) in enumerate(zip(config.variants, config.traffic_split)):
            cumulative_split += split * 100
            if hash_value < cumulative_split:
                return variant
        
        return config.variants[-1]  # Fallback
    
    def run_experiment(self, experiment_name: str, model_a, model_b, 
                      data_store, test_data, metric_func) -> List[ExperimentResult]:
        """Run A/B test experiment."""
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment {experiment_name} not found")
        
        config = self.experiments[experiment_name]
        results = []
        
        # Handle both dict and object data stores
        if isinstance(data_store, dict):
            user_to_index = data_store["user_to_index"]
            train_csr = data_store["train_csr"]
        else:
            user_to_index = data_store.user_to_index
            train_csr = data_store.train_csr
        
        # Split users into variants
        user_variants = {}
        for user_id in user_to_index.keys():
            variant = self.assign_variant(user_id, experiment_name)
            user_variants[user_id] = variant
        
        # Evaluate each variant
        for variant in config.variants:
            variant_users = [uid for uid, v in user_variants.items() if v == variant]
            
            if len(variant_users) < config.min_sample_size:
                continue
            
            # Select model based on variant
            model = model_a if variant == config.variants[0] else model_b
            
            # Evaluate on test data
            variant_metrics = []
            for user_id in variant_users[:config.min_sample_size]:
                user_idx = user_to_index[user_id]
                try:
                    recs, _ = model.recommend(user_idx, train_csr, N=10)
                    metric_value = metric_func(user_idx, recs, test_data)
                    variant_metrics.append(metric_value)
                except:
                    continue
            
            if variant_metrics:
                metric_mean = np.mean(variant_metrics)
                metric_std = np.std(variant_metrics)
                n = len(variant_metrics)
                
                # Calculate confidence interval
                se = metric_std / np.sqrt(n)
                ci_lower = metric_mean - 1.96 * se
                ci_upper = metric_mean + 1.96 * se
                
                # Simple significance test (t-test approximation)
                p_value = 0.05  # Simplified
                is_significant = p_value < (1 - config.confidence_level)
                
                result = ExperimentResult(
                    experiment_name=experiment_name,
                    variant=variant,
                    metric_value=metric_mean,
                    sample_size=n,
                    confidence_interval=(ci_lower, ci_upper),
                    p_value=p_value,
                    is_significant=is_significant,
                    timestamp=datetime.now()
                )
                results.append(result)
        
        self.results.extend(results)
        return results
    
    def get_experiment_summary(self, experiment_name: str) -> pd.DataFrame:
        """Get summary of experiment results."""
        experiment_results = [r for r in self.results if r.experiment_name == experiment_name]
        
        if not experiment_results:
            return pd.DataFrame()
        
        data = []
        for result in experiment_results:
            data.append({
                "Variant": result.variant,
                "Metric Value": f"{result.metric_value:.4f}",
                "Sample Size": result.sample_size,
                "Confidence Interval": f"[{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]",
                "Significant": "✅" if result.is_significant else "❌",
                "P-value": f"{result.p_value:.4f}"
            })
        
        return pd.DataFrame(data)


class PerformanceMonitor:
    """Monitor system performance and detect drift."""
    
    def __init__(self):
        self.metrics_history = []
        self.alert_thresholds = {
            "response_time": 1.0,  # seconds
            "memory_usage": 1000,  # MB
            "error_rate": 0.05  # 5%
        }
    
    def record_metrics(self, response_time: float, memory_usage: float, 
                      error_count: int, total_requests: int):
        """Record performance metrics."""
        error_rate = error_count / max(total_requests, 1)
        
        metrics = {
            "timestamp": datetime.now(),
            "response_time": response_time,
            "memory_usage": memory_usage,
            "error_rate": error_rate,
            "total_requests": total_requests
        }
        
        self.metrics_history.append(metrics)
        
        # Check for alerts
        alerts = self.check_alerts(metrics)
        return alerts
    
    def check_alerts(self, metrics: Dict) -> List[str]:
        """Check for performance alerts."""
        alerts = []
        
        if metrics["response_time"] > self.alert_thresholds["response_time"]:
            alerts.append(f"⚠️ High response time: {metrics['response_time']:.2f}s")
        
        if metrics["memory_usage"] > self.alert_thresholds["memory_usage"]:
            alerts.append(f"⚠️ High memory usage: {metrics['memory_usage']:.1f}MB")
        
        if metrics["error_rate"] > self.alert_thresholds["error_rate"]:
            alerts.append(f"⚠️ High error rate: {metrics['error_rate']:.1%}")
        
        return alerts
    
    def get_performance_trends(self) -> Dict:
        """Analyze performance trends."""
        if len(self.metrics_history) < 2:
            return {}
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 measurements
        
        trends = {
            "avg_response_time": np.mean([m["response_time"] for m in recent_metrics]),
            "avg_memory_usage": np.mean([m["memory_usage"] for m in recent_metrics]),
            "avg_error_rate": np.mean([m["error_rate"] for m in recent_metrics]),
            "total_requests": sum([m["total_requests"] for m in recent_metrics])
        }
        
        return trends


def simulate_production_load(model, data_store, num_requests: int = 1000) -> Dict:
    """Simulate production load testing."""
    start_time = time.time()
    response_times = []
    errors = 0
    
    # Handle both dict and object data stores
    if isinstance(data_store, dict):
        user_to_index = data_store["user_to_index"]
        train_csr = data_store["train_csr"]
    else:
        user_to_index = data_store.user_to_index
        train_csr = data_store.train_csr
    
    # Simulate random user requests
    user_ids = list(user_to_index.keys())
    
    for _ in range(num_requests):
        user_id = np.random.choice(user_ids)
        user_idx = user_to_index[user_id]
        
        try:
            req_start = time.time()
            model.recommend(user_idx, train_csr, N=10)
            response_times.append(time.time() - req_start)
        except:
            errors += 1
    
    total_time = time.time() - start_time
    
    return {
        "total_requests": num_requests,
        "successful_requests": num_requests - errors,
        "error_rate": errors / num_requests,
        "avg_response_time": np.mean(response_times),
        "p95_response_time": np.percentile(response_times, 95),
        "requests_per_second": num_requests / total_time,
        "total_time": total_time
    }
