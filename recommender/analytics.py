from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, List, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Business analytics module for user segmentation, insights, and business metrics


def load_user_demographics(data_root: str) -> pd.DataFrame:
    """Load user demographics from MovieLens-100k dataset."""
    import os
    from pathlib import Path
    
    data_path = Path(data_root) / "ml-100k" / "u.user"
    if not data_path.exists():
        return pd.DataFrame()
    
    users = pd.read_csv(
        data_path,
        sep="|",
        names=["user_id", "age", "gender", "occupation", "zip_code"],
        engine="python"
    )
    return users


def analyze_user_segments(ratings: pd.DataFrame, items: pd.DataFrame, users: pd.DataFrame) -> Dict:
    """Analyze user behavior patterns by demographics."""
    if users.empty:
        return {}
    
    # Merge data
    full_data = ratings.merge(users, on="user_id", how="left")
    full_data = full_data.merge(items[["item_id", "title"]], on="item_id", how="left")
    
    # Age segments
    full_data["age_group"] = pd.cut(full_data["age"], 
                                   bins=[0, 25, 35, 50, 100], 
                                   labels=["Young", "Adult", "Middle-aged", "Senior"])
    
    # Rating patterns by demographics
    segment_analysis = {}
    
    # Gender analysis
    gender_stats = full_data.groupby("gender").agg({
        "rating": ["mean", "std", "count"],
        "user_id": "nunique"
    }).round(3)
    segment_analysis["gender"] = gender_stats
    
    # Age group analysis
    age_stats = full_data.groupby("age_group").agg({
        "rating": ["mean", "std", "count"],
        "user_id": "nunique"
    }).round(3)
    segment_analysis["age_group"] = age_stats
    
    # Occupation analysis (top 10)
    occupation_stats = full_data.groupby("occupation").agg({
        "rating": ["mean", "std", "count"],
        "user_id": "nunique"
    }).round(3).sort_values(("rating", "count"), ascending=False).head(10)
    segment_analysis["occupation"] = occupation_stats
    
    return segment_analysis


def analyze_genre_preferences(ratings: pd.DataFrame, items: pd.DataFrame, users: pd.DataFrame) -> Dict:
    """Analyze genre preferences by user segments."""
    if users.empty:
        return {}
    
    # Get genre columns
    genre_cols = [col for col in items.columns if col not in ["item_id", "title", "release_date", "video_release_date", "imdb_url"]]
    
    # Merge data
    full_data = ratings.merge(users, on="user_id", how="left")
    full_data = full_data.merge(items, on="item_id", how="left")
    
    # Age groups
    full_data["age_group"] = pd.cut(full_data["age"], 
                                   bins=[0, 25, 35, 50, 100], 
                                   labels=["Young", "Adult", "Middle-aged", "Senior"])
    
    genre_insights = {}
    
    # Genre preferences by gender
    gender_genre = {}
    for gender in full_data["gender"].unique():
        if pd.isna(gender):
            continue
        gender_data = full_data[full_data["gender"] == gender]
        genre_prefs = {}
        for genre in genre_cols:
            if genre in gender_data.columns:
                # Weight by rating
                weighted_pref = (gender_data[genre] * gender_data["rating"]).sum() / gender_data[genre].sum()
                genre_prefs[genre] = weighted_pref
        gender_genre[gender] = dict(sorted(genre_prefs.items(), key=lambda x: x[1], reverse=True)[:5])
    
    genre_insights["gender_genre"] = gender_genre
    
    # Genre preferences by age
    age_genre = {}
    for age_group in full_data["age_group"].unique():
        if pd.isna(age_group):
            continue
        age_data = full_data[full_data["age_group"] == age_group]
        genre_prefs = {}
        for genre in genre_cols:
            if genre in age_data.columns:
                weighted_pref = (age_data[genre] * age_data["rating"]).sum() / age_data[genre].sum()
                genre_prefs[genre] = weighted_pref
        age_genre[age_group] = dict(sorted(genre_prefs.items(), key=lambda x: x[1], reverse=True)[:5])
    
    genre_insights["age_genre"] = age_genre
    
    return genre_insights


def calculate_business_metrics(ratings: pd.DataFrame, items: pd.DataFrame, model, user_to_index: Dict) -> Dict:
    """Calculate business impact metrics."""
    metrics = {}
    
    # User engagement metrics
    ratings_per_user = ratings.groupby("user_id").size()
    metrics["avg_ratings_per_user"] = ratings_per_user.mean()
    metrics["active_users"] = len(ratings_per_user[ratings_per_user >= 10])  # Users with 10+ ratings
    metrics["total_users"] = ratings["user_id"].nunique()
    metrics["engagement_rate"] = metrics["active_users"] / metrics["total_users"]
    
    # Rating distribution
    rating_dist = ratings["rating"].value_counts().sort_index()
    metrics["avg_rating"] = ratings["rating"].mean()
    metrics["rating_distribution"] = rating_dist.to_dict()
    
    # Popularity bias analysis
    item_popularity = ratings.groupby("item_id").size().sort_values(ascending=False)
    metrics["popularity_gini"] = calculate_gini_coefficient(item_popularity.values)
    metrics["top_10_percent_items"] = len(item_popularity.head(int(len(item_popularity) * 0.1)))
    
    # Cold start analysis
    user_rating_counts = ratings.groupby("user_id").size()
    cold_start_users = len(user_rating_counts[user_rating_counts < 5])
    metrics["cold_start_ratio"] = cold_start_users / metrics["total_users"]
    
    return metrics


def calculate_gini_coefficient(values: np.ndarray) -> float:
    """Calculate Gini coefficient for popularity distribution."""
    sorted_values = np.sort(values)
    n = len(sorted_values)
    cumsum = np.cumsum(sorted_values)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n


def analyze_temporal_patterns(ratings: pd.DataFrame) -> Dict:
    """Analyze temporal patterns in user behavior."""
    if "ts" not in ratings.columns:
        return {}
    
    # Convert timestamp to datetime
    ratings["datetime"] = pd.to_datetime(ratings["ts"], unit="s")
    ratings["hour"] = ratings["datetime"].dt.hour
    ratings["day_of_week"] = ratings["datetime"].dt.day_name()
    ratings["month"] = ratings["datetime"].dt.month
    
    temporal_insights = {}
    
    # Hourly patterns
    hourly_ratings = ratings.groupby("hour")["rating"].agg(["mean", "count"]).reset_index()
    temporal_insights["hourly"] = hourly_ratings
    
    # Day of week patterns
    daily_ratings = ratings.groupby("day_of_week")["rating"].agg(["mean", "count"]).reset_index()
    temporal_insights["daily"] = daily_ratings
    
    # Monthly patterns
    monthly_ratings = ratings.groupby("month")["rating"].agg(["mean", "count"]).reset_index()
    temporal_insights["monthly"] = monthly_ratings
    
    return temporal_insights


def generate_business_insights(segment_analysis: Dict, genre_insights: Dict, 
                              business_metrics: Dict, temporal_insights: Dict) -> List[str]:
    """Generate human-readable business insights."""
    insights = []
    
    # User engagement insights
    if business_metrics:
        engagement_rate = business_metrics.get("engagement_rate", 0)
        insights.append(f"📊 **User Engagement**: {engagement_rate:.1%} of users are highly active (10+ ratings)")
        
        avg_rating = business_metrics.get("avg_rating", 0)
        insights.append(f"⭐ **Average Rating**: {avg_rating:.2f}/5.0 - {'High satisfaction' if avg_rating > 3.5 else 'Room for improvement'}")
        
        cold_start = business_metrics.get("cold_start_ratio", 0)
        insights.append(f"🆕 **Cold Start Challenge**: {cold_start:.1%} of users have <5 ratings - need better onboarding")
    
    # Demographic insights
    if "gender" in segment_analysis:
        gender_data = segment_analysis["gender"]
        if not gender_data.empty:
            male_avg = gender_data.loc["M", ("rating", "mean")] if "M" in gender_data.index else 0
            female_avg = gender_data.loc["F", ("rating", "mean")] if "F" in gender_data.index else 0
            if male_avg > 0 and female_avg > 0:
                insights.append(f"👥 **Gender Preferences**: {'Men' if male_avg > female_avg else 'Women'} rate movies {abs(male_avg - female_avg):.2f} points higher on average")
    
    # Genre insights
    if "gender_genre" in genre_insights:
        gender_genre = genre_insights["gender_genre"]
        if "M" in gender_genre and "F" in gender_genre:
            male_top = list(gender_genre["M"].keys())[:3]
            female_top = list(gender_genre["F"].keys())[:3]
            insights.append(f"🎬 **Genre Preferences**: Men prefer {', '.join(male_top)}, Women prefer {', '.join(female_top)}")
    
    # Temporal insights
    if "hourly" in temporal_insights:
        hourly = temporal_insights["hourly"]
        peak_hour = hourly.loc[hourly["count"].idxmax(), "hour"]
        insights.append(f"⏰ **Peak Activity**: Most ratings happen at {peak_hour}:00 - prime recommendation time")
    
    return insights
