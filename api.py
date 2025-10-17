from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from recommender.data import prepare_data_bundle
from recommender.als import ALSConfig, ImplicitALS
from recommender.baselines import PopularityRecommender, RandomRecommender, ContentBasedRecommender

# Initialize FastAPI app
app = FastAPI(
    title="Movie Recommendation API",
    description="REST API for movie recommendations using collaborative filtering",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and data
model = None
data_store = None

# Pydantic models for API
class RecommendationRequest(BaseModel):
    user_id: int
    num_recommendations: int = 10
    model_type: str = "als"  # als, popularity, random, content

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[Dict[str, float]]
    model_type: str
    num_recommendations: int

class SimilarItemsRequest(BaseModel):
    item_id: int
    num_similar: int = 10

class SimilarItemsResponse(BaseModel):
    item_id: int
    similar_items: List[Dict[str, float]]
    num_similar: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    data_loaded: bool

@app.on_event("startup")
async def startup_event():
    """Initialize model and data on startup."""
    global model, data_store
    
    try:
        # Load data and train model
        data_store = prepare_data_bundle("data")
        config = ALSConfig(factors=48, regularization=0.02, iterations=12, alpha=40.0)
        model = ImplicitALS(config).fit(data_store.train_csr)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None
        data_store = None

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        data_loaded=data_store is not None
    )

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get movie recommendations for a user."""
    if model is None or data_store is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate user ID
    if request.user_id not in data_store.user_to_index:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_idx = data_store.user_to_index[request.user_id]
    
    # Get recommendations based on model type
    if request.model_type == "als":
        recs, scores = model.recommend(
            user_idx, 
            data_store.train_csr, 
            N=request.num_recommendations, 
            filter_seen=True
        )
    elif request.model_type == "popularity":
        pop_model = PopularityRecommender().fit(data_store.train_csr)
        recs, scores = pop_model.recommend(
            user_idx, 
            data_store.train_csr, 
            N=request.num_recommendations, 
            filter_seen=True
        )
    elif request.model_type == "random":
        rand_model = RandomRecommender().fit(data_store.train_csr)
        recs, scores = rand_model.recommend(
            user_idx, 
            data_store.train_csr, 
            N=request.num_recommendations, 
            filter_seen=True
        )
    elif request.model_type == "content":
        content_model = ContentBasedRecommender().fit(data_store.train_csr, data_store.items)
        recs, scores = content_model.recommend(
            user_idx, 
            data_store.train_csr, 
            N=request.num_recommendations, 
            filter_seen=True
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid model type")
    
    # Convert to item IDs and get movie titles
    recommendations = []
    for rec_idx, score in zip(recs, scores):
        item_id = data_store.index_to_item[rec_idx]
        movie_info = data_store.items[data_store.items["item_id"] == item_id]
        if not movie_info.empty:
            title = movie_info.iloc[0]["title"]
            recommendations.append({"item_id": int(item_id), "title": title, "score": float(score)})
    
    return RecommendationResponse(
        user_id=request.user_id,
        recommendations=recommendations,
        model_type=request.model_type,
        num_recommendations=len(recommendations)
    )

@app.post("/similar", response_model=SimilarItemsResponse)
async def get_similar_items(request: SimilarItemsRequest):
    """Get similar movies to a given movie."""
    if model is None or data_store is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate item ID
    if request.item_id not in data_store.item_to_index:
        raise HTTPException(status_code=404, detail="Movie not found")
    
    item_idx = data_store.item_to_index[request.item_id]
    similar_idx, scores = model.similar_items(item_idx, N=request.num_similar)
    
    # Convert to item IDs and get movie titles
    similar_items = []
    for sim_idx, score in zip(similar_idx, scores):
        item_id = data_store.index_to_item[sim_idx]
        movie_info = data_store.items[data_store.items["item_id"] == item_id]
        if not movie_info.empty:
            title = movie_info.iloc[0]["title"]
            similar_items.append({"item_id": int(item_id), "title": title, "similarity": float(score)})
    
    return SimilarItemsResponse(
        item_id=request.item_id,
        similar_items=similar_items,
        num_similar=len(similar_items)
    )

@app.get("/users/{user_id}/ratings")
async def get_user_ratings(user_id: int):
    """Get rating history for a user."""
    if data_store is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    if user_id not in data_store.user_to_index:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_idx = data_store.user_to_index[user_id]
    start, end = data_store.train_csr.indptr[user_idx], data_store.train_csr.indptr[user_idx + 1]
    
    ratings = []
    for i in range(start, end):
        item_idx = data_store.train_csr.indices[i]
        rating = data_store.train_csr.data[i]
        item_id = data_store.index_to_item[item_idx]
        
        movie_info = data_store.items[data_store.items["item_id"] == item_id]
        if not movie_info.empty:
            title = movie_info.iloc[0]["title"]
            ratings.append({"item_id": int(item_id), "title": title, "rating": float(rating)})
    
    return {"user_id": user_id, "ratings": ratings}

@app.get("/stats")
async def get_dataset_stats():
    """Get dataset statistics."""
    if data_store is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    return {
        "num_users": len(data_store.user_to_index),
        "num_items": len(data_store.item_to_index),
        "num_ratings": len(data_store.ratings),
        "avg_rating": float(data_store.ratings["rating"].mean()),
        "sparsity": 1 - (len(data_store.ratings) / (len(data_store.user_to_index) * len(data_store.item_to_index)))
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
