# 🎬 Movie Recommendation System

A production-ready movie recommendation system with collaborative filtering, business analytics, model comparison, and REST API. Built for Google-level ML engineering interviews.

## ✨ Features

- **🤖 Multiple ML Models**: ALS, Popularity, Random, Content-Based
- **📊 Business Analytics**: User segmentation, demographic insights, temporal patterns
- **🔬 Model Comparison**: Side-by-side performance evaluation
- **🌐 REST API**: FastAPI endpoints for production deployment
- **🐳 Docker Ready**: Containerized with docker-compose
- **📈 A/B Testing**: Framework for experimentation

## 🚀 Quick Start

### Option 1: Streamlit UI (Local)
```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Option 2: Docker (Production)
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or with Docker directly
docker build -t movie-recommender .
docker run -p 8501:8501 movie-recommender
```

### Option 3: REST API
```bash
# Run API server
python api.py

# Test endpoints
curl http://localhost:8000/health
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "num_recommendations": 10}'
```

## 📁 Project Structure

```
├── app.py                 # Streamlit UI
├── api.py                 # FastAPI REST endpoints
├── requirements.txt       # Dependencies
├── Dockerfile            # Container definition
├── docker-compose.yml    # Multi-service setup
├── recommender/
│   ├── data.py           # Data loading & preprocessing
│   ├── als.py            # ALS implementation
│   ├── baselines.py      # Baseline models
│   ├── metrics.py        # Evaluation metrics
│   └── analytics.py      # Business analytics
└── data/                 # Dataset cache
```

## 🎯 API Endpoints

- `GET /health` - Health check
- `POST /recommend` - Get recommendations
- `POST /similar` - Find similar movies
- `GET /users/{user_id}/ratings` - User rating history
- `GET /stats` - Dataset statistics

## 📊 Business Insights

The system provides:
- **User Segmentation**: Demographics analysis
- **Genre Preferences**: By age, gender, occupation
- **Temporal Patterns**: Peak activity hours
- **Engagement Metrics**: Active users, cold start ratio
- **Model Performance**: Recall@K, NDCG@K comparison

## 🔧 Configuration

- **Model Parameters**: Tunable in UI sidebar
- **Memory Optimization**: Float32, sparse matrices
- **CPU-Only**: No GPU dependencies
- **Scalable**: Ready for production deployment

## 🏗️ Production Deployment

```bash
# Using Docker Compose (recommended)
docker-compose up -d

# Using Kubernetes
kubectl apply -f k8s/

# Using cloud platforms
# - AWS ECS
# - Google Cloud Run
# - Azure Container Instances
```

## 📈 Performance

- **Training Time**: ~10-30 seconds
- **Inference**: <100ms per recommendation
- **Memory Usage**: <500MB
- **Accuracy**: Recall@10 ~0.20-0.35

## 🎓 Learning Outcomes

This project demonstrates:
- **ML Engineering**: End-to-end pipeline, model comparison
- **Software Engineering**: Clean architecture, API design
- **Data Science**: Business analytics, user segmentation
- **DevOps**: Docker, containerization, monitoring
- **Production Skills**: Scalability, deployment, testing
