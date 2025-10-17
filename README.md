# Local Movie Recommender (Streamlit, CPU-only)

A small, end-to-end recommender that runs locally on CPU and serves a Streamlit app. It downloads MovieLens-100k automatically, trains an implicit ALS model, and provides top-N recommendations with basic metrics.

## Quickstart
1) Create venv (optional)
```
python -m venv .venv
.venv\Scripts\activate
```
2) Install
```
pip install --upgrade pip
pip install -r requirements.txt
```
3) Run
```
streamlit run app.py
```

The first run downloads and caches the dataset and model artifacts.

## Layout
- `app.py` — Streamlit UI
- `recommender/data.py` — download, parse, sparse matrices
- `recommender/als.py` — lightweight implicit ALS (CPU)
- `recommender/metrics.py` — Recall@K, NDCG@K

## Notes
- CPU-only, small memory footprint (float32, sparse CSR).
- You can tune factors/iterations in the UI if your machine allows.
