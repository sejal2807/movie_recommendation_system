import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from recommender.data import prepare_data_bundle
from recommender.als import ALSConfig, ImplicitALS
from recommender.metrics import recall_at_k, ndcg_at_k

st.set_page_config(page_title="Local Movie Recommender", page_icon="🎬", layout="wide")

DATA_DIR = Path("data")

# Keep threads conservative for laptops.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


@st.cache_data(show_spinner=True)
def load_data() -> dict:
    bundle = prepare_data_bundle(DATA_DIR)
    return {
        "ratings": bundle.ratings,
        "items": bundle.items,
        "user_to_index": bundle.user_to_index,
        "item_to_index": bundle.item_to_index,
        "index_to_item": bundle.index_to_item,
        "train_csr": bundle.train_csr,
        "test_csr": bundle.test_csr,
    }


@st.cache_resource(show_spinner=True)
def train_model(_train_csr) -> ImplicitALS:
    cfg = ALSConfig(factors=48, regularization=0.02, iterations=12, alpha=40.0)
    model = ImplicitALS(cfg).fit(_train_csr)
    return model


def sidebar_controls(items_df: pd.DataFrame, user_to_index: dict) -> dict:
    st.sidebar.header("Controls")
    st.sidebar.caption("Lean CPU recommender. Defaults are safe for laptops.")

    user_ids = sorted(user_to_index.keys())
    selected_user = st.sidebar.number_input("User ID", min_value=int(min(user_ids)), max_value=int(max(user_ids)), value=int(user_ids[0]))

    topn = st.sidebar.slider("Top-N 🔢", 5, 30, 10)

    st.sidebar.subheader("Model knobs")
    factors = st.sidebar.select_slider("Factors", options=[16, 32, 48, 64], value=48)
    iters = st.sidebar.select_slider("Iterations", options=[8, 10, 12, 15], value=12)

    return {
        "user_id": int(selected_user),
        "topn": int(topn),
        "factors": int(factors),
        "iterations": int(iters),
    }


st.title("🎬 Local Movie Recommender (ALS, CPU)")
st.caption("Downloads ML-100k on first run, trains a small ALS model, and serves recommendations.")

store = load_data()
model = train_model(store["train_csr"])

controls = sidebar_controls(store["items"], store["user_to_index"])  # items not used in controls yet

user_idx = store["user_to_index"].get(controls["user_id"])  # map to internal index


TAB_RECS, TAB_USER, TAB_ITEM, TAB_METRICS, TAB_ABOUT = st.tabs(["Recommendations", "Explore User", "Item Details", "Metrics", "About"]) 

with TAB_RECS:
    st.subheader("Recommendations")
    if user_idx is None:
        st.warning("Unknown user id. Pick one from the valid range in the sidebar.")
    else:
        idxs, scores = model.recommend(user_idx, store["train_csr"], N=controls["topn"], filter_seen=True)
        item_ids = [store["index_to_item"][i] for i in idxs]
        frame = store["items"][store["items"]["item_id"].isin(item_ids)][["item_id", "title", "release_date", "Comedy", "Drama", "Action"]].copy()
        order = pd.Series(scores, index=item_ids).sort_values(ascending=False)
        frame["score"] = frame.item_id.map(order)
        st.dataframe(frame.sort_values("score", ascending=False), use_container_width=True)

with TAB_USER:
    st.subheader("Explore User")
    if user_idx is None:
        st.info("Pick a valid user id in the sidebar.")
    else:
        # Show user's top rated items in train
        start, end = store["train_csr"].indptr[user_idx], store["train_csr"].indptr[user_idx + 1]
        item_indices = store["train_csr"].indices[start:end]
        ratings = store["train_csr"].data[start:end]
        user_items = pd.DataFrame({
            "item_index": item_indices,
            "rating": ratings,
            "item_id": [store["index_to_item"][i] for i in item_indices],
        })
        user_items = user_items.merge(store["items"][['item_id','title']], on='item_id', how='left')
        st.dataframe(user_items.sort_values("rating", ascending=False).head(30), use_container_width=True)

with TAB_ITEM:
    st.subheader("Item Details & Similar")
    example_item_id = int(store["items"]["item_id"].iloc[0])
    picked_item_id = st.number_input("Movie ID", min_value=int(store["items"]["item_id"].min()), max_value=int(store["items"]["item_id"].max()), value=example_item_id)
    item_index = store["item_to_index"].get(picked_item_id)
    if item_index is None:
        st.info("Enter a valid movie id.")
    else:
        sims_idx, sims = model.similar_items(item_index, N=10)
        sim_ids = [store["index_to_item"][i] for i in sims_idx]
        frame = store["items"][store["items"]["item_id"].isin(sim_ids)][["item_id", "title", "release_date"]].copy()
        order = pd.Series(sims, index=sim_ids).sort_values(ascending=False)
        frame["similarity"] = frame.item_id.map(order)
        st.dataframe(frame.sort_values("similarity", ascending=False), use_container_width=True)

with TAB_METRICS:
    st.subheader("Offline Metrics (quick)")
    k = st.slider("K", 5, 20, 10)
    # Generate top-K for a slice of users for speed
    num_users = store["train_csr"].shape[0]
    sample_users = np.arange(min(300, num_users))
    preds = np.zeros((len(sample_users), k), dtype=int)
    for idx, u in enumerate(sample_users):
        recs, _ = model.recommend(u, store["train_csr"], N=k, filter_seen=True)
        if len(recs) < k:
            fill = np.full(k, -1, dtype=int)
            fill[:len(recs)] = recs
            preds[idx] = fill
        else:
            preds[idx] = recs[:k]
    # Trim test matrix to sampled users for metric computation
    test_trim = store["test_csr"][sample_users]
    r = recall_at_k(preds, test_trim, k)
    n = ndcg_at_k(preds, test_trim, k)
    col1, col2 = st.columns(2)
    col1.metric("Recall@K", f"{r:.3f}")
    col2.metric("NDCG@K", f"{n:.3f}")

with TAB_ABOUT:
    st.subheader("About")
    st.markdown(
        "This demo keeps things intentionally small and reproducible: implicit ALS on ML-100k, float32, sparse CSR, and a simple UI."
    )
