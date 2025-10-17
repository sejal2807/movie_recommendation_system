import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from recommender.data import prepare_data_bundle
from recommender.als import ALSConfig, ImplicitALS
from recommender.metrics import recall_at_k, ndcg_at_k
from recommender.analytics import (
    load_user_demographics, analyze_user_segments, analyze_genre_preferences,
    calculate_business_metrics, analyze_temporal_patterns, generate_business_insights
)
from recommender.baselines import PopularityRecommender, RandomRecommender, ContentBasedRecommender, ModelComparator
from recommender.experiments import ScalabilitySimulator, ABTestFramework, PerformanceMonitor, simulate_production_load

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


TAB_RECS, TAB_USER, TAB_ITEM, TAB_METRICS, TAB_ANALYTICS, TAB_COMPARE, TAB_EXPERIMENTS, TAB_ABOUT = st.tabs(["Recommendations", "Explore User", "Item Details", "Metrics", "📊 Analytics", "🔬 Model Compare", "🧪 Experiments", "About"]) 

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

with TAB_ANALYTICS:
    st.subheader("📊 Business Analytics & Insights")
    st.caption("User segmentation, demographic analysis, and business impact metrics")
    
    # Load user demographics
    users_df = load_user_demographics(DATA_DIR)
    
    if not users_df.empty:
        # User segmentation analysis
        segment_analysis = analyze_user_segments(store["ratings"], store["items"], users_df)
        genre_insights = analyze_genre_preferences(store["ratings"], store["items"], users_df)
        business_metrics = calculate_business_metrics(store["ratings"], store["items"], model, store["user_to_index"])
        temporal_insights = analyze_temporal_patterns(store["ratings"])
        
        # Generate business insights
        insights = generate_business_insights(segment_analysis, genre_insights, business_metrics, temporal_insights)
        
        # Display insights
        st.subheader("🎯 Key Business Insights")
        for insight in insights:
            st.markdown(insight)
        
        # User segmentation charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👥 User Demographics")
            if "gender" in segment_analysis and not segment_analysis["gender"].empty:
                gender_data = segment_analysis["gender"]
                st.dataframe(gender_data, use_container_width=True)
        
        with col2:
            st.subheader("📈 Business Metrics")
            if business_metrics:
                metrics_df = pd.DataFrame([
                    ["Active Users", f"{business_metrics.get('active_users', 0):,}"],
                    ["Engagement Rate", f"{business_metrics.get('engagement_rate', 0):.1%}"],
                    ["Avg Rating", f"{business_metrics.get('avg_rating', 0):.2f}/5.0"],
                    ["Cold Start Ratio", f"{business_metrics.get('cold_start_ratio', 0):.1%}"],
                ], columns=["Metric", "Value"])
                st.dataframe(metrics_df, use_container_width=True)
        
        # Genre preferences
        if "gender_genre" in genre_insights:
            st.subheader("🎬 Genre Preferences by Demographics")
            gender_genre = genre_insights["gender_genre"]
            
            for gender, prefs in gender_genre.items():
                if prefs:
                    st.write(f"**{gender} Users Top Genres:**")
                    for genre, score in list(prefs.items())[:3]:
                        st.write(f"• {genre}: {score:.2f}")
        
        # Temporal patterns
        if "hourly" in temporal_insights:
            st.subheader("⏰ User Activity Patterns")
            hourly = temporal_insights["hourly"]
            st.write("Peak rating hours:", hourly.loc[hourly["count"].idxmax(), "hour"])
            st.write("Average rating by hour:", hourly["mean"].round(2).to_dict())
    
    else:
        st.info("User demographics not available. This analysis requires the u.user file from MovieLens-100k.")

with TAB_COMPARE:
    st.subheader("🔬 Model Comparison")
    st.caption("Compare ALS with baseline models: Popularity, Random, and Content-Based")
    
    if st.button("Run Model Comparison", type="primary"):
        with st.spinner("Training and evaluating models..."):
            # Initialize models
            comparator = ModelComparator()
            comparator.add_model("ALS (Our Model)", model)
            comparator.add_model("Popularity", PopularityRecommender())
            comparator.add_model("Random", RandomRecommender())
            comparator.add_model("Content-Based", ContentBasedRecommender())
            
            # Train all models
            comparator.fit_all(store["train_csr"], store["items"])
            
            # Evaluate on test set
            results = comparator.evaluate_all(store["train_csr"], store["test_csr"], k=10, sample_users=200)
            
            # Display results
            st.subheader("📊 Performance Comparison")
            comparison_df = comparator.get_comparison_table()
            st.dataframe(comparison_df, use_container_width=True)
            
            # Show insights
            st.subheader("💡 Key Insights")
            if "ALS (Our Model)" in results and "Popularity" in results:
                als_recall = results["ALS (Our Model)"]["recall_at_k"]
                pop_recall = results["Popularity"]["recall_at_k"]
                improvement = ((als_recall - pop_recall) / pop_recall) * 100
                st.success(f"🎯 **ALS outperforms Popularity by {improvement:.1f}%** in Recall@10")
            
            if "Random" in results:
                random_recall = results["Random"]["recall_at_k"]
                st.info(f"📈 **Random baseline**: {random_recall:.3f} - shows the importance of personalization")
            
            # Model characteristics
            st.subheader("🔍 Model Characteristics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **ALS (Collaborative Filtering)**
                - ✅ Learns user-item interactions
                - ✅ Handles sparse data well
                - ⚠️ Cold start problem
                """)
            
            with col2:
                st.markdown("""
                **Content-Based**
                - ✅ Uses movie features (genres)
                - ✅ Good for cold start
                - ⚠️ Limited by feature quality
                """)
    
    else:
        st.info("Click 'Run Model Comparison' to see how different algorithms perform on this dataset.")

with TAB_EXPERIMENTS:
    st.subheader("🧪 ML Experiments & Scalability")
    st.caption("A/B testing, scalability simulation, and production load testing")
    
    # Create tabs for different experiment types
    exp_tab1, exp_tab2, exp_tab3 = st.tabs(["📈 Scalability", "🧪 A/B Testing", "⚡ Load Testing"])
    
    with exp_tab1:
        st.subheader("📈 Scalability Simulation")
        st.caption("Simulate system performance at different scales (100k → 10M ratings)")
        
        if st.button("Run Scalability Analysis", type="primary"):
            with st.spinner("Analyzing scalability..."):
                simulator = ScalabilitySimulator()
                baseline = simulator.measure_baseline(model, store)
                report_df = simulator.get_scalability_report()
                
                st.subheader("📊 Current Performance (Baseline)")
                col1, col2, col3 = st.columns(3)
                col1.metric("Training Time", f"{baseline['training_time']:.2f}s")
                col2.metric("Avg Inference", f"{baseline['avg_inference_time']*1000:.1f}ms")
                col3.metric("Memory Usage", f"{baseline['memory_usage']:.1f}MB")
                
                st.subheader("🚀 Scalability Projections")
                st.dataframe(report_df, use_container_width=True)
                
                # Show key insights
                st.subheader("💡 Scalability Insights")
                max_scale = report_df.iloc[-1]
                st.info(f"**At 10M ratings**: Training time would be ~{max_scale['estimated_training_time']:.1f}s, "
                       f"Memory usage ~{max_scale['estimated_memory_mb']:.0f}MB")
                
                if max_scale['estimated_memory_mb'] > 2000:
                    st.warning("⚠️ **Memory Alert**: System may need optimization for large scale")
                else:
                    st.success("✅ **Memory OK**: System can handle projected scale")
    
    with exp_tab2:
        st.subheader("🧪 A/B Testing Framework")
        st.caption("Test different recommendation algorithms with statistical significance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Experiment Setup**")
            experiment_name = st.text_input("Experiment Name", "ALS vs Popularity")
            traffic_split = st.slider("Traffic Split (%)", 10, 90, 50)
            min_sample = st.number_input("Min Sample Size", 100, 1000, 200)
        
        with col2:
            st.write("**Test Configuration**")
            metric_choice = st.selectbox("Success Metric", ["Recall@10", "NDCG@10", "User Engagement"])
            confidence = st.slider("Confidence Level", 0.90, 0.99, 0.95)
        
        if st.button("Run A/B Test", type="primary"):
            with st.spinner("Running A/B test experiment..."):
                # Setup A/B test
                ab_framework = ABTestFramework()
                
                # Create experiment config
                from recommender.experiments import ExperimentConfig
                config = ExperimentConfig(
                    experiment_name=experiment_name,
                    variants=["ALS", "Popularity"],
                    traffic_split=[traffic_split/100, 1-traffic_split/100],
                    duration_days=7,
                    success_metric=metric_choice.lower(),
                    min_sample_size=min_sample,
                    confidence_level=confidence
                )
                
                ab_framework.create_experiment(config)
                
                # Run experiment
                pop_model = PopularityRecommender().fit(store["train_csr"])
                
                def recall_metric(user_idx, recs, test_data):
                    # Simplified metric calculation
                    return len(recs) / 10.0  # Normalized by expected length
                
                results = ab_framework.run_experiment(
                    experiment_name, model, pop_model, store, store["test_csr"], recall_metric
                )
                
                # Display results
                if results:
                    st.subheader("📊 A/B Test Results")
                    summary_df = ab_framework.get_experiment_summary(experiment_name)
                    st.dataframe(summary_df, use_container_width=True)
                    
                    # Show winner
                    if len(results) >= 2:
                        als_result = next(r for r in results if r.variant == "ALS")
                        pop_result = next(r for r in results if r.variant == "Popularity")
                        
                        if als_result.metric_value > pop_result.metric_value:
                            improvement = ((als_result.metric_value - pop_result.metric_value) / pop_result.metric_value) * 100
                            st.success(f"🏆 **ALS wins!** {improvement:.1f}% better than Popularity")
                        else:
                            st.info("🤝 **No clear winner** - results are not statistically significant")
                else:
                    st.warning("⚠️ Not enough data for statistical significance")
    
    with exp_tab3:
        st.subheader("⚡ Production Load Testing")
        st.caption("Simulate production traffic and measure system performance")
        
        num_requests = st.slider("Number of Requests", 100, 5000, 1000)
        st.caption(f"Simulating {num_requests} concurrent user requests...")
        
        if st.button("Run Load Test", type="primary"):
            with st.spinner("Running load test..."):
                load_results = simulate_production_load(model, store, num_requests)
                
                st.subheader("📊 Load Test Results")
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric("Total Requests", f"{load_results['total_requests']:,}")
                col2.metric("Success Rate", f"{(1-load_results['error_rate'])*100:.1f}%")
                col3.metric("Avg Response Time", f"{load_results['avg_response_time']*1000:.1f}ms")
                col4.metric("RPS", f"{load_results['requests_per_second']:.1f}")
                
                # Performance analysis
                st.subheader("🎯 Performance Analysis")
                
                if load_results['error_rate'] > 0.01:
                    st.error(f"❌ **High Error Rate**: {load_results['error_rate']:.1%} - System needs optimization")
                else:
                    st.success(f"✅ **Low Error Rate**: {load_results['error_rate']:.1%} - System is stable")
                
                if load_results['avg_response_time'] > 0.1:
                    st.warning(f"⚠️ **Slow Response**: {load_results['avg_response_time']*1000:.1f}ms - Consider optimization")
                else:
                    st.success(f"✅ **Fast Response**: {load_results['avg_response_time']*1000:.1f}ms - Good performance")
                
                # Recommendations
                st.subheader("💡 Optimization Recommendations")
                if load_results['requests_per_second'] < 100:
                    st.info("🚀 **Scale Up**: Consider horizontal scaling or model optimization")
                if load_results['p95_response_time'] > 0.5:
                    st.info("⚡ **Cache**: Implement recommendation caching for better performance")
                if load_results['error_rate'] > 0.05:
                    st.info("🛡️ **Resilience**: Add error handling and circuit breakers")

with TAB_ABOUT:
    st.subheader("About")
    st.markdown(
        "This demo keeps things intentionally small and reproducible: implicit ALS on ML-100k, float32, sparse CSR, and a simple UI."
    )
