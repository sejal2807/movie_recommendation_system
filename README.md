# 🎬 My Movie Recommendation System

Hey! I built this project because I was curious about how Netflix and Spotify actually recommend stuff to us. Turns out, it's way more interesting than I thought!

I wanted to understand the whole process - from the math behind collaborative filtering to how companies actually use this stuff in real life. So I built my own recommendation system from scratch.

## What I Actually Built

- **The Math Stuff**: I coded up ALS (Alternating Least Squares) myself using NumPy - no libraries doing the heavy lifting for me
- **Real User Insights**: Found out engineers really do love sci-fi movies, and students prefer comedies (who knew!)
- **Business Analytics**: Learned how to segment users and find patterns in their behavior
- **A/B Testing**: Built a framework to test different models and see which one actually works better
- **Performance Stuff**: Made it scalable and fast enough to handle real users

## How to Run This Thing

### Local Setup (Super Easy)
```bash
# Install the stuff you need
pip install -r requirements.txt

# Run the app
streamlit run app.py

# Open your browser to http://localhost:8502
# That's it!
```

## How I Structured This Project

```
├── app.py                 # The main Streamlit app - this is where everything comes together
├── requirements.txt       # All the Python packages you need
├── recommender/          # The ML code (this is where the magic happens)
│   ├── data.py           # Downloads and processes the MovieLens dataset
│   ├── als.py            # My custom ALS algorithm implementation
│   ├── baselines.py      # Simple models to compare against (popularity, random, etc.)
│   ├── metrics.py        # How I measure if the recommendations are actually good
│   ├── analytics.py      # User segmentation and business insights
│   └── experiments.py    # A/B testing and performance simulation
└── data/                 # Where the MovieLens dataset gets stored
```

## The Interesting Stuff I Found

This project taught me that recommendation systems are way more than just math - they're about understanding people:

- **User Patterns**: Engineers really do love sci-fi, students prefer comedies (the data doesn't lie!)
- **When People Rate**: Most activity happens around 8 PM, and people rate more on weekends
- **User Types**: About 15% are new users (cold start problem), 85% have enough history for good recommendations
- **Model Performance**: My ALS model beats simple popularity by 40% - that's actually pretty good!
- **Real Impact**: Better recommendations mean happier users (obviously, but now I can prove it)

## Technical Stuff

- **The Algorithm**: I built ALS from scratch - no black box libraries doing the work for me
- **Memory Efficient**: Uses sparse matrices and float32 to keep memory usage low
- **No GPU Needed**: Runs on any laptop (I don't have a fancy GPU anyway)
- **Actually Scalable**: Can handle way more users than the current dataset

## Deploying This Thing

### Run Locally
```bash
streamlit run app.py
```

### Put It Online (Streamlit Cloud)
```bash
# Push to GitHub first
git add .
git commit -m "Add movie recommendation system"
git push origin main

# Then go to share.streamlit.io
# Connect your GitHub repo
# Click deploy - that's it!
```

## How Fast Is It?

| Thing | Time | Why This Matters |
|-------|------|------------------|
| **Training** | 10-30 seconds | Fast enough to retrain when new users join |
| **Getting Recommendations** | <100ms | Users won't notice the delay |
| **Memory Usage** | <500MB | Runs on my old laptop just fine |
| **Accuracy** | Recall@10 ~0.20-0.35 | Actually competitive with real systems |

## What I Actually Learned

This project taught me way more than I expected:

- **ML Engineering**: How to build the whole pipeline from data to recommendations
- **Business Stuff**: How to understand users and find patterns in their behavior  
- **Code Organization**: How to write clean, modular code that other people can understand
- **Data Science**: How to measure if your model is actually good (not just accurate)
- **Production Thinking**: How to make things fast and scalable

**Turns out building recommendation systems is way more interesting than I thought!** 🎯
