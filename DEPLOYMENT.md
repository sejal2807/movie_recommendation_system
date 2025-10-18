# 🚀 Deployment Guide

## Quick Start

### 1. **Local Development**
```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

# Open http://localhost:8502
```

### 2. **Deploy to Streamlit Cloud**

#### Step 1: Push to GitHub
```bash
# Initialize git (if not already done)
git init
git add .
git commit -m "Add movie recommendation system"

# Create GitHub repo and push
git remote add origin https://github.com/yourusername/movie-recommendation-system.git
git push -u origin main
```

#### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub account
4. Select your repository
5. Set main file path to `app.py`
6. Click "Deploy!"

#### Step 3: Configure (Optional)
- **App URL**: Your app will be available at `https://your-app-name.streamlit.app`
- **Resources**: Streamlit Cloud provides free hosting with 1GB RAM
- **Auto-deploy**: Every push to main branch automatically updates the app

## 🎯 Production Tips

### **Performance Optimization**
- The app is optimized for CPU-only execution
- Memory usage is kept under 500MB
- All models use efficient sparse matrices

### **Data Management**
- MovieLens-100k dataset is auto-downloaded on first run
- Data is cached locally for faster subsequent runs
- No external database required

### **Monitoring**
- Built-in performance monitoring in the Experiments tab
- Scalability simulation from 100k to 10M ratings
- A/B testing framework for model comparison

## 🔧 Troubleshooting

### **Common Issues**
1. **Import errors**: Make sure all dependencies are installed
2. **Memory issues**: The app is designed to run on any modern laptop
3. **Slow loading**: First run downloads the dataset (takes ~30 seconds)

### **Streamlit Cloud Issues**
1. **App won't start**: Check that `app.py` is in the root directory
2. **Import errors**: Ensure all dependencies are in `requirements.txt`
3. **Timeout**: The app should start within 2-3 minutes

## 📊 What's Included

- **Custom ML Implementation**: ALS from scratch
- **Business Analytics**: User segmentation and insights
- **Model Comparison**: Multiple baselines with statistical testing
- **A/B Testing**: Experimental framework
- **Scalability Analysis**: Performance optimization
- **Interactive UI**: Streamlit dashboard with all features

---

*Ready to deploy your movie recommendation system!* 🎬
