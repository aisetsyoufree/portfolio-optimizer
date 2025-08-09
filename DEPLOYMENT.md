# 🌐 Portfolio Optimizer Deployment Guide

## 🚀 Quick Deploy to Streamlit Community Cloud (Recommended)

### Prerequisites
1. **GitHub Account** - Create at [github.com](https://github.com)
2. **Streamlit Account** - Sign up at [share.streamlit.io](https://share.streamlit.io)

### Step-by-Step Deployment

#### 1. Create GitHub Repository
```bash
cd /Users/siddhesh/CascadeProjects/portfolio-optimizer
git init
git add .
git commit -m "Initial commit: Portfolio Optimizer with Excel export and raw data features"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/portfolio-optimizer.git
git push -u origin main
```

#### 2. Deploy on Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub account
4. Select your `portfolio-optimizer` repository
5. Set main file path: `portfolio_app.py`
6. Click "Deploy!"

#### 3. Your App Will Be Live At:
`https://YOUR_USERNAME-portfolio-optimizer-portfolio-app-xyz123.streamlit.app`

## 🛠️ Alternative Deployment Options

### Railway Deployment
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

### Render Deployment
1. Connect GitHub repo at [render.com](https://render.com)
2. Select "Web Service"
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `streamlit run portfolio_app.py --server.port $PORT --server.address 0.0.0.0`

### Heroku Deployment
```bash
# Install Heroku CLI and login
heroku create your-portfolio-optimizer
git push heroku main
```

## 📋 Deployment Checklist

### ✅ Files Ready for Deployment
- [x] `portfolio_app.py` - Main application
- [x] `requirements.txt` - All dependencies
- [x] `.streamlit/config.toml` - Streamlit configuration
- [x] `README.md` - Documentation

### ✅ Features Included
- [x] Portfolio optimization with 3 strategies
- [x] Excel export functionality
- [x] Raw data viewing (4 tabs)
- [x] Interactive charts and visualizations
- [x] Individual fund performance toggle
- [x] Comprehensive test suite (26 tests)

### ✅ Production Ready
- [x] All tests passing (100% success rate)
- [x] Error handling implemented
- [x] Dependencies optimized
- [x] UI responsive and stable

## 🎯 Recommended: Streamlit Community Cloud

**Why Streamlit Community Cloud?**
- ✅ **Free hosting** for public apps
- ✅ **Zero configuration** required
- ✅ **Automatic deployments** from GitHub
- ✅ **Perfect for Streamlit apps**
- ✅ **Custom subdomain** included
- ✅ **SSL certificate** included

Your app will be accessible worldwide at a URL like:
`https://portfolio-optimizer.streamlit.app`
