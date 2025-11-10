# YouTube Video Popularity Prediction and Engagement Analysis

A comprehensive machine learning project that predicts video engagement and identifies key factors influencing YouTube video popularity using two different data collection methods: web scraping and YouTube Data API.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Models](#models)
- [Results](#results)
- [Requirements](#requirements)

## Project Overview

This project develops and compares two machine learning models to predict YouTube video view counts and engagement:

1. **Model 1**: Trained on data collected via web scraping
2. **Model 2**: Trained on data collected via YouTube Data API v3

**Key Objectives:**
- Collect 3000+ videos using each method
- Build regression models to predict video engagement
- Compare model performance between data sources
- Identify the most important features influencing video popularity

## Features

### Data Collection
- **Web Scraping**: Automated scraper using Selenium and BeautifulSoup
  - Collects: title, views, channel, duration, upload date
  - Handles dynamic content loading
  - Robust error handling

- **YouTube API**: Official API integration
  - Collects: comprehensive metadata, statistics, channel info
  - Includes: likes, comments, tags, category, subscriber count
  - Efficient quota management

### Machine Learning
- **Models**: Random Forest and XGBoost regressors
- **Target Variable**: Log-transformed view count
- **Evaluation Metrics**: RMSE, MAE, R², MAPE
- **Feature Importance**: Identifies key engagement drivers

### Analysis & Visualization
- Model performance comparison
- Feature importance analysis
- Engagement trends by category, duration, and upload time
- Correlation analysis
- Distribution analysis

## Installation

### Prerequisites
- Python 3.8 or higher
- Google Chrome browser (for web scraping)
- YouTube Data API v3 key (for API collection)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd YouTube-Video-Popularity-Prediction-and-Engagement-Analysis
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up YouTube API Key** (for API collection)
   ```bash
   cp .env.example .env
   # Edit .env and add: YOUTUBE_API_KEY=your_api_key_here
   ```

   Get API key from [Google Cloud Console](https://console.cloud.google.com/)

## Usage

### Quick Start

Run the entire pipeline with one command:

```bash
python run_pipeline.py
```

This will:
1. Test API connection
2. Collect 3000+ videos via API and web scraping
3. Preprocess and clean the data
4. Engineer features
5. Train both Random Forest and XGBoost models
6. Generate all visualizations

**Note:** Web scraping takes 30-60 minutes. You can monitor progress in the terminal.

### Manual Pipeline Steps

Run each stage individually:

```bash
# 0. Test API connection first
python test_api.py

# 1. Collect data (3000+ videos each)
python src/scraper.py          # Web scraping (30-60 min)
python src/api_collector.py    # API collection (~5 min)

# 2. Process data
python src/preprocessor.py     # Clean and normalize

# 3. Engineer features
python src/feature_engineer.py # Create ML features

# 4. Train models
python src/model.py            # Train Random Forest & XGBoost

# 5. Generate visualizations
python src/visualizer.py       # Create all plots
```

### Using Jupyter Notebooks

For interactive analysis:
```bash
jupyter notebook
# Open notebooks in order: 01, 02, 03, etc.
```

### Inference (Using Trained Models)

```python
import joblib
import pandas as pd
import numpy as np

# Load model
model = joblib.load('models/model_api_random_forest.pkl')

# Prepare features (must match training features)
# Make predictions
predictions = model.predict(X)

# Convert from log space
predicted_views = np.expm1(predictions)
```

## Project Structure

```
youtube-popularity-prediction/
│
├── data/
│   ├── raw/                    # Raw collected data
│   └── processed/              # Cleaned & feature-engineered data
│
├── notebooks/                  # Jupyter notebooks for each stage
│
├── src/                        # Source code modules
│   ├── scraper.py             # Web scraping
│   ├── api_collector.py       # YouTube API
│   ├── preprocessor.py        # Data cleaning
│   ├── feature_engineer.py    # Feature engineering
│   ├── model.py               # ML models
│   └── visualizer.py          # Visualization
│
├── models/                     # Trained models (.pkl files)
│
├── reports/                    # Results and visualizations
│   ├── figures/               # Generated plots
│   └── results_*.json         # Performance metrics
│
├── .env.example               # API key template
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Models

### Model 1: Scraped Data

**Features:**
- Duration (seconds, log-transformed)
- Days since upload
- Title features (length, word count, uppercase ratio)
- Binary flags (clickbait, numbers, questions)
- Categorical features (video age, duration category)

### Model 2: API Data

**Additional Features:**
- Engagement metrics (likes, comments, engagement rate)
- Channel features (subscribers, video count)
- Content features (tags, description, category)
- Derived metrics (likes/views, comments/views)

**Both Models Use:**
- Random Forest Regressor (200 trees, max_depth=20)
- XGBoost Regressor (200 estimators, learning_rate=0.1)

### Trained Models

All models saved in `models/` directory:
- `model_scraped_random_forest.pkl` (14MB) - Best scraped data model
- `model_scraped_xgboost.pkl` (1.9MB)
- `model_api_random_forest.pkl` (15MB)
- `model_api_xgboost.pkl` (1.5MB) - **Best overall model** (R²=0.94)

## Results

### Dataset Statistics
- **Scraped Data**: 3,050 videos collected → 3,034 records after cleaning
- **API Data**: 2,972 videos collected → 2,911 records after cleaning

### Model Performance

#### Model 1: Scraped Data Models
| Model | Test R² | Test RMSE | Test MAE | Cross-Val R² |
|-------|---------|-----------|----------|--------------|
| Random Forest | **0.5663** | 2.5565 | 1.8903 | 0.5213 ± 0.0295 |
| XGBoost | 0.5116 | 2.7129 | 2.0036 | 0.4376 ± 0.0331 |

**Top 5 Features (Random Forest):**
1. days_since_upload (0.299)
2. log_days_since_upload (0.235)
3. is_recent (0.125)
4. title_uppercase_ratio (0.120)
5. title_length (0.118)

#### Model 2: API Data Models (Significantly Better!)
| Model | Test R² | Test RMSE | Test MAE | Cross-Val R² |
|-------|---------|-----------|----------|--------------|
| Random Forest | 0.9082 | 0.8494 | 0.3593 | 0.9418 ± 0.0198 |
| XGBoost | **0.9419** | 0.6757 | 0.1579 | 0.9789 ± 0.0180 |

**Top 5 Features (XGBoost):**
1. log_likes (0.577)
2. engagement_rate (0.215)
3. likes_per_view (0.066)
4. log_comments (0.027)
5. comments_per_view (0.019)

### Key Findings
1. **API models vastly outperform scraped models** (R² 0.94 vs 0.57)
2. **Engagement metrics are crucial**: Likes and comments are the strongest predictors
3. **Scraped data limitations**: Without engagement metrics, model accuracy is significantly reduced
4. **Recency matters**: Videos uploaded recently perform differently
5. **Title characteristics** have moderate predictive power even without engagement data

### Visualizations

All visualizations saved in `reports/figures/`:

1. **model_comparison.png** - Side-by-side comparison of model performance metrics (R², RMSE, MAE)
2. **feature_importance_scraped.png** - Top features influencing predictions for scraped data models
3. **feature_importance_api.png** - Top features influencing predictions for API data models
4. **engagement_by_category.png** - Video engagement trends across different content categories (API data)
5. **engagement_by_duration.png** - Relationship between video length and engagement
6. **engagement_by_upload_time.png** - How video age affects engagement metrics
7. **correlation_heatmap_api.png** - Feature correlations in API dataset
8. **correlation_heatmap_scraped.png** - Feature correlations in scraped dataset
9. **data_distribution_api.png** - Distribution of key metrics in API data
10. **data_distribution_scraped.png** - Distribution of key metrics in scraped data

## Requirements

### Key Dependencies
- selenium, beautifulsoup4 (web scraping)
- google-api-python-client (YouTube API)
- pandas, numpy (data processing)
- scikit-learn, xgboost (ML)
- matplotlib, seaborn, plotly (visualization)

See `requirements.txt` for complete list with versions.

### System Requirements
- RAM: 4GB minimum, 8GB recommended
- Disk: 500MB for data/models
- Internet connection required
- Chrome browser (latest)

## Data Samples

### Scraped Data Sample
| video_id | title | views | duration_seconds | days_since_upload |
|----------|-------|-------|------------------|-------------------|
| abc123 | Python Tutorial | 150000 | 600 | 45 |
| xyz789 | Travel Vlog | 50000 | 900 | 12 |

### API Data Sample (Additional Columns)
| likes | comments | category_name | channel_subscribers |
|-------|----------|---------------|---------------------|
| 5000 | 250 | Education | 100000 |
| 2000 | 150 | Travel | 50000 |

## Ethical Considerations

- **Web Scraping**: Educational use, respects rate limits
- **API**: Official API with proper authentication
- **Privacy**: Only public metadata, no personal data
- **Compliance**: Review YouTube Terms of Service

## Troubleshooting

**Web Scraping:**
- Update Chrome and chromedriver if errors occur
- Adjust scroll settings if too slow

**API:**
- Check API key in .env
- Monitor quota (10,000 units/day limit)

**Training:**
- Reduce dataset if out of memory
- Check data quality if poor performance

## Future Improvements
- Sentiment analysis
- Deep learning models
- Real-time prediction API
- Web dashboard
- Automated pipeline

## License

Educational purposes. Review YouTube's Terms of Service before use.

---

**Developed as a machine learning course project to demonstrate data collection, preprocessing, feature engineering, and model development skills.**
