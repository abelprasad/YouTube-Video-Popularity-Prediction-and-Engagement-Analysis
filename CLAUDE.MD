# YouTube Video Popularity Prediction and Engagement Analysis

## Project Overview

**Objective:** Develop two machine learning models to predict video category and identify key factors influencing engagement. One model will use data collected via web scraping, and the other will use data from the YouTube Data API.

---

## Data Sources

### 1. Web Scraping

- Scrape trending or search result pages from YouTube
- Extract data such as:
  - Video title
  - Channel name
  - Views
  - Category
  - Upload date
  - Duration
  - Tags (if visible)
- **Note:** You can add more attributes of each video if you can collect them
- **Example data structure:**
  ```json
  {
    "title": "Rick Astley - Never Gonna Give You Up (Official Video) (4K Remaster)",
    "views": "1,696,931,258",
    "likes": "Hidden by YouTube",
    "upload_date": "2009.10.24",
    "category": "Music"
  }
  ```

### 2. YouTube Data API

- Use the API to collect structured metadata for the same or similar videos
- Extract data such as:
  - Title
  - Description
  - Tags
  - Category
  - Statistics (views, likes, comments)
  - Publish date
  - Channel info

---

## Required Works

### 1. Data Collection

- Use web scraping tools to collect job postings from the YouTube websites
- Use the YouTube Data API with an API key to collect structured data
- Ensure overlap or similarity in the video sets for fair comparison
- **Collect at least 3000 data for each method**

### 2. Data Preprocessing

- Clean and normalize both datasets
- Handle missing or inconsistent values
- Standardize features (e.g., convert duration to minutes, normalize view counts)

### 3. Feature Engineering

- Create features like:
  - Video length
  - Time since upload
  - Keyword frequency in title/description
  - Channel subscriber count (API only)
  - Engagement rate = (likes + comments) / views

### 4. Model Development

- Train two regression models (e.g., Random Forest, XGBoost) to predict view count or engagement rate
- Compare performance between models trained on scraped vs. API data
- Use feature importance techniques to identify the most important attributes (You can simply use the function supported by SKLearn library)

### 5. Visualization

- Compare prediction accuracy between the two models
- Visualize feature importance for each model
- Show engagement trends by video category, length, or upload time

---

## Mandatory Items in the Report

### 1. Description of the Project

Provide a comprehensive overview of the project objectives, scope, and methodology.

### 2. How to Use

#### Training

Explain the steps to train the models, including:

- Environment setup
- Dependencies installation
- Data preparation
- Model training commands

#### Inferencing

Explain how to use the trained models for prediction on new data.

### 3. Data Collection

- **Used tools:** Specify web scraping libraries and API tools used
- **Collected attributes:** List all features extracted from both sources
- **Number of data samples:** Total count for scraped and API datasets
- **API usage:** Describe authentication and request methodology
- **2 Sample data after preprocessing:** Provide examples from both datasets

### 4. Data Preprocessing

- **Data cleaning:** Methods used to handle duplicates, errors, and outliers
- Explain normalization and standardization techniques applied

### 5. Feature Engineering

- **How data is processed and prepared** for the machine learning model after loading from data repositories
- **3 collected data and corresponding features** after data preprocessing and feature engineering

### 6. Model Development and Evaluation

#### Train and Test Data Partition

Describe the split ratio and methodology (e.g., 80/20, stratified sampling)

#### Model-1 Based on Scraped Data

1. **Machine learning model:** Name and justification
2. **Input to model:** List of features used
3. **Size of train data:** Number of samples
4. **Attributes to the machine learning model:** Feature list
5. **Performance with training data:** Metrics (MSE, R², etc.)
6. **Performance with test data:** Metrics on unseen data

#### Model-2 Based on API Usage

1. **Machine learning model:** Name and justification
2. **Input to model:** List of features used
3. **Size of train data:** Number of samples
4. **Attributes to the machine learning model:** Feature list
5. **Performance with training data:** Metrics (MSE, R², etc.)
6. **Performance with test data:** Metrics on unseen data

#### Feature Importance

- **Description of the exploited feature importance techniques** to identify the most important attributes
- Visualizations comparing feature importance across both models

### 7. Visualization

- Compare prediction accuracy between the two models
- Visualize feature importance for each model
- Show engagement trends by video category, length, or upload time **for API data**

### 8. Discussion and Conclusions

- **Project findings:** Including insights gained from data analysis
- **Challenges encountered** during model development
- **Ethical and legal considerations** of data collection
- **Recommendations** for improving the performance of the model

---

## Deliverables

1. **Code:** Well-documented Python scripts for:

   - Web scraping
   - API data collection
   - Data preprocessing
   - Feature engineering
   - Model training and evaluation

2. **Data:**

   - Raw scraped data
   - Raw API data
   - Preprocessed datasets

3. **Report:** Comprehensive document covering all mandatory items listed above

---

## Evaluation Criteria

The project will be evaluated based on four main components:

- **Code (20%):** The functionalities, quality, readability and effectiveness of the code developed for data preprocessing, feature engineering, and model development.

- **Data (10%):** The quantity and quality of your data

- **Report (60%):** The completeness, thoroughness and clarity of the report, which should cover all mandatory items

- **Github (10%):** Setting the repository as public and the containment of data, code, and report in the repository

---

## How to Submit

Your project submission should include a GitHub repository link containing all the required deliverables for evaluation.

---

## Notes

_Everything here can be subject to change with common-sense reasoning._

---

## Project Structure Recommendation

```
youtube-popularity-prediction/
│
├── data/
│   ├── raw/
│   │   ├── scraped_data.csv
│   │   └── api_data.csv
│   └── processed/
│       ├── scraped_processed.csv
│       └── api_processed.csv
│
├── notebooks/
│   ├── 01_web_scraping.ipynb
│   ├── 02_api_collection.ipynb
│   ├── 03_data_preprocessing.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_training.ipynb
│   └── 06_visualization.ipynb
│
├── src/
│   ├── scraper.py
│   ├── api_collector.py
│   ├── preprocessor.py
│   ├── feature_engineer.py
│   ├── model.py
│   └── visualizer.py
│
├── models/
│   ├── model_scraped.pkl
│   └── model_api.pkl
│
├── reports/
│   ├── figures/
│   └── final_report.pdf
│
├── requirements.txt
├── README.md
└── claude.md
```
