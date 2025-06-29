# Nifty 50 Predictor using News Sentiment Analysis

This project predicts short-term directional movements in the NIFTY 50 index by analyzing the sentiment of financial news articles. It combines news scraping, sentiment scoring, and machine learning models to classify the market movement as Up, Down, or Neutral.

## Features

* News scraping from financial websites related to the NIFTY 50 index
* Sentiment analysis using VADER and TextBlob
* Integration of sentiment features with historical NIFTY index data
* Classification models: Logistic Regression, Random Forest, and XGBoost
* Performance evaluation through accuracy, precision, confusion matrix, and backtesting

## Project Workflow

1. **News Collection**
   Scrapes headlines or summaries from financial news sources using `requests` and `BeautifulSoup`.

2. **Preprocessing**
   Applies standard NLP preprocessing: cleaning text, removing stopwords, and lemmatization.

3. **Sentiment Scoring**
   Computes polarity scores using VADER and TextBlob to quantify market sentiment.

4. **Labeling**
   Labels each day based on the next-day NIFTY 50 index movement (up/down/neutral) using a predefined threshold.

5. **Model Training**
   Trains classification models using historical price and sentiment data.

6. **Prediction and Evaluation**
   Predicts the next day's market direction and evaluates the results using common classification metrics.

## Directory Structure

```
Nifty-50-Predictor-using-News-Sentiment-Analysis/
├── data/
│   ├── nifty_price_data.csv
│   └── news_data.csv
├── models/
│   └── trained_model.pkl
├── predictor.py
├── sentiment_analysis.py
├── data_scraper.py
├── model_training.py
├── backtest.py
└── README.md
```

## Installation

Install the required libraries using:

```bash
pip install -r requirements.txt
```

### Main Libraries

* `pandas`, `numpy`
* `scikit-learn`, `xgboost`
* `nltk`, `vaderSentiment`, `textblob`
* `matplotlib`, `seaborn`
* `requests`, `beautifulsoup4`

## Results

* Sentiment scores showed a measurable correlation with NIFTY 50 daily changes.
* Classification accuracy of approximately 70% on the test set.
* Backtesting indicated potential trading profitability on high-confidence predictions.

## Sample Prediction Output

```python
Input: "RBI hikes repo rate by 25bps to tame inflation"
Sentiment Score: -0.31
Model Prediction: NIFTY likely to go Down
```

## Future Work

* Use real-time APIs like NewsAPI for live prediction
* Replace basic sentiment models with FinBERT or other finance-tuned transformers
* Integrate technical indicators (moving averages, RSI) alongside sentiment
* Deploy as a web app using Streamlit or Flask

## Author

**Nishika Kakrecha**
GitHub: [Nish-011-100](https://github.com/Nish-011-100)
