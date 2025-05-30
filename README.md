# Financial News Sentiment Analysis

Nova Financial Solutions aims to enhance its predictive analytics capabilities to improve forecasting accuracy and operational efficiency. This project focuses on analyzing the sentiment in financial news and its correlation with stock price movements.

## 🚀 Project Overview

This repository is part of a multi-phase analysis project to:

- Perform **sentiment analysis** on financial news headlines.
- Correlate **sentiment signals** with **stock price movements**.
- Develop **investment strategies** based on news sentiment data.
- Establish a **reproducible, production-ready data science environment** with CI/CD and version control.

## 📊 Dataset

I use the **Financial News and Stock Price Integration Dataset (FNSPID)**, which contains:
- `headline`: Title of the news article.
- `url`: Link to the full article.
- `publisher`: Publisher name or email domain.
- `date`: Publication date/time (UTC-4 timezone).
- `stock`: Ticker symbol (e.g., AAPL, TSLA, AMZN).

## 🧠 Objectives

### Phase 1: Environment and Git Setup
- Create GitHub repository with standardized structure.
- Use Python virtual environments.
- Integrate GitHub Actions for CI/CD.

### Phase 2: Exploratory Data Analysis (EDA)
- Descriptive statistics on headlines, publishers, and publication times.
- NLP-based topic modeling and keyword extraction.
- Publisher and time-based trend analysis.

### Phase 3: Sentiment Analysis
- Apply NLP models to derive sentiment scores from headlines.
- Associate sentiment with respective stock symbols and dates.

### Phase 4: Correlation & Predictive Modeling
- Analyze relationship between sentiment and stock movements.
- Build models to test predictability of sentiment on price action.
- Generate actionable trading signals based on sentiment.

---


## ⚙️ Setup Instructions

```bash
git clone https://github.com/your-username/KAIM-WEEK1.git
cd KAIM-WEEK1
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

--- 

## 🧪 Testing

```bash
pytest tests
```

--- 

## 📌 Project Status

- Task 1: Environment Setup & EDA – ✅ In Progress
- Task 2: Setiment analysis – ⏳ Upcoming
- Task 3: Correlation Modeling – ⏳ Upcoming


