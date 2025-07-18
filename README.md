# 📈 LSTM-BERT Stock Market Predictor with Web Scraper

This project is a hybrid model that combines **time series forecasting** using LSTM/TCN and **news sentiment analysis** using BERT. It aims to predict future market behavior by analyzing both historical stock prices and real-world economic news.

---

## 🧠 Core Features

- **LSTM / TCN Models**: Predict short-term movements of stock indices like Nifty 50.
- **Web Scraper (Selenium)**: Gathers economic news from major financial sites.
- **BERT Integration**: Processes financial articles to extract contextual sentiment.
- **Automation**: Automatically scrapes, processes, and stores article data.
- **Custom Preprocessing**: Log scaling, robust normalization, and dataset-specific treatment.

---

Install All Dependencies with

pip install -r requirements.txt


## 📁 Project Structure

```bash
.
├── models/
│   ├── lstm_model.py
│   ├── tcn_model.py
├── scraper/
│   ├── selenium_scraper.py
│   └── utils.py
├── preprocessing/
│   └── data_preprocessing.py
├── main.py
├── requirements.txt
└── README.md

