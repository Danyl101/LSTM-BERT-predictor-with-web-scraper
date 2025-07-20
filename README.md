# LSTM-BERT Predictor with Web Scraper 🧠📈🕸️

A hybrid AI model that combines LSTM for time-series prediction and BERT for text-based sentiment analysis, powered by a custom web scraper that collects real-time data from news articles and online sources. This project aims to enhance stock market prediction by fusing numerical and textual data.

---

## 🔧 Features

- 📉 **LSTM Model**: Predicts future stock prices using historical price trends.
- 🧠 **BERT/NLP Engine**: Analyzes sentiment and relevance of news headlines and articles.
- 🕸️ **Custom Web Scraper**: Scrapes financial news from the web for live updates.
- 🔀 **Fusion Logic**: Integrates both models' outputs for improved prediction accuracy.
- 📊 **Preprocessed Datasets**: Scaled and cleaned datasets for efficient training.
- 🌐 **React Frontend** *(planned)*: A visual dashboard to display predictions and scraped news.

---

## 🗂️ Project Structure

├── Datasets/
│ └── [train/test/val/nifty]_scaled.csv
├── Documentation/
│ └── Debug logs and production notes
├── Models/
│ └── LSTM_Model.py, BERT.py
├── Scrapers/
│ └── Scraper.py,Loader.py


---

## 🚀 Getting Started

### 1. Clone the Repo
```bash
git clone https://github.com/Danyl101/LSTM-BERT-predictor-with-web-scraper.git
cd LSTM-BERT-predictor-with-web-scraper
``` 
###2. Set Up Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

###3. Train the LSTM
```bash
python Models/LSTM_Model.py
```

###4. Run Web Scraper
```bash
python Scrapers/Scraper.py
```


