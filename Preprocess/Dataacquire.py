import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

#Fetch historical data for Nifty(AAPL)
ticker = "^NSEI"
data = yf.download(ticker, start="2005-01-01", end="2024-12-31",interval="1d")
print(data.head())

# Save the data to a CSV file
data.to_csv("Datasets/nifty_data.csv")

# Plot the values in a graph
plt.plot(data['Close'])
plt.title(f"{ticker} Stock Price")
plt.xlabel("Date")
plt.ylabel("Close Price")
