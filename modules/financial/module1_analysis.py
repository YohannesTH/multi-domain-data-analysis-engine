import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression

# Ensure output directories exist
os.makedirs('data', exist_ok=True)
os.makedirs('output', exist_ok=True)

def main():
    print("Fetching stock data...")
    # 1. Load stock price data
    ticker = 'AAPL'
    data = yf.download(ticker, start='2020-01-01', end='2024-01-01', progress=False)
    
    # yfinance sometimes returns a MultiIndex column dataframe. Flatten it if needed.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
        
    df = data[['Close']].copy()
    df.index = pd.to_datetime(df.index)
    
    # 2. Perform Calculations
    # Returns calculation
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Moving averages
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    
    # Volatility analysis (21-day rolling standard deviation of returns)
    df['Volatility_21d'] = df['Daily_Return'].rolling(window=21).std() * np.sqrt(252) # Annualized
    
    df.dropna(inplace=True)
    
    # Save output for MATLAB
    df.to_csv('data/financial_data.csv')
    print("Saved prepared financial data to data/financial_data.csv")
    
    # 3. Modeling
    # ARIMA (Statsmodels)
    print("Training ARIMA model...")
    train_size = int(len(df) * 0.8)
    train, test = df['Close'][:train_size], df['Close'][train_size:]
    
    # Fit ARIMA(1,1,1)
    model_arima = ARIMA(train, order=(1, 1, 1))
    arima_result = model_arima.fit()
    
    # Forecast
    arima_pred = arima_result.forecast(steps=len(test))
    
    # Linear Regression (Predicting Close based on MAs and Volatility)
    print("Training Linear Regression model...")
    features = ['MA_50', 'MA_200', 'Volatility_21d']
    X = df[features]
    y = df['Close']
    
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    
    # 4. Visualization
    print("Generating visualizations...")
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Close Price & Moving Averages
    plt.subplot(3, 1, 1)
    plt.plot(df.index, df['Close'], label='Close Price')
    plt.plot(df.index, df['MA_50'], label='50-Day MA')
    plt.plot(df.index, df['MA_200'], label='200-Day MA')
    plt.title(f'{ticker} Stock Analysis')
    plt.ylabel('Price')
    plt.legend()
    
    # Plot 2: Volatility
    plt.subplot(3, 1, 2)
    plt.plot(df.index, df['Volatility_21d'], color='orange', label='21-Day Annualized Volatility')
    plt.ylabel('Volatility')
    plt.legend()
    
    # Plot 3: Predictions
    plt.subplot(3, 1, 3)
    plt.plot(test.index, test, label='Actual Close Price')
    plt.plot(test.index, arima_pred, label='ARIMA Forecast', color='red')
    plt.plot(test.index, lr_pred, label='Linear Regression Prediction', color='green', linestyle='--')
    plt.ylabel('Price')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('output/module1_financial_analysis.png')
    print("Saved visualization to output/module1_financial_analysis.png")
    
if __name__ == "__main__":
    main()
