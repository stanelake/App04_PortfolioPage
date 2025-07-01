import streamlit as st
import pandas as pd
import numpy as np
import os
import yfinance as yf

DATA_DIR = "./Data/Sources"

def displayFeatures():# Macro and Market Indicators
    macro_features = {
        "Feature": [
            "TB3MS",
            "DGS10",
            "AAA",
            "BAA",
            'csp',
            "GS10",
            "T10Y2Y",
            "CPIAUCSL",
            "PCEPILFE",
            "VIXCLS",
            "PAYEMS",
            "UNRATE",
            "GDPC1",
            "INDPRO",
            "NCBEILQ027S",
            "DGS3MO"
        ],
        "Definition": [
            "3-Month Treasury Bill Secondary Market Rate, Discount Basis",
            "Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity",
            "Moody’s AAA Corporate Bond Yield",
            "Moody’s BAA Corporate Bond Yield",
            "CSP: Corporate Spread Premium (BAA - AAA)",
            "10-Year Treasury Rate",
            "10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity",
            "Consumer Price Index for All Urban Consumers: All Items in U.S. City Average",
            "Personal Consumption Expenditures Excluding Food and Energy (Chain-Type Price Index)",
            "CBOE Volatility Index (VIX)",
            "All Employees, Total Nonfarm",
            "Unemployment Rate",
            "Real Gross Domestic Product",
            "Industrial Production: Total Index",
            "Nonfinancial Corporate Business: Corporate Equities, Liabilities",
            "3-Month Treasury Constant Maturity Rate (Investment Basis)"
        ]
    }

    # Ticker-specific Features (e.g., for ticker 'MMM')
    ticker_features = {
        "Feature": [
            "Ticker.Volume",
            "Ticker.beta",
            "Ticker.betasq",
            "Ticker.mom1m",
            "Ticker.mom6m",
            "Ticker.mom12m",
            "Ticker.dolvol",
            "Ticker.pricedelay",
            "Ticker.Ret",
            "Ticker.std"
        ],
        "Definition": [
            "Daily trading volume of the ticker",
            "Beta of the ticker (market sensitivity)",
            "Square of the beta (non-linear risk exposure)",
            "1-month price momentum",
            "6-month price momentum",
            "12-month price momentum",
            "Dollar trading volume (price × volume)",
            "Price delay (proxy for market efficiency)",
            "Daily return of the ticker",
            "Rolling standard deviation of returns (volatility)"
        ]
    }

    # Combine into a single DataFrame
    df_features = pd.concat([
        pd.DataFrame(macro_features),
        pd.DataFrame(ticker_features)
    ], ignore_index=True)
    return df_features

def data_loading(end_date='2024-12-31', ticker = 'AAPL'):
    def fetch_yahoo_data(ticker= 'AAPL', START_DATE='1990-01-01', END_DATE = end_date,
                        technicals=False):
        """
        Fetch monthly stock market data from Yahoo Finance and compute additional metrics.

        Parameters:
            ticker (str): The stock ticker symbol (e.g., 'AAPL').
            START_DATE (str): Start date for the data in 'YYYY-MM-DD' format.
            END_DATE (str): End date for the data in 'YYYY-MM-DD' format.

        Returns:
            DataFrame: A Pandas DataFrame containing the following columns:
                - date: The date of the data point.
                - {ticker}: Adjusted closing price for the ticker.
                - {ticker}.M18: 18-month rolling mean of adjusted closing price.
                - {ticker}.M36: 36-month rolling mean of adjusted closing price.
                - {ticker}.S18: 18-month rolling standard deviation.
                - {ticker}.S36: 36-month rolling standard deviation.
                - {ticker}BBU: Upper Bollinger Band (mean + 2 standard deviations).
                - {ticker}BBL: Lower Bollinger Band (mean - 2 standard deviations).
                - {ticker}.E12: 12-period exponential moving average.
                - {ticker}.E26: 26-period exponential moving average.
                - {ticker}.Ret: Logarithmic returns.
                - {ticker}.Vol: Rolling annualized volatility (252 trading days).

        Returns None if no data is available.
        """
        try:
            # Download data from Yahoo Finance
            data = yf.download(ticker, start=START_DATE, end=END_DATE, interval="1mo")            
            if not data.empty:
                # Prepare the DataFrame
                df = data.reset_index()
                df.rename(columns={"Adj Close": ticker, "Date": "date"}, inplace=True) 
                df[f'{ticker}.Ret'] = np.log(df[ticker] / df[ticker].shift(1))
                df[f'{ticker}.Vol'] = df[f'{ticker}.Ret'].rolling(12).std() * np.sqrt(12)    
                if technicals:         
                    # Compute additional metrics
                    df[f'{ticker}.M18'] = df[ticker].rolling(18).mean()
                    df[f'{ticker}.M36'] = df[ticker].rolling(36).mean()
                    df[f'{ticker}.S18'] = df[ticker].rolling(18).std()
                    df[f'{ticker}.S36'] = df[ticker].rolling(36).std()
                    df[f'{ticker}BBU'] = df[f'{ticker}.M18'] + 2 * df[f'{ticker}.S18']
                    df[f'{ticker}BBL'] = df[f'{ticker}.M18'] - 2 * df[f'{ticker}.S36']
                    df[f'{ticker}.E12'] = df[ticker].ewm(span=12, adjust=False).mean()
                    df[f'{ticker}.E26'] = df[ticker].ewm(span=26, adjust=False).mean()          
                return df
            else:
                print("No data available for the given ticker and date range.")
                return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def get_economic_data(start_date  = '1957-03-01', end_date    = '2020-12-31'):    
        fname = 'EconomicData.csv'
        file_path = os.path.join(DATA_DIR,fname)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, parse_dates=["date"])  # Load from CSV
            return df
        else:
            print("Fetching economic data from FRED...")
    def get_data(start_date='1990-01-01',end_date='2024-12-31',ticker='AAPL'):
        """
        Fetches economic data and stock data for a given ticker.
        Returns a DataFrame with the following columns:
            - date: The date of the data point.
            - {ticker}: Adjusted closing price for the ticker.
            - {ticker}.M18: 18-month rolling mean of adjusted closing price.
            - {ticker}.M36: 36-month rolling mean of adjusted closing price.
            - {ticker}.S18: 18-month rolling standard deviation.
            - {ticker}.S36: 36-month rolling standard deviation.
            - {ticker}BBU: Upper Bollinger Band (mean + 2 standard deviations).
            - {ticker}BBL: Lower Bollinger Band (mean - 2 standard deviations).
            - {ticker}.E12: 12-period exponential moving average.
            - {ticker}.E26: 26-period exponential moving average.
            - {ticker}.Ret: Logarithmic returns.
            - {ticker}.Vol: Rolling annualized volatility (252 trading days).   
        """     
        fname = f'ModellingData{ticker}.csv'
        file_path = os.path.join(DATA_DIR,fname)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, parse_dates=["date"])  # Load from CSV
            return df
        else:
            macro = get_economic_data(start_date, end_date)
            print(macro.shape)
            print('%%%%%%%%%%%%%%%%%%%%%%%')
            snp500 = fetch_yahoo_data(ticker='^GSPC',
                                    START_DATE=start_date, END_DATE=end_date)
            print(snp500.shape)   
            print(f"Fetching data for {ticker} from Yahoo Finance...")
            print('%%%%%%%%%%%%%%%%%%%%%%%')
            firm = fetch_yahoo_data(ticker=ticker,
                                    START_DATE=start_date, END_DATE=end_date,
                                    technicals=True,
                                    snp500=snp500)
            print(firm.shape)
            print('%%%%%%%%%%%%%%%%%%%%%%%')
            macro.set_index("date", inplace=True)
            firm.set_index("date", inplace=True)
            snp500.set_index("date", inplace=True)
            # Perform a join instead of concat to match dates
            result = firm.join(snp500, how="left")
            result = result.join(macro, how="left")
            print(result.head())
            # Save your DataFrame to a CSV file
            result.to_csv(file_path, index=True)
        return result
    df = get_data(ticker, end_date)
    return df


st.markdown(
    """
    <style>
    .project-title {
        font-size: 36px !important;
        font-weight: bold !important;
        text-align: center;
        color: #2E86C1;
        margin-bottom: 10px;
    }
    .project-by {
        font-size: 20px !important;
        font-style: italic;
        text-align: center;
        color: #5D6D7E;
        margin-top: 0px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <div class="project-title">Stock Price Modelling</div>
    """,
    unsafe_allow_html=True
)
st.write("""
         ## Introduction: 
         - This app trains and diploys a models to forecast stock prices using historical data.
         - It aims to reproduce the results from "Empirical Asset Pricing via Machine Learning" by Gu, Kelly, and Xiu (2020).
         - It uses a combination of macroeconomic indicators and ticker-specific features to predict future stock prices
         - The app allows users to choose between a default analysis of Apple Inc. (AAPL) or a custom ticker analysis.

         """)

st.sidebar.header('User Input Parameters')
with st.sidebar:
    st.sidebar.subheader('Choose the process you want:')
    feature_check = st.checkbox("Display Feature explanations?")
    data_check = st.checkbox("Run AAPL EDA?")
    default_analysis = st.checkbox("Run AAPL stock forecasting?")
    custom_ticker_analysis = st.checkbox("Logistic Regression?")

if feature_check:
    st.subheader("Feature Explanations")
    df_features = displayFeatures()
    st.dataframe(df_features)
if default_analysis or data_check:
    st.subheader("AAPL Stock Price Forecasting")
    df = data_loading()
    st.dataframe(df.head())
    st.write("This section will contain the analysis and forecasting results for AAPL stock prices.")
