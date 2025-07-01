import streamlit as st
import pandas as pd
import numpy as np
import os
import yfinance as yf
import statsmodels.api as sm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Gets current script's directory
DATA_DIR = os.path.join(BASE_DIR, "Data", "Sources")

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

def data_loading(ticker = 'MSFT',end_date='2024-12-31'):
    ####################################
    # HELPER FUNCTIONS TO PROCESS FIRM DATA
    ####################################
    def returns(df, col,lag=1):
        """
        Computes lagged returns for a specified column in the DataFrame.

        Parameters:
            df (DataFrame): A Pandas DataFrame containing the data.
            col (str): The column name for which returns are calculated.
            lag (int): The lag period for calculating returns (default is 1).

        Returns:
            Series: A Pandas Series containing the lagged returns, calculated as:
                    (current_value / previous_value) - 1.
        """
        return df[col]/df[col].shift(lag) - 1

    def rolling_var(data1,N):
        """
        Computes the rolling variance over a specified window size.

        Parameters:
            data1 (Series): A Pandas Series containing the data.
            N (int): The rolling window size.

        Returns:
            Series: A Pandas Series with the rolling variance over the window `N`.
        """
        return data1.rolling(N).var()

    def rolling_cov(data1, data2, N):
        """
        Computes the rolling covariance between two data series over a specified window size.

        Parameters:
            data1 (Series): A Pandas Series containing the first data series.
            data2 (Series): A Pandas Series containing the second data series.
            N (int): The rolling window size.

        Returns:
            Series: A Pandas Series with the rolling covariance over the window `N`.
        """
        return data1.rolling(N).cov(data2)

    def regress_on_mkt(data, snp500, ticker, window_size=60):
        """
        Perform rolling regression to compute the price delay ratio based on market return lags.

        Parameters:
            data (DataFrame): Input data containing stock and market returns.
            ticker (str): Stock ticker for which price delay is computed.
            window_size (int): Size of the rolling window for regression.

        Returns:
            DataFrame: Original data with added columns for price delay ratios.
        """

        snpTicker = '^GSPC'
        # Check for the required column availability
        if f'{snpTicker}.Ret' not in snp500.columns:
            raise ValueError(f"Column '{snpTicker}.Ret' is missing from the input data.")
        if f'{ticker}.Ret' not in data.columns:
            raise ValueError(f"Column '{ticker}.Ret' is missing from the input data.")
        if f"{ticker}.mom1m" not in data.columns:
            raise ValueError(f"Column '{ticker}.mom1m' is missing from the input data.")
        # Ensure data is properly copied to avoid modifying the original DataFrame
        data1 = data[['date', 'MMM.Ret', 'MMM.mom1m']].copy()
        data1 = data1.merge(snp500[['date',f'{snpTicker}.Ret']], 
                            on='date', how = 'inner')
        # Add market return lags
        for lg in range(1, 5):  # Updated to include lag 4, as it's used in the 4-lag regression
            data1[f'mkt.ret.lag{lg}'] = data1[f'{snpTicker}.Ret'].shift(lg)

        # Drop rows with NaN values
        data1 = data1.dropna()
        # Initialize result storage
        price_delay_series = []
        dates = list(data1['date'])

        # Apply rolling regression
        for i in range(window_size, len(data1)):
            # Select rolling window data
            window_df = data1.iloc[i - window_size:i]
            # 1-lag regression
            X1 = sm.add_constant(window_df["mkt.ret.lag1"])
            y = window_df[f"{ticker}.mom1m"]
            model1 = sm.OLS(y, X1).fit()
            R2_1 = model1.rsquared

            # 4-lag regression
            X4 = sm.add_constant(window_df[['mkt.ret.lag1', 'mkt.ret.lag2', 
                                            'mkt.ret.lag3', 'mkt.ret.lag4']])
            model4 = sm.OLS(y, X4).fit()
            R2_4 = model4.rsquared

            # Compute price delay ratio
            price_delay_ratio = 1 - (R2_1 / R2_4) if R2_4 > 0 else None
            price_delay_series.append((dates[i], price_delay_ratio))

        # Convert results to DataFrame
        price_delay_df = pd.DataFrame(price_delay_series, 
                                    columns=["date", f"{ticker}.pricedelay"])
        # set date as datetime
        price_delay_df['date'] = pd.to_datetime(price_delay_df['date'])
        # Ensure alignment based on dates
        price_delay_df.set_index("date", inplace=True)
        data.set_index('date', inplace=True)
        # Perform a join instead of concat to match dates
        result = data.join(price_delay_df, how="left")

        # Reset index after the join
        result.reset_index(inplace=True)

        return result

    def tech_indicators(data, ticker, snp500):
        """
        Calculates various technical indicators for a given ticker and S&P500 market data.

        Parameters:
            data (DataFrame): A DataFrame containing the stock data for the ticker.
            ticker (str): The stock ticker symbol (e.g., 'AAPL').
            snp500 (DataFrame): A DataFrame containing S&P500 data, including market returns.

        Returns:
            DataFrame: A DataFrame with the following additional calculated indicators:
                - {ticker}.M18: 18-period rolling mean of the closing price.
                - {ticker}.M36: 36-period rolling mean of the closing price.
                - {ticker}.S18: 18-period rolling standard deviation of the closing price.
                - {ticker}.S36: 36-period rolling standard deviation of the closing price.
                - {ticker}.E9 or {ticker}.E18: Exponential moving averages with spans of 9 or 18.
                - {ticker}BBU: Upper Bollinger Band (18-period mean + 2 std deviations).
                - {ticker}BBL: Lower Bollinger Band (18-period mean - 2 std deviations).
                - {ticker}.direction: Directional change in the closing price.
                - {ticker}.momXm: Momentum indicators (1, 6, 12, and 36-month lags).
                - {ticker}.chmom: Change in 12-month momentum.
                - {ticker}.dolvol: Dollar volume (Closing price * Volume).
                - {ticker}.var: Rolling variance of the 1-month momentum over 36 periods.
                - {ticker}.cov: Rolling covariance of 1-month momentum with S&P500 returns.
                - {ticker}.beta: Beta (covariance divided by variance).
                - {ticker}.betasq: Beta squared.

        Notes:
            - Bollinger Bands and other indicators are calculated using 18-period metrics.
            - The function calculates beta and its square over a 3-year (36-period) rolling window.

        Example:
            df = tech_indicators(stock_data, 'AAPL', sp500_data)
        """         
        # Compute additional metrics
        df = data.copy()
        close = ticker+'.Close'
        for m in [18,36]:
            df[f'{ticker}.M{m}'] = df[close].rolling(m).mean()
            df[f'{ticker}.S{m}'] = df[close].rolling(m).std()
            m = 9 if m == 18 else 18
            df[f'{ticker}.E{m}'] = df[close].ewm(span=m, adjust=False).mean()
        df[f'{ticker}BBU'] = df[f'{ticker}.M18'] + 2 * df[f'{ticker}.S18']
        df[f'{ticker}BBL'] = df[f'{ticker}.M18'] - 2 * df[f'{ticker}.S18']
        df[f'{ticker}.direction'] = df[close]-df[close].shift(1)
        for m in [1,6,12,36]:
            df[f'{ticker}.mom{m}m'] = returns(df,close,m)
        df[f'{ticker}.chmom'] = df[f'{ticker}.mom1m']-df[f'{ticker}.mom1m'].shift(12)
        df[f'{ticker}.dolvol'] = df[close]*df[f'{ticker}.Volume']
        df[f'{ticker}.var'] = rolling_var(df[f'{ticker}.mom1m'],36) # over 3 year peiod
        df[f'{ticker}.cov'] = rolling_cov(df[f'{ticker}.mom1m'], snp500['^GSPC.Ret'],36) # over 3 year peiod
        df[f'{ticker}.beta'] = df[f'{ticker}.cov']/df[f'{ticker}.var'] # over 3 year peiod
        df[f'{ticker}.betasq'] = df[f'{ticker}.beta']*df[f'{ticker}.beta']
        # ticker_df['xs_ret'] = ticker_df['mom1m'] - macro['DGS10']
        # ticker_df = pd.concat([ticker_df,snp500], axis=1)
        return df#regress_on_mkt(df,snp500, ticker, window_size=36)
    def fetch_yahoo_data(ticker, START_DATE='1990-01-01', END_DATE = end_date,
                        snp500 = None, technicals=False):
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
            df = yf.download(ticker, start=START_DATE, end=END_DATE, interval="1mo")
            
            if not df.empty:
                # Prepare the DataFrame
                close = ticker+'.Close'
                df.columns = df.columns.droplevel('Ticker')
                new_colz = {old_col: ticker+'.'+old_col for old_col in list(df.columns)}
                df = df.rename(columns = new_colz)
                df =df.reset_index()
                df.rename(columns={"Date": "date"}, inplace=True) 
                df[f'{ticker}.Ret'] = np.log(df[close] / df[close].shift(1))
                df[f'{ticker}.Vol'] = df[f'{ticker}.Ret'].rolling(12).std() * np.sqrt(12) 
                if technicals:
                    df = tech_indicators(data=df,ticker=ticker,snp500=snp500)   
                return df
            else:
                print("No data available for the given ticker and date range.")
                return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def get_economic_data(start_date  = '1957-03-01', end_date    = '2020-12-31'):            
        file_path = os.path.join(DATA_DIR, "EconomicData.csv")
        print(f"File path: {file_path}")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, parse_dates=["date"])  # Load from CSV
            return df
        else:
            print("Fetching economic data from FRED...")

    def get_data(ticker, start_date='1990-01-01',end_date='2024-12-31'):
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
            macro = get_economic_data(start_date=start_date, end_date=end_date)
            snp500 = fetch_yahoo_data(ticker='^GSPC',
                                        START_DATE=start_date, END_DATE=end_date)
            firm = fetch_yahoo_data(ticker=ticker,
                                    START_DATE=start_date, END_DATE=end_date,
                                    snp500=snp500,
                                    technicals=True)
            macro.set_index("date", inplace=True)
            firm.set_index("date", inplace=True)
            snp500.set_index("date", inplace=True)
            # Make sure date indexes are datetime
            macro.index = pd.to_datetime(macro.index)
            firm.index = pd.to_datetime(firm.index)
            snp500.index = pd.to_datetime(snp500.index)

            # Add 'YearMonth' column to each for join
            macro["YearMonth"] = macro.index.to_period("M")
            firm["YearMonth"] = firm.index.to_period("M")
            snp500["YearMonth"] = snp500.index.to_period("M")

            # Merge firm and snp500 on YearMonth
            result = pd.merge(firm, snp500, on="YearMonth", how="left", suffixes=("", "_snp"))
            # Merge in macro data
            result = pd.merge(result, macro, on="YearMonth", how="left", suffixes=("", "_macro"))
            result.set_index("YearMonth", inplace=True)
            # Convert index to PeriodIndex if needed
            result["date"] = result.index.to_timestamp()  # If using PeriodIndex
            # Ensure 'date' is in datetime format
            result["date"] = pd.to_datetime(result["date"])
            # Reorder columns to have 'date' first  
            result.set_index("date", inplace=True)
            # Drop the YearMonth column if not needed   
            result.drop(columns=["YearMonth"], inplace=True, errors='ignore')
            # Save the result to a CSV file
            print(result.head())
            # Save your DataFrame to a CSV file
            result.to_csv(file_path, index=True)
        return result
    df = get_data(ticker=ticker, start_date='1990-01-01',end_date=end_date)
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
    data_check = st.checkbox("Run MSFT EDA?")
    default_analysis = st.checkbox("Run MSFT stock forecasting?")
    custom_ticker_analysis = st.checkbox("Logistic Regression?")

if feature_check:
    st.subheader("Feature Explanations")
    df_features = displayFeatures()
    st.dataframe(df_features)
if default_analysis or data_check:
    st.subheader("Microsoft Stock Price Forecasting")
    df = data_loading(ticker = 'MSFT')
    st.dataframe(df.head())
    st.write("This section will contain the analysis and forecasting results for AAPL stock prices.")
