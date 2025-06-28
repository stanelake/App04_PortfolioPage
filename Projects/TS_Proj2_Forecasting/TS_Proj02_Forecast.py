import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import fredapi
import os
import statsmodels.api as sm

from my_key import Fred_API_key

DATA_DIR = "./Data"

os.makedirs(DATA_DIR, exist_ok=True)  # Ensure directory exists
series_tickers = [
            'TB3MS',    # 01 3-Month Treasury Bill Secondary Market Rate, Discount Basis
           'DGS10',     # 02 Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity           
            'AAA',      # 03 'Moody’s AAA Corporate Bond Yield',
            'BAA',      # 04 'Moody’s BAA Corporate Bond Yield',
            'GS10',     # 05 '10-Year Treasury Rate',
           'T10Y2Y',    # 06 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity
           'CPIAUCSL',  # 07 Consumer Price Index for All Urban Consumers: All Items in U.S. City Average
           'PCEPILFE',  # 08 Personal Consumption Expenditures Excluding Food and Energy (Chain-Type Price Index)
           'VIXCLS',    # 10 CBOE Volatility Index: VIX
           'PAYEMS',    # 11 All Employees, Total Nonfarm
           'UNRATE',    # 12 Unemployment Rate
           'GDPC1',     # 13 Real Gross Domestic Product
            'INDPRO',    # 14 Industrial Production: Total Index
            'NCBEILQ027S', # 15 Nonfinancial Corporate Business; Corporate Equities; Liability, Level
            'DGS3MO',      # 16 Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity, Quoted on an Investment Basis
           ]
quarterly_list = ['GDPC1', 'INDPRO', 'NCBEILQ027S']

####################################
# FETCH MACRO DATA FROM FRED DATABASE
####################################
def extend_quarterly_to_monthly(df):
    """
    Extends quarterly data to monthly using forward-fill.

    Parameters:
        df (DataFrame): A Pandas DataFrame containing quarterly data. 
                        It must have a column named 'date' with datetime values.

    Returns:
        DataFrame: A DataFrame with monthly data, where the values from the
                   quarterly data are forward-filled to fill missing months.

    Example:
        If quarterly data includes rows for March, June, September, and December,
        this function will fill intermediate months (e.g., April and May) with
        the March data values.
    """
    # Ensure 'date' column is of datetime type
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Drop rows with invalid or missing 'date' values
    df = df.dropna(subset=['date'])

    # Generate a complete range of monthly start dates
    all_months = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='MS')

    # Reindex to include all months and forward-fill missing data
    df = (
        df.set_index("date")
        .reindex(all_months)
        .ffill()
        .infer_objects()
        .reset_index()
    )

    # Rename the index column to 'date'
    df.rename(columns={"index": "date"}, inplace=True)

    return df
def get_FredData(series_id, api_key, start,end, data_freq = 'm'):
    """
    Fetches data from the FRED (Federal Reserve Economic Data) API or a local CSV file.

    Parameters:
        series_id (str): The FRED series identifier (e.g., 'GDP' for Gross Domestic Product).
        api_key (str): Your API key for accessing the FRED API.
        start (str): The start date for the data in 'YYYY-MM-DD' format.
        end (str): The end date for the data in 'YYYY-MM-DD' format.
        data_freq (str): The frequency of data retrieval ('m' for monthly, 'q' for quarterly).
                         Defaults to 'm'.

    Returns:
        DataFrame: A DataFrame with the requested data containing two columns:
                   - 'date': The observation dates as datetime values.
                   - 'value': The series data for the corresponding dates.

                   If the series is quarterly, the data is extended to monthly using forward-fill.

    Notes:
        - If the request to the FRED API is successful, the data is fetched and converted into
          a Pandas DataFrame.
        - If the series is defined as quarterly, it will be transformed into monthly data.
        - Errors during API requests are printed to the console.

    Example:
        data = get_FredData("GDP", "your_api_key", "1950-01-01", "2020-12-31")
    # https://www.youtube.com/watch?v=M_jswxN3iwI&t=18s
    """
    base_url    = 'https://api.stlouisfed.org/fred/series/observations'
    series_id   = series_id
    if series_id in quarterly_list:
        data_freq = 'q'
    obs_params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json',
        'observation_start': start,
        'observation_end': end,
        'frequency': data_freq
        }
    # make request
    response = requests.get(base_url,
                            params=obs_params)
    # Check response
    df = None
    if response.status_code == 200:
        print(f'Lets download {series_id}')
        data = response.json()
        df = pd.DataFrame(data['observations'])
        #df['date'] = pd.to_datetime(df['date'])
        #df.value = df.value.astype('float')
        df = df[['date','value']]
        #df.to_csv(file_path, index=False)
        #print(df.head())  # Print retrieved data        
        if series_id in quarterly_list:
            df = extend_quarterly_to_monthly(df)            
    else:
        print(f"Error {response.status_code}: {response.text}")  # Print error message            
    if series_id in quarterly_list:
        df = extend_quarterly_to_monthly(df)
    return df

def get_economic_data(start_date  = '1957-03-01', end_date = '2020-12-31'):
    """
    Retrieves economic data for a specified date range either from a local CSV file
    or by downloading it from the FRED API.

    Returns:
        DataFrame: A DataFrame containing economic data for multiple series. 
                   Derived indicators like the credit spread ('csp') are also computed.

    Notes:
        - If a local file named 'EconomicData.csv' exists in the data directory, 
          the data is loaded from this file.
        - If the file does not exist, the function fetches data from the FRED API
          for a predefined list of series identifiers (`series_tickers`) and merges the results.

    Additional Computations:
        - The function computes the credit spread ('csp') if both 'BAA' and 'AAA' series
          are present in the data. The credit spread is calculated as:
            csp = BAA - AAA

    Example:
        df = get_economic_data()

    Dependencies:
        - Requires a valid FRED API key via the `Fred_API_key()` function.
        - Requires the list of series identifiers in `series_tickers`.
        - Requires a defined `DATA_DIR` for file operations.
    """
    fname = 'EconomicData.csv'
    file_path = os.path.join(DATA_DIR,fname)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, parse_dates=["date"])  # Load from CSV
        return df
    else:    
        API_KEY = Fred_API_key()
        df = None
        for series_id in series_tickers:
            temp = get_FredData(series_id,API_KEY,start_date,end_date)
            if temp is not None:
                temp = temp.rename(columns={'value': series_id})
                print(temp.shape)
                print('%%%%%%%%%%%%%%%%%%%%%%%')
                temp['date'] = pd.to_datetime(temp['date'], errors='coerce')
                if df is None:
                    df = temp
                else:
                    df = df.merge(temp, on='date', how = 'inner')
    
    #   Computes derived indicators like credit spread."""
    if (('BAA' in df.columns) and ('AAA' in df.columns)):
        df['BAA'] = df['BAA'].astype(float) 
        df['AAA'] = df['AAA'].astype(float)
        df['csp'] = df['BAA'] - df['AAA']  # 15 Credit Spread
    return df

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
    data1 = data[['date', f'{ticker}.Ret', f'{ticker}.mom1m']].copy()
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
    return regress_on_mkt(df,snp500, ticker, window_size=36)

####################################
# FETCH FIRM DATA FROM YFINANCE DATABASE
####################################
def fetch_yahoo_data(ticker= 'MMM', START_DATE='2000-03-01', END_DATE = '2015-12-31',
                     technicals=False, snp500=None):
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

def compile_data(ticker='MMM', start_date='2000-03-01', end_date='2015-12-31'):
    """
    Compiles data from multiple sources (firm-level data, market data, and macroeconomic data)
    into a single DataFrame, aligned by date.

    Parameters:
        ticker (str): The stock ticker symbol for the firm-level data.
                      Defaults to 'MMM' (3M Company).
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
                          Defaults to '2000-03-01'.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format.
                        Defaults to '2015-12-31'.

    Returns:
        DataFrame: A merged DataFrame containing firm-level data, S&P500 market data, 
                   and macroeconomic data, aligned by date.

                   The DataFrame includes:
                   - Firm-level data with calculated technical indicators (if enabled).
                   - S&P500 market data.
                   - Macroeconomic indicators.

    Notes:
        - Firm-level data is fetched using the `fetch_yahoo_data()` function.
        - Market data is fetched for the S&P500 (`^GSPC`) using `fetch_yahoo_data()`.
        - Macroeconomic data is retrieved using `get_economic_data()`.
        - All datasets are merged based on their 'date' columns using a left join to preserve
          firm-level data as the base.

    Example:
        result = compile_data(ticker='AAPL', start_date='2010-01-01', end_date='2020-12-31')

    Output:
        Prints the first five rows of the resulting merged DataFrame to the console.
    """
    fname = 'ModellingData.csv'
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

if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)
    start_date  = '1990-03-01'
    end_date    = '1999-12-31'
    ticker      = 'MMM'
    data_df = compile_data(ticker= 'MMM',
                           start_date='2000-03-01', end_date = '2015-12-31')
    colz = list(data_df.columns)
    print(len(colz))
    print('%%%%%%%%%%%%%%%%%%%%%%%')
    print(colz)

    
        

        