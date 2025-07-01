import pandas as pd
import numpy as np
import requests
import fredapi
import os
import yfinance as yf
from bs4 import BeautifulSoup
from my_key import Fred_API_key


DATA_DIR = "./Data/Sources/"

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
           'SP500',     # 09 SnP500
           'VIXCLS',    # 10 CBOE Volatility Index: VIX
           'PAYEMS',    # 11 All Employees, Total Nonfarm
           'UNRATE',    # 12 Unemployment Rate
           'GDPC1',     # 13 Real Gross Domestic Product
            'INDPRO',    # 14 Industrial Production: Total Index
            'NCBEILQ027S', # 15 Nonfinancial Corporate Business; Corporate Equities; Liability, Level
            'DGS3MO',      # 16 Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity, Quoted on an Investment Basis
           ]
quarterly_list = ['GDPC1']
snp500_tickers = ['MMM', 'AOS', 'ABT', 'ABBV', 'ACN', 'ADBE', 'AMD', 'AES', 'AFL', 
               'A', 'APD', 'ABNB', 'AKAM', 'ALB', 'ARE', 'ALGN', 'ALLE', 'LNT', 
               'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AMCR', 'AEE', 'AEP', 'AXP', 
               'AIG', 'AMT', 'AWK', 'AMP', 'AME', 'AMGN', 'APH', 'ADI', 'ANSS', 
               'AON', 'APA', 'APO', 'AAPL', 'AMAT', 'APTV', 'ACGL', 'ADM', 'ANET', 
               'AJG', 'AIZ', 'T', 'ATO', 'ADSK', 'ADP', 'AZO', 'AVB', 'AVY', 'AXON', 
               'BKR', 'BALL', 'BAC', 'BAX', 'BDX', 'BRK.B', 'BBY', 'TECH', 'BIIB', 
               'BLK', 'BX', 'BK', 'BA', 'BKNG', 'BWA', 'BSX', 'BMY', 'AVGO', 'BR', 
               'BRO', 'BF.B', 'BLDR', 'BG', 'BXP', 'CHRW', 'CDNS', 'CZR', 'CPT', 'CPB', 
               'COF', 'CAH', 'KMX', 'CCL', 'CARR', 'CAT', 'CBOE', 'CBRE', 'CDW', 'CE', 
               'COR', 'CNC', 'CNP', 'CF', 'CRL', 'SCHW', 'CHTR', 'CVX', 'CMG', 'CB', 
               'CHD', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 'CLX', 'CME', 'CMS', 
               'KO', 'CTSH', 'CL', 'CMCSA', 'CAG', 'COP', 'ED', 'STZ', 'CEG', 'COO', 
               'CPRT', 'GLW', 'CPAY', 'CTVA', 'CSGP', 'COST', 'CTRA', 'CRWD', 'CCI', 
               'CSX', 'CMI', 'CVS', 'DHR', 'DRI', 'DVA', 'DAY', 'DECK', 'DE', 'DELL', 
               'DAL', 'DVN', 'DXCM', 'FANG', 'DLR', 'DFS', 'DG', 'DLTR', 'D', 'DPZ', 
               'DOV', 'DOW', 'DHI', 'DTE', 'DUK', 'DD', 'EMN', 'ETN', 'EBAY', 'ECL', 
               'EIX', 'EW', 'EA', 'ELV', 'EMR', 'ENPH', 'ETR', 'EOG', 'EPAM', 'EQT', 
               'EFX', 'EQIX', 'EQR', 'ERIE', 'ESS', 'EL', 'EG', 'EVRG', 'ES', 'EXC', 
               'EXPE', 'EXPD', 'EXR', 'XOM', 'FFIV', 'FDS', 'FICO', 'FAST', 'FRT', 'FDX', 
               'FIS', 'FITB', 'FSLR', 'FE', 'FI', 'FMC', 'F', 'FTNT', 'FTV', 'FOXA', 'FOX', 
               'BEN', 'FCX', 'GRMN', 'IT', 'GE', 'GEHC', 'GEV', 'GEN', 'GNRC', 'GD', 
               'GIS', 'GM', 'GPC', 'GILD', 'GPN', 'GL', 'GDDY', 'GS', 'HAL', 'HIG', 
               'HAS', 'HCA', 'DOC', 'HSIC', 'HSY', 'HES', 'HPE', 'HLT', 'HOLX', 'HD', 
               'HON', 'HRL', 'HST', 'HWM', 'HPQ', 'HUBB', 'HUM', 'HBAN', 'HII', 'IBM', 
               'IEX', 'IDXX', 'ITW', 'INCY', 'IR', 'PODD', 'INTC', 'ICE', 'IFF', 'IP', 
               'IPG', 'INTU', 'ISRG', 'IVZ', 'INVH', 'IQV', 'IRM', 'JBHT', 'JBL', 'JKHY', 
               'J', 'JNJ', 'JCI', 'JPM', 'JNPR', 'K', 'KVUE', 'KDP', 'KEY', 'KEYS', 'KMB', 
               'KIM', 'KMI', 'KKR', 'KLAC', 'KHC', 'KR', 'LHX', 'LH', 'LRCX', 'LW', 'LVS', 
               'LDOS', 'LEN', 'LII', 'LLY', 'LIN', 'LYV', 'LKQ', 'LMT', 'L', 'LOW', 'LULU', 
               'LYB', 'MTB', 'MPC', 'MKTX', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MTCH', 'MKC', 
               'MCD', 'MCK', 'MDT', 'MRK', 'META', 'MET', 'MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 
               'MAA', 'MRNA', 'MHK', 'MOH', 'TAP', 'MDLZ', 'MPWR', 'MNST', 'MCO', 'MS', 'MOS', 
               'MSI', 'MSCI', 'NDAQ', 'NTAP', 'NFLX', 'NEM', 'NWSA', 'NWS', 'NEE', 'NKE', 
               'NI', 'NDSN', 'NSC', 'NTRS', 'NOC', 'NCLH', 'NRG', 'NUE', 'NVDA', 'NVR', 
               'NXPI', 'ORLY', 'OXY', 'ODFL', 'OMC', 'ON', 'OKE', 'ORCL', 'OTIS', 'PCAR', 
               'PKG', 'PLTR', 'PANW', 'PARA', 'PH', 'PAYX', 'PAYC', 'PYPL', 'PNR', 'PEP', 
               'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PNC', 'POOL', 'PPG', 'PPL', 'PFG', 'PG', 
               'PGR', 'PLD', 'PRU', 'PEG', 'PTC', 'PSA', 'PHM', 'PWR', 'QCOM', 'DGX', 'RL', 
               'RJF', 'RTX', 'O', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RVTY', 'ROK', 'ROL', 
               'ROP', 'ROST', 'RCL', 'SPGI', 'CRM', 'SBAC', 'SLB', 'STX', 'SRE', 'NOW', 
               'SHW', 'SPG', 'SWKS', 'SJM', 'SW', 'SNA', 'SOLV', 'SO', 'LUV', 'SWK', 'SBUX', 
               'STT', 'STLD', 'STE', 'SYK', 'SMCI', 'SYF', 'SNPS', 'SYY', 'TMUS', 'TROW', 
               'TTWO', 'TPR', 'TRGP', 'TGT', 'TEL', 'TDY', 'TFX', 'TER', 'TSLA', 'TXN', 
               'TPL', 'TXT', 'TMO', 'TJX', 'TSCO', 'TT', 'TDG', 'TRV', 'TRMB', 'TFC', 'TYL', 
               'TSN', 'USB', 'UBER', 'UDR', 'ULTA', 'UNP', 'UAL', 'UPS', 'URI', 'UNH', 'UHS', 
               'VLO', 'VTR', 'VLTO', 'VRSN', 'VRSK', 'VZ', 'VRTX', 'VTRS', 'VICI', 'V', 
               'VST', 'VMC', 'WRB', 'GWW', 'WAB', 'WBA', 'WMT', 'DIS', 'WBD', 'WM', 'WAT', 
               'WEC', 'WFC', 'WELL', 'WST', 'WDC', 'WY', 'WMB', 'WTW', 'WDAY', 'WYNN', 
               'XEL', 'XYL', 'YUM', 'ZBRA', 'ZBH', 'ZTS']
def extend_quarterly_to_monthly(df):
    """
    Extends quarterly data to monthly using forward-fill.
    """
    all_months = pd.date_range(df['date'].min(), df['date'].max(), freq='MS')  # Monthly start dates
    df = df.set_index("date").reindex(all_months).ffill().reset_index()
    df.rename(columns={"index": "date"}, inplace=True)
    return df

def get_FredData0(series_id, api_key, start= None,end=None, data_freq = 'm'):
    fred = fredapi.Fred(api_key)
    df = fred.get_series(series_id)
    df["date"] = df.index.strftime("%Y%m").astype(int)
    return df

def get_FredData(series_id, api_key, start,end, data_freq = 'm'):
    """
    Fetches data from a local CSV file if available, otherwise downloads from FRED API and saves it.
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
        print('Lets download it...')
        data = response.json()
        df = pd.DataFrame(data['observations'])
        df['date'] = pd.to_datetime(df['date'])
        #df.value = df.value.astype('float')
        df = df[['date','value']]
        #df.to_csv(file_path, index=False)
        #print(df.head())  # Print retrieved data
    else:
        print(f"Error {response.status_code}: {response.text}")  # Print error message            
    if series_id in quarterly_list:
        df = extend_quarterly_to_monthly(df)
    return df

def get_economic_data(start_date  = '1957-03-01', end_date    = '2020-12-31'):    
    fname = 'EconomicData.csv'
    file_path = os.path.join(DATA_DIR,fname)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, parse_dates=["date"])  # Load from CSV
        return df
    else:    
        from my_key import Fred_API_key
        API_KEY = Fred_API_key()
        df = None
        for series_id in series_tickers:
            temp = get_FredData(series_id,API_KEY,start_date,end_date)
            if temp is not None:
                temp = temp.rename(columns={'value': series_id})
                if df is None:
                    df = temp.copy()
                else:
                    df = df.merge(temp, on='date', how = 'inner')
    
        #   Computes derived indicators like credit spread."""
        if 'BAA' in df.columns and 'AAA' in df.columns:
            df['csp'] = df['BAA'] - df['AAA']  # 15 Credit Spread
        df.to_csv(file_path,index=False)
    print(df.head())
    return df

def fetch_yahoo_data(ticker= 'AAPL', START_DATE='1957-03-01', END_DATE = '1960-12-31',
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


def get_sp500_tickers():
    """
    Fetches the list of S&P 500 tickers from Wikipedia.
    Returns: list of tickers
    """
    res = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    soup = BeautifulSoup(res.content,'html')
    table = soup.find_all('table')[0] 
    df = pd.read_html(str(table))
    tickers = list(df[0].Symbol)
    return tickers

def get_data(start='1990-01-01',end='2024-12-31',ticker='AAPL'):
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

if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)
    start_date  = '1957-03-01'
    end_date    = '1960-12-31'
    ticker = 'STX'
    macro_data = get_economic_data()
   
    if True:
        df = get_FredData0(series_id='DGS3MO',
                           api_key=Fred_API_key(),
                           start=start_date,
                           end=end_date)
        print(df.head())

