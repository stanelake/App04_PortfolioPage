# README

## Overview
This project fetches and processes macroeconomic data from the Federal Reserve Economic Data (FRED) database using the `fredapi` and other financial indicators. The dataset includes multiple economic indicators such as interest rates, unemployment rates, GDP, and stock market volatility. The data is then transformed and extended for further financial analysis.

## Features
- Fetches economic data from FRED using an API key
- Extends quarterly data to monthly frequency
- Computes derived financial indicators such as credit spreads
- Provides helper functions for time-series data processing, including:
  - Rolling variance and covariance calculations
  - Stock return calculations
  - Regression on market returns
  - Technical indicators for stock analysis

## Installation
To run this project, install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Usage
### 1. Fetching Economic Data
The `get_economic_data()` function retrieves economic indicators from FRED. It either loads existing data from a local CSV file or downloads fresh data from the API if the file is not available.

```python
from my_key import Fred_API_key
data = get_economic_data()
```

### 2. Fetching Individual Series Data
You can fetch individual series from FRED using `get_FredData`:

```python
data = get_FredData('GDP', Fred_API_key(), '1950-01-01', '2020-12-31')
```

### 3. Extending Quarterly Data to Monthly
```python
monthly_data = extend_quarterly_to_monthly(quarterly_data)
```

### 4. Computing Stock Return
```python
stock_data['returns'] = returns(stock_data, 'Close')
```

### 5. Running Regression on Market Data
```python
regressed_data = regress_on_mkt(stock_data, snp500_data, 'AAPL')
```

## Dependencies
- Python 3.x
- `yfinance` for stock market data
- `fredapi` for fetching FRED economic indicators
- `pandas` and `numpy` for data manipulation
- `requests` for API calls
- `statsmodels` for statistical analysis

## File Structure
- `main.py`: Main script to fetch and process data
- `my_key.py`: Stores API key for FRED
- `Data/Sources/`: Directory to store retrieved data files

## Notes
- Ensure you have a valid FRED API key and store it in `my_key.py`.
- The script automatically creates necessary directories if they don’t exist.
- Some transformations, such as extending quarterly data to monthly, may affect data accuracy.

## License
This project is licensed under the MIT License.

