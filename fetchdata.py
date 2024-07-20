import yfinance as yf
import pandas as pd
from loguru import logger

def fetch_stock_data(tickers, start_date, end_date):
    try:
        data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
        data_clean = data.fillna(method='ffill')
        return data_clean
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        return None
