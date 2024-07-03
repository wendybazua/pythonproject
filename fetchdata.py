
import yfinance as yf
import pandas as pd


def fetch_stock_data(tickers, start_date, end_date):
    try:
        data = yf.download(tickers, start=start_date, end=end_date) #this would make it work for only 1 at the time, no more than 1 ticker
        data_clean = data.fillna(method='ffill') #it should be more specific/maybe use another method
        return data_clean
    except Exception as e: 
        print(f"Error fetching data: {str(e)}") 
        return None #not return none
    #we want to make sure to catch specific errors
    except Exception as e: 
        print(f"Error fetching data: {str(e)}")
        return None


