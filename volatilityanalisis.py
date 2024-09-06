import pandas as pd
import numpy as np
import yfinance as yf
from arch import arch_model
import plotly.graph_objs as go
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from loguru import logger
import matplotlib.pyplot as plt

# Set up logger configuration
logger.add("file_{time}.log")

# List of stock tickers to analyze
tickers = ["AAPL", "MSFT", "TSLA", "JPM", "XOM"]
# Define the start and end date for fetching historical stock data
start_date = "2015-01-01"
end_date = "2023-01-01"

def fetch_stock_data(tickers, start_date, end_date):
    """
    Fetches historical stock data for a list of tickers from Yahoo Finance.

    Parameters:
    tickers (list): A list of stock ticker symbols (e.g., ['AAPL', 'MSFT']).
    start_date (str): The start date for fetching historical data in 'YYYY-MM-DD' format.
    end_date (str): The end date for fetching historical data in 'YYYY-MM-DD' format.

    Returns:
    DataFrame: A pandas DataFrame containing the historical stock data with filled missing values.
    """
    try:
        # Fetch data from Yahoo Finance for specified tickers and date range
        data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
        # Fill missing values using forward fill method
        data_clean = data.fillna(method='ffill')
        return data_clean
    except Exception as e:
        # Log any errors that occur during data fetching
        logger.error(f"Error fetching data: {str(e)}")
        return None

def adf_test(series):
    """
    Perform Augmented Dickey-Fuller test for stationarity on a time series.

    Parameters:
    series (Series): A pandas Series containing the time series data.

    Returns:
    tuple: A tuple containing the ADF statistic and p-value.
    """
    result = adfuller(series.dropna())
    return result

def prepare_stock_data(data):
    """
    Prepare stock data by checking for stationarity and applying necessary transformations.

    Parameters:
    data (DataFrame): A pandas DataFrame containing stock data.

    Returns:
    tuple: A tuple containing the prepared DataFrame and a message indicating the best transformation applied.
    """
    # Ensure the index is in datetime format
    data.index = pd.to_datetime(data.index)
    # Perform ADF test on the original adjusted close price
    result = adf_test(data['Adj Close'])
    logger.info('ADF test for Original Series:')
    logger.info(f'ADF Statistic: {result[0]}')
    logger.info(f'p-value: {result[1]}')

    # Check if the series is already stationary
    if result[1] <= 0.01:
        data['Best Transformation'] = data['Adj Close']
        logger.info('The original data is stationary, so we can continue with this for the analysis')
        return data, f"Best transformation: Original Series (p-value: {result[1]})"
    
    # If non-stationary, compute daily returns
    logger.info('The data is non-stationary, we will compute the returns to address this:')
    data['Daily Returns'] = data['Adj Close'].pct_change()
    result_daily = adf_test(data['Daily Returns'])
    logger.info('ADF test for Daily Returns:')
    logger.info(f'ADF Statistic: {result_daily[0]}')
    logger.info(f'p-value: {result_daily[1]}')
    
    # Check if daily returns are stationary
    if result_daily[1] <= 0.01:
        logger.info('The returns are stationary, so we can continue with this data for the analysis')
        data['Best Transformation'] = data['Daily Returns']
        return data, f"Best transformation: Daily Returns (p-value: {result_daily[1]})"
    
    # Try log returns and squared returns if necessary
    try:
        logger.info('Both the original data and returns are non-stationary, so we will try to compute the log and squared transformations of returns to solve the non-stationarity')
        data['Log Returns'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
        data['Squared Returns'] = data['Log Returns'] ** 2
        
        result_log = adf_test(data['Log Returns'])
        result_squared = adf_test(data['Squared Returns'])
        
        logger.info('ADF test for Log Returns:')
        logger.info(f'ADF Statistic: {result_log[0]}')
        logger.info(f'p-value: {result_log[1]}')
        
        logger.info('ADF test for Squared Returns:')
        logger.info(f'ADF Statistic: {result_squared[0]}')
        logger.info(f'p-value: {result_squared[1]}')
        
        if result_log[1] <= 0.01 or result_squared[1] <= 0.01:
            if result_log[1] < result_squared[1]:
                logger.info('The log-returns transformation solves the non-stationarity issue, we will continue with this for the analysis')
                data['Best Transformation'] = data['Log Returns']
                return data, f"Best transformation: Log Returns (p-value: {result_log[1]})"
            else:
                logger.info('The squared-returns transformation solves the non-stationarity issue, we will continue with this for the analysis')
                data['Best Transformation'] = data['Squared Returns']
                return data, f"Best transformation: Squared Returns (p-value: {result_squared[1]})"
    except Exception as e:
        logger.error(f"Error in preparing data: {str(e)}")
    
    return None, "Non-stationarity was not solved. Data is likely not suitable for GARCH or TARCH modeling."

def correlation_analysis(stock_data):
    """
    Perform correlation analysis between volatility and volume for a given stock.

    Parameters:
    stock_data (DataFrame): A pandas DataFrame containing stock data with 'Best Transformation' and 'Volume' columns.
    """
    ticker = stock_data.columns[0]
    logger.info(f"\nPerforming Correlation Analysis for {ticker}")
    
    if 'Best Transformation' not in stock_data.columns:
        logger.info(f"No suitable transformation found for {ticker}. Skipping correlation analysis.")
        return
    
    if 'Adj Close' not in stock_data.columns or 'Volume' not in stock_data.columns:
        logger.info(f"Required columns 'Adj Close' or 'Volume' not found for {ticker}. Skipping correlation analysis.")
        return
    
    try:
        # Calculate daily returns and volatility
        daily_returns = stock_data['Best Transformation'].pct_change()
        volatility = daily_returns.rolling(window=252).std() * np.sqrt(252)
        
        # Normalize volume
        volume = (stock_data['Volume'] - stock_data['Volume'].mean()) / stock_data['Volume'].std()
        
        # Combine volatility and volume into a DataFrame
        correlation_data = pd.concat([volatility, volume], axis=1)
        correlation_data.columns = ['Volatility', 'Volume']
        
        # Calculate Pearson and Spearman correlations
        correlation_pearson = correlation_data.corr(method='pearson').iloc[0, 1]
        correlation_spearman = correlation_data.corr(method='spearman').iloc[0, 1]
        
        logger.info(f"Pearson Correlation for {ticker}: {correlation_pearson}")
        logger.info(f"Spearman Correlation for {ticker}: {correlation_spearman}")
        
    except KeyError as e:
        logger.error(f"KeyError: {e}. Check if {ticker} is present in stock data.")
        
    except Exception as e:
        logger.error(f"Error occurred during correlation analysis for {ticker}: {str(e)}")

def decompose_time_series(stock_data, ticker):
    """
    Decompose a time series for visualization.

    Parameters:
    stock_data (DataFrame): A pandas DataFrame containing stock data.
    ticker (str): The stock ticker symbol.

    Returns:
    None
    """
    ts_data = stock_data['Adj Close'].dropna()
    decomposition = seasonal_decompose(ts_data, model='multiplicative', period=252)
    fig = decomposition.plot()
    fig.suptitle(f'Time Series Decomposition of {ticker}', fontsize=16)
    plt.show()

def calculate_volatility(data, window=21):
    """
    Calculate rolling volatility for a given stock.

    Parameters:
    data (DataFrame): A pandas DataFrame containing stock data with a 'Best Transformation' column.
    window (int): The rolling window size for volatility calculation.

    Returns:
    DataFrame: A DataFrame with the calculated volatility.
    """
    data['Volatility'] = data['Best Transformation'].rolling(window=window, min_periods=1).std() * np.sqrt(252)
    data['Volatility'].fillna(method='bfill', inplace=True)  # Backward fill to handle NaN values
    logger.info(f"Calculated Volatility Head:\n{data['Volatility'].head(20)}")
    return data

def fit_best_garch_tarch_models(data, max_p=5, max_q=5):
    """
    Fit GARCH and TARCH models to find the best fit for the given data.

    Parameters:
    data (DataFrame): A pandas DataFrame containing stock data with 'Best Transformation' column.
    max_p (int): Maximum order for GARCH model's p parameter.
    max_q (int): Maximum order for GARCH model's q parameter.

    Returns:
    tuple: A tuple containing the best model type, model order, and model summary.
    """
    returns = data['Best Transformation'].dropna()
    best_garch_aic = np.inf
    best_tarch_aic = np.inf
    best_garch_model = None
    best_tarch_model = None
    best_garch_fit = None
    best_tarch_fit = None

    for p in range(1, max_p + 1):
        for q in range(1, max_q + 1):
            try:
                # Fit GARCH model
                garch_model = arch_model(returns, vol='Garch', p=p, q=q, rescale=False)
                garch_fit = garch_model.fit(disp="off")
                if garch_fit.aic < best_garch_aic:
                    best_garch_aic = garch_fit.aic
                    best_garch_model = (p, q)
                    best_garch_fit = garch_fit
            except:
                continue

            try:
                # Fit TARCH model
                tarch_model = arch_model(returns, vol='Garch', p=p, q=q, o=1, rescale=False)
                tarch_fit = tarch_model.fit(disp="off")
                if tarch_fit.aic < best_tarch_aic:
                    best_tarch_aic = tarch_fit.aic
                    best_tarch_model = (p, q)
                    best_tarch_fit = tarch_fit
            except:
                continue

    if best_garch_fit and best_tarch_fit:
        if best_garch_fit.aic < best_tarch_fit.aic:
            best_model_type = "GARCH"
            best_model = best_garch_model
            best_fit = best_garch_fit
        else:
            best_model_type = "TARCH"
            best_model = best_tarch_model
            best_fit = best_tarch_fit
    elif best_garch_fit:
        best_model_type = "GARCH"
        best_model = best_garch_model
        best_fit = best_garch_fit
    elif best_tarch_fit:
        best_model_type = "TARCH"
        best_model = best_tarch_model
        best_fit = best_tarch_fit
    else:
        best_model_type = None
        best_model = None
        best_fit = None

    return best_model_type, best_model, best_fit.summary() if best_fit else None

def plot_volatility_volume(stock_data, ticker):
    """
    Plot the volatility and volume for a given stock.

    Parameters:
    stock_data (DataFrame): A pandas DataFrame containing stock data with 'Best Transformation' and 'Volume' columns.
    ticker (str): The stock ticker symbol.

    Returns:
    None
    """
    daily_returns = stock_data['Best Transformation']
    volatility = daily_returns.rolling(window=252, min_periods=1).std() * np.sqrt(252)
    volatility.fillna(method='bfill', inplace=True)

    # Normalize data for better comparison
    volatility_normalized = (volatility - volatility.mean()) / volatility.std()
    volume_normalized = (stock_data['Volume'] - stock_data['Volume'].mean()) / stock_data['Volume'].std()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=volatility.index,
        y=volatility_normalized,
        mode='lines',
        name=f'Volatility_{ticker}',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=volume_normalized.index,
        y=volume_normalized,
        mode='lines',
        name=f'Volume_{ticker}',
        yaxis='y2',
        line=dict(color='red')
    ))

    fig.update_layout(
        title=f'{ticker} Stock Volatility and Volume Analysis',
        xaxis_title='Date',
        yaxis=dict(
            title='Volatility',
            side='left'
        ),
        yaxis2=dict(
            title='Volume',
            side='right',
            overlaying='y',
            type='linear'
        )
    )

    fig.show()

# Main Execution Flow
if __name__ == "__main__":
    # Fetch and prepare stock data
    stock_data = fetch_stock_data(tickers, start_date, end_date)
    if stock_data is not None:
        stock_data.fillna(method='ffill', inplace=True)  # Forward fill to handle missing data

        # Process each stock ticker
        for stock_ticker in tickers:
            logger.info(f"Processing data for {stock_ticker}")

            if (stock_ticker, 'Adj Close') in stock_data.columns:
                stock_ticker_data = stock_data.loc[:, (stock_ticker, slice(None))]
                stock_ticker_data.columns = stock_ticker_data.columns.droplevel(0)  # Drop the ticker level to simplify column names

                result = prepare_stock_data(stock_ticker_data)

                if result is not None:
                    prepared_data, message = result
                    logger.info(message)

                    if 'Best Transformation' in prepared_data.columns:
                        logger.info("Transformations Applied:")
                        logger.info(prepared_data[['Adj Close', 'Best Transformation']].head())

                    analysis_data = calculate_volatility(prepared_data)
                    logger.info(analysis_data)

                    correlation_analysis(prepared_data)
                    
                    best_model_type, best_model, best_model_summary = fit_best_garch_tarch_models(prepared_data)  
                    logger.info(f"Best model type: {best_model_type}")
                    logger.info(f"Best model: {best_model}")
                    logger.info(best_model_summary)
                    
                    plot_volatility_volume(analysis_data, stock_ticker)
                else:
                    logger.error(f"Data preparation failed for {stock_ticker}.")
            else:
                logger.error(f"{stock_ticker} data is not present in stock_data.")
    else:
        logger.error("Failed to fetch stock data.")
