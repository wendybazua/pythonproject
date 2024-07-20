import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from loguru import logger

def adf_test(series):
    result = adfuller(series.dropna())
    return result

def prepare_stock_data(data):
    data.index = pd.to_datetime(data.index)
    result = adf_test(data['Adj Close'])
    logger.info('ADF test for Original Series:')
    logger.info(f'ADF Statistic: {result[0]}')
    logger.info(f'p-value: {result[1]}')

    if result[1] <= 0.01:
        data['Best Transformation'] = data['Adj Close']
        logger.info('The original data is stationary, so we can continue with this for the analysis')
        return data, f"Best transformation: Original Series (p-value: {result[1]})"
    
    logger.info('The data is non-stationary, we will compute the returns to address this:')
    
    data['Daily Returns'] = data['Adj Close'].pct_change()
    result_daily = adf_test(data['Daily Returns'])
    logger.info('ADF test for Daily Returns:')
    logger.info(f'ADF Statistic: {result_daily[0]}')
    logger.info(f'p-value: {result_daily[1]}')
    
    if result_daily[1] <= 0.01:
        logger.info('The returns are stationary, so we can continue with this data for the analysis')
        data['Best Transformation'] = data['Daily Returns']
        return data, f"Best transformation: Daily Returns (p-value: {result_daily[1]})"
    
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
    ticker = stock_data.columns[0]
    logger.info(f"\nPerforming Correlation Analysis for {ticker}")
    
    if 'Best Transformation' not in stock_data.columns:
        logger.info(f"No suitable transformation found for {ticker}. Skipping correlation analysis.")
        return
    
    if 'Adj Close' not in stock_data.columns or 'Volume' not in stock_data.columns:
        logger.info(f"Required columns 'Adj Close' or 'Volume' not found for {ticker}. Skipping correlation analysis.")
        return
    
    try:
        daily_returns = stock_data['Best Transformation'].pct_change()
        volatility = daily_returns.rolling(window=252).std() * np.sqrt(252)
        
        volume = (stock_data['Volume'] - stock_data['Volume'].mean()) / stock_data['Volume'].std()
        
        correlation_data = pd.concat([volatility, volume], axis=1)
        correlation_data.columns = ['Volatility', 'Volume']
        
        correlation_pearson = correlation_data.corr(method='pearson').iloc[0, 1]
        correlation_spearman = correlation_data.corr(method='spearman').iloc[0, 1]
        
        logger.info(f"Pearson Correlation for {ticker}: {correlation_pearson}")
        if correlation_pearson <= 1 and correlation_pearson >= -1:
            if correlation_pearson > 0:
                if correlation_pearson == 1:
                    logger.info('There is a perfect positive linear relationship. As one variable increases, the other variable increases in a perfectly linear manner.')
                elif 0.5 < correlation_pearson < 1:
                    logger.info('Moderate to strong positive linear relationship between the volatility and volume for the stock. There might be some strong degree of linear association.') 
                else:
                    logger.info('Weak to moderate positive linear relationship between the volatility and volume for the stock. While there is some degree of linear association, it is not very strong.')    
            else:
                if correlation_pearson == -1:
                    logger.info('There is a perfect negative linear relationship. As one variable increases, the other variable decreases in a perfectly linear manner.')
                elif -0.5 < correlation_pearson < 1:
                    logger.info('Weak to moderate negative linear relationship between the volatility and volume for the stock. There might be some degree of linear association.') 
                else:
                    logger.info('Moderate to strong negative linear relationship between the volatility and volume for the stock. While there is some degree of linear association, it is not very strong.')    
        else:
            logger.error('Error in calculation of Pearson correlation.')

        logger.info(f"Spearman Correlation for {ticker}: {correlation_spearman}")
        if -1 <= correlation_spearman <= 1:
            if correlation_spearman > 0:
                if correlation_spearman == 1:
                    logger.info('Perfect positive monotonic relationship. As one variable increases, the other variable consistently increases.')
                elif 0.5 < correlation_spearman < 1:
                    logger.info('Moderate to strong positive monotonic relationship between the volatility and volume. It suggests that as one of the variables increases, the other variable tends to increase as well, but not necessarily in a perfectly linear fashion.') 
                else:
                    logger.info('Weak to moderate positive monotonic relationship between the volatility and volume. It suggests that as one of the variables increases, the other variable tends to increase as well, but not necessarily in a perfectly linear fashion.')    
            else:
                if correlation_spearman == -1:
                    logger.info('Perfect negative monotonic relationship. As one variable increases, the other variable consistently decreases.')
                elif -0.5 < correlation_spearman < 1:
                    logger.info('Weak to moderate negative monotonic relationship between the volatility and volume. It suggests that as one of the variables increases, the other variable tends to decrease as well, but not necessarily in a perfectly linear fashion.') 
                else:
                    logger.info('Moderate to strong negative monotonic relationship between the volatility and volume. It suggests that as one of the variables increases, the other variable tends to decrease as well, but not necessarily in a perfectly linear fashion.')    
        else:
            logger.error('Error in calculation of Spearman correlation.')
            
    except KeyError as e:
        logger.error(f"KeyError: {e}. Check if {ticker} is present in stock data.")
        
    except Exception as e:
        logger.error(f"Error occurred during correlation analysis for {ticker}: {str(e)}")

def decompose_time_series(stock_data, ticker):
    ts_data = stock_data['Adj Close'].dropna()
    decomposition = seasonal_decompose(ts_data, model='multiplicative', period=252)
    fig = decomposition.plot()
    fig.suptitle(f'Time Series Decomposition of {ticker}', fontsize=16)
    plt.show()
