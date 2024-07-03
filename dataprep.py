import pandas as pd
setattr(pd, "Int64Index", pd.Index)
setattr(pd, "Float64Index", pd.Index)
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import norm

def adf_test(series):
    result = adfuller(series.dropna()) #not necessary to make it a function
    return result

def prepare_stock_data(data):
    data.index = pd.to_datetime(data.index)
    #check if the function requires no missing data -> check with assert data and asking for a false
    result = adf_test(data['Adj Close']) #change funct to adfuller
    print('ADF test for Original Series:')
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])

    #****** We should use returns even if the original data is stationary
    if result[1] <= 0.01:   
        data['Best Transformation'] = data['Adj Close'] 
        print('The original data is stationary so we can continue with this for the analysis')
        return data, f"Best transformation: Original Series (p-value: {result[1]})"
    
    print('*The data is non-stationary, we will compute the returns to adress this:')
    
    data['Daily Returns'] = data['Adj Close'].pct_change()
    result_daily = adf_test(data['Daily Returns'])
    print('ADF test for Daily Returns:')
    print('ADF Statistic: %f' % result_daily[0])
    print('p-value: %f' % result_daily[1])
 
    
    if result_daily[1] <= 0.01:  
        print('*The returns are stationary so we can continue with this data for the analysis')
        data['Best Transformation'] = data['Daily Returns']
        return data, f"Best transformation: Daily Returns (p-value: {result_daily[1]})"
    
    try:
        print('*Both the original data and returns are non-stationaty, so we will try to compute the log and squared transformations of returns to try and solve the non-stationarity')
        data['Log Returns'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
        data['Squared Returns'] = data['Log Returns'] ** 2
        
        result_log = adf_test(data['Log Returns'])
        result_squared = adf_test(data['Squared Returns'])
        
        print('ADF test for Log Returns:')
        print('ADF Statistic: %f' % result_log[0])
        print('p-value: %f' % result_log[1])
        
        
        print('ADF test for Squared Returns:')
        print('ADF Statistic: %f' % result_squared[0])
        print('p-value: %f' % result_squared[1])
        
        if result_log[1] <= 0.01 or result_squared[1] <= 0.01: 
            if result_log[1] < result_squared[1]:
                print('*The log-returns transformation solves the non-stationarity issue, we will continue with this for the analysis')
                data['Best Transformation'] = data['Log Returns']
                return data, f"Best transformation: Log Returns (p-value: {result_log[1]})"
            else:
                print('*The squares-returns transformation solves the non-stationarity issue, we will continue with this for the analysis')
                data['Best Transformation'] = data['Squared Returns']
                return data, f"Best transformation: Squared Returns (p-value: {result_squared[1]})"
    except Exception as e: #also not catching the error 
        print(f"Error in preparing data: {str(e)}")
    
    return None, "*Non stationarity was not solved. Data is likely not suitable for GARCH or TARCH modeling. "



def correlation_analysis(stock_data):
    ticker = stock_data.columns[0][1]  #better to use the column name instead of the #'s
    
    print(f"\nPerforming Correlation Analysis for {ticker}")
    
    if 'Best Transformation' not in stock_data.columns:
        print(f"No suitable transformation found for {ticker}. Skipping correlation analysis.")
        return
    
    if 'Adj Close' not in stock_data.columns or 'Volume' not in stock_data.columns:
        print(f"Required columns 'Adj Close' or 'Volume' not found for {ticker}. Skipping correlation analysis.")
        return
    
    try:
        daily_returns = stock_data['Best Transformation'].pct_change() 
        #*****HERE I DONT KNOW IF DOING DAILY RETURNS OF THE BEST TRANSFORMATION IS CORRECT???
        volatility = daily_returns.rolling(window=252).std() * np.sqrt(252)
        
        volume = (stock_data['Volume'] - stock_data['Volume'].mean()) / stock_data['Volume'].std()
        
        correlation_data = pd.concat([volatility, volume], axis=1)
        correlation_data.columns = ['Volatility', 'Volume']
        
        correlation_pearson = correlation_data.corr(method='pearson').iloc[0, 1]
        correlation_spearman = correlation_data.corr(method='spearman').iloc[0, 1]
        
        #better the plot, not necesarily the values 
        print(f"Pearson Correlation for {ticker}: {correlation_pearson}")
        if correlation_pearson <=1 and correlation_pearson >=-1:
            if correlation_pearson>0:
                if correlation_pearson==1:
                    print('*There is perfect positive linear relationship. As one variable increases, the other variable increases in a perfectly linear manner.')
                if correlation_pearson >0.5 and correlation_pearson<1:
                    print('*Moderate to strong positive linear relationship between the volatility and volume for the stock. There might be some strong degree of linear association.') 
                if correlation_pearson<0.5:
                    print('*Weak to moderate positive linear relationship between the volatility and volume for the stock. While there is some degree of linear association, it is not very strong.')    
            if correlation_pearson<0:
                if correlation_pearson==-1:
                    print('*There is perfect negative linear relationship. As one variable increases, the other variable decreases in a perfectly linear manner.')
                if correlation_pearson >-0.5 and correlation_pearson<1:
                    print('*Weak to moderate  negative linear relationship between the volatility and volume for the stock. There might be some strong degree of linear association.') 
                if correlation_pearson<-0.5:
                    print('*Moderate to strong negative linear relationship between the volatility and volume for the stock. While there is some degree of linear association, it is not very strong.')    
            if correlation_pearson==0:
                print('*No linear relationship. Changes in one variable do not predict changes in the other variable.')
        else:
            print('Error in calculation of Pearson correlation ')

        print(f"Spearman Correlation for {ticker}: {correlation_spearman}")
        if correlation_spearman>=-1 and correlation_spearman<=1:
            if correlation_spearman>0:
                if correlation_spearman==1:
                        print('*Perfect positive monotonic relationship. As one variable increases, the other variable consistently increases.')
                if correlation_spearman >0.5 and correlation_pearson<1:
                        print('*Moderate to strong positive monotonic relationship between the volatility and volume. It suggests that as one of the variables increases, the other variable tends to increase as well, but not necessarily in a perfectly linear fashion.') 
                if correlation_spearman<0.5:
                        print('*Weak to moderate positive monotonic relationship between the volatility and volume. It suggests that as one of the variables increases, the other variable tends to increase as well, but not necessarily in a perfectly linear fashion.')    
            if correlation_spearman<0:
                if correlation_spearman==-1:
                    print('*Perfect negative monotonic relationship. As one variable increases, the other variable consistently decreases.')
                if correlation_spearman >-0.5 and correlation_pearson<1:
                    print('*weak to moderate negative monotonic relationship between the volatility and volume. It suggests that as one of the variables increases, the other variable tends to decrease as well, but not necessarily in a perfectly linear fashion.') 
                if correlation_spearman<-0.5:
                    print('*Moderate to strong negative monotonic relationship between the volatility and volume. It suggests that as one of the variables increases, the other variable tends to decrease as well, but not necessarily in a perfectly linear fashion.')    
            if correlation_pearson==0:
                print('*No monotonic relationship. Changes in one variable do not predict changes in the other variable in a consistent order.')
        else:
            print('Error in calculation of Spearman correlation')
        
    except KeyError as e:
        print(f"KeyError: {e}. Check if {ticker} is present in stock data.")
        
    except Exception as e:
        print(f"Error occurred during correlation analysis for {ticker}: {str(e)}")



def decompose_time_series(stock_data, ticker):
    ts_data = stock_data['Adj Close'].dropna()
    decomposition = seasonal_decompose(ts_data, model='multiplicative', period=252)
    fig = decomposition.plot()
    fig.suptitle(f'Time Series Decomposition of {ticker}', fontsize=16)
    plt.show()