




get_ipython().system('pip install numpy pandas matplotlib plotly yfinance statsmodels')


# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.seasonal import seasonal_decompose


# In[ ]:





# In[14]:


import yfinance as yf

def fetch_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    return data
tickers = ["AAPL", "MSFT", "TSLA", "JPM", "XOM"]

stock_data = fetch_data(tickers, "2015-01-01", "2023-01-01")
stock_data.fillna(method='ffill', inplace=True)  # Forward fill to handle missing data


# In[15]:


print(stock_data.head())


# In[16]:


stock_data


# In[17]:


for ticker in tickers:
    print(f"Analyzing {ticker}")


# In[18]:


for ticker in tickers:
    daily_returns = stock_data['Adj Close'][ticker].pct_change()
    volatility = daily_returns.rolling(window=252).std() * np.sqrt(252)
    volume = (stock_data['Volume'][ticker] - stock_data['Volume'][ticker].mean()) / stock_data['Volume'][ticker].std()



# In[19]:


from statsmodels.tsa.stattools import adfuller

def perform_adf_test(series):
    result = adfuller(series.dropna())
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))


stock_data = fetch_data(tickers, "2015-01-01", "2023-01-01")

for ticker in tickers:
    print(f"Performing ADF Test for {ticker}")
    perform_adf_test(stock_data['Adj Close'][ticker])


# In[20]:


from statsmodels.tsa.seasonal import seasonal_decompose


for ticker in tickers:
    print(f"Decomposing Time Series of {ticker}")

    ts_data = stock_data['Adj Close'][ticker].dropna()
    
    decomposition = seasonal_decompose(ts_data, model='multiplicative', period=252)
    
    decomposition.plot()
    plt.suptitle(ticker)
    plt.show()


# In[21]:


stock_data = fetch_data(tickers, "2015-01-01", "2023-01-01")

for ticker in tickers:
    daily_returns = stock_data['Adj Close'][ticker].pct_change()
    volatility = daily_returns.rolling(window=252).std() * np.sqrt(252)
    
    volume = (stock_data['Volume'][ticker] - stock_data['Volume'][ticker].mean()) / stock_data['Volume'][ticker].std()

    correlation_data = pd.concat([volatility, volume], axis=1)
    correlation_data.columns = ['Volatility', 'Volume']

    correlation_pearson = correlation_data.corr(method='pearson').iloc[0, 1]
    correlation_spearman = correlation_data.corr(method='spearman').iloc[0, 1]

    print(f"Pearson Correlation for {ticker}: {correlation_pearson}")
    print(f"Spearman Correlation for {ticker}: {correlation_spearman}")


# In[22]:


for ticker in tickers:
    print(f"\nPerforming OLS Regression for {ticker}")

    daily_returns = stock_data['Adj Close'][ticker].pct_change()
    volatility = daily_returns.rolling(window=252).std() * np.sqrt(252)
    
    volume = (stock_data['Volume'][ticker] - stock_data['Volume'][ticker].mean()) / stock_data['Volume'][ticker].std()
    
    aligned_data = pd.concat([volatility, volume], axis=1).dropna()
    aligned_volatility = aligned_data.iloc[:, 0]
    aligned_volume = aligned_data.iloc[:, 1]
    
    X = sm.add_constant(aligned_volume)  # Adds a constant term to the predictor
    y = aligned_volatility

    model = sm.OLS(y, X).fit()

    print(model.summary())


# In[23]:



for ticker in tickers:
    daily_returns = stock_data['Adj Close'][ticker].pct_change()
    volatility = daily_returns.rolling(window=252).std() * np.sqrt(252)
    
    volume = (stock_data['Volume'][ticker] - stock_data['Volume'][ticker].mean()) / stock_data['Volume'][ticker].std()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=volatility.index,
        y=volatility,
        mode='lines',
        name=f'Volatility_{ticker}'
    ))

    fig.add_trace(go.Scatter(
        x=volume.index,
        y=volume,
        mode='lines',
        name=f'Volume_{ticker}',
        yaxis='y2'
    ))

    fig.update_layout(
        title=f'{ticker} Stock Volatility and Volume Analysis',
        xaxis_title='Date',
        yaxis=dict(title='Volatility', side='left'),
        yaxis2=dict(title='Volume', side='right', overlaying='y', type='linear')
    )

    fig.show()

    input("Press Enter to continue to the next ticker analysis...")


# In[ ]:





