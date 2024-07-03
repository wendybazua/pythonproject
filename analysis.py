import pandas as pd
import numpy as np
from arch import arch_model
import plotly.graph_objs as go

def calculate_volatility(data, window=21): 
    data['Volatility'] = data['Best Transformation'].rolling(window=window).std()
    return data
 
#https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwiNseeI4rCGAxUUQ0EAHfGSD88QFnoECA8QAQ&url=https%3A%2F%2Farch.readthedocs.io%2Fen%2Flatest%2Funivariate%2Fintroduction.html&usg=AOvVaw36nBThBOb_BGaOov2WSS4r&opi=89978449
#this is very wrong**
def fit_best_garch_tarch_models(data, max_p=5, max_q=5): 
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
                garch_model = arch_model(returns * 100, vol='Garch', p=p, q=q, rescale=False) #not 100, should be just normal returns
                garch_fit = garch_model.fit(disp="off")
                if garch_fit.aic < best_garch_aic:
                    best_garch_aic = garch_fit.aic
                    best_garch_model = (p, q)
                    best_garch_fit = garch_fit
            except:
                continue

            try:
                tarch_model = arch_model(returns * 100, vol='Garch', p=p, q=q, o=1, rescale=False)
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
    daily_returns = stock_data['Best Transformation'].pct_change()
    volatility = daily_returns.rolling(window=252).std() * np.sqrt(252)
    
    volume = (stock_data['Volume'] - stock_data['Volume'].mean()) / stock_data['Volume'].std()

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
        yaxis2=dict(title='Volume', side='right', overlaying='y', type='linear'),
        width=1000,  
        height=600,  
        showlegend=True
    )

    fig.show()
