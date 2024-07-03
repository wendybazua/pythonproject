import pandas as pd
from fetchdata import fetch_stock_data
from dataprep import prepare_stock_data, correlation_analysis, decompose_time_series
from analysis import calculate_volatility, fit_best_garch_tarch_models, plot_volatility_volume
#***** https://github.com/Delgan/loguru instead of prints 

stock_ticker = 'AMZN'
start_date = "2015-01-01"
end_date = "2023-01-01"

stock_data = fetch_stock_data(stock_ticker, start_date, end_date)

result = prepare_stock_data(stock_data)



if result is not None:
    prepared_data, message = result
    print(message)

    if 'Best Transformation' in prepared_data.columns:
        print("Transformations Applied:")
        print(prepared_data[['Adj Close', 'Best Transformation']].head())

    analysis_data = calculate_volatility(prepared_data)
    print(analysis_data)

    correlation_analysis(prepared_data)
    
    best_model_type, best_model, best_model_summary = fit_best_garch_tarch_models(prepared_data)  
    print(f"Best model type: {best_model_type}")
    print(f"Best model: {best_model}")
    print(best_model_summary)
    
    #****FIX VOLATILITY-VOLUME PLOT
    #plot_volatility_volume(analysis_data, stock_ticker)

    

else:
    print("Data preparation failed.")



#