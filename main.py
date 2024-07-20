import pandas as pd
from loguru import logger
from fetchdata import fetch_stock_data
from dataprep import prepare_stock_data, correlation_analysis, decompose_time_series
from analysis import calculate_volatility, fit_best_garch_tarch_models, plot_volatility_volume

tickers = ["AAPL", "MSFT", "TSLA", "JPM", "XOM"]
start_date = "2015-01-01"
end_date = "2023-01-01"

stock_data = fetch_stock_data(tickers, start_date, end_date)
stock_data.fillna(method='ffill', inplace=True)  # Forward fill to handle missing data

print(stock_data.head())  # Print the structure of stock_data for debugging

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
