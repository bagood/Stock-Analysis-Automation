import numpy as np
import pandas as pd

def _load_emiten_model_performance_and_forecast(forecast_dd: int, label_type: str):
    model_performances_path = f'database/modelPerformances/{label_type}/{forecast_dd}dd.csv'
    model_performances = pd.read_csv(model_performances_path)

    stock_forecasts_path = f'database/forecastedStocks/{label_type}/{forecast_dd}dd.csv'
    stock_forecasts = pd.read_csv(stock_forecasts_path)

    return (model_performances, stock_forecasts)

def _process_emiten_forecasts(forecast_dd: int, label_type: str):
    model_performances, stock_forecasts = _load_emiten_model_performance_and_forecast(forecast_dd, label_type)

    positive_label_column = stock_forecasts.columns[-1]
    
    stock_forecasts_performances = pd.merge( 
                                            model_performances,
                                            stock_forecasts.sort_values(positive_label_column, ascending=False),
                                            on='Kode',
                                            how='inner'
                                        )

    stock_forecasts_performances = stock_forecasts_performances[stock_forecasts_performances['Threshold'] > 0] \
                                        .sort_values('Threshold', ascending=False) \
                                        .reset_index(drop=True)

    selected_emiten = stock_forecasts_performances['Kode'].values

    return (stock_forecasts_performances, selected_emiten)

def _process_emiten_risks(forecast_dd: int, label_type: str, selected_emtien: list):
    model_performances, stock_forecasts = _load_emiten_model_performance_and_forecast(forecast_dd, label_type)

    stock_forecasts_performances =  pd.merge(
                                                model_performances,
                                                stock_forecasts,
                                                on='Kode',
                                                how='inner'
                                            )

    stock_forecasts_performances = stock_forecasts_performances[stock_forecasts_performances['Kode'].isin(selected_emtien)] \
                                        .sort_values('Threshold', ascending=False) \
                                        .reset_index(drop=True)

    return stock_forecasts_performances