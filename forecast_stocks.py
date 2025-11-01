from warnings import simplefilter
simplefilter('ignore')

from performStockAnalysis.main import forecast_using_the_developed_models

all_forecast_dd = [10, 15]
label_types = ['medianGain', 'maxLoss']
development_date = '20251101'
min_test_gini=None
forecast_using_the_developed_models(all_forecast_dd, label_types, development_date, min_test_gini)