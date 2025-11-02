import os
from dotenv import load_dotenv
from warnings import simplefilter
simplefilter('ignore')

from performStockAnalysis.main import forecast_using_the_developed_models

load_dotenv()

all_forecast_dd = [int(val) for val in os.getenv("forecast_stocks_all_forecast_dd").split(',')]
label_types = os.getenv("forecast_stocks_label_types").split(',')

if os.getenv("forecast_stocks_min_test_gini") == 'None':
    min_test_gini = None
else:
    min_test_gini = float(os.getenv("forecast_stocks_min_test_gini"))

forecast_using_the_developed_models(all_forecast_dd, label_types, min_test_gini)