import os
from dotenv import load_dotenv
from warnings import simplefilter
simplefilter('ignore')

from performStockAnalysis.main import select_emiten_to_model, develop_models_for_selected_emiten

load_dotenv()

quantile_threshold = float(os.getenv("develop_model_quantile_threshold"))
if os.getenv("develop_model_bypass_bool") == 'False':
    _ = select_emiten_to_model(quantile_threshold)

label_types = os.getenv("develop_model_label_types").split(',')
rolling_windows = [int(val) for val in os.getenv("develop_model_rolling_windows").split(',')]

develop_models_for_selected_emiten(label_types, rolling_windows)