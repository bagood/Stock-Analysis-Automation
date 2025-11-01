from warnings import simplefilter
simplefilter('ignore')

from performStockAnalysis.main import select_emiten_to_model, develop_models_for_selected_emiten

# quantile_threshold = 0.6
# _ = select_emiten_to_model(quantile_threshold)

# label_types = ['median_gain', 'max_loss']
# rolling_windows = [10, 15]

label_types = ['median_gain']
rolling_windows = [10]

develop_models_for_selected_emiten(label_types, rolling_windows)