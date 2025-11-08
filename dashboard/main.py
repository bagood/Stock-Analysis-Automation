import numpy as np
import pandas as pd
import streamlit as st

from helper import _process_emiten_forecasts, _process_emiten_risks

st.header("Analysis on Stocks' 10 Days Forecasts")
forecasts_10dd, selected_emtien_10dd = _process_emiten_forecasts(10, 'medianGain')
risks_10dd = _process_emiten_risks(10, 'maxLoss', selected_emtien_10dd)


st.dataframe(forecasts_10dd)

st.dataframe(risks_10dd)