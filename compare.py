import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import re

# Import bourbon csv #
bourbon = pd.read_csv('bourbon_data.csv')
bourbon = bourbon.sort_values('Date', ascending = False)

# Pull today #
today_bourbon_value = bourbon.iloc[0]['Bourbon_today']
today_bourbon = pd.DataFrame({'Bourbon_today': [today_bourbon_value]})
del today_bourbon_value

# Import predictions #
today_prediction = pd.read_csv('tom_predictions.csv')

# Concat #
today_compare = pd.concat([today_prediction, today_bourbon], axis=1)

# Check model performance #
today_compare['My_Correct'] = today_compare['My_Prediction'] == today_compare['Bourbon_today']
today_compare['They_Correct'] = today_compare['Their_Prediction'] == today_compare['Bourbon_today']

# Append to historical data #
historical = pd.read_csv('historical_model_comparisons.csv')
historical = pd.concat([historical, today_compare], axis = 0)
historical.to_csv('historical_model_comparisons.csv', index = False)