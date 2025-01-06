
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import re
import requests

# Import dataset #

# Import csv #
bourbon = pd.read_csv('bourbon_data.csv')
bourbon = bourbon.sort_values('Date', ascending = False)

# Grab tomorrow's date #
bourbon['Date'] = pd.to_datetime(bourbon['Date'])
date = bourbon['Date'].iloc[0] + timedelta(days=1)
date = date.strftime('%m-%d-%Y')

# Drop closed time, date #
bourbon = bourbon.drop(['Date', 'temp', 'Day', 'Month'], axis = 1)

# Pull out prediction #
bourbon_pred = pd.DataFrame(bourbon.iloc[0]).transpose()
bourbon_pred = bourbon_pred.drop('Bourbon_tomorrow', axis = 1)

# Import best model #
with open('bourbon_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('preprocess.pkl', 'rb') as file:
    preprocess = pickle.load(file)

with open('labeler.pkl', 'rb') as file:
    labeler = pickle.load(file)

# Transform training data #
index_of_last_entry = bourbon.index[0]
bourbon_t = bourbon.drop(index_of_last_entry, axis=0)
bourbon_t = bourbon_t.dropna()
y_train = labeler.transform(bourbon_t['Bourbon_tomorrow'])

bourbon_t = bourbon_t.drop('Bourbon_tomorrow', axis = 1)

X_train = pd.DataFrame(
    preprocess.transform(bourbon_t),
    columns = preprocess.get_feature_names_out(),
    index = bourbon_t.index)

model = model.fit(X_train, y_train)

# Transform prediciton data #
input = pd.DataFrame(
    preprocess.transform(bourbon_pred),
    columns = preprocess.get_feature_names_out(),
    index = bourbon_pred.index)

# Predict probabilities #
pred_bourbon_probs = model.predict_proba(input)
classes = labeler.classes_

# Create a DataFrame with class labels and corresponding probabilities
result_df = pd.DataFrame(pred_bourbon_probs, columns=classes)

# Plot probabilities #
probs = result_df.transpose().reset_index()
probs.columns = ['Bourbon', 'Probability']
probs = probs.sort_values(by='Probability', ascending=False)
sns.catplot(x='Bourbon', y='Probability', data = probs, kind='bar')
plt.xticks(fontsize = 9)
plt.subplots_adjust(top=0.94)  # Adjust the value as needed
plt.title(f'Gift Shop Bourbon Probabilities - {date}')
plt.savefig('Pred_plot.png')
plt.show()

# Make Bourbon prediction #
pred_bourbon = labeler.inverse_transform(model.predict(input))

# Make dataframe #
pred_df = pd.DataFrame({'Date': [datetime.now() + timedelta(days=1)],
                        'My_Prediction': pred_bourbon})


# Get max date plus 1 day of dataframe and match ULR #
day_1 = pred_df['Date'].max() + timedelta(days=-1)
day_1_form = day_1.strftime('%Y/%m/%d')

# Get next day's date and match URL #
day_2 = day_1 + timedelta(days=1)
day_2 = day_2.strftime('%B-%d-%Y').lower()
day_2 = day_2.replace('-0', '-')

# URL skeleton #
# Get url for days #
url = f'https://buffalotracedaily.com/{day_1_form}/what-buffalo-trace-is-selling-today-and-predictions-for-tomorrow-{day_2}/'

# Get HTML table #
response = requests.get(url)

# Break if URL does not exhist. Likely not updated today #
if response.status_code != 200:
    print(f"URL does not exist or there was an issue for {day_1_form}.")

# Parse the HTML #
soup = BeautifulSoup(response.content, 'html.parser')

# Find all tables on the page #
tables = soup.find_all('table')

# Find the correct table #
table = tables[2]

# Make a dataframe to fill #
rows = []

# Pull info #
for tr in table.find_all('tr'):
    row_data = [td.get_text(strip=True) for td in tr.find_all('td')]
    if row_data:
        rows.append(row_data)

# Get predicted bourbon #
rows = pd.DataFrame(rows, columns = ['Bourbon', 'Probability'])
rows = pd.DataFrame(rows.iloc[0]).transpose()


def rename_categories(x):
    # Lower case values
    x = x['Bourbon'].str.lower()

    # Replace values based on conditions
    def replace_value(value, pattern, replacement):
        if pd.notna(value) and re.match(pattern, value):
            return replacement
        return value

    x = x.apply(lambda value: replace_value(value, r'.*blanton.*', 'Blantons'))
    x = x.apply(lambda value: replace_value(value, r'.*close.*', 'Closed'))
    x = x.apply(lambda value: replace_value(value, r'.*weller.*', 'Weller'))
    x = x.apply(lambda value: replace_value(value, r'.*eagle.*', 'Eagle Rare'))
    x = x.apply(lambda value: replace_value(value, r'.*sazerac.*', 'Other'))
    x = x.apply(lambda value: replace_value(value, r'.*taylor.*', 'Taylor'))
    x = x.apply(lambda value: replace_value(value, r'.*oak.*', 'Other'))
    x = x.apply(lambda value: replace_value(value, r'.*none.*', 'Closed'))

    return x

# Rename bourbon #
rows = pd.DataFrame(rename_categories(rows))
rows = rows.rename(columns = {'Bourbon' : 'Their_Prediction'})

# Add to prediction frame #
pred_df = pd.concat([pred_df, rows], axis = 1)
 
# Write dataset to CSV #
pred_df.to_csv('tom_predictions.csv', index=False)
