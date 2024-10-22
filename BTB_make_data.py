# -*- coding: utf-8 -*-
"""
Code to make data for Buffalo Trace Prediction.
"""

# Libraries #
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from meteostat import Point, Daily
import holidays
from dateutil import easter


#
#### Scrape Historical Bourbon Data ####
#

# Create function for data scraping #
def scrape_historical_bourbon_data(urls):
    """
    Function to pull bourbon gift shop data from a list of URLs.

    Parameters
    ----------
    urls : list of strings
        List of URLs to scrape.

    Returns
    -------
    df : DataFrame
        DataFrame containing combined info from HTML tables across all URLs.
    """
    # Make an empty dataframe to concatenate results
    df = pd.DataFrame()

    # Loop through list of URLs
    for url in urls:
        # Get the HTML table with desired info
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table')

        # Empty list to fill with info
        rows = []
        for tr in table.find_all('tr'):
            row_data = [td.get_text(strip=True) for td in tr.find_all('td')]
            if row_data:
                rows.append(row_data)

        # Convert list to a DataFrame and concatenate with the main DataFrame
        x = pd.DataFrame(rows)
        df = pd.concat([df, x], ignore_index=True)

    # Provide column labels
    df.columns = ['Date', 'DOW', 'B1', 'B2']

    # Return final DataFrame
    return df


# URL of the pages to scrape
urls = [
    "https://buffalotracedaily.com/2024-gift-shop-releases/",
    "https://buffalotracedaily.com/2023-gift-shop-releases/",
    "https://buffalotracedaily.com/2022-gift-shop-releases/",
    "https://buffalotracedaily.com/2021-gift-shop-releases/",
    "https://buffalotracedaily.com/2020-gift-shop-releases/"
]

# Call the function to get bourbon data
bourbon = scrape_historical_bourbon_data(urls)
del urls

# Set datatime #
bourbon['Date'] = pd.to_datetime(bourbon['Date'])

#
#### Update Historical Data ####
#

def update_historical_bourbon_data(df):

    # Get BT holiday dates
    us_holidays = holidays.US(years=df['Date'].dt.year.max())
    us_holidays[easter.easter(df['Date'].dt.year.max())] = 'Easter Sunday'
    us_holidays[pd.Timestamp(f'{df["Date"].dt.year.max()}-12-24')] = 'Christmas Eve'
    holidays_df = pd.DataFrame(list(us_holidays.items()), columns=['Date', 'Holiday']).sort_values(by='Date')
    holidays_df = holidays_df[holidays_df['Holiday'].isin(['New Year\'s Day', 'Easter Sunday', 'Independence Day',
                                                           'Thanksgiving', 'Christmas Eve', 'Christmas Day'])]

    # Ensure 'Date' column is in datetime format
    holidays_df['Date'] = pd.to_datetime(holidays_df['Date'])

    while df['Date'].max() < datetime.now():

        # Holiday Check
        max_date = df['Date'].max().date()  # Convert max date to datetime.date for comparison
        holiday_dates = holidays_df['Date'].dt.date  # Convert holidays_df['Date'] to datetime.date

        print(max_date)

        if max_date in holiday_dates.values:  # Use 'in' with converted date values

            # Holiday fill row
            holiday = pd.DataFrame({'Date': df['Date'].max() + timedelta(days=1),
                                    'DOW': (df['Date'].max() + timedelta(days=1)).strftime('%a'),
                                    'B1': 'Closed',
                                    'B2': ' '}, index=[0])

            # Combine with dataframe
            df = pd.concat([df, holiday], ignore_index=True)

        else:
            found_data = False

            # Loop to check up to 3 days in the future
            for day_offset in range(3):
                day_1 = df['Date'].max() + timedelta(days=day_offset + 1)
                day_1_form = day_1.strftime('%Y/%m/%d')

                # Get next day's date and match URL
                day_2 = day_1 + timedelta(days=1)
                day_2_form = day_2.strftime('%B-%d-%Y').lower().replace('-0', '-')

                # URL skeleton
                url = f'https://buffalotracedaily.com/{day_1_form}/what-buffalo-trace-is-selling-today-and-predictions-for-tomorrow-{day_2_form}/'

                # Get HTML table
                response = requests.get(url)

                if response.status_code == 200:
                    # Parse the HTML
                    soup = BeautifulSoup(response.content, 'html.parser')

                    # Find all tables on the page
                    tables = soup.find_all('table')

                    # Find the correct table
                    table = tables[2]

                    # Make a dataframe to fill
                    rows = []

                    # Pull info #
                    for tr in table.find_all('tr'):
                        row_data = [td.get_text(strip=True) for td in tr.find_all('td')]
                        if row_data:
                            rows.append(row_data)

                    # Make dataframe #
                    soup_df = pd.DataFrame(rows, columns=['Date', 'DOW', 'B1', 'Medal'])

                    # Drop the 'Medal' column #
                    soup_df = soup_df.drop('Medal', axis=1)

                    # Make 'Date' into datetime
                    soup_df['Date'] = pd.to_datetime(soup_df['Date'], format='%m/%d/%y')

                    # Filter to keep rows in soup_df with dates not already in df
                    soup_df = soup_df[~soup_df['Date'].isin(df['Date'])]

                    # Append to the original df #
                    df = pd.concat([soup_df, df], ignore_index=True)
                    df = df.sort_values(by=['Date'])

                    found_data = True
                    break
                else:
                    print(f"URL does not exist or there was an issue for {day_1_form}.")

            # Break the outer loop if no data was found after checking 3 days
            if not found_data:
                print("No valid data found for up to 3 days in the future. Exiting loop.")
                break

    return df


bourbon = update_historical_bourbon_data(bourbon)

# Pull year, month, day, and DOW #
bourbon['Year'] = bourbon['Date'].dt.year
bourbon['Month'] = bourbon['Date'].dt.month
bourbon['Day'] = bourbon['Date'].dt.day
bourbon['Weekday'] = bourbon['Date'].dt.weekday

# Drop date variables #
bourbon = bourbon.drop(['DOW'], axis=1)

# Split B2 by '/' #
bourbon[['B2', 'B3']] = bourbon['B2'].str.split('/', expand=True)


# Function to detect alphanumeric in column
def has_alphanum(s):
    return isinstance(s, str) and any(c.isalnum() for c in s)


# Apply the function to replace values with NaN if no alphanumeric characters are present
bourbon[['B2', 'B3']] = bourbon[['B2', 'B3']].apply(
    lambda col: col.map(lambda x: np.nan if pd.isna(x) or not has_alphanum(x) else x))

# Make function to limit category values and aggregate #
def rename_categories(x):
    """Lowercase and replace category names based on regex patterns."""
    # Lowercase values
    x = x.str.lower()

    # Apply replacements directly
    x = x.apply(lambda value: 'Blantons' if pd.notna(value) and re.match(r'.*blanton.*', value) else value)
    x = x.apply(lambda value: 'Closed' if pd.notna(value) and re.match(r'.*close.*', value) else value)
    x = x.apply(lambda value: 'Weller' if pd.notna(value) and re.match(r'.*weller.*', value) else value)
    x = x.apply(lambda value: 'Eagle Rare' if pd.notna(value) and re.match(r'.*eagle.*', value) else value)
    x = x.apply(lambda value: 'Other' if pd.notna(value) and re.match(r'.*sazerac.*', value) else value)
    x = x.apply(lambda value: 'Taylor' if pd.notna(value) and re.match(r'.*taylor.*', value) else value)
    x = x.apply(lambda value: 'Other' if pd.notna(value) and re.match(r'.*oak.*', value) else value)
    x = x.apply(lambda value: 'Closed' if pd.notna(value) and re.match(r'.*none.*', value) else value)

    return x


# Rename categories #
bourbon['B1'] = rename_categories(bourbon['B1'])
bourbon['B2'] = rename_categories(bourbon['B2'])
bourbon['B3'] = rename_categories(bourbon['B3'])

# Add variable for multiple bourbon day #
bourbon['Multi'] = 0
bourbon['Multi'] = np.where(pd.isnull(bourbon['B2']), 0, 1)

# Create a list of unique categories from each column and merge them
unique_categories_b1 = bourbon['B1'].dropna().unique()
unique_categories_b2 = bourbon['B2'].dropna().unique()
unique_categories_b3 = bourbon['B3'].dropna().unique()

# Combine unique categories #
unique_categories = np.union1d(np.union1d(unique_categories_b1, unique_categories_b2), unique_categories_b3)

# One-hot encode B1, B2, and B3 columns separately
one_hot_b1 = pd.get_dummies(bourbon['B1'], columns=unique_categories, prefix='B1')
one_hot_b2 = pd.get_dummies(bourbon['B2'], columns=unique_categories, prefix='B2')
one_hot_b3 = pd.get_dummies(bourbon['B3'], columns=unique_categories, prefix='B3')

# Merge the one-hot encoded B1 and B2 columns and original dataframe #
bourbon = pd.concat([bourbon, one_hot_b1, one_hot_b2, one_hot_b3], axis=1)

# Replace NaN values with 0 in the one-hot encoded columns #
bourbon.fillna(0, inplace=True)

# Iterate through unique categories and create a new column for each
for category in unique_categories:
    # Find columns that contain the category name
    columns_with_category = [col for col in bourbon.columns if category in col]

    # Check if the sum of columns with the category name is greater than or equal to 1
    if columns_with_category:
        bourbon[category] = np.where(bourbon[columns_with_category].sum(axis=1) >= 1, 1, 0)

    # Drop old categories
    for col in columns_with_category:
        bourbon.drop(col, axis=1, inplace=True)

# Clear variables #
del one_hot_b1, one_hot_b2, one_hot_b3, col, category, unique_categories_b1, unique_categories_b2, unique_categories_b3

# Initialize a dictionary to store the time since last occurrence for each category
time_since_last_occurrence = {}

# Iterate through unique categories
for category in unique_categories:

    # Sort by date #
    bourbon = bourbon.sort_values('Date')

    # Start time counter #
    t = 0  # Initialize the time counter

    # Create a new column name for time since last occurrence #
    column_name = f'{category}_time'

    # Initialize an empty list to store time values
    time_values = []

    # Iterate through the rows of the DataFrame
    for index, row in bourbon.iterrows():
        if row[category] == 1:
            t = 0
        else:
            t += 1
        time_values.append(t)

    # Add the time values as a new column in the DataFrame
    bourbon[column_name] = time_values
del category, column_name, columns_with_category, index, row, t, time_since_last_occurrence, time_values, unique_categories

#
#### Add Local Weather ####
#


# Set time period #
start = bourbon['Date'].min()
end = bourbon['Date'].max()

# Create point for Frankfort, KY #
location = Point(38.2009, -84.8733)

# Get daily data #
data = Daily(location, start, end)
ky_data = data.fetch()
ky_data = ky_data.reset_index()

# Make time into datetime #
ky_data['Date'] = pd.to_datetime(ky_data['time'])
ky_data = ky_data.drop('time', axis=1)

# Remove rows where 'tavg' is NaN
ky_data = ky_data.dropna(subset=['tavg'])

# Convert tavg into Fahrenheit #
ky_data['temp'] = (ky_data['tavg'] * 9 / 5) + 32

# Keep variables #
ky_data = ky_data[['Date', 'temp']]

# Merge #
bourbon = pd.merge(bourbon, ky_data, how='left', on='Date')
del data, end, ky_data, location, start

# Lead bourbon #
bourbon['B1_tomorrow'] = bourbon['B1'].shift(-1)
bourbon['B2_tomorrow'] = bourbon['B2'].shift(-1)

# Lag Boubrbon #
bourbon['B1_lag'] = bourbon['B1'].shift(1)
bourbon['B2_lag'] = bourbon['B2'].shift(1)

#
#### Expand days with >1 bourbons #
#

# Make '0' values with nan #
bourbon['B2'] = bourbon['B2'].replace(0, np.nan)
bourbon['B3'] = bourbon['B3'].replace(0, np.nan)
bourbon['B1_tomorrow'] = bourbon['B1_tomorrow'].replace(0, np.nan)
bourbon['B2_tomorrow'] = bourbon['B2_tomorrow'].replace(0, np.nan)

# Drop B3 #
bourbon = bourbon.drop(['B3'], axis=1)

# Check if B2 is not equal to '0' for any element in the Series.
# Make new row if yes
for index, row in bourbon.iterrows():
    if pd.notna(row['B2']):
        new_row = row.copy()
        new_row['B1'] = row['B2']
        bourbon = pd.concat([bourbon, pd.DataFrame([new_row])], axis=0)
del index, new_row, row

# Make new row to break up tomorrow outcome #
for index, row in bourbon.iterrows():
    if pd.notna(row['B2_tomorrow']):
        new_row = row.copy()
        new_row['B1_tomorrow'] = row['B2_tomorrow']
        bourbon = pd.concat([bourbon, pd.DataFrame([new_row])], axis=0)
del index, new_row, row

# Drop B2 #
bourbon = bourbon.drop(['B2', 'B2_tomorrow'], axis=1)

# Rename B1 #
bourbon.rename(columns={'B1': 'Bourbon_today'}, inplace=True)
bourbon.rename(columns={'B1_lag': 'Bourbon_1_lag'}, inplace=True)
bourbon.rename(columns={'B2_lag': 'Bourbon_2_lag'}, inplace=True)
bourbon.rename(columns={'B1_tomorrow': 'Bourbon_tomorrow'}, inplace=True)

# Replace 0 with No Bottles #
bourbon['Bourbon_2_lag'] = bourbon['Bourbon_2_lag'].replace(0, 'No Extra')

# Drop Bourbon binary #
bourbon = bourbon.drop(['Blantons', 'Closed', 'Eagle Rare', 'Other', 'Taylor', 'Weller'], axis=1)


#
#### Save Data
#

# Drop Closed and Other outcomes #
bourbon = bourbon[bourbon['Bourbon_today'] != 'Closed']
bourbon = bourbon[bourbon['Bourbon_today'] != 'Other']
bourbon = bourbon.drop(["Other_time", "Closed_time"], axis=1)

# Sort by date #
bourbon = bourbon.sort_values('Date')

# Write dataset to CSV #
bourbon.to_csv('bourbon_data.csv', index=False)