# -*- coding: utf-8 -*-
"""
Code to make Buffalo Trace Bourbon prediction
"""

# Libraries #
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
from datetime import date
from sklearn.preprocessing import OneHotEncoder

#
#### Scrape data ####
#

# Create function for data scraping #
def scrape_bourbon_data(urls):
    def get_bourbon_data(url):
        """
        Function to pull bourbon gift shop data. 

        Parameters
        ----------
        urls : string
            List of URLs to scrape. 
            
        url : string
            URL for the website.

        Returns
        -------
        x : Dataframe 
            Dataframe of info from HTML table.
        
        df : Dataframe 
            Dataframe of all info from HTML tables. 

        """
        
        # Get the HTML table with desired info #
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table')
        
        # Empty list to fill with info #
        rows = []
        for tr in table.find_all('tr'):
            row_data = [td.get_text(strip=True) for td in tr.find_all('td')]
            if row_data:
                rows.append(row_data)

        # Make empty list into a dataframe #
        x = pd.DataFrame(rows)
        
        # Return dataframe #
        return x
    
    # Make empty dataframe to concatenate #
    df = pd.DataFrame()

    # Loop through list of urls #
    for url in urls:
        df = pd.concat([df, get_bourbon_data(url)], ignore_index=True)
    
    # Provide columns labels 
    df.columns = ['Date', 'DOW', 'B1', 'B2']
    
    # Return final dataframe #
    return df


# URL of the pages to scrape
urls = [
    "https://buffalotracedaily.com/2023-gift-shop-releases/",
    "https://buffalotracedaily.com/2022-gift-shop-releases/",
    "https://buffalotracedaily.com/2021-gift-shop-releases/",
    "https://buffalotracedaily.com/2020-gift-shop-releases/"
]

# Call the function to get bourbon data 
bourbon = scrape_bourbon_data(urls)
del urls


#
#### Prepare dataframe ####
# 

# Set datatime #
bourbon['Date'] = pd.to_datetime(bourbon['Date'])


# Pull year, month, day, and DOW #
bourbon['Year'] = bourbon['Date'].dt.year
bourbon['Month'] = bourbon['Date'].dt.month
bourbon['Day'] = bourbon['Date'].dt.day
bourbon['Weekday'] = bourbon['Date'].dt.weekday


# Drop date variables #
bourbon = bourbon.drop(['DOW'], axis = 1)


# Split B2 by '/' #
bourbon[['B2', 'B3']] = bourbon['B2'].str.split('/', expand = True)


# Function to detect alphanumeric in column
def has_alphanum(s):
    return isinstance(s, str) and any(c.isalnum() for c in s)


# Apply the function to replace values with NaN if no alphanumeric characters are present
bourbon[['B2', 'B3']] = bourbon[['B2', 'B3']].\
    applymap(lambda x: np.nan if pd.isna(x) or not has_alphanum(x) else x)
  

# Make function to limit category values and aggregate #
def rename_categories(x):
    # Lower case values
    x = x.str.lower()
    
    # Replace values based on conditions
    def replace_value(value, pattern, replacement):
        if pd.notna(value) and re.match(pattern, value):
            return replacement
        return value
    
    x = x.apply(lambda value: replace_value(value, r'.*blanton.*', 'Blantons'))
    x = x.apply(lambda value: replace_value(value, r'.*close.*', 'Closed'))
    x = x.apply(lambda value: replace_value(value, r'.*weller.*', 'Weller'))
    x = x.apply(lambda value: replace_value(value, r'.*eagle.*', 'Eagle Rare'))
    x = x.apply(lambda value: replace_value(value, r'.*sazerac.*', 'Sazerac'))
    x = x.apply(lambda value: replace_value(value, r'.*taylor.*', 'Taylor'))
    x = x.apply(lambda value: replace_value(value, r'.*oak.*', 'Oak'))
    x = x.apply(lambda value: replace_value(value, r'.*none.*', 'Closed'))
    
    return x


# Rename categories # 
bourbon['B1'] = rename_categories(bourbon['B1'])
bourbon['B2'] = rename_categories(bourbon['B2'])
bourbon['B3'] = rename_categories(bourbon['B3'])


# Add variable for multiple bourbon day #
bourbon['Multi'] = 0
bourbon['Multi'] = np.where(pd.isnull(bourbon['B2']), 0, 1)








# Check if B2 has a value and add a new row if it does
for index, row in bourbon.iterrows():
    if pd.notna(row['B2']):
        new_row = row.copy()  
        new_row['B1'] = row['B2']  
        bourbon = bourbon.append(new_row, ignore_index=True)


# Check if B2 has a value and add a new row if it does
for index, row in bourbon.iterrows():
    if pd.notna(row['B3']):
        new_row = row.copy()
        new_row['B1'] = row['B3']
        bourbon = bourbon.append(new_row, ignore_index=True)
del index, new_row, row






# One Hot Encode B1, B2, and B3 #
one_hot = OneHotEncoder(sparse_output = False)


# Fit transform #
bourbon_encoded = one_hot.fit_transform(bourbon[['B1', 'B2', 'B3']])

bourbon_encoded = one_hot.fit(bourbon[['B1']])

trans = one_hot.transform(bourbon[['B1', 'B2', 'B3']])


# Create a DataFrame from the encoded data
encoded_df = pd.DataFrame(bourbon_encoded, columns=one_hot.get_feature_names_out(['B1', 'B2', 'B3']))

