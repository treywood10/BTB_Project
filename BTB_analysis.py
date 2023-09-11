#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code to predict Buffalo Trace bourbon. 
@author: treywood
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


#
#### Import dataset 
#

# Import csv #
bourbon = pd.read_csv('bourbon_data.csv')


# Drop closed time, oak time, and sazerac time #
bourbon = bourbon.drop(['Closed_time', 'Oak_time', 'Sazerac_time'], axis = 1)


#
#### Train-Test Split
#

# List of targets #
targets = ['Blantons', 'Eagle Rare', 'Taylor', 'Weller']


# Split data #
X_train, X_test, y_train, y_test = train_test_split(bourbon.drop(targets, axis = 1), 
                                                    bourbon[targets],
                                                    test_size = 0.2,
                                                    random_state = 42)

#
#### Pre-process 
#

# Variables to categorize #
cat_vars = ['Year', 'Month', 'Weekday']


# Categorical pipeline #
cat_pipe = Pipeline([
    ('one_hot', OneHotEncoder(sparse_output = False))
])


# Numerical variables #
num_vars = ['Day', 'Blantons_time', 'Eagle Rare_time', 'Taylor_time', 'Weller_time', 'temp']


# Numerical pipeline #
num_pipe = Pipeline([
    ('scaler', StandardScaler())
])


# Make column transformer #
preprocess = ColumnTransformer([
    ('num', num_pipe, num_vars),
    ('cat', cat_pipe, cat_vars)],
    remainder = 'drop',
    verbose_feature_names_out = False)

X_train = pd.DataFrame(
    preprocess.fit_transform(X_train),
    columns = preprocess.get_feature_names_out(),
    index = X_train.index)

X_test = pd.DataFrame(
    preprocess.transform(X_test),
    columns = preprocess.get_feature_names_out(),
    index = X_test.index)
