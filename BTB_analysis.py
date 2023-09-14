#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code to predict Buffalo Trace bourbon. 
@author: treywood
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
from bayes_opt import BayesianOptimization



#
#### Import dataset 
#

# Run script to update dataset #
with open ('BTB_make_data.py') as f:
    exec(f.read())
del f


# Set seed. Selected 90 for the proof of Buffalo Trace Flagship bourbon #
seed = 90

# Import csv #
bourbon = pd.read_csv('bourbon_data.csv')


# Drop closed time, oak time, and sazerac time #
bourbon = bourbon.drop(['Closed_time', 'Oak_time', 'Sazerac_time'], axis = 1)


# Make matrix to compare models #
train_compare = pd.DataFrame(columns = ['Model', 'F1', 'hypers'])


#
#### Train-Test Split
#

# List of targets #
#targets = ['Blantons', 'Eagle Rare', 'Taylor', 'Weller']


# Split data #
X_train, X_test, y_train, y_test = train_test_split(bourbon.drop('Bourbon', axis = 1), 
                                                    bourbon['Bourbon'],
                                                    test_size = 0.2,
                                                    random_state = seed)


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


# Clear variables #
del cat_pipe, cat_vars, num_pipe, num_vars, preprocess


#
#### Logistic Regression
#

# Objective function for Logistic Regression #
def log_tuning(X_train, y_train, pbounds, n_init, n_iter, seed):
    def obj_log(penalty, C, l1_ratio, multi_class):
        """
        Function for bayesian search of best hyperparameters. 

        Parameters
        ----------
        penalty : string
            Penalty for regularization.
            C : float
            Regularization strenth. Smaller values, stronger reg.
            l1_ratio : float
            Regularization for 'elasticnet'. Only for 'saga' solver.

        Returns
        -------
        f_score : float
            F1 score to measure model performance.
            """
        
        # Penalty
        if penalty < 0.3:
            penalty = 'l1'

        elif penalty < 0.6:
            penalty = 'l2'

        else:
            penalty = 'elasticnet'
        
        # Multiclass option #
        if multi_class < 0.5:
            multi_class = 'ovr'
        else :
            multi_class = 'multinomial'

        # Instantiate model #
        if penalty == 'elasticnet':
            model = LogisticRegression(C=C, penalty=penalty, solver='saga',
                                       l1_ratio=l1_ratio, random_state=seed,
                                       max_iter=20000)
        else:
            model = LogisticRegression(C=C, penalty=penalty,
                                       solver='saga', random_state=seed,
                                       max_iter=20000)

        # Cross validation #
        pred = cross_val_predict(model, X_train, y_train, cv = 5)

        # F1 Score #
        f_score = f1_score(y_train, pred, average = 'weighted')

        return f_score
    
    optimizer = BayesianOptimization(f = obj_log, pbounds = pbounds,
                                     random_state = seed)
    optimizer.maximize(init_points = n_init, n_iter = n_iter)

    # Pull best info #
    best_hypers = optimizer.max['params']
    best_f1 = optimizer.max['target']

    # Adjust hypers #
    if best_hypers['penalty'] < 0.3:
        best_hypers['penalty'] = 'l1'
    elif best_hypers['penalty'] < 0.6:
        best_hypers['penalty'] = 'l2'
    else:
        best_hypers['penalty'] = 'elasticnet'

    if best_hypers['penalty'] != 'elasticnet':
        best_hypers.pop('l1_ratio')
        
    if best_hypers['penalty'] == 'elasticnet':
        best_hypers['solver'] = 'saga'

    
    # Multiclass option #
    if best_hypers['multi_class'] < 0.5:
        best_hypers['multi_class'] = 'ovr'
    else :
        best_hypers['multi_class'] = 'multinomial'
        

    best_model = LogisticRegression(**best_hypers, max_iter=20000)
    
    return best_f1, best_model, best_hypers
    

# Define the search space #
pbounds = {
    'C' : (0.00001, 10),
    'penalty' : (0, 1),
    'l1_ratio' : (0.01, 0.99),
    'multi_class' : (0, 1)
}

best_f1, best_model, hypes = log_tuning(X_train, y_train, 
                                 pbounds, n_init = 25, 
                                 n_iter = 5, seed = seed)



# Fill comparison matrix #
train_compare = pd.concat([train_compare,
                           pd.DataFrame({'Model' : 'Logistic',
                            'F1': best_f1,
                            'hypers': [best_model]})], 
                          ignore_index = True)




# Fit the best logistic regression model to your data
best_model.fit(X_train, y_train)


# Get list of coefficients #
coefficients = best_model.coef_  # This will be a 2D array with shape (n_classes, n_features)


# Get the absolute values of coefficients for each class
abs_coefficients = np.abs(coefficients)


# Get the feature names (assuming you have them in a list)
feature_names = X_train.columns.tolist()


# Create a feature importance plot for all classes in one graph
n_classes = abs_coefficients.shape[0]


plt.figure(figsize=(12, 8))


# Set up colors for each class
colors = plt.cm.Set1(np.linspace(0, 1, n_classes))

for i in range(n_classes):
    plt.barh(np.arange(len(feature_names)) + i * 0.2, abs_coefficients[i, :], height=0.2, label=f'Class {i}', color=colors[i])

plt.xlabel('Absolute Coefficient Value')
plt.ylabel('Feature Names')
plt.yticks(np.arange(len(feature_names)), feature_names)
plt.title('Feature Importance for All Classes')
plt.gca().invert_yaxis()  # Invert the y-axis to display the most important features at the top
plt.legend()
plt.show()