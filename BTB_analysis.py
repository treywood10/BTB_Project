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
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
from bayes_opt import BayesianOptimization
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier




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
bourbon = bourbon.drop(['Closed_time', 'Date'], axis = 1)


# Make matrix to compare models #
train_compare = pd.DataFrame(columns = ['Model', 'F1', 'hypers'])


#
#### Train-Test Split
#

# Split data #
X_train, X_test, y_train, y_test = train_test_split(bourbon.drop('Bourbon_1', axis = 1), 
                                                    bourbon['Bourbon_1'],
                                                    test_size = 0.2,
                                                    stratify = bourbon['Bourbon_1'],
                                                    random_state = seed)


#
#### Pre-process 
#

# Variables to categorize #
cat_vars = ['Year', 'Month', 'Weekday', 'Bourbon_1_lag', 'Bourbon_2_lag']


# Categorical pipeline #
cat_pipe = Pipeline([
    ('one_hot', OneHotEncoder(sparse_output = False))
])


# Numerical variables #
num_vars = ['Day', 'Blantons_time', 'Eagle Rare_time', 'Taylor_time', 'Weller_time', 'Other_time', 'temp']


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
            model = LogisticRegression(C=C, penalty=penalty, solver='saga',
                                           random_state=seed,
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
    
    # Multiclass option #
    if best_hypers['multi_class'] < 0.5:
        best_hypers['multi_class'] = 'ovr'
    else :
        best_hypers['multi_class'] = 'multinomial'
        

    best_model = LogisticRegression(**best_hypers, solver = 'saga', max_iter=20000)
    
    return best_f1, best_model
    

# Define the search space #
pbounds = {
    'C' : (0.00001, 5),
    'penalty' : (0, 1),
    'l1_ratio' : (0.1, 0.9),
    'multi_class' : (0, 1)
}

best_f1, best_model = log_tuning(X_train, y_train, 
                                 pbounds, n_init = 25, 
                                 n_iter = 75, seed = seed)


# Fill comparison matrix #
train_compare = pd.concat([train_compare,
                           pd.DataFrame({'Model' : 'Logistic',
                            'F1': best_f1,
                            'hypers': [best_model]})], 
                          ignore_index = True)


#
#### Support Vector Machine
#

# Objective function for SVM #
def svm_tuning(X_train, y_train, pbounds, n_init, n_iter, seed): 
    def obj_svm(C, kernel, degree, gamma, shrinking):
        """
        Function for bayesian search of best hyperparameters. 

        Parameters
        ----------
        C : float
            Regularization strenth. Smaller values, stronger reg.
        kernel : string
            Kernel type for algorithm.
        degree : int
            Degree of polynomial function for 'poly' kernel.
        gamma : string
            Kernel coefficient for non-linear kernels.
        shrinking : bool
            Use shrinking heuristic.

        Returns
        -------
        f_score : float
            F1 score to measure model performance.

        """
    
        # Kernel #
        if kernel <= 1:
            kernel = 'linear'
        elif kernel <= 2:
            kernel = 'poly'
        elif kernel <= 3:
            kernel = 'rbf'
        else:
            kernel = 'sigmoid'
    
        # Gamma #
        if gamma <= 0.5:
            gamma = 'scale'
        else:
            gamma = 'auto'
    
        # Shrinking #
        shrinking = bool(round(shrinking))
    
        # Instantiate modlel #
        model = SVC(C = C, kernel = kernel,
                    degree = int(degree), gamma = gamma,
                    shrinking = shrinking, random_state = seed)
    
        # Cross validation #
        pred = cross_val_predict(model, X_train, y_train, cv = 5)
    
        # F1 Score #
        f_score = f1_score(y_train, pred, average = 'weighted')
    
        return f_score
    
    optimizer = BayesianOptimization(f = obj_svm, pbounds = pbounds,
                                     random_state = seed)
    optimizer.maximize(init_points = n_init, n_iter = n_iter)
    
    
    # Pull best info #
    best_hypers = optimizer.max['params']
    best_f1 = optimizer.max['target']


    # Adjust hypers #
    if best_hypers['kernel'] <= 1:
        best_hypers['kernel']  = 'linear'
    elif best_hypers['kernel']  <= 2:
        best_hypers['kernel']  = 'poly'
    elif best_hypers['kernel']  <= 3:
        best_hypers['kernel']  = 'rbf'
    else:
        best_hypers['kernel']  = 'sigmoid'
        
    if best_hypers['gamma']  <= 0.5:
        best_hypers['gamma'] = 'scale'
    else:
        best_hypers['gamma'] = 'auto'
        
    best_hypers['shrinking'] = bool(round(best_hypers['shrinking']))
    
    best_hypers['degree'] = round(best_hypers['degree'])

    best_model = SVC(**best_hypers)
    
    return best_f1, best_model 

# Set search space # 
pbounds = {
    'C' : (0.00001, 1),
    'kernel' : (0, 4),
    'degree' : (1, 5),
    'gamma' : (0, 1),
    'shrinking': (0, 1)}


best_f1, best_model = svm_tuning(X_train, y_train, 
                                 pbounds, n_init = 25, 
                                 n_iter = 75, seed = seed)

# Fill comparison matrix #
train_compare = pd.concat([train_compare,
                           pd.DataFrame({'Model' : 'SVM',
                            'F1': best_f1,
                            'hypers': [best_model]})], 
                          ignore_index = True)


# Objective function for random forest #
def rf_tuning(X_train, y_train, pbounds, n_init, n_iter, seed):
    def obj_forest(n_estimators, criterion,
                   max_depth, max_features,
                   bootstrap) :
        """
        Function for bayesian search of best hyperparameters. 

        Parameters
        ----------
        n_estimators : int
            Number of trees in forest.
        criterion : string
            Function to measure quality of split.
        max_depth : int
            Max tree depth.
        max_features : string
            Number of features to include.
        bootstrap : bool
            Whether to bootstrap samples for trees.

        Returns
        -------
        f_score : float
            F1 score to measure model performance.
        
        """
    
        # Vary criterion #
        if criterion <= 1:
            criterion = 'gini'
        elif criterion <= 2:
            criterion = 'entropy'
        else:
            criterion = 'log_loss'
    
        # Vary max features #
        if max_features <= 1:
            max_features = 'sqrt'
        else:
            max_features = 'log2'
        
        # Vary bootstrap #
        bootstrap = bool(round(bootstrap))
    
        # Instantiate modlel #
        model = RandomForestClassifier(n_estimators = int(n_estimators),
                                       criterion = criterion, 
                                       max_depth =  int(max_depth), 
                                       max_features = max_features,
                                       bootstrap = bootstrap)
    
        # Cross validation #
        pred = cross_val_predict(model, X_train, y_train, cv = 5)
    
        # F1 Score #
        f_score = f1_score(y_train, pred, average = 'weighted')
    
        return f_score
    
    # Set optimizer #
    optimizer = BayesianOptimization(f = obj_forest, pbounds = pbounds,
                                     random_state = seed)


    # Call maximizer #
    optimizer.maximize(init_points = n_init, n_iter = n_iter)

    # Pull best info #
    best_hypers = optimizer.max['params']
    best_f1 = optimizer.max['target']


    # Adjust hypers #
    best_hypers['n_estimators'] = round(best_hypers['n_estimators'])

    best_hypers['max_depth'] = round(best_hypers['max_depth'])

    if best_hypers['criterion'] <= 1:
        best_hypers['criterion'] = 'gini'
    elif best_hypers['criterion'] <= 2:
        best_hypers['criterion'] = 'entropy'
    else:
        best_hypers['criterion'] = 'log_loss'
        
    if best_hypers['max_features'] <= 1:
        best_hypers['max_features'] = 'sqrt'
    else:
        best_hypers['max_features'] = 'log2'
            
    best_hypers['bootstrap'] = bool(round(best_hypers['bootstrap']))   

    best_model = RandomForestClassifier(**best_hypers)
    
    return best_f1, best_model

# Set search space #
pbounds = {
    'n_estimators' : (50, 1000),
    'criterion' : (0, 3),
    'max_depth' : (3, 8),
    'max_features' : (0, 2),
    'bootstrap' : (0, 1)
}


best_f1, best_model = rf_tuning(X_train, y_train, 
                                 pbounds, n_init = 25, 
                                 n_iter = 75, seed = seed)

# Fill comparison matrix #
train_compare = pd.concat([train_compare,
                           pd.DataFrame({'Model' : 'Random Forest',
                            'F1': best_f1,
                            'hypers': [best_model]})], 
                          ignore_index = True)