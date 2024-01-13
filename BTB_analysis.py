#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code to predict Buffalo Trace bourbon. 
@author: treywood
"""

import pandas as pd
import numpy as np
from beepy import beep
import pickle
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score, make_scorer
from bayes_opt import BayesianOptimization
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras import optimizers
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils import np_utils
import tensorflow as tf


#
#### Import dataset 
#

# Run script to update dataset #
with open ('BTB_make_data.py') as f:
    exec(f.read())
del f


# Set seed. Selected 90 for the proof of Buffalo Trace Flagship bourbon #
seed = 90


# Set  searches #
n_init = 50
n_iter = 150


# Import csv #
bourbon = pd.read_csv('bourbon_data2.csv')


# Drop closed time, date #
#bourbon = bourbon.drop(['Closed_time', 'Date'], axis = 1)
bourbon = bourbon.drop(['Date', 'temp'], axis = 1)

# Make matrix to compare models #
train_compare = pd.DataFrame(columns = ['Model', 'Train_F1', 'Test_F1', 'Model_Specs'])


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
#num_vars = ['Day', 'Blantons_time', 'Eagle Rare_time', 'Taylor_time', 'Weller_time', 'Other_time', 'temp']
num_vars = ['Day', 'Blantons_time', 'Eagle Rare_time', 'Taylor_time', 'Weller_time']


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


# Optimize model #
best_f1, best_model = log_tuning(X_train, y_train, 
                                 pbounds, n_init = n_init, 
                                 n_iter = n_iter, seed = seed)

# Notify #
beep(6)


# Generate test score #
best_model.fit(X_train, y_train)
test_f1 = f1_score(y_test, best_model.predict(X_test), average = 'weighted')


# Fill comparison matrix #
train_compare = pd.concat([train_compare,
                           pd.DataFrame({'Model' : 'Logistic',
                            'Train_F1': best_f1,
                            'Test_F1': test_f1,
                            'Model_Specs': [best_model]})], 
                          ignore_index = True)


# Define the scoring metric as 'f1_micro' (or 'f1_macro' or 'f1_weighted' based on your preference)
scorer = make_scorer(f1_score, average='weighted')


# Calculate permutation importance using 'f1_micro' as the scoring metric
perm_log = permutation_importance(best_model, X_train, y_train, n_repeats=10,
                              random_state=seed, scoring=scorer, n_jobs=2)


# Get feature names #
feature_names = X_train.columns.tolist()


# Get the indices that would sort importances_mean in descending order
sorted_indices = np.argsort(perm_log.importances_mean)


# Sort both feature_names and importances_mean based on the sorted_indices
sorted_feature_names = [feature_names[i] for i in sorted_indices]
sorted_importances_mean = [perm_log.importances_mean[i] for i in sorted_indices]


# Create the bar plot with sorted values
plt.figure(figsize=(10, 8))
plt.barh(range(len(sorted_feature_names)), sorted_importances_mean, align="center")
plt.yticks(range(len(sorted_feature_names)), sorted_feature_names)
plt.xlabel("Permutation Importance")
plt.ylabel("Feature")
plt.title("Permutation Importance for Each Feature (Sorted)")
plt.savefig('Importances/Logit.png')
plt.show()


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


# Optimize model #
best_f1, best_model = svm_tuning(X_train, y_train, 
                                 pbounds, n_init = n_init, 
                                 n_iter = n_iter, seed = seed)


# Notify #
beep(6)


# Generate test score #
best_model.fit(X_train, y_train)
test_f1 = f1_score(y_test, best_model.predict(X_test), average = 'weighted')


# Fill comparison matrix #
train_compare = pd.concat([train_compare,
                           pd.DataFrame({'Model' : 'SVM',
                            'Train_F1': best_f1,
                            'Test_F1': test_f1,
                            'Model_Specs': [best_model]})], 
                          ignore_index = True).sort_values('Test_F1', ascending = False)


# Define the scoring metric as 'f1_micro' (or 'f1_macro' or 'f1_weighted' based on your preference)
scorer = make_scorer(f1_score, average='weighted')


# Calculate permutation importance using 'f1_micro' as the scoring metric
perm_svm = permutation_importance(best_model, X_train, y_train, n_repeats=10,
                              random_state=seed, scoring=scorer, n_jobs=2)


# Get feature names #
feature_names = X_train.columns.tolist()


# Get the indices that would sort importances_mean in descending order
sorted_indices = np.argsort(perm_svm.importances_mean)


# Sort both feature_names and importances_mean based on the sorted_indices
sorted_feature_names = [feature_names[i] for i in sorted_indices]
sorted_importances_mean = [perm_svm.importances_mean[i] for i in sorted_indices]


# Create the bar plot with sorted values
plt.figure(figsize=(10, 8))
plt.barh(range(len(sorted_feature_names)), sorted_importances_mean, align="center")
plt.yticks(range(len(sorted_feature_names)), sorted_feature_names)
plt.xlabel("Permutation Importance")
plt.ylabel("Feature")
plt.title("Permutation Importance for Each Feature (Sorted)")
plt.savefig('Importances/SVM.png')
plt.show()


#
#### Random Forest
#

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



# Optimize model #
best_f1, best_model = rf_tuning(X_train, y_train, 
                                 pbounds, n_init = n_init, 
                                 n_iter = n_iter, seed = seed)


# Notify #
beep(6)


# Generate test score #
best_model.fit(X_train, y_train)
test_f1 = f1_score(y_test, best_model.predict(X_test), average = 'weighted')


# Fill comparison matrix #
train_compare = pd.concat([train_compare,
                           pd.DataFrame({'Model' : 'Random Forest',
                            'Train_F1': best_f1,
                            'Test_F1': test_f1,
                            'Model_Specs': [best_model]})], 
                          ignore_index = True).sort_values('Test_F1', ascending = False)


# Define the scoring metric as 'f1_micro' (or 'f1_macro' or 'f1_weighted' based on your preference)
scorer = make_scorer(f1_score, average='weighted')


# Calculate permutation importance using 'f1_micro' as the scoring metric
perm_rf = permutation_importance(best_model, X_train, y_train, n_repeats=10,
                              random_state=seed, scoring=scorer, n_jobs=2)


# Get feature names #
feature_names = X_train.columns.tolist()


# Get the indices that would sort importances_mean in descending order
sorted_indices = np.argsort(perm_rf.importances_mean)


# Sort both feature_names and importances_mean based on the sorted_indices
sorted_feature_names = [feature_names[i] for i in sorted_indices]
sorted_importances_mean = [perm_rf.importances_mean[i] for i in sorted_indices]


# Create the bar plot with sorted values
plt.figure(figsize=(10, 8))
plt.barh(range(len(sorted_feature_names)), sorted_importances_mean, align="center")
plt.yticks(range(len(sorted_feature_names)), sorted_feature_names)
plt.xlabel("Permutation Importance")
plt.ylabel("Feature")
plt.title("Permutation Importance for Each Feature (Sorted)")
plt.savefig('Importances/Forest.png')
plt.show()


#
#### XGBoost
#

# Fix target values for XGBoost #
labeler = LabelEncoder()
y_train_encode = labeler.fit_transform(y_train)
y_test_encode = labeler.transform(y_test)


# Objective function for XGBoost #
def boost_tuning(X_train, y_train_encode, pbounds, n_init, n_iter, seed):
    def obj_xgb(n_estimators, max_depth, 
                learning_rate, subsample, 
                colsample_bytree, 
                reg_alpha, reg_lambda):
        """
        Function for bayesian search of best hyperparameters. 

        Parameters
        ----------
        n_estimators : int
            Number of boosting rounds.
        max_depth : int
            Max tree depth for base learners.
        learning_rate : float
            Boosting learning rate.
        subsample : float
            Subsample ratio of training data.
        colsample_bytree : float
            Subsample ratio of columns used.
        reg_alpha : float
            L1 regularization.
        reg_lambda : float
            L2 regularization.

        Returns
        -------
        f_score : float
            F1 score to measure model performance.
        
        """
    

        # Instantiate modlel #
        model = XGBClassifier(n_estimators = int(n_estimators),
                                       max_depth = int(max_depth),
                                       learning_rate = learning_rate,
                                       subsample = subsample,
                                       colsample_bytree = colsample_bytree,
                                       reg_alpha = reg_alpha,
                                       reg_lambda = reg_lambda,
                                       random_state = seed,
                                       n_jobs = 2)
    
        # Cross validation #
        pred = cross_val_predict(model, X_train, y_train_encode, cv = 5)
    
        # F1 Score #
        f_score = f1_score(y_train_encode, pred, average = 'weighted')
        
        return f_score

    # Set optimizer #
    optimizer = BayesianOptimization(f = obj_xgb, pbounds = pbounds,
                                     random_state = seed)


    # Call maximizer #
    optimizer.maximize(init_points = n_init, n_iter = n_iter)
    
    # Pull best info #
    best_hypers = optimizer.max['params']
    best_f1 = optimizer.max['target']


    # Adjust hyperparamters #
    best_hypers['n_estimators'] = round(best_hypers['n_estimators'])

    best_hypers['max_depth'] = round(best_hypers['max_depth'])

    best_model = XGBClassifier(**best_hypers)
    
    return best_f1, best_model 


# Set search space #
pbounds = {
    'n_estimators' : (50, 1000),
    'max_depth' : (3, 8),
    'learning_rate' : (0.0001, 1),
    'subsample' : (0.2, 0.8),
    'colsample_bytree' : (0.2, 1),
    'reg_alpha' : (0.0001, 1),
    'reg_lambda' : (0.0001, 1)
}


# Optimize model #
best_f1, best_model = boost_tuning(X_train, y_train_encode, 
                                 pbounds, n_init = n_init, 
                                 n_iter = n_iter, seed = seed)


# Notify #
beep(6)


# Generate test score #
best_model.fit(X_train, y_train_encode)
test_f1 = f1_score(y_test_encode, best_model.predict(X_test), average = 'weighted')


# Fill comparison matrix #
train_compare = pd.concat([train_compare,
                           pd.DataFrame({'Model' : 'XGBoost',
                            'Train_F1': best_f1,
                            'Test_F1': test_f1,
                            'Model_Specs': [best_model]})], 
                          ignore_index = True).sort_values('Test_F1', ascending = False)



# Define the scoring metric as 'f1_micro' (or 'f1_macro' or 'f1_weighted' based on your preference)
scorer = make_scorer(f1_score, average='weighted')


# Calculate permutation importance using 'f1_micro' as the scoring metric
perm_boost = permutation_importance(best_model, X_train, y_train_encode, n_repeats=10,
                              random_state=seed, scoring=scorer, n_jobs=2)


# Get feature names #
feature_names = X_train.columns.tolist()


# Get the indices that would sort importances_mean in descending order
sorted_indices = np.argsort(perm_boost.importances_mean)


# Sort both feature_names and importances_mean based on the sorted_indices
sorted_feature_names = [feature_names[i] for i in sorted_indices]
sorted_importances_mean = [perm_boost.importances_mean[i] for i in sorted_indices]


# Create the bar plot with sorted values
plt.figure(figsize=(10, 8))
plt.barh(range(len(sorted_feature_names)), sorted_importances_mean, align="center")
plt.yticks(range(len(sorted_feature_names)), sorted_feature_names)
plt.xlabel("Permutation Importance")
plt.ylabel("Feature")
plt.title("Permutation Importance for Each Feature (Sorted)")
plt.savefig('Importances/Boost.png')
plt.show()


#
#### KNN
#

# Objective function for KNN #
def knn_tuning(X_train, y_train, pbounds, n_init, n_iter, seed):
    def obj_knn(n_neighbors, weights, algorithm, leaf_size, p):
        
        """
        Function for bayesian search of best hyperparameters. 

        Parameters
        ----------
        n_neighbors : int
            Number of neighbors to use.
        weights : string
            Weight function.
        algorithm : string
            Algorithm to compute nearest neighbors.
        leaf_size : int
            Leaf size passed to BallTree and KDTree.
        p : int
            Power parameter for Minkowski metric.

        Returns
        -------
        f_score : float
            F1 score to measure model performance.

        """
        
        # Vary weights #
        if weights <= 0.5:
            weights = 'uniform'
        else:
            weights = 'distance'
        
        # Vary algorithm #
        if algorithm <= 1:
            algorithm = 'auto'
        elif algorithm <= 2:
            algorithm = 'ball_tree'
        elif algorithm <= 3:
            algorithm = 'kd_tree'
        else:
            algorithm = 'brute'
        
        # Vary p #
        # Variation on p #
        if p <= 1.0:
            p = 1
        elif p <= 1.0 and algorithm != 'brute':
            p = 1
        else:
            p = 2
            
        
        # Instantiate modlel #
        model = KNeighborsClassifier(n_neighbors = int(n_neighbors),
                                     weights = weights,
                                     algorithm = algorithm,
                                     leaf_size = int(leaf_size),
                                     p = p,
                                     n_jobs = 2)
        
        # Cross validation #
        pred = cross_val_predict(model, X_train, y_train, cv = 5)
        
        # F1 Score #
        f_score = f1_score(y_train, pred, average = 'weighted')
        
        return f_score

    optimizer = BayesianOptimization(f=obj_knn, pbounds=pbounds, random_state=seed)
    optimizer.maximize(init_points=n_init, n_iter=n_iter)
    
    best_hypers = optimizer.max['params']
    best_f1 = optimizer.max['target']

    if best_hypers['weights'] <= 0.5:
        best_hypers['weights'] = 'uniform'
    else:
        best_hypers['weights'] = 'distance'
        
    # Vary algorithm #
    if best_hypers['algorithm'] <= 1:
        best_hypers['algorithm'] = 'auto'
    elif best_hypers['algorithm'] <= 2:
        best_hypers['algorithm'] = 'ball_tree'
    elif best_hypers['algorithm'] <= 3:
        best_hypers['algorithm'] = 'kd_tree'
    else:
        best_hypers['algorithm'] = 'brute'

    if best_hypers['p'] <= 1.0 and best_hypers['algorithm'] != 'brute':
        best_hypers['p'] = 1
    else:
        best_hypers['p'] = 2
        
    best_hypers['n_neighbors'] = int(round(best_hypers['n_neighbors']))
    best_hypers['leaf_size'] = int(round(best_hypers['leaf_size']))

    best_model = KNeighborsClassifier(**best_hypers, n_jobs=2)
    
    return best_f1, best_model

pbounds = {
    'n_neighbors': (2, 8),
    'weights' : (0, 1),
    'algorithm' : (0, 4),
    'leaf_size' : (20, 40),
    'p' : (0, 2)}


# Optimize model #
best_f1, best_model = knn_tuning(X_train, y_train, 
                                 pbounds, n_init = n_init, 
                                 n_iter = n_iter, seed = seed)


# Notify #
beep(6)


# Generate test score #
best_model.fit(X_train, y_train)
test_f1 = f1_score(y_test, best_model.predict(X_test), average = 'weighted')


# Fill comparison matrix #
train_compare = pd.concat([train_compare,
                           pd.DataFrame({'Model' : 'KNN',
                            'Train_F1': best_f1,
                            'Test_F1': test_f1,
                            'Model_Specs': [best_model]})], 
                          ignore_index = True).sort_values('Test_F1', ascending = False)


# Define the scoring metric as 'f1_micro' (or 'f1_macro' or 'f1_weighted' based on your preference)
scorer = make_scorer(f1_score, average='weighted')


# Calculate permutation importance using 'f1_micro' as the scoring metric
perm_knn = permutation_importance(best_model, X_train, y_train, n_repeats=10,
                              random_state=seed, scoring=scorer, n_jobs=2)


# Get feature names #
feature_names = X_train.columns.tolist()


# Get the indices that would sort importances_mean in descending order
sorted_indices = np.argsort(perm_knn.importances_mean)


# Sort both feature_names and importances_mean based on the sorted_indices
sorted_feature_names = [feature_names[i] for i in sorted_indices]
sorted_importances_mean = [perm_knn.importances_mean[i] for i in sorted_indices]


# Create the bar plot with sorted values
plt.figure(figsize=(10, 8))
plt.barh(range(len(sorted_feature_names)), sorted_importances_mean, align="center")
plt.yticks(range(len(sorted_feature_names)), sorted_feature_names)
plt.xlabel("Permutation Importance")
plt.ylabel("Feature")
plt.title("Permutation Importance for Each Feature (Sorted)")
plt.savefig('Importances/KNN.png')
plt.show()



#
#### Neural Net 
#

# Fix target values for XGBoost #
labeler = LabelEncoder()
y_train_encode = labeler.fit_transform(y_train)
y_train_dums = np_utils.to_categorical(y_train_encode)

y_test_encode = labeler.transform(y_test)
y_test_dums = np_utils.to_categorical(y_test_encode)


# Set the random seed for TensorFlow #
tf.random.set_seed(seed)


# Create optimizer function #
def net_tuning(X_train, y_train_dums, pbounds, n_init, n_iter, seed):
    def obj_net(batch_size, epochs, activation, num_nodes,
                num_hidden_layers, learning_rate, rate, optimizer):
        """
    The objective of this function is to minimize the error of the
    neural network

    Parameters
    ----------
    batch_size : Int
        The number of cases to include in each batch.
    epochs : Int
        Number of runs through the data when updating weights.
    activation : String
        Type of activation function for the layer.
    num_nodes : Int
        Number of nodes to include in the hidden layer.
    num_hidden_layers : Int
        Number of hideen layers in the model.
    learning_rate : Float
        How much to change the model with each model update.
    rate : Float
        Dropout rate for each hidden layer to prevent overfitting.
    optimizer : String
        Optimizer to use for the model.

    Returns
    -------
    error : Float
        Cross validation returns root mean error that is later
        convereted into RMSE in the comparison frame.

    """
    
        # Set Optimizer #
        if optimizer <= 0.33:
            optimizer = optimizers.Adam(learning_rate = learning_rate)
    
        elif optimizer <= 0.66:
            optimizer = optimizers.Adagrad(learning_rate = learning_rate)
    
        else:
            optimizer = optimizers.RMSprop(learning_rate = learning_rate)
        
        # Set activation function #
        if activation <= 0.33:
            activation = 'relu'
            
        elif activation <= 0.66:
            activation = 'sigmoid'
       
        else:
            activation = 'tanh'
       
        # Instantiate model
        model = Sequential()
    
        # Set input layer #
        model.add(Dense(int(num_nodes), activation = activation, 
                    input_shape = (X_train.shape[1],)))
        
        model.add(BatchNormalization())
    
        # Set hidden layer with batch normalizer #
        for _ in range(int(num_hidden_layers)):
            model.add(Dense(int(num_nodes), activation = activation))
            model.add(Dropout(rate = rate, seed = seed))
    
        # Add output layer #
        model.add(Dense(len(labeler.classes_), activation='softmax'))

    
        # Set compiler #
        model.compile(optimizer = optimizer,
                  loss = 'categorical_crossentropy')
    
        # Set early stopping #
        early_stopping = EarlyStopping(monitor='val_loss', 
                                       patience=15, 
                                       restore_best_weights=True)
    
        # Create model #
        net_model = KerasClassifier(model = lambda : model,
                             batch_size = int(batch_size),
                             epochs = int(epochs),
                             validation_split = 0.2,
                             callbacks = [early_stopping],
                             random_state = seed)

        # Cross validation #
        pred = cross_val_predict(net_model, X_train, y_train_dums, cv = 5)
        
        
        # F1 Score #
        f_score = f1_score(y_train_dums, pred, average = 'weighted')
    
        return f_score
    

    optimizer = BayesianOptimization(f=obj_net, pbounds=pbounds, random_state=seed)
    optimizer.maximize(init_points=n_init, n_iter=n_iter)

    # Pull best info #
    best_hypers = optimizer.max['params']
    best_f1 = optimizer.max['target']
    
    # Replace optimizer and learning rate #
    if best_hypers['optimizer'] <= 0.33:
        best_hypers['optimizer'] = 'Adam'
    elif best_hypers['optimizer'] <= 0.66:
        best_hypers['optimizer'] = 'Adagrad'
    else:
        best_hypers['optimizer'] = 'RMSprop'
    
    
    if best_hypers['optimizer'] == 'Adam':
        optimizer = optimizers.Adam(learning_rate=best_hypers['learning_rate'])
    elif best_hypers['optimizer'] == 'Adagrad':
        optimizer = optimizers.Adagrad(learning_rate=best_hypers['learning_rate'])
    else:
        optimizer = optimizers.RMSprop(learning_rate=best_hypers['learning_rate'])
    
    
    # Replace activation with string #
    if best_hypers['activation'] <= 0.33:
        best_hypers['activation'] = 'relu'
        
    elif best_hypers['activation'] <= 0.66:
        best_hypers['activation'] = 'sigmoid'
       
    else:
        best_hypers['activation'] = 'tanh'

    final_model = Sequential()

    final_model.add(Dense(int(best_hypers['num_nodes']), activation=best_hypers['activation'], 
                          input_shape=(X_train.shape[1],)))
    
    final_model.add(BatchNormalization())

    for _ in range(int(best_hypers['num_hidden_layers'])):
        final_model.add(Dense(int(best_hypers['num_nodes']), activation=best_hypers['activation']))
        final_model.add(Dropout(rate=best_hypers['rate'], seed=seed))

    # Add output layer with the correct number of units
    final_model.add(Dense(len(labeler.classes_), activation='softmax'))

    final_model.compile(optimizer=optimizer, loss='binary_crossentropy')

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    final_model.fit(X_train, y_train_dums, batch_size=int(best_hypers['batch_size']), epochs=int(best_hypers['epochs']),
                    validation_split=0.2, callbacks=[early_stopping])

    return best_f1, final_model



# Define the search space #
pbounds = {
    'batch_size': (100, 800),
    'epochs': (20, 80),
    'optimizer': (0, 1),
    'rate': (0.001, 0.9),
    'activation': (0, 1),
    'learning_rate': (0.0001, 0.3),
    'num_hidden_layers': (1, 200),
    'num_nodes': (1, 50),
}

# Optimize model #
best_f1, best_model = net_tuning(X_train, y_train_dums, 
                                 pbounds, n_init = n_init, 
                                 n_iter = n_iter, seed = seed)


# Notify #
beep(6)


# Generate test score #
best_model.fit(X_train, y_train_dums)


# Predicted probabilities of each class #
probs = best_model.predict(X_test)


# Find the column index with the highest value for each row #
max_column_indices = np.argmax(probs, axis=1)


# Create a new array with zeros, with the same shape as the original data #
result = np.zeros_like(probs)


# Replace 0 with 1 for highest probability #
result[np.arange(result.shape[0]), max_column_indices] = 1


# Get f1 score #
test_f1 = f1_score(y_test_dums, result, average = 'weighted')


# Fill comparison matrix #
train_compare = pd.concat([train_compare,
                           pd.DataFrame({'Model' : 'Neural Net',
                            'Train_F1': best_f1,
                            'Test_F1': test_f1,
                            'Model_Specs': [best_model]})], 
                          ignore_index = True).sort_values('Test_F1', ascending = False)


# Stack model #
from sklearn.ensemble import StackingClassifier


# Pull top 3 models #
model_1 = train_compare['Model_Specs'].loc[0]
model_2 = train_compare['Model_Specs'].loc[1]
model_3 = train_compare['Model_Specs'].loc[2]


stack = StackingClassifier(estimators = [
    ('M1', model_1),
    ('M2', model_2),
    ('M3', model_3)
])

stack.fit(X_train, y_train)

pred = stack.predict(X_test)

test_f1 = f1_score(y_test, pred, average = 'weighted')


# Fill comparison matrix #
train_compare = pd.concat([train_compare,
                           pd.DataFrame({'Model' : 'Stacked',
                            'Train_F1': 0.0,
                            'Test_F1': test_f1,
                            'Model_Specs': [stack]})], 
                          ignore_index = True).sort_values('Test_F1', ascending = False)


#
#### Save best model 
#

# Pull best model #
best_model = train_compare['Model_Specs'].iloc[0]


# Pickle best model #
with open('bourbon_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)
