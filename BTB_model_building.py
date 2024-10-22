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
from sklearn.metrics import f1_score, make_scorer
import tensorflow as tf

# Import model tuning functions
from models.logistic_train import log_tuning
from models.svm_train import svm_tuning
from models.rand_forest_train import rf_tuning
from models.xgboost_train import boost_tuning
from models.knn_train import knn_tuning
from models.neural_net_train import net_tuning

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
n_init = 10
n_iter = 10

# Import csv #
bourbon = pd.read_csv('bourbon_data.csv')
bourbon = bourbon.sort_values('Date', ascending = False)

# Drop closed time, date #
bourbon = bourbon.drop(['Date', 'temp', 'Day', 'Month'], axis = 1)

bourbon_pred = pd.DataFrame(bourbon.iloc[0]).transpose()
bourbon_pred = bourbon_pred.drop('Bourbon_tomorrow', axis = 1)
bourbon = bourbon.dropna()

# Make matrix to compare models #
train_compare = pd.DataFrame(columns = ['Model', 'Train_F1', 'Test_F1', 'Model_Specs'])

#
#### Train-Test Split
#

# Split data #
X_train, X_test, y_train, y_test = train_test_split(bourbon.drop('Bourbon_tomorrow', axis = 1), 
                                                    bourbon['Bourbon_tomorrow'],
                                                    test_size = 0.2,
                                                    stratify = bourbon['Bourbon_tomorrow'],
                                                    random_state = seed)


#
#### Pre-process 
#

# Variables to categorize #
#cat_vars = ['Year', 'Month', 'Weekday',
#            'Bourbon_today', 'Bourbon_1_lag', 'Bourbon_2_lag']
cat_vars = ['Year', 'Weekday',
            'Bourbon_today', 'Bourbon_1_lag', 'Bourbon_2_lag']

# Categorical pipeline #
cat_pipe = Pipeline([
    ('one_hot', OneHotEncoder(sparse_output = False))
])

# Numerical variables #
#num_vars = ['Day', 'Blantons_time', 'Eagle Rare_time', 'Taylor_time', 'Weller_time']
num_vars = ['Blantons_time', 'Eagle Rare_time', 'Taylor_time', 'Weller_time']

# Numerical pipeline #
num_pipe = Pipeline([
    ('scaler', StandardScaler())
])

# Make column transformer #
preprocess = ColumnTransformer([
    ('num', num_pipe, num_vars),
    ('cat', cat_pipe, cat_vars)],
    remainder = 'passthrough',
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
del cat_pipe, cat_vars, num_pipe, num_vars

labeler = LabelEncoder()
y_train_encode = labeler.fit_transform(y_train)
y_test_encode = labeler.transform(y_test)

#
#### Logistic Regression
#

# Define the search space #
pbounds = {
    'C' : (0.00001, 5),
    'penalty' : (0, 1),
    'l1_ratio' : (0.1, 0.9),
    'multi_class' : (0, 1)
}

# Optimize logistic model #
best_f1, best_model = log_tuning(X_train, y_train_encode,
                                 pbounds, n_init = n_init, 
                                 n_iter = n_iter, seed = seed)

# Notify #
beep(6)

# Generate test score #
best_model.fit(X_train, y_train_encode)
test_f1 = f1_score(y_test_encode, best_model.predict(X_test), average = 'weighted')

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
perm_log = permutation_importance(best_model, X_train, y_train_encode, n_repeats=10,
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

# Set search space # 
pbounds = {
    'C' : (0.00001, 1),
    'kernel' : (0, 4),
    'degree' : (1, 5),
    'gamma' : (0, 1),
    'shrinking': (0, 1)}


# Optimize model #
best_f1, best_model = svm_tuning(X_train, y_train_encode, 
                                 pbounds, n_init = n_init, 
                                 n_iter = n_iter, seed = seed)

# Notify #
beep(6)

# Generate test score #
best_model.fit(X_train, y_train_encode)
test_f1 = f1_score(y_test_encode, best_model.predict(X_test), average = 'weighted')

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
perm_svm = permutation_importance(best_model, X_train, y_train_encode, n_repeats=10,
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

# Set search space #
pbounds = {
    'n_estimators' : (50, 1000),
    'criterion' : (0, 3),
    'max_depth' : (3, 8),
    'max_features' : (0, 2),
    'bootstrap' : (0, 1)
}

# Optimize model #
best_f1, best_model = rf_tuning(X_train, y_train_encode,
                                 pbounds, n_init = n_init, 
                                 n_iter = n_iter, seed = seed)

# Notify #
beep(6)

# Generate test score #
best_model.fit(X_train, y_train_encode)
test_f1 = f1_score(y_test_encode, best_model.predict(X_test), average = 'weighted')

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
perm_rf = permutation_importance(best_model, X_train, y_train_encode, n_repeats=10,
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

# Define the scoring metric as 'f1_micro' (or 'f1_macro' or 'f1_weighted')
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

pbounds = {
    'n_neighbors': (2, 8),
    'weights' : (0, 1),
    'algorithm' : (0, 4),
    'leaf_size' : (20, 40),
    'p' : (0, 2)}

# Optimize model #
best_f1, best_model = knn_tuning(X_train, y_train_encode,
                                 pbounds, n_init = n_init, 
                                 n_iter = n_iter, seed = seed)

# Notify #
beep(6)

# Generate test score #
best_model.fit(X_train, y_train_encode)
test_f1 = f1_score(y_test_encode, best_model.predict(X_test), average = 'weighted')

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
perm_knn = permutation_importance(best_model, X_train, y_train_encode, n_repeats=10,
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
y_train_dums = pd.get_dummies(y_train_encode).values
y_test_dums = pd.get_dummies(y_test_encode).values

# Set the random seed for TensorFlow #
tf.random.set_seed(seed)

# Define the search space #
pbounds = {
    'batch_size': (100, 800),
    'epochs': (20, 80),
    'optimizer': (0, 1),
    'rate': (0.001, 0.9),
    'activation': (0, 1),
    'learning_rate': (0.0001, 0.3),
    'num_hidden_layers': (1, 200),
    'num_nodes': (1, 100),
}

# Optimize model #
best_f1, best_model = net_tuning(X_train, y_train_dums, 
                                 pbounds, n_init = n_init, 
                                 n_iter = n_iter, seed = seed,
                                 labeler = labeler)

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

bourbon_pred_trans = pd.DataFrame(
    preprocess.transform(bourbon_pred),
    columns = preprocess.get_feature_names_out(),
    index = bourbon_pred.index)

preds = best_model.predict_proba(bourbon_pred_trans)

classes = labeler.classes_

# Create a DataFrame with class labels and corresponding probabilities
result_df = pd.DataFrame(preds, columns=classes)

# Pickle best model #
with open('bourbon_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

# Pickle preprocessor #
with open('preprocess.pkl', 'wb') as file:
    pickle.dump(preprocess, file)

# Pickle encoder #
with open('labeler.pkl', 'wb') as file:
    pickle.dump(labeler, file)