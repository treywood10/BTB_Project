# Import libraries
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
from bayes_opt import BayesianOptimization

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
                                     random_state = seed, allow_duplicate_points = True)


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