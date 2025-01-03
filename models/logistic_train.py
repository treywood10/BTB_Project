from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
from bayes_opt import BayesianOptimization

# Objective function for Logistic Regression
def log_tuning(X_train, y_train_encode, pbounds, n_init, n_iter, seed):
    def obj_log(penalty, C, l1_ratio):
        """
        Function for bayesian search of best hyperparameters.

        Parameters
        ----------
        penalty : string
            Penalty for regularization.
        C : float
            Regularization strength. Smaller values, stronger regularization.
        l1_ratio : float
            Regularization for 'elasticnet'. Only for 'saga' solver.

        Returns
        -------
        f_score : float
            F1 score to measure model performance.
        """
        # Map penalty to string
        if penalty < 0.3:
            penalty = 'l1'
        elif penalty < 0.6:
            penalty = 'l2'
        else:
            penalty = 'elasticnet'

        # Ensure l1_ratio is only passed for elasticnet
        if penalty == 'elasticnet':
            model = LogisticRegression(C=C, penalty=penalty, solver='saga',
                                       l1_ratio=l1_ratio, random_state=seed, max_iter=20000)
        else:
            model = LogisticRegression(C=C, penalty=penalty, solver='saga',
                                       random_state=seed, max_iter=20000)

        # Cross-validation
        pred = cross_val_predict(model, X_train, y_train_encode, cv=5)

        # F1 score
        f_score = f1_score(y_train_encode, pred, average='weighted')
        return f_score

    # Run Bayesian Optimization
    optimizer = BayesianOptimization(f=obj_log, pbounds=pbounds,
                                     random_state=seed, allow_duplicate_points=True)
    optimizer.maximize(init_points=n_init, n_iter=n_iter)

    # Pull best hyperparameters
    best_hypers = optimizer.max['params']
    best_f1 = optimizer.max['target']

    # Adjust hyperparameters
    if best_hypers['penalty'] < 0.3:
        best_hypers['penalty'] = 'l1'
    elif best_hypers['penalty'] < 0.6:
        best_hypers['penalty'] = 'l2'
    else:
        best_hypers['penalty'] = 'elasticnet'

    if best_hypers['penalty'] != 'elasticnet':
        best_hypers.pop('l1_ratio')

    # Final model with best hyperparameters
    best_model = LogisticRegression(**best_hypers, solver='saga', max_iter=20000)

    return best_f1, best_model