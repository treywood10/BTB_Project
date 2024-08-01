# Import libraries
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
from bayes_opt import BayesianOptimization

# Objective function for Logistic Regression #
def log_tuning(X_train, y_train_encode, pbounds, n_init, n_iter, seed):
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
        else:
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
        pred = cross_val_predict(model, X_train, y_train_encode, cv=5)

        # F1 Score #
        f_score = f1_score(y_train_encode, pred, average='weighted')

        return f_score

    optimizer = BayesianOptimization(f=obj_log, pbounds=pbounds,
                                     random_state=seed, allow_duplicate_points=True)
    optimizer.maximize(init_points=n_init, n_iter=n_iter)

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
    else:
        best_hypers['multi_class'] = 'multinomial'

    best_model = LogisticRegression(**best_hypers, solver='saga', max_iter=20000)

    return best_f1, best_model