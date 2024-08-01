# Import libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
from bayes_opt import BayesianOptimization


# Objective function for random forest #
def rf_tuning(X_train, y_train_encode, pbounds, n_init, n_iter, seed):
    def obj_forest(n_estimators, criterion,
                   max_depth, max_features,
                   bootstrap):
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
        model = RandomForestClassifier(n_estimators=int(n_estimators),
                                       criterion=criterion,
                                       max_depth=int(max_depth),
                                       max_features=max_features,
                                       bootstrap=bootstrap)

        # Cross validation #
        pred = cross_val_predict(model, X_train, y_train_encode, cv=5)

        # F1 Score #
        f_score = f1_score(y_train_encode, pred, average='weighted')

        return f_score

    # Set optimizer #
    optimizer = BayesianOptimization(f=obj_forest, pbounds=pbounds,
                                     random_state=seed, allow_duplicate_points=True)

    # Call maximizer #
    optimizer.maximize(init_points=n_init, n_iter=n_iter)

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