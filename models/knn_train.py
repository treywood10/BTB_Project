# Import libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
from bayes_opt import BayesianOptimization

# Objective function for KNN #
def knn_tuning(X_train, y_train_encode, pbounds, n_init, n_iter, seed):
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
        pred = cross_val_predict(model, X_train, y_train_encode, cv = 5)

        # F1 Score #
        f_score = f1_score(y_train_encode, pred, average = 'weighted')

        return f_score

    optimizer = BayesianOptimization(f=obj_knn, pbounds=pbounds, random_state=seed, allow_duplicate_points = True)
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
