# Import libraries
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
from bayes_opt import BayesianOptimization

# Objective function for SVM #
def svm_tuning(X_train, y_train_encode, pbounds, n_init, n_iter, seed):
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
        model = SVC(C=C, kernel=kernel,
                    degree=int(degree), gamma=gamma,
                    shrinking=shrinking, random_state=seed,
                    probability=True)

        pred_proba = cross_val_predict(model, X_train, y_train_encode, cv=5, method='predict_proba')
        pred = np.argmax(pred_proba, axis=1)  # Choose the class with the highest probability

        # F1 Score #
        f_score = f1_score(y_train_encode, pred, average='weighted')

        return f_score

    optimizer = BayesianOptimization(f=obj_svm, pbounds=pbounds,
                                     random_state=seed, allow_duplicate_points=True)
    optimizer.maximize(init_points=n_init, n_iter=n_iter)

    # Pull best info #
    best_hypers = optimizer.max['params']
    best_f1 = optimizer.max['target']

    # Adjust hypers #
    if best_hypers['kernel'] <= 1:
        best_hypers['kernel'] = 'linear'
    elif best_hypers['kernel'] <= 2:
        best_hypers['kernel'] = 'poly'
    elif best_hypers['kernel'] <= 3:
        best_hypers['kernel'] = 'rbf'
    else:
        best_hypers['kernel'] = 'sigmoid'

    if best_hypers['gamma'] <= 0.5:
        best_hypers['gamma'] = 'scale'
    else:
        best_hypers['gamma'] = 'auto'

    best_hypers['shrinking'] = bool(round(best_hypers['shrinking']))

    best_hypers['degree'] = round(best_hypers['degree'])

    best_model = SVC(**best_hypers, probability=True)

    return best_f1, best_model