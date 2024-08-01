# Import libraries
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras import optimizers
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
from bayes_opt import BayesianOptimization

# Create optimizer function #
def net_tuning(X_train, y_train_dums, pbounds, n_init, n_iter, seed, labeler):
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


    optimizer = BayesianOptimization(f=obj_net, pbounds=pbounds, random_state=seed, allow_duplicate_points = True)
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