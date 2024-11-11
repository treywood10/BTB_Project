from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
from tensorflow.keras import optimizers
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
from bayes_opt import BayesianOptimization


def net_tuning(X_train, y_train_dums, pbounds, n_init, n_iter, seed, labeler):
    def obj_net(batch_size, epochs, activation, num_nodes,
                num_hidden_layers, learning_rate, rate, optimizer):
        print("Running obj_net with parameters:")
        print(f"batch_size: {batch_size}, epochs: {epochs}, activation: {activation}, "
              f"num_nodes: {num_nodes}, num_hidden_layers: {num_hidden_layers}, "
              f"learning_rate: {learning_rate}, rate: {rate}, optimizer: {optimizer}")

        # Set Optimizer #
        if optimizer <= 0.33:
            optimizer = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer <= 0.66:
            optimizer = optimizers.Adagrad(learning_rate=learning_rate)
        else:
            optimizer = optimizers.RMSprop(learning_rate=learning_rate)

        # Set activation function #
        if activation <= 0.33:
            activation = 'relu'
        elif activation <= 0.66:
            activation = 'sigmoid'
        else:
            activation = 'tanh'

        # Instantiate model
        model = Sequential()
        model.add(Input(shape=(X_train.shape[1],)))
        model.add(Dense(int(num_nodes), activation=activation))
        model.add(BatchNormalization())

        # Set hidden layers with batch normalization and dropout #
        for _ in range(int(num_hidden_layers)):
            model.add(Dense(int(num_nodes), activation=activation))
            if 0 < rate < 1:
                model.add(Dropout(rate=rate, seed=seed))

        # Output layer
        model.add(Dense(len(labeler.classes_), activation='softmax'))

        # Compile model
        model.compile(optimizer=optimizer, loss='categorical_crossentropy')

        # Early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Wrap model for cross-validation
        net_model = KerasClassifier(
            model=lambda: model,
            batch_size=int(batch_size),
            epochs=int(epochs),
            validation_split=0.2,
            callbacks=[early_stopping],
            random_state=seed
        )

        # Cross-validation with error handling
        try:
            pred = cross_val_predict(net_model, X_train, y_train_dums, cv=5)
            print("Cross-validation successful.")
            print("Prediction shape:", pred.shape, "Label shape:", y_train_dums.shape)
        except Exception as e:
            print("Error during cross-validation:", e)
            return 0  # Return a low score to avoid optimizing invalid configs

        # Calculate F1 Score
        try:
            f_score = f1_score(y_train_dums, pred, average='weighted')
            print("F1 score calculated:", f_score)
        except Exception as e:
            print("Error calculating F1 score:", e)
            f_score = 0  # Return a low score if calculation fails

        return f_score

    # Bayesian optimization
    optimizer = BayesianOptimization(f=obj_net, pbounds=pbounds, random_state=seed, allow_duplicate_points=True)
    optimizer.maximize(init_points=n_init, n_iter=n_iter)

    # Extract best parameters and model
    best_hypers = optimizer.max['params']
    best_f1 = optimizer.max['target']

    # Replace optimizer and activation values for final model
    best_hypers['optimizer'] = 'Adam' if best_hypers['optimizer'] <= 0.5 else 'Adagrad'
    optimizer = optimizers.Adam(learning_rate=best_hypers['learning_rate']) if best_hypers[
                                                                                   'optimizer'] == 'Adam' else optimizers.Adagrad(
        learning_rate=best_hypers['learning_rate'])

    best_hypers['activation'] = 'relu' if best_hypers['activation'] <= 0.33 else 'sigmoid' if best_hypers[
                                                                                                  'activation'] <= 0.66 else 'tanh'

    # Build and compile the final model
    final_model = Sequential()
    final_model.add(
        Dense(int(best_hypers['num_nodes']), activation=best_hypers['activation'], input_shape=(X_train.shape[1],)))
    final_model.add(BatchNormalization())

    for _ in range(int(best_hypers['num_hidden_layers'])):
        final_model.add(Dense(int(best_hypers['num_nodes']), activation=best_hypers['activation']))
        final_model.add(Dropout(rate=best_hypers['rate'], seed=seed))

    final_model.add(Dense(len(labeler.classes_), activation='softmax'))
    final_model.compile(optimizer=optimizer, loss='categorical_crossentropy')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train final model
    final_model.fit(X_train, y_train_dums, batch_size=int(best_hypers['batch_size']), epochs=int(best_hypers['epochs']),
                    validation_split=0.2, callbacks=[early_stopping])

    return best_f1, final_model