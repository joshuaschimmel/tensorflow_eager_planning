import os
from collections import deque
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
import json

tf.enable_eager_execution()

print(f"TensorFlow version: {tf.__version__}")
print(f"Eager execution: {tf.executing_eagerly()}")

# data ranges
data_ranges = [
    [-1.0, 1.0],
    [-1.0, 1.0],
    [-8.0, 8.0],
    [-2.0, 2.0]
]

# +++ i/o functions +++

def save_model(model: tf.keras.models.Model,
               file_path: str = "forward_model.h5"
               ):
    """Saves the given Keras Model in the given File.

    :param model: Model to be saved
    :param file_path: File the Model should be saved in
    :return: None
    """
    # save test
    model.save(file_path)


def load_model(file_path: str,
               pre_compile: bool = False
               ) -> tf.keras.models.Model:
    """Loads a Keras Model from the given file.

    :param file_path: Path to file
    :param pre_compile: Whether the model should be compiled before returning
    :return: the Keras Model
    """
    model = tf.keras.models.load_model(file_path, compile=pre_compile)
    print(model.summary())
    return model


# +++ preprocessing functions +++

def get_data(file_path: str,
             test_partition: float,
             ) -> (np.array, np.array, np.array, np.array):

    # read the datafile
    df = pd.read_csv(file_path)

    # shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)

    # partition the data into a training and testing set
    partition_index = int(test_partition * len(df.index))
    train_df = df.iloc[partition_index:, :]
    test_df = df.iloc[:partition_index, :]

    # strip the data and labes from the training sets
    features = train_df.loc[:, [
                                   "s_0_cos(theta)",
                                   "s_0_sin(theta)",
                                   "s_0_theta_dot",
                                   "action"
                               ]].values

    labels = train_df.loc[:, [
                                 "s_1_cos(theta)",
                                 "s_1_sin(theta)",
                                 "s_1_theta_dot"
                             ]].values

    # do the same for the test data
    test_features = test_df.loc[:, [
                                       "s_0_cos(theta)",
                                       "s_0_sin(theta)",
                                       "s_0_theta_dot",
                                       "action"
                                   ]].values

    test_labels = test_df.loc[:, ["s_1_cos(theta)",
                                  "s_1_sin(theta)",
                                  "s_1_theta_dot"
                                  ]].values

    return features, labels, test_features, test_labels


def normalize_dot() -> list:
    """Normalizes the world state + action for the pendulum task.

    :param state_list: state of the world + action
    :return: normalized state + action in shape of input
    """




def get_model(hidden_layers: int = 2,
              neurons: int = 20,
              dropout_rate: float = 0.5
              ) -> tf.keras.models.Model:
    """Creates a sequential Keras model with the given parameters.

    Using the parameters, this function creates a Keras model with
    at least one fully connected Dense followed by a Dropout layer.
    After that, the rest of the Dense layers will be added if
    layers > 1.

    :param hidden_layers: Number of fully connected Dense layers
    :param neurons: Number of neuron in each hidden layer
    :param dropout_rate: float for the dropout chance in the
        second "layer"
    :return: the finished Keras model
    """
    _layers = 0
    if hidden_layers > 0:
        _layers = hidden_layers

    # prepare model by gettin a simple sequential object
    model = tf.keras.Sequential()
    # add first hidden layer with input shape
    model.add(tf.keras.layers.Dense(neurons,
                                    input_shape=(4,),
                                    activation=tf.nn.relu
                                    ))
    # add dropout layer
    model.add(tf.keras.layers.Dropout(dropout_rate))

    # add any additional dense layers
    for i in range(_layers):
        model.add(tf.keras.layers.Dense(neurons,
                                        activation=tf.nn.relu
                                        ))

    # add output layer
    model.add(tf.keras.layers.Dense(3))

    # return the model
    return model


# +++ loss functions +++

def mse_loss(model: tf.keras.Model,
             model_input: tf.Tensor,
             model_target: tf.Tensor):
    """Calculates MSE of the model given output and expected values

    :param model: a model the mse is to be calculated for
    :param model_input: input
    :param model_target: teacher value
    :returns loss value
    """
    y_ = model(model_input)
    _reduction_string = "weighted_sum_over_batch_size"
    return tf.losses.mean_squared_error(labels=model_target,
                                        predictions=y_,
                                        reduction=_reduction_string
                                        )


def abs_loss(model: tf.keras.Model,
             model_input: tf.Tensor,
             model_target: tf.Tensor):
    """Calculates the absolute difference loss for a model.

    :param model: the model used for the prediction
    :param model_input: input tensor to be used for the prediction
    :param model_target: wanted output tensor of the prediction
    :returns loss float Tensor with the shape of target
    """
    y_ = model(model_input)
    return tf.losses.absolute_difference(labels=model_target,
                                         predictions=y_,
                                         reduction="none"
                                         )


def mae_loss(model: tf.keras.Model,
             model_input: tf.Tensor,
             model_target: tf.Tensor):
    """Calculates the Mean Absolute Error for a model given input and target.

    :param model: the model used for the prediction
    :param model_input: input tensor to be used for the prediction
    :param model_target: wanted output tensor of the prediction
    :returns loss float Tensor with the shape of a scalar
    """
    _y = model(model_input)
    _reduction_string= "weighted_sum_over_batch_size"
    return tf.losses.absolute_difference(labels=model_target,
                                         predictions=_y,
                                         reduction=_reduction_string
                                         )


def rmse_loss(model: tf.keras.Model,
              model_input: tf.Tensor,
              model_target: tf.Tensor):
    """Calculates RMSE of the model given input and teacher values

    :param model: a kerase model for which to calculate the RMSE
    :param model_input: input for the model
    :param model_target: target|teacher values
    :return: RMSE value
    """
    model_output = model(model_input)
    return tf.sqrt(
        tf.losses.mean_squared_error(labels=model_target,
                                     predictions=model_output,
                                     reduction="weighted_sum_over_batch_size"
                                     ))


@PendingDeprecationWarning
def single_rmse(target: list, value: list) -> float:
    """Calculates the RMSE for a single value target pair

    :param value: predicted value
    :param target: target value
    :return: RMSE
    """
    return np.sqrt(np.mean(np.square(np.subtract(target, value))))


@PendingDeprecationWarning
def single_absolute_error(target: list, prediction: list):
    """Calculates the absolute error of an array given a target,
    by addition of absolut differences

    :param target: target|teacher values
    :param prediction: actual value
    :return: the absolut error
    """
    differences = np.subtract(target, prediction)
    absolute_diffs = np.absolute(differences)
    absolute_error = np.sum(absolute_diffs, axis=None, dtype=np.float32)
    return absolute_error


def list_rmse(values: list, targets: list) -> list:
    """Calculates the RMSE over lists of values and targets.

    Calculates multiple RMSE for a list of values and targets.
    Returns a list calculated RMSEs

    :param values: List of predicted values
    :param targets: List with wanted target values
    :return: list with RMSE values
    """
    _calculated_rmses = []

    for _v, _t in zip(values, targets):
        # RMSE function
        _calculated_rmses.append(single_rmse(value=_v, target=_t))

    return _calculated_rmses


# +++ training functions +++

def train_function(model: tf.keras.Model,
                   data: tf.data.Dataset,
                   set_len: int,
                   loss,
                   optimizer: tf.train.Optimizer,
                   epochs: int,
                   validation_split: float,
                   batch_size: int = 32,
                   shuffle: bool = True
                   ) -> (list, list):
    """Trains a Keras Model using data.

    :param model: The model to be trained
    :param data: Dataset in the shape (feature, label)
    :param set_len: The length of the Dataset (sorry)
    :param loss: The loss function to be applied
    :param optimizer: The optimizer to apply the loss
    :param epochs: # of iterations over the training set
    :param validation_split: Value in [0,1] defining the split of training
        and validation set.
    :param batch_size: Size of the batches for a single pass through
        the network. Default 32.
    :param shuffle: Whether the Dataset should be shuffled (for each
        iteration). Default True.
    :return:
    """

    # list of losses total
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # starting epoch

        # check if shuffle is enabled
        if shuffle:
            data.shuffle(100, reshuffle_each_iteration=True)

        # split the validation set off
        train_data = data.skip(set_len * validation_split)
        val_data = data.take(set_len * validation_split)

        # batch the data accordingly
        train_data = train_data.batch(batch_size)
        val_data = val_data.batch(batch_size)

        for feature, label in train_data:
            with tf.GradientTape() as tape:
                loss_value = loss(
                    model,
                    feature,
                    label
                )
            train_losses.append(loss_value)

            # get the gradients from the tape
            grads = tape.gradient(loss_value, model.trainable_variables)

            # apply the gradients using the optimizer
            optimizer.apply_gradients(
                zip(grads, model.variables),
                global_step=tf.train.get_or_create_global_step()
            )

        # calculate validation loss
        epoch_val_losses = []
        # for every batch in val_data: calculate and add the loss
        for feature, label in val_data:
            epoch_val_losses.append(
                loss(model, feature, label)
            )
        # then use np.mean to calc the mean val loss
        val_loss = np.mean(epoch_val_losses)
        val_losses.append(val_loss)

        # Output the validation loss
        print(f"Validation loss in Epoch {epoch + 1}: {val_loss}")

    return train_losses, val_losses


def test_model(model: tf.keras.Model,
               test_set: tf.data.Dataset,
               loss
               ) -> list:
    """Calculates the test loss for every batch in the test_set

    :param model: Keras model
    :param test_set: test_data set
    :param loss: loss function
    :return: list of losses
    """
    test_losses = []
    for feature, label in test_set:
        test_losses.append(loss(model, feature, label))
    print(f"Mean test loss: {np.mean(test_losses)}")
    return test_losses


def plot_model_performance(training_x, training_y,
                           validation_x, validation_y,
                           test_x, test_y):
    """Prints the metrices for the model

    :param training_x: Training Step in all epoch
    :param training_y: Error during training
    :param validation_x: Validation after each epoch
    :param validation_y: Error during validation
    :param test_x: Values of max epoch
    :param test_y: Test_Error after last epoch
    :return:
    """
    plt.figure()
    # Train loss
    plt.plot(training_x,
             training_y,
             label="Training Loss")

    # Validation loss
    plt.plot(validation_x,
             validation_y,
             "--",
             label="Validation Loss",
             linewidth=2)

    # Test loss
    plt.plot(test_x,
             test_y,
             "r+",
             label="Test Loss",
             markersize=10,
             linewidth=10,
             )

    plt.xlabel("End of Epoch #")
    plt.ylabel("Error Value")
    plt.grid(b=True, alpha=0.25, linestyle="--")
    plt.tick_params(axis="both", which="major", direction="out")
    plt.legend()

    plt.show()


def build_forward_model(
    epochs: int = 10,
    batch_size: int = 32,
    test_split: float = 0.1,
    validation_split: float = 0.1,
    learning_rate: float = 0.001,
    shuffle: bool = True,
    neurons: int = 20,
    dropout_rate: float = 0.5,
    hidden_layers: int = 1,
    loss=rmse_loss,
    plot_performance: bool = True
):
    """This function is the main function for the training.
    It takes the hyperparameters as input and feeds them into
    the model and training functions.
    Returns the trained model.

    :param epochs: # of iterations over the same dataset
    :param batch_size: # of entries for a single passthrough
    :param test_split: in [0,1], proportion of dataset to be saved
        for testing after the training is complete
    :param validation_split: in [0,1], proportion of data to be saved
        during each epoch for validation after the epoch is finished
    :param learning_rate: rate at which the optimizer will apply the
        gradients
    :param shuffle: Whether the dataset should be shuffled before
        each epoch
    :param neurons: # of neurons in each hidden layer
    :param dropout_rate: rate of which connections in the layer
        will be blocked
    :param hidden_layers: # of hidden layers in the model
    :param loss: loss function used for calculating difference
        between prediction and target values
    :param plot_performance: whether a final plot with the performance
        should be made
    :return: A trained Keras model
    """

    # Reward function in pendulum environment:
    # -(theta^2 + 0.1*theta_dt^2 + 0.001*action^2)

    # +++ get data +++
    (features, labels,
     test_features, test_labels) = get_data(
        file_path="data/pendulum_data_dot_stretched.csv",
        test_partition=test_split
    )

    # reformat data into tensors and pre-shuffle them
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(features, dtype=tf.float32),
         tf.cast(labels, dtype=tf.float32)
         ))
    train_dataset = train_dataset.shuffle(100)
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(test_features, dtype=tf.float32),
         tf.cast(test_labels, dtype=tf.float32)
         ))
    test_dataset = test_dataset.shuffle(100)

    # +++ get the model +++
    model = get_model(hidden_layers=hidden_layers,
                      neurons=neurons,
                      dropout_rate=dropout_rate
                      )
    # print the models summary
    print(model.summary())

    # choose an optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)

    train_losses, val_losses = train_function(
        model=model,
        data=train_dataset,
        set_len=len(labels),
        loss=loss,
        optimizer=optimizer,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        shuffle=shuffle
    )

    # +++ calculate test loss +++
    test_dataset = test_dataset.batch(batch_size)
    test_losses = test_model(model,
                             test_dataset,
                             loss)

    # +++ plot the metrics +++
    # use epochs as scale where a tick represents the end of an epoch
    train_loss_x_axis = np.arange(0,
                                  epochs,
                                  np.divide(epochs, len(train_losses))
                                  )

    val_loss_x_axis = np.arange(1,
                                epochs + 1,
                                np.divide(epochs, len(val_losses)))

    if plot_performance:
        # Plot the performance of the model
        plot_model_performance(train_loss_x_axis, train_losses,
                               val_loss_x_axis, val_losses,
                               [epochs] * len(test_losses), test_losses
                               )

    return model


def predict_states(model: tf.keras.models.Model,
                   state_0: list,
                   plan: list
                   ) -> list:
    """Uses the model to predict the state after each step.

    :param model: Model used for prediction
    :param state_0: initial state s_0: [cos, sin, dot]
    :param plan: list of actions [a_1, ..., a_n]
    :return: list of predicted states [s_0, ..., s_n]
    """
    # initial state does not need to be predicted
    predicted_states = [state_0]

    # use deque for efficient deconstruction of the list
    plan = deque(plan)

    # get the current state
    current_state = state_0

    while plan:
        # get the next action
        next_action = plan.popleft()

        # merge it with the current state
        next_input = np.append(current_state, next_action)

        # shape the input for the model (because of expected batching)
        # into the form [[current_state]]
        next_input = np.array(next_input).reshape(1, 4)

        # let the model predict the next state
        prediction = model(next_input).numpy()[0]

        # add the prediction to the return list
        predicted_states.append(prediction)

        # reassign current state to prediction
        current_state = prediction

    return predicted_states


def predict_simulation(predictor: tf.keras.models.Model,
                       loss,
                       steps: int
                       ) -> list:
    """Uses the predictor to predict the simulation states.

    :param predictor: Model used for prediction.
    :param loss: A function calculation the loss.
    :param steps: # of steps to be done in the simulation.
    :return: List of losses calculated by the loss function.
    """
    # build the simulation
    env = gym.make("Pendulum-v0")
    prediction_losses = []

    # get the initial state
    cos, sin, dot = env.reset()
    for i in range(steps):
        # render the environment
        env.render(mode="human")
        # get a random action
        # action = get_action(env.action_space)

        action = np.array([0])
        # build the model input
        s_0 = np.array([cos, sin, dot, action]).reshape(1, 4)
        # do a step and get the next state
        s_1, _, _, _ = env.step(action)
        # reassign for next state
        cos, sin, dot = s_1
        # reshape s_1 into a label
        s_1 = s_1.reshape(1, 3)
        # compare the models prediction to reality
        prediction_loss = loss(predictor, s_0, s_1)
        print(f"Prediction Loss is: {prediction_loss}")
        prediction_losses.append(prediction_loss)

    env.close()
    return prediction_losses
