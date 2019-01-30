import os
from collections import deque
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
import gym
import json

tf.enable_eager_execution()

print(f"TensorFlow version: {tf.__version__}")
print(f"Eager execution: {tf.executing_eagerly()}")


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

    partition_index = int(test_partition * len(df.index))

    # partition the data into a training and testing set
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


# +++ loss functions +++

def mean_loss(model: tf.keras.Model,
              model_input: tf.Tensor,
              model_target: tf.Tensor):
    """Calculates MSE of the model given output and expected values

    :param model: a model the mse is to be calculated for
    :param model_input: input
    :param model_target: teacher value
    :returns loss value
    """
    y_ = model(model_input)
    return tf.losses.mean_squared_error(labels=model_target, predictions=y_)


def abs_loss(model: tf.keras.Model,
              model_input: tf.Tensor,
              model_target: tf.Tensor):
    """Calculates the absolute difference loss for a model.

    :param model: a model the mae is to be calculated for
    :param model_input: input
    :param model_target: teacher value
    :returns loss value
    """
    y_ = model(model_input)
    return tf.losses.absolute_difference(labels=model_target, predictions=y_)


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
    return tf.sqrt(tf.losses.mean_squared_error(labels=model_target,
                                                 predictions=model_output
                                                 ))


def single_rmse(target: list, value: list) -> float:
    """Calculates the RMSE for a single value target pair

    :param value: predicted value
    :param target: target value
    :return: RMSE
    """
    return np.sqrt(np.mean(np.square(np.subtract(target, value))))


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
                mean_loss(model, feature, label)
            )
        # then use np.mean to calc the mean val loss
        val_loss = np.mean(epoch_val_losses)
        val_losses.append(val_loss)

        # Output the validation loss
        print(f"Validation loss in Epoch {epoch + 1}: {val_loss}")

    return train_losses, val_losses


def build_forward_model():
    # hyperparameter
    _epochs = 20
    _batch_size = 32
    _test_split = 0.1
    _validation_split = 0.1
    _learning_rate = 0.0002
    _shuffle = True

    # Reward function in pendulum environment:
    # -(theta^2 + 0.1*theta_dt^2 + 0.001*action^2)

    # +++ get data +++
    (features, labels,
     test_features, test_labels) = get_data("pendulum_data.csv",
                                            _test_split)

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

    # get the model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(40, input_shape=(4,), activation=tf.nn.relu),
        tf.keras.layers.Dense(40, activation=tf.nn.relu),
        tf.keras.layers.Dense(3)
    ])

    # print the models summary
    print(model.summary())

    # choose an optimizer
    optimizer = tf.train.AdamOptimizer(_learning_rate)

    train_losses, val_losses = train_function(
        model=model,
        data=train_dataset,
        set_len=len(labels),
        loss=mean_loss,
        optimizer=optimizer,
        epochs=_epochs,
        batch_size=_batch_size,
        validation_split=_validation_split,
        shuffle=_shuffle
    )

    # +++ save the model +++
    # save_model(model, "forward_model.h5")

    # +++ calculate test loss +++

    test_losses = []
    test_dataset = test_dataset.batch(_batch_size)
    for feature, label in test_dataset:
        test_losses.append(mean_loss(model, feature, label))

    print(f"Test loss: {np.mean(test_losses)}")

    # +++ plot the metrics +++

    # use epochs as scale where a tick represents the end of an epoch
    train_loss_x_axis = np.arange(0,
                                  _epochs,
                                  np.divide(_epochs, len(train_losses))
                                  )

    val_loss_x_axis = np.arange(1,
                                _epochs + 1,
                                np.divide(_epochs, len(val_losses)))

    plt.figure()
    # Train loss
    plt.plot(train_loss_x_axis, train_losses, label="Training Loss")

    # Validation loss
    plt.plot(val_loss_x_axis,
             val_losses,
             "--",
             label="Validation Loss",
             linewidth=2)

    # Test loss
    plt.plot([_epochs],
             [np.mean(test_losses)],
             "r+",
             label="Test Loss",
             markersize=10,
             linewidth=10,
             )

    plt.xlabel("End of Epoch #")
    plt.ylabel("MSE")
    plt.grid(b=True, alpha=0.25, linestyle="--")
    plt.tick_params(axis="both", which="major", direction="out")
    plt.legend()

    plt.show()

    return model


def predict_states(model: tf.keras.models.Model,
                   state_0: list,
                   plan: list
                   ) -> list:
    """Uses the model to predict the state after each step.

    :param model: Model used for prediction
    :param s_0: initial state s_0: [cos, sin, dot]
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
        current_state.append(next_action)

        # shape the input for the model (because of expected batching)
        # into the form [[current_state]]
        next_input = np.array(current_state).reshape(1,4)

        # let the model predict the next state
        prediction = model(next_input).numpy()

        # add the prediction to the return list
        predicted_states.append(prediction)

        # reassign current state to prediction
        current_state = prediction

    return predicted_states
