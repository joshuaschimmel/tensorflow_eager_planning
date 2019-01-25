import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
import gym
import time
import json
import os

tf.enable_eager_execution()

print(f"TensorFlow version: {tf.__version__}")
print(f"Eager execution: {tf.executing_eagerly()}")


def get_action(action_space) -> float:
    """returns an action in the actionspace"""
    return action_space.sample()


def min_max_normalization(old_value: float,
                          old_range: dict,
                          new_range: dict) -> float:
    new_value = ((old_value - old_range["min"])
                 / (old_range["max"] - old_range["min"])) \
                * (new_range["max"] - new_range["min"]) \
                + new_range["min"]
    return new_value


def save_model(model: tf.keras.models.Model,
               file_path: str):
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


def run_simulation(steps: int = 10) -> list:
    """Runs the simulation for steps steps.

    :param steps: number of steps
    :return: list of states as numpy arrays
    """
    env = gym.make('Pendulum-v0')
    simulation_states = []

    cos, sin, dot = env.reset()
    for i in range(steps):
        env.render(mode="human")
        action = get_action(env.action_space)
        s_0 = np.array([cos, sin, dot, action])
        s_0 = s_0.reshape(1, 4)
        simulation_states.append(s_0)

        s_1, _, _, _ = env.step(action)
        cos, sin, dot = s_1

    env.close()
    return simulation_states


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
        action = get_action(env.action_space)
        # build the model input
        s_0 = np.array([cos, sin, dot, action]).reshape(1, 4)
        # do a step and get the next state
        s_1,_,_,_ = env.step(action)
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


def forward_pass():
    # hyperparameter
    _epochs = 10
    _batch_size = 32
    _test_split = 0.1
    _validation_split = 0.1
    _learning_rate = 0.001
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
        tf.keras.layers.Dense(20, input_shape=(4,), activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(20, activation=tf.nn.sigmoid),
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
    save_model(model, "forward_model.h5")

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
