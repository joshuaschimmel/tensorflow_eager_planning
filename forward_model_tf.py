import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
import gym
import time
import json


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


# headers = ["s_0_cos(theta)", "s_0_sin(theta)", "s_0_theta_dot",
#                "s_1_cos(theta)", "s_1_sin(theta)", "s_1_theta_dot",
#                "action", "reward"
#                ]
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

    #    for i in range(25):
    #     x = env.reset()
    #     env.render(mode="human")
    #     action = get_action(env.action_space)
    #     x = np.hstack((x, action))
    #     y = x.reshape(1, 4)
    #
    #     prediction = model.predict(y)
    #     print("Preditction: ", prediction)
    #     x, reward, done, info = env.step(action)
    #     x_test = x.reshape(1, 3)
    #     print(x_test - prediction)
    #     # for o, p in zip(x_test, prediction[0]):
    #     #      print("Diff: ", o-p)
    #     # result = model.test_on_batch(y, x_test)
    #     # print(model.metrics_names, "\n", result)
    #     print("x_test: ", x_test)
    #
    # env.close()


def plot_history(history) -> None:
    """Plots the history returned by the fit function."""
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Square Error")
    plt.plot(history.epoch, np.array(
        history.history['loss']), label="Train Loss")
    plt.plot(history.epoch, np.array(
        history.history['val_loss']), label="Validation Loss")
    plt.legend()
    # plt.ylim([0, 5])

    plt.show()


def mean_loss(model: tf.keras.Model,
              x: tf.Tensor,
              y: tf.Tensor,
              training: bool = False):
    """Calculates MSE of the model given output and expected values

    :param model: a model the mse is to be calculated for
    :param x: input
    :param y: teacher value
    :param training: whether the model is being trained
    :returns loss value
    """
    y_ = model(x)
    return tf.losses.mean_squared_error(labels=y, predictions=y_)


# +++ script starts here +++

# Reward function in pendulum environment:
# -(theta^2 + 0.1*theta_dt^2 + 0.001*action^2)

# read the datafile
df = pd.read_csv("pendulum_data.csv")

# shuffle the data
df = df.sample(frac=1).reset_index(drop=True)

partition = 0.1
partition_index = int(partition * df.shape[0])

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

train_dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(features, dtype=tf.float32),
     tf.cast(labels, dtype=tf.float32)
     ))
train_dataset = train_dataset.shuffle(100)
# del df, train_df, features, labels

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


# get the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(20, input_shape=(4,), activation=tf.nn.sigmoid),
    tf.keras.layers.Dense(20, activation=tf.nn.sigmoid),
    tf.keras.layers.Dense(3)
])

# print the models summary
print(model.summary())

# choose an optimizer
optimizer = tf.train.AdamOptimizer(0.001)

# hyperparameter
_epochs = 100
_batch_size = 32
_validation_split = 0.1
_shuffle = True


def train_function(model: tf.keras.Model,
                   data: tf.data.Dataset,
                   loss,
                   optimizer: tf.train.Optimizer,
                   epochs: int,
                   validation_split: float,
                   batch_size: int = 32,
                   shuffle: bool = True
                   ) -> (list, list):
    """ Trains a keras model using input x and target y

    :param model: A Keras model
    :param x: input
    :param y: target
    :param loss: loss function
    :param optimizer: optimizer to apply the gradients
    :param epochs: # of repetitions
    :param batch_size: # of input in a single forward pass
    :param validation_split: [0,1] float for validation after each epoch
    :param shuffle: whether the data should be shuffled before each epoch
    :return: list of losses after each epoch
    """
    # list of losses total
    loss_history = []
    loss_mean_history = []
    # check if shuffle is enabled
    if shuffle:
        data.shuffle(100, reshuffle_each_iteration=True)
    # batch the data accordingly
    data = data.batch(batch_size)

    for epoch in range(epochs):
        # starting epoch
        # error in each batch
        batch_loss = []
        for feature, label in data:

            with tf.GradientTape() as tape:
                loss_value = loss(
                    model,
                    feature,
                    label
                )
            batch_loss.append(loss_value)
            loss_history.append(loss_value)

            grads = tape.gradient(loss_value, model.trainable_variables)

            optimizer.apply_gradients(
                zip(grads, model.variables),
                global_step=tf.train.get_or_create_global_step()
            )

            loss_mean_history.append(np.mean(batch_loss))

        print(f"Epoch {epoch} finished!")
    return loss_history, loss_mean_history


losses, losses_mean = train_function(
    model=model,
    data=train_dataset,
    loss=mean_loss,
    optimizer=optimizer,
    epochs=10,
    batch_size=_batch_size,
    validation_split=0.1,
    shuffle=True

)
print(f"Length of losses: {len(losses)}")

plt.figure()
plt.plot(losses)
plt.plot(losses_mean)
plt.xlabel("Batch #")
plt.ylabel("MSE")
plt.show()
