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
              input: tf.Tensor,
              label: tf.Tensor):
    """Calculates MSE of the model given output and expected values

    :param model: a model the mse is to be calculated for
    :param input: input
    :param label: teacher value
    :param training: whether the model is being trained
    :returns loss value
    """
    y_ = model(input)
    return tf.losses.mean_squared_error(labels=label, predictions=y_)


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
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # starting epoch

        # check if shuffle is enabled
        if shuffle:
            data.shuffle(100, reshuffle_each_iteration=True)

        # split of the validation set
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

            grads = tape.gradient(loss_value, model.trainable_variables)

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


# +++ script starts here +++
# hyperparameter
_epochs = 10
_batch_size = 64
_test_split = 0.1
_validation_split = 0.1
_learning_rate = 0.0005
_shuffle = True

# Reward function in pendulum environment:
# -(theta^2 + 0.1*theta_dt^2 + 0.001*action^2)

# +++ get data +++
features, labels, test_features, test_labels = get_data("pendulum_data.csv",
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
# checkpoint_dir = "/home/joshua/Projects/BA/models"
# os.makedirs(checkpoint_dir, exist_ok=True)
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# root = tf.train.Checkpoint(model=model)

# root.save(checkpoint_prefix)

# root.restore(tf.train.latest_checkpoint(checkpoint_dir))

# save test
model.save("save_test.h5")


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
