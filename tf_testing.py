import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
import gym
import time
import json


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


def build_model():
    """Builds the model and returns it."""
    # build model
    model = tf.keras.Sequential()
    # input shape 4 for 3 state values + 1 action
    model.add(layers.Dense(20, input_shape=(4,), activation="sigmoid"))
    model.add(layers.Dense(20, activation="sigmoid"))

    model.add(layers.Dense(3, activation="linear"))

    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss="mse",
                  metrics=["mae"])
    return model


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
train_data = train_df.loc[:, ["s_0_cos(theta)",
                              "s_0_sin(theta)",
                              "s_0_theta_dot",
                              "action"
                              ]].values
train_labels = train_df.loc[:, ["s_1_cos(theta)",
                                "s_1_sin(theta)",
                                "s_1_theta_dot"
                                ]].values

# do the same for the test data
test_data = test_df.loc[:, ["s_0_cos(theta)",
                            "s_0_sin(theta)",
                            "s_0_theta_dot",
                            "action"
                            ]].values
test_labels = test_df.loc[:, ["s_1_cos(theta)",
                              "s_1_sin(theta)",
                              "s_1_theta_dot"
                              ]].values


# get the model
model = build_model()
# print the models summary
print(model.summary())

# if validation loss does not change for 20 steps, stop training
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss",
                                           patience=20,
                                           min_delta=1/1000
                                           )

# start the training and save the logs
history = model.fit(train_data,
                    train_labels,
                    epochs=100,
                    batch_size=32,
                    validation_split=0.1,
                    shuffle=True,
                    callbacks=[early_stop]
                    )

# evaluate the model using the test data
[loss, mae] = model.evaluate(test_data,
                             test_labels,
                             batch_size=32,
                             verbose=1
                             )

# print the evaluated metrics
print(f"The used metrics are: {model.metrics_names}")
print(f"Evaluation of model:\nloss: {loss}\nmae: {mae}")

plot_history(history)


# +++ test model on new data +++
plan_length = 10

states = run_simulation(plan_length+1)
# all states are data and label except for the first and last one
# get all states except the last
simulation_data = states[:-1]
simulation_label = []
# align the states by ignoring the first element
for state in states[1:]:
    # cut away the action
    simulation_label.append(state[0][:3].reshape((1, 3)))

simulation_loss = []
simulation_mae = []
for i in range(plan_length):
    [s_loss, s_mae] = model.evaluate(simulation_data[i],
                                     simulation_label[i],
                                     batch_size=1,
                                     verbose=1
                                     )
    simulation_loss.append(s_loss)
    simulation_mae.append(s_mae)
plan_steps = np.array(range(plan_length+1)[1:])

plt.figure()
plt.xlabel("Steps")
plt.ylabel("Error")
plt.plot(plan_steps, np.array(
    simulation_loss), label="MSE")
plt.plot(plan_steps, np.array(
    simulation_mae), label="MAE")
plt.legend()
# plt.ylim([0, 2])

plt.show()
