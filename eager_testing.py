import os
import matplotlib.pyplot as plt
import pandas as  pd
import numpy as np
import gym

import tensorflow as tf

tf.enable_eager_execution()

print(f"TensorFlow version: {tf.__version__}")
print(f"Eager execution: {tf.executing_eagerly()}")


# +++ helper functions +++
def loss(model, x, y):
    y_ = model(x)
    return tf.losses.mean_squared_error(labels=y, predictions=y_)


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)




# +++ get data +++
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

model = tf.keras.Sequential([
    tf.keras.layers.Dense(20, input_shape=(4,), activation=tf.nn.sigmoid),
    tf.keras.layers.Dense(20, activation=tf.nn.sigmoid),
    tf.keras.layers.Dense(3)
])
print(model.summary())

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

global_step = tf.Variable(0)

# single optimization step
loss_value, grads = grad(model, train_data, train_labels)

print(f"Step: {global_step.numpy()}, Initial Loss: {loss_value.numpy()}")

optimizer.apply_gradients(zip(grads, model.variables), global_step)

print(f"Step: {global_step.numpy()}, Loss: "
      f"{loss(model, train_data, train_labels).numpy()}")

