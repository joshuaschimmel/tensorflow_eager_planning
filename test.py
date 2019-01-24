import pandas as pd
import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

print(f"TensorFlow version: {tf.__version__}")
print(f"Eager execution: {tf.executing_eagerly()}")

model = tf.keras.Sequential([
    tf.keras.layers.Dense(20, input_shape=(4,), activation=tf.nn.sigmoid),
    tf.keras.layers.Dense(20, activation=tf.nn.sigmoid),
    tf.keras.layers.Dense(3)
])
print(model.summary())


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
print(f"Postshuffle: {train_dataset.output_shapes}")
i = 1
x = train_dataset.take(1000)
for n in train_dataset:
    print(i)
    i += 1

