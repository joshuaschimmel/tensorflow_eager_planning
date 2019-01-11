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

default_records = [tf.float32] * 8

ds_file = tf.data.experimental.CsvDataset(
    filenames="pendulum_data.csv",
    record_defaults=default_records,
    header=True
)

#ds_test1 = ds_file.shuffle(2)

print("types", ds_file.output_types)
print("shapes", ds_file.output_shapes)

iter = ds_file.__iter__()

print(iter)
print(type(iter))

print(iter.next())

i = 0
for x in ds_file:
    print(f"{i}: ", x)
    i += 1
    if i >= 3:
        break
