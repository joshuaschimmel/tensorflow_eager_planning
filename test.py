import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import helper_functions as _hp

tf.enable_eager_execution()

print(f"TensorFlow version: {tf.__version__}")
print(f"Eager execution: {tf.executing_eagerly()}")


s_0 = tf.random.uniform(shape=(3, 1))
action = tf.random.uniform(shape=(1,))

with tf.GradientTape() as tape:
    tape.watch(action)
    next_input = tf.concat([tf.squeeze(s_0), action], axis=0)

grads = tape.gradient(next_input, action)
print(grads)

