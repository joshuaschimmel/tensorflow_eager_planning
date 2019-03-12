import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import helper_functions as _hp

tf.enable_eager_execution()

print(f"TensorFlow version: {tf.__version__}")
print(f"Eager execution: {tf.executing_eagerly()}")


tensor = tf.convert_to_tensor([1], dtype=tf.float32)
#print(tf.acos(tensor))

x = tf.constant(3.0)
with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    y = x * x

print(tape.gradient(y, x))
