import pendulum as p
import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import seaborn as sns
import world_model
import pendulum
import plan_optimizer as po
import gym
# import helper_functions as _hp

tf.enable_eager_execution()

print(f"TensorFlow version: {tf.__version__}")
print(f"Eager execution: {tf.executing_eagerly()}")

optimizer = tf.train.GradientDescentOptimizer(0.1)
step = tf.Variable(2)
a = tf.Variable([[10, 23, 2],
                 [8, 4.3, 4]
                 ],
                dtype="float32")
g = tf.constant([[1, 0.5, 0.25],
                 [0.2, 1, 1]
                 ],  dtype="float32")

zipped = zip([g], [a])
optimizer.apply_gradients(zipped, step)
print(a)
