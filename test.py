import pendulum as p
import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
#import helper_functions as _hp

tf.enable_eager_execution()

print(f"TensorFlow version: {tf.__version__}")
print(f"Eager execution: {tf.executing_eagerly()}")


for x in p.get_state_generator(5):
    print(x)
    break
