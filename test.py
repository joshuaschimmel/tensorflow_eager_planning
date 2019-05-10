import pendulum as p
import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import seaborn as sns
#import helper_functions as _hp

tf.enable_eager_execution()

print(f"TensorFlow version: {tf.__version__}")
print(f"Eager execution: {tf.executing_eagerly()}")


def f():
    tips = sns.load_dataset("tips")
    g = sns.relplot(x="total_bill", y="tip", data=tips)
    return g.fig


f1 = f()
f2 = f()
print("hello")
plt.show()
