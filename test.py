import pandas as pd
import numpy as np
import tensorflow as tf
import forward_model_tf as fm
import matplotlib.pyplot as plt


tf.enable_eager_execution()

print(f"TensorFlow version: {tf.__version__}")
print(f"Eager execution: {tf.executing_eagerly()}")

#model = tf.keras.Sequential([
#    tf.keras.layers.Dense(20, input_shape=(4,), activation=tf.nn.sigmoid),
#    tf.keras.layers.Dense(20, activation=tf.nn.sigmoid),
#    tf.keras.layers.Dense(3)
#])

# load saved model

model = tf.keras.models.load_model("forward_model.h5", compile=False)
print(model.summary())

# test the model
losses = fm.predict_simulation(model,
                               fm.mean_loss,
                               steps=10)
print(f"Overall losses: {np.array(losses)}")

plt.figure()
plt.plot(losses, label="Prediction Losses")
plt.legend()
plt.show()




