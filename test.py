import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import forward_model_tf as fm
import pendulum as pend


tf.enable_eager_execution()

print(f"TensorFlow version: {tf.__version__}")
print(f"Eager execution: {tf.executing_eagerly()}")

#model = tf.keras.Sequential([
#    tf.keras.layers.Dense(20, input_shape=(4,), activation=tf.nn.sigmoid),
#    tf.keras.layers.Dense(20, activation=tf.nn.sigmoid),
#    tf.keras.layers.Dense(3)
#])

# load saved model

model = tf.keras.models.load_model("prediction_model.h5", compile=False)
#model = fm.build_forward_model()
print(model.summary())

#save_answer = input("save model? [yes|no]")

#if save_answer == "yes":
#   fm.save_model(model, "prediction_model.h5")
i = 0
for i in range (5):
    # test the model
    losses = pend.predict_simulation(model,
                                     fm.mean_loss,
                                     steps=200)
    print(f"Overall losses: {np.array(losses)}")

    plt.figure()
    plt.plot(losses, label="Prediction Losses")
    plt.legend()
    plt.show()


#fm.predict_states(model, [1,1,1], [2, -1, 3])




