import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import forward_model_tf as fm
import pendulum as pend
import helper_functions as hf


tf.enable_eager_execution()

print(f"TensorFlow version: {tf.__version__}")
print(f"Eager execution: {tf.executing_eagerly()}")

#model = tf.keras.Sequential([
#    tf.keras.layers.Dense(20, input_shape=(4,), activation=tf.nn.sigmoid),
#    tf.keras.layers.Dense(20, activation=tf.nn.sigmoid),
#    tf.keras.layers.Dense(3)
#])

# load saved model

#model = tf.keras.models.load_model("models/prediction_model.h5",
#                                   compile=False
#                                   )
model = fm.build_forward_model()
print(model.summary())
#
# #save_answer = input("save model? [yes|no]")
#
# #if save_answer == "yes":
# #   fm.save_model(model, "models/prediction_model.h5")
# i = 0
# for i in range(1):
#     # test the model
#     losses = fm.predict_simulation(model,
#                                    fm.rmse_loss,
#                                    steps=200)
#     print(f"Overall losses: {np.array(losses)}")
#
#     plt.figure()
#     plt.plot(losses, label="Prediction Losses")
#     plt.legend()
#     plt.show()

steps = 100
plan = [0] * (steps - 1) # because we count s_0 as a "step"
sim_states = pend.run_simulation(steps=steps)
s_0 = sim_states[0]
pred_states = fm.predict_states(model=model, state_0=s_0, plan=plan)

sim_cos = []
sim_sin = []
sim_dot = []
pred_cos = []
pred_sin = []
pred_dot = []

for state in sim_states:
    sim_cos.append(state[0])
    sim_sin.append(state[1])
    sim_dot.append(state[2])

for state in pred_states:
    pred_cos.append(state[0])
    pred_sin.append(state[1])
    pred_dot.append(state[2])


plot_list = [
    { "values": sim_cos, "label": "sim_cos", "format": "g-"},
    { "values": sim_sin, "label": "sim_sin", "format": "b-"},
    { "values": sim_dot, "label": "sim_dot", "format": "r-"},
    {"values": pred_cos, "label": "pred_cos", "format": "g--"},
    {"values": pred_sin, "label": "pred_sin", "format": "b--"},
    {"values": pred_dot, "label": "pred_dot", "format": "r--"},]

hf.plot_graphs(plot_list)