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

# hyperparameters
_neurons = 40
_hidden_layer = 1
_epochs = 4
_loss_function = fm.rmse_loss
_drop_rate = 0.0
_load_model = False
_save_model = True

drop_text = "nodrop"
if _drop_rate != 0.0:
    drop_text = "drop"
neuron_text = (str(_neurons) + '-') \
              * _hidden_layer \
              + str(_neurons) + "_" \
              + str(_loss_function.__name__)

# load saved model

model_name = f"model_{neuron_text}_{_epochs}e_{drop_text}"
model_path = f"models/{model_name}.h5"

# TODO check if model exists
if _load_model:
    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )
else:
    model = fm.build_forward_model(epochs=_epochs,
                                   neurons=_neurons,
                                   hidden_layers=_hidden_layer,
                                   loss=_loss_function,
                                   dropout_rate=_drop_rate,
                                   validation_split=0.05,
                                   test_split=0.05,
                                   plot_performance=True)

print(model.summary())
print(f"Using {model_name}")

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

steps = 200
# because we count s_0 as a "step"
plan = [0] * (steps - 1)
sim_states = pend.run_simulation(steps=steps)
s_0 = sim_states[0]
pred_states = fm.predict_states(model=model, state_0=s_0, plan=plan)

# plot error functions
mse_list = []
rmse_list = []
mae_list = []
zipped_states = zip(pred_states, sim_states)
for prediction, target in zipped_states:
    mse = tf.losses.mean_squared_error(
        labels=target,
        predictions=prediction,
        reduction="weighted_sum_over_batch_size"
    )
    mse_list.append(mse)
    rmse_list.append(tf.sqrt(mse))
    mae_list.append(tf.losses.absolute_difference(
        labels=target,
        predictions=prediction,
        reduction="weighted_sum_over_batch_size"
    ))
    

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
    {"values": sim_cos, "label": "sim_cos", "format": "g-"},
    {"values": sim_sin, "label": "sim_sin", "format": "b-"},
    {"values": sim_dot, "label": "sim_dot", "format": "r-"},
    {"values": pred_cos, "label": "pred_cos", "format": "g--"},
    {"values": pred_sin, "label": "pred_sin", "format": "b--"},
    {"values": pred_dot, "label": "pred_dot", "format": "r--"},
    {"values": mse_list, "label": "mse", "format": "c:"},
    {"values": rmse_list, "label": "rmse", "format": "m:"},
    {"values": mae_list, "label": "mae", "format": "y:"}
]

hf.plot_graphs(title=model_name, plot_list=plot_list)


if _save_model and not _load_model:
    save_answer = input("save model? [yes|no]")

    if save_answer == "yes":
        fm.save_model(model, model_path)
        print("saved")
