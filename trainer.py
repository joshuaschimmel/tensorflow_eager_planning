import copy
import time
import numpy as np
import tensorflow as tf

import gym
import seaborn as sns
import matplotlib.pyplot as plt

import pendulum
import world_model
import plan_optimizer
import planning_cases
# import helper_functions as hf


tf.enable_eager_execution()

print(f"TensorFlow version: {tf.__version__}")
print(f"Eager execution: {tf.executing_eagerly()}")

# hyperparameters for the world model
_neurons = 40
_hidden_layer = 2
_epochs = 1
_loss_function = world_model.rmse_loss
_drop_rate = 0.5
_load_model = True


_steps = 100
_test_runs = 50

# get the model identifier
drop_text = "nodrop" if _drop_rate == 0 else f"{_drop_rate}drop"
neuron_text = (str(_neurons) + '-') \
    * _hidden_layer \
    + str(_neurons) + "_" \
    + str(_loss_function.__name__)

# load saved model
model_name = f"model_{neuron_text}_{_epochs}e_{drop_text}"
model_path = f"models/{model_name}.h5"

env = pendulum.Pendulum()
env.close()

wm = world_model.WorldModelWrapper()
wm.load_model()

# wm.build_keras_model(neurons=20, hidden_layers=1, dropout_rate=0)
# wm.train_model(env=env, max_iterations=5000, steps=20)

speeds = [-8, 0, 8]
angles = np.arange(-45, 46, 5)

planning_cases.angle_test(angles, speeds, wm)


def test_world_model(wmr: world_model.WorldModelWrapper):
    # TODO implement pipeline to show the quality of the model
    pass

#planning_cases.eval_model_predictions(10, wm)

# planning_cases.plan_convergence(wm.get_model())
# planning_cases.model_quality_analysis(test_runs=50,
#                                      wmr=wm,
#                                      steps=50,
#                                      visualize=True
#                                      )

# df = planning_cases.prediction_accuracy(model=wm,
#                                        rollouts = 200,
#                                        steps = 25
#                                        )
# df.to_parquet("data/world_model_prediction.parquet", engine="pyarrow")

# planning_cases.plan_convergence(model)
# wm = world_model.WorldModelWrapper()
# wm.build_keras_model(neurons=_neurons, hidden_layers=_hidden_layer)
# env = gym.make("Pendulum-v0")
# loss, test_loss = wm.train_model(env=env,
#                                  steps=3,
#                                  epochs=2
#                                  )

# print(loss, "\n")
# print(test_loss)
