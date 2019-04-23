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

wm = world_model.WorldModelWrapper()
# wm.load_model()
wm.build_keras_model(neurons=20, hidden_layers=1, dropout_rate=0)
wm.train_model(env=env, max_iterations=5000, steps=20)

# planning_cases.plan_convergence(wm.get_model())
# planning_cases.model_quality_analysis(test_runs=50,
#                                      wmr=wm,
#                                      steps=50,
#                                      visualize=True
#                                      )

df = planning_cases.prediction_accuracy(model=wm,
                                        rollouts=200,
                                        steps=25
                                        )
df.to_parquet("data/world_model_prediction.parquet", engine="pyarrow")

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
# +++ do the planning +++

# hyperparameters for Optimizer
# _learning_rate = 0.5
# _iterations = 15

# _test_length = 3000

# # get the environment
# env = pend.Pendulum()

# # initialize a random plan
# plan = hf.get_random_plan(10)

# # test the old optimizer
# print("Old Optimizer")
# old_times = []

# # get the starting state
# env_state = env.get_env_state()
# starting_state = env.get_state()
# next_state = tf.convert_to_tensor(starting_state, dtype=tf.float32)

# # intialize the optimizer object
# plan_optimizer = optimizer.Optimizer(world_model=model,
#                                      learning_rate=_learning_rate,
#                                      iterations=_iterations,
#                                      initial_plan=plan,
#                                      fill_function=hf.get_random_action
#                                      )

# score = 0
# for i in range(_test_length):
#     # check current loss value of the state
#     loss = optimizer.reinforcement(next_state).numpy()
#     # append current loss to total score
#     score += loss
#     # print metrices
#     print(f"+++ Step {i} +++\n"
#           f"Loss gained {loss}\n"
#           f"Current score {score}"
#           )
#     # measure the time
#     start_time = time.time()
#     # execute a single planning step
#     next_action, logs = plan_optimizer(next_state)
#     next_state = env(next_action)
#     # measure the time
#     total_time = time.time() - start_time
#     print(f"Total Time: {total_time}\n")
#     old_times.append(total_time)
# env.close()

# # test the new optimizer
# print("New Optimizer")
# new_times = []
#
#
# env.set_env_state(env_state)
# starting_state = env.get_state()
# next_state = tf.convert_to_tensor(starting_state, dtype=tf.float32)
#
# # intialize the optimizer object
# plan_optimizer = optimizer.Optimizer(world_model=model,
#                                      adaptation_rate=_learning_rate,
#                                      iterations=_iterations,
#                                      initial_plan=plan,
#                                      fill_function=hf.get_random_action,
#                                      use_test_function=True
#                                      )
#
# score = 0
# for i in range(_test_length):
#     # check current loss value of the state
#     loss = optimizer.reinforcement(next_state).numpy()
#     # append current loss to total score
#     score += loss
#     # print metrices
#     print(f"+++ Step {i} +++\n"
#           f"Loss gained {loss}\n"
#           f"Current score {score}"
#           )
#     # measure the time
#     start_time = time.time()
#     # execute a single planning step
#     next_action = plan_optimizer(next_state)
#     next_state = env(next_action)
#     # measure the time
#     total_time = time.time() - start_time
#     print(f"Total Time: {total_time}\n")
#     new_times.append(total_time)
# env.close()
#
# old_times = np.array(old_times)
# print(f"Old mean: {old_times.mean()}")
# print(f"Old meadian: {np.median(old_times)}")
#
# new_times = np.array(new_times)
# print(f"New mean: {new_times.mean()}")
# print(f"New meadian: {np.median(new_times)}")
