import tensorflow as tf
import numpy as np
import forward_model_tf as fm
import helper_functions as hf
import pendulum as pend
import optimizer

import copy
import time

tf.enable_eager_execution()

print(f"TensorFlow version: {tf.__version__}")
print(f"Eager execution: {tf.executing_eagerly()}")

# hyperparameters for the world model
_neurons = 40
_hidden_layer = 2
_epochs = 1
_loss_function = fm.rmse_loss
_drop_rate = 0.5
_load_model = False


_steps = 100
_test_runs = 50


drop_text = "nodrop"
if _drop_rate > 0.0:
    drop_text = f"{_drop_rate}drop"
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

# save the model if it is new
if not _load_model:
    fm.save_model(model, model_path)
    print("saved")

# +++ do the planning +++

# hyperparameters for Optimizer
_learning_rate = 0.1
_iterations = 20

# get the environment
env = pend.Pendulum()

# get the starting state
starting_state = env.get_state()
next_state = tf.convert_to_tensor(starting_state, dtype=tf.float32)


# initialize a random plan
plan = hf.get_random_plan(25)

# intialize the optimizer object
plan_optimizer = optimizer.Optimizer(world_model=model,
                                     learning_rate=_learning_rate,
                                     iterations=_iterations,
                                     initial_plan=plan,
                                     fill_function=hf.get_random_action
                                     )

score = 0
for i in range(1000):
    loss = optimizer.reinforcement(next_state)
    score += loss
    print(f"Step {i}\n"
          f"Loss gained {loss}\n"
          f"Current score {score}"
          )
    next_action = plan_optimizer(next_state)
    next_state = env(next_action)
env.close()


# TODO Put this into its own function
# time old model first:

# start_1 = time.time()
# next_action = plan_optimizer(next_state)
# next_state = env(next_action)
# next_action = plan_optimizer(next_state)
# next_state = env(next_action)
# next_action = plan_optimizer(next_state)
# next_state = env(next_action)
# end_1 = time.time()
# time_1 = end_1 - start_1
# print(f"Old Function needed {time_1}")
#
# starting_state = env.set_env_state(env_state)
# next_state = tf.convert_to_tensor(starting_state, dtype=tf.float32)
#
# start_2 = time.time()
# next_action = modified_plan_optimizer(next_state)
# next_state = env(next_action)
# next_action = modified_plan_optimizer(next_state)
# next_state = env(next_action)
# next_action = modified_plan_optimizer(next_state)
# next_state = env(next_action)
# end_2 = time.time()
# time_2 = end_2 - start_2
# print(f"Old Function needed {time_2}")
#
# print(f"Diff 1-2: {time_1 - time_2}")
# print(f"time_1 > time_2: {time_1 > time_2}")
