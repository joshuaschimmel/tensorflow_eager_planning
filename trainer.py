import tensorflow as tf
import numpy as np
import forward_model_tf as fm
import helper_functions as hf
import pendulum as pend
import optimizer

tf.enable_eager_execution()

print(f"TensorFlow version: {tf.__version__}")
print(f"Eager execution: {tf.executing_eagerly()}")

# hyperparameters for the world model
_neurons = 40
_hidden_layer = 1
_epochs = 1
_loss_function = fm.rmse_loss
_drop_rate = 0.5
_load_model = True
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
_learning_rate = 0.5
_iterations = 2

# get the environment
env = pend.Pendulum()

# get the starting state
starting_state = env.get_state()
s_0 = tf.convert_to_tensor(starting_state, dtype=tf.float32)

# initialize a random plan
plan = hf.get_random_plan(10)

# intialize the optimizer object
optimizer = optimizer.Optimizer(world_model=model,
                                learning_rate=_learning_rate,
                                iterations=_iterations,
                                initial_plan=hf.get_random_plan(20),
                                fill_function=hf.get_random_action
                                )

for i in range(1000):
    print(f"Step {i}")
    next_action = optimizer(s_0)
    next_state = env(next_action)
env.close()



