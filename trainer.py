import tensorflow as tf
import numpy as np
import forward_model_tf as fm
import helper_functions as hf
import pendulum as pend
import optimizer as opti

tf.enable_eager_execution()

print(f"TensorFlow version: {tf.__version__}")
print(f"Eager execution: {tf.executing_eagerly()}")

# hyperparameters
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


def reward_f(state: tf.Tensor):
    """Calculates the reward for a state

    :param state: A Tensor of shape 3
    :return: a scalar reinforcement value
    """
    reward = -(tf.square(tf.acos(state[0]))
               + 0.1 * tf.square(state[2]))
    return reward


# +++ do the planning +++

# define the optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
# get a plan with length 1
plan = hf.get_random_plan(1)
# initialize the simulation
sim_states = pend.run_simulation_plan(plan=plan)
# get the current state
s_0 = tf.convert_to_tensor(sim_states[0], dtype=tf.float32)

action = tf.convert_to_tensor(plan[0], dtype=tf.float32)


#s_0 = tf.random.uniform(shape=(3, 1))
#action = tf.random.uniform(shape=(1,))

with tf.GradientTape() as tape:
    # watch the action variable
    tape.watch(action)
    action = tf.reshape(action, shape=(1,))
    # concat the state with the action to get the model input
    next_input = tf.concat([tf.squeeze(s_0), action], axis=0)
    # reshape for the model (list of inputs but we only do one)
    next_input = tf.reshape(next_input, shape=(1, 4))
    # get the next state prediction
    s_1 = model(next_input)
    # flatten the state
    s_1 = tf.squeeze(s_1)
    # calculate the loss
    loss_value = reward_f(s_1)


grads = tape.gradient(loss_value, action)
optimizer.apply_gradients(zip(grads, action),
                          )

