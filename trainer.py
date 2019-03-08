import tensorflow as tf
import numpy as np
import forward_model_tf as fm
import helper_functions as hf
import pendulum as pend
import optimizer as ot
import copy

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
    reward = -(tf.square(tf.subtract(state[0], 1))
               + tf.square(state[1])
               + 0.1 * tf.square(state[2]))
    return reward


# +++ do the planning +++

# define the optimizer
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
_learning_rate = 0.1
_iterations = 10
# get a plan with length 1
plan = hf.get_random_plan(20)
old_plan = copy.deepcopy(plan)
# initialize the simulation
sim_states = pend.run_simulation_plan(plan=plan)
# get the starting state
s_0 = tf.convert_to_tensor(sim_states[0], dtype=tf.float32)

optimizer = ot.Optimizer(world_model=model,
                         learning_rate=_learning_rate,
                         iterations=_iterations,
                         initial_plan=hf.get_random_plan(20),
                         starting_state=s_0,
                         fill_function=hf.get_random_action
                         )



# actual_action =
#action = tf.Variable([plan[0]], dtype=tf.float32)
#print(action)

#s_0 = tf.random.uniform(shape=(3, 1))
#action = tf.random.uniform(shape=(1,))



# algorithm begins here
# for e in range(_iterations):
#     print(f"Iteration {e + 1}")
#
#     # set up the inital states
#     current_state = s_0
#     taken_actions = []
#     derivatives = []
#     grads = []
#
#     # collect the rewards and calculations using the gradient tape
#     with tf.GradientTape(persistent=True) as tape:
#         # iterate over all actions
#         for action in plan:
#             # watch the action variable
#             tape.watch(action)
#             action = tf.reshape(action, shape=(1,))
#             # concat the state with the action to get the model input
#             # for this, squeeze the state by one axis into a list
#             next_input = tf.concat(
#                 [tf.squeeze(current_state), action],
#                 axis=0
#             )
#             # reshape the input for the model
#             next_input = tf.reshape(next_input, shape=(1, 4))
#             # get the next state prediction
#             current_state = model(next_input)
#             # update the list of already taken actions
#             taken_actions.append(action)
#             # flatten the state and calculate the loss
#             loss_value = reward_f(tf.squeeze(current_state))
#             # add the loss value together with the actions that led up to
#             # it and add them to the list of derivatives
#             derivatives.append([taken_actions.copy(), loss_value])
#
#     # now iterate over all derivative pairs and add the gradients to
#     # the the grads list
#     for actions, loss in derivatives:
#         # initialize counter for list accessing
#         i = 0
#
#         for action in actions:
#             # calculate the gradients for each action
#             grad = tape.gradient(loss, action)
#             # add the gradient to the position in grads corresponding to
#             # the actions position in plan
#             # a bit of EAFP
#             try:
#                 # add the grad to the existing one
#                 grads[i].append(grad)
#             except IndexError:
#                 # initialize a new one
#                 grads.append([grad])
#
#             # update counter
#             i += 1
#
#     # iterate over grads and fill the lists with tf constants to get
#     # an even shape
#     # first list is the longes
#     max_len = len(grads[0])
#     for grad_list in grads:
#         while len(grad_list) < max_len:
#             # add zero constants until the lengths are the same
#             grad_list.append(tf.zeros(1))
#
#     # reduce the the grads by summing them up
#     sums = tf.reduce_sum(grads, axis=0)
#
#     # apply the sums to each action
#     for grad, action in zip(sums, plan):
#         # add learning rate
#         action.assign_add(grad * _learning_rate)

print(old_plan)
print(plan)
