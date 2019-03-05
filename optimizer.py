import tensorflow as tf
import numpy as np
import forward_model_tf as fm
import helper_functions as hf
import pendulum as pend



def optimize_plan(model: tf.keras.Model,
                  s_0: list,
                  plan: list
                  ) -> list:
    """Optimizes the plan using the given predicting forward model.

    :param model: a keras model
    :param s_0: the initial state [cos_theta, sin_theta, theta_dot]
    :param plan: a list of actions in the interval [-2 , 2]
    :return: the optimized plan
    """
    action = plan.pop(0)
    next_input = np.append(s_0, action).reshape(1, 4)


    with tf.GradientTape() as tape:
        s_1 = model(next_input)[0]
        loss_value = pend.reward_function(s_1)

    grads = tape.gradient(loss_value, action)

    return [grads]

