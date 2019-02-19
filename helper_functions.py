import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pendulum as pend
import forward_model_tf as fm


def min_max_norm(v : float,
                 v_min: float, v_max: float,
                 n_min: float = 0, n_max: float = 1
                 ) -> float:
    """Does min/max normalization.

    :param v: value
    :param v_min: min value
    :param v_max: max value
    :param n_min: new min value default 0
    :param n_max: new max value default 1
    :return: normalised value
    """
    new_v = np.multiply(np.divide((v - v_min), (v_max - v_min)),
                        (n_max - n_min)
                        ) + n_min
    return new_v


def plot_graphs(title: str, plot_list: list) -> None:
    """Plots the elements in plot dicts.

    Acces every plot plot_list's dict and uses "value" and "label"
    keywords for the plotting.

    :param plot_dicts: list of dicts with "values" and "label" kwds.
    :return: returns nothing
    """
    plt.figure()
    plt.suptitle(title)
    for plot_data in plot_list:
        plt.plot(plot_data["values"],
                 plot_data["format"],
                 label=plot_data["label"]
                 )

    plt.ylim(-10, 10)
    plt.grid(b=True, alpha=0.25, linestyle="--")
    plt.tick_params(axis="both", which="major", direction="out")
    plt.legend()
    plt.show()


def get_random_plan(steps: int) -> list:
    """TODO Comment

    :param steps:
    :return:
    """
    # initialize a random plan of with _steps steps
    plan = []
    for i in range(steps - 1):
        plan.append(np.random.uniform(-2, 2))

    return plan


def get_plot_losses(predictions, realities):
    # TODO Comment
    zipped_states = zip(predictions, realities)
    rmse_list = []
    for prediction, target in zipped_states:
        mse = tf.losses.mean_squared_error(
            labels=target,
            predictions=prediction,
            reduction="weighted_sum_over_batch_size"
        )
        rmse_list.append(tf.sqrt(mse))

    # TODO learn to unpack this list
    sim_cos, sim_sin, sim_dot = realities
    pred_cos = []
    pred_sin = []
    pred_dot = []

    for state in realities:
        sim_cos.append(state[0])
        sim_sin.append(state[1])
        sim_dot.append(state[2])

    for state in predictions:
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
        {"values": rmse_list, "label": "rmse", "format": "c-"},
    ]
    return plot_list


def plot_model_prediction(steps, model, model_name):
    """TODO Comment"""
    # create a new random plan
    plan = get_random_plan(steps)

    # let the simulation run on the plan to create
    # the expected states as well as the starting state
    # s_0
    sim_states = pend.run_simulation_plan(plan=plan)
    # get starting state
    s_0 = sim_states[0]
    # let the model predict the states
    pred_states = fm.predict_states(model=model, state_0=s_0, plan=plan)

    # plot error functions (good for a single pass)
    plot_list = get_plot_losses(pred_states, sim_states)
    plot_graphs(title=model_name, plot_list=plot_list)