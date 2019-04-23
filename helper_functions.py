import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

import world_model
import pendulum


def min_max_norm(v: float,
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


@PendingDeprecationWarning
def calc_simulation_rmse(predictions, targets):
    """Calculates the RMSE and returns a plot dict with all values.

    Unpacks the lists into a list for each attribute and observation
    type and adds them to a dict for printing together with the
    calculated RMSE.

    :param predictions: list of predictions
    :param targets: list of the actual values
    :return: dict for function plot_graphs
    """
    zipped_states = zip(predictions, targets)
    rmse_list = []
    sim_cos = []
    sim_sin = []
    sim_dot = []
    pred_cos = []
    pred_sin = []
    pred_dot = []

    for pred, targ in zipped_states:
        mse = tf.losses.mean_squared_error(
            labels=targ,
            predictions=pred,
            reduction="weighted_sum_over_batch_size"
        )
        rmse_list.append(tf.sqrt(mse))
        # unpack simulation values
        sim_cos.append(targ[0])
        sim_sin.append(targ[1])
        sim_dot.append(targ[2])
        # unpack predicted values
        pred_cos.append(pred[0])
        pred_sin.append(pred[1])
        pred_dot.append(pred[2])

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


# TODO move to planning cases
def eval_model_predictions(steps, world_model, model_name):
    """Evalutes the model for a number of consecutive steps.

    Creates a random plan with length steps. Initiates the simulation
    in a random starting state and lets it uses the plan as inputs
    and uses the resulting state as the values to compare the
    models predictions against.
    Uses the randomly initialized state as starting state for the
    model and inputs the action.
    Then the RMSE will be calculated and plotted against the
    actual and the predicted states.

    :param steps: # of predictions to make
    :param model: the model to make the predictions
    :param model_name: the name of the model (for the plot)
    :return: nothing
    """
    # create a new random plan
    plan = pendulum.get_random_plan(steps)

    # let the simulation run on the plan to create
    # the expected states as well as the starting state
    # s_0
    sim_states = pendulum.run_simulation_plan(plan=plan)
    # get starting state
    s_0 = sim_states[0]
    # let the model predict the states
    pred_states = world_model.predict_states(
        initial_state=s_0, plan=plan)

    # plot error functions (good for a single pass)
    plot_list = calc_simulation_rmse(pred_states, sim_states)
    plot_graphs(title=model_name, plot_list=plot_list)
