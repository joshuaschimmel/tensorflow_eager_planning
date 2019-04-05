"""This Module contatins different planning scenarios.

This Module defines planning scenarios for the agent. For this,
it needs to set the starting state sometimes.

:Version: 26-03-2019
:Author: Joshua Schimmelpfennig
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import copy
# import optimizer as Plan Optimizer
import optimizer as po
import pendulum as environment
import helper_functions as hf


def plan_convergence(model: tf.keras.models.Model) -> pd.DataFrame:
    """This function tests convergence for different adapation rates.

    :param optimizer: a model for the plan optimizer
    :type optimizer: tf.keras.models.Model
    :return: the gradients in a pandas dataframe
    :rtype: pd.DataFrame
    """
    # setup the scenario
    # do 70 iterations
    iterations = 200
    #iterations = 3
    plan_length = 10
    starting_state = np.array([2, 4])
    env = environment.Pendulum(render=False, state=starting_state)
    # use plan length 10
    init_plan = hf.get_random_plan(plan_length)
    # iterate through these adaptation rates
    adaptation_rates = [10, 5, 1, 0.1]
    #adaptation_rates = [1]
    # save result arrays in here
    log_list = []
    for rate in adaptation_rates:
        plan_optimizer = po.Optimizer(model,
                                      rate,
                                      iterations,
                                      copy.deepcopy(init_plan),
                                      hf.get_random_action
                                      )
        _, logs = plan_optimizer.plan_next_step(env.get_state())
        # extract gradient_log and append to full log list
        for iteration_log in logs:
            log_list.append(iteration_log["gradient_log"])

    # flatten results
    log_list = [log for adaptation_r in log_list
                for log in adaptation_r
                ]

    # log structure
    """np.array([
        # objects adaptation rate
        self.adaptation_rate,
        # epsilon, current iteration
        e,
        # loss for this action
        loss,
        # the position of the loss
        loss_pos,
        # the gradient
        grad,
        # the position of the action
        taken_action_i,
    ])
    """

    # titles for the columns
    new_columns = ["adaptation_rate",
                   "iteration",
                   "loss",
                   "loss_nr",
                   "grad",
                   "action_nr",
                   ]
    # turn the result into a pandas dataframe for easy visualization
    # and storage
    result_df = pd.DataFrame(log_list)
    # swap running index with string names
    #result_df.rename(index=str, columns=str, inplace=True)
    result_df.columns = new_columns
    print("saving data")
    result_df.to_parquet("data/grad_logs.parquet", engine="pyarrow")
