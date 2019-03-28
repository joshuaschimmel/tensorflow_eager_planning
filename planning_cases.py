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
    iterations = 70
    starting_state = np.array([2, 4])
    env = environment.Pendulum(render=False, state=starting_state)
    # use plan length 10
    init_plan = hf.get_random_plan(10)
    # iterate through these adaptation rates
    adaption_rates = [10, 1, 0.1, 0.01, 0.001]
    # save result arrays in here
    results = []
    for rate in adaption_rates:
        plan_optimizer = po.Optimizer(model, rate, iterations,
                                      copy.deepcopy(init_plan),
                                      hf.get_random_action
                                      )
        _, logs = plan_optimizer.plan_next_step(env.get_state())
        for l in logs:
            l["adaptation_rate"] = rate

        results.append(logs)
    # flatten results
    results = [log for row in results for log in row]
    # change results into list of arrays
    temp = []
    for r in results:
        row = np.array([r["adaptation_rate"], r["iteration"]])
        temp.append(np.concatenate((row, r["gradients"])))

    result = pd.DataFrame(temp)
    # swap running index with string names
    result.rename(index=str, columns=str, inplace=True)
    print("saving data")
    result.to_parquet("adaptation_rate.parquet",
                      engine="pyarrow"
                      )
