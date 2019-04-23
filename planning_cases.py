"""This Module contatins different planning scenarios.

This Module defines planning scenarios for the agent. For this,
it needs to set the starting state sometimes.

:Version: 26-03-2019
:Author: Joshua Schimmelpfennig
"""
import tensorflow as tf
import pandas as pd
import numpy as np
import copy

import pendulum
import world_model
import plan_optimizer as po
import matplotlib.pyplot as plt
# import helper_functions as hf


def plan_convergence(model: tf.keras.models.Model) -> pd.DataFrame:
    """This function tests convergence for different adapation rates.

    :param optimizer: a model for the plan optimizer
    :type optimizer: tf.keras.models.Model
    :return: the gradients in a pandas dataframe
    :rtype: pd.DataFrame
    """
    # TODO return df
    # setup the scenario
    # do 70 iterations
    iterations = 200
    #iterations = 3
    # use plan length 10
    plan_length = 10
    starting_state = np.array([2, 4])
    env = pendulum.Pendulum(render=False, state=starting_state)
    init_plan = po.get_random_plan(plan_length)
    # iterate through these adaptation rates
    adaptation_rates = [10, 5, 1, 0.1]
    #adaptation_rates = [1]
    # save result arrays in here
    log_list = []
    for rate in adaptation_rates:
        plan_optimizer = po.Planner(model,
                                    rate,
                                    iterations,
                                    copy.deepcopy(init_plan),
                                    po.get_random_action
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
    # result_df.rename(index=str, columns=str, inplace=True)
    result_df.columns = new_columns

    print("saving data")
    result_df.to_parquet("data/grad_logs.parquet", engine="pyarrow")


def prediction_accuracy(model: world_model.WorldModelWrapper,
                        rollouts: int,
                        steps: int
                        ) -> pd.DataFrame:
    """Predicts the truth and saves both in a DataFrame.

    Runs multiple rollouts for mutiple consecutive steps
    to create a pandas DataFrame for each step with both
    the prediction and the truth. The states are
    [cos_theta, sin_theta, theta_dot] and are prepended
    with "sim_" for the truth and "pre_" for prediction
    values.

    :param model: A World Model wrapper
    :type model: world_model.WorldModelWrapper
    :param rollouts: Number of random initializations
    :type rollouts: int
    :param steps: Number of consecutive steps
    :type steps: int
    :return: DataFrame with the states
    :rtype: pd.DataFrame
    """
    columns = [
        "rollout", "step",
        "source", "state_type",
        "value"
    ]

    observations = []
    for r in range(rollouts):
        plan = pendulum.get_random_plan(steps)
        # get true states
        true_states = pendulum.run_simulation_plan(plan)
        # extract initial state
        state_0 = true_states[0]
        # get predictions
        predicted_states = model.predict_states(initial_state=state_0,
                                                plan=plan
                                                )
        step = 1
        # create observations from the state lists (clean format for df)
        for true_state, predicted_state in zip(true_states, predicted_states):
            observations.append(_unpack_state(r,
                                              step,
                                              "simulation",
                                              true_state
                                              ))
            observations.append(_unpack_state(r,
                                              step,
                                              "prediction",
                                              predicted_state
                                              ))
            step += 1
    # flatten the observations and create the df
    observations = [obs for state in observations for obs in state]
    observations_df = pd.DataFrame(data=observations,
                                   columns=columns,
                                   copy=True
                                   )
    return observations_df


def _unpack_state(rollout: int, step: int,
                  source: str, state: list
                  ) -> list:
    """Unpacks the state into observations and returns the list.

    :param rollout: rollout the state was generated in
    :type rollout: int
    :param step: the step this state was generated in
    :type step: int
    :param source: the source (either simulation or prediction) this state
        was generated in
    :type source: str
    :param state: the state in qestion
    :type state: list
    :return: list of observations generated from this state
    :rtype: list
    """
    # state consists of cos, sin, dot
    state_type = ["cos", "sin", "dot"]
    observations = []
    for s_type, value in zip(state_type, state):
        # create observations
        observation = [rollout, step, source, s_type, value]
        observations.append(observation)

    return observations


def model_quality_analysis(test_runs: int,
                           wmr: world_model.WorldModelWrapper,
                           steps: int,
                           visualize: bool = True,
                           plot_title: str = ""
                           ):
    """Calculates the mean rmse over multiple random instances.

    Calculates the mean of the rmse values for test_runs number of
    runs with steps number of steps. The calculated mean is the mean
    over all test runs. If visualize is true, the rmse values and
    the mean will be plotted against each other in a single plot.

    :param test_runs: # of random tries
    :param wmr: the model wrapper to be tested
    :param steps: # of steps in a single try
    :param visualize: whether the result should be visualized
    :param plot_title: the title of the plot
    :return:
    """
    all_rmse_values = []
    plot_list = []
    for i in range(test_runs):
        # create a new random plan
        plan = pendulum.get_random_plan(steps)

        # let the simulation run on the plan to create
        # the expected states as well as
        # the starting state s_0
        sim_states = pendulum.run_simulation_plan(plan=plan)
        # get starting state
        s_0 = sim_states[0]
        # let the model predict the states
        pred_states = wmr.predict_states(initial_state=s_0, plan=plan)

        current_rmse_values = []
        # calculate the rmse
        step = 0
        for pred_state, sim_state in zip(pred_states, sim_states):
            current_rmse_values.append([
                i,
                step,
                world_model.single_rmse_loss(pred_state,
                                             sim_state
                                             )
            ])
            step += 1
        # append the values to the list of values as an numpy array
        all_rmse_values.append(np.array(current_rmse_values))

    #mean_values = np.mean(all_rmse_values, axis=0)
    # drop the index

    if visualize:
        # TODO use seaborn
        # initialize figure
        plt.figure()
        # plot title
        plt.suptitle(plot_title)
        # iterate through list and plot all entries
        for plot_data in plot_list:
            plt.plot(plot_data["values"],
                     plot_data["format"],
                     label=plot_data["label"],
                     linewidth=1,
                     alpha=0.5
                     )
        # plot the mean
        mean_plot = plt.plot(mean_dict["values"],
                             mean_dict["format"],
                             label=mean_dict["label"],
                             linewidth=2
                             )

        plt.ylim(0, 10)
        plt.grid(b=True, alpha=0.25, linestyle="--")
        plt.tick_params(axis="both", which="major", direction="out")
        plt.legend(mean_plot, ["Mean"], loc=1)
        plt.show()

    return all_rmse_values


def angle_test(angles: list,
               speeds: list,
               wmr: world_model.WorldModelWrapper
               ) -> pd.DataFrame:
    """Initializes with speeds and angles, returns true theta and dot values.

    This case tests the planning algorithm for a set of starting angles
    and speeds. The agent then has to solve the task. Observations of 
    true theta and thetadot values will be returned as observations in a
    DataFrame.

    :param angles: List of starting angles in DEGREES
    :type angles: list
    :param speeds: List of speeds in [-8, 8]
    :type speeds: list
    :return: Results
    :rtype: pd.DataFrame
    """
    #  hyperparameters
    _plan_length = 10
    _steps = 50
    _logs = []
    _columns = ["angle", "speed", "theta", "thetadot"]
    # iterate over both lists
    for angle in angles:
        for speed in speeds:
            rad = np.radians(angle)

            # initialize the environment with the given parameters
            env = pendulum.Pendulum(state=[rad, speed])
            plan = po.get_random_plan(_plan_length)

            # initialize the plan optimizer
            plan_optimizer = po.Planner(world_model=wmr.get_model(),
                                        learning_rate=1,
                                        iterations=10,
                                        initial_plan=plan,
                                        fill_function=po.get_random_action
                                        )
            current_state = env.get_state()
            # create first entry
            _logs.append([angle, speed, *env.get_env_state()])
            # run the rollout
            for _ in range(_steps):
                next_action, _ = plan_optimizer.plan_next_step(current_state)
                current_state = env(next_action)
            # close the environment after the rollout
            env.close()
    return pd.DataFrame(data=_logs, columns=_columns)


def eval_model_predictions(steps: int,
                           world_model_wrapper: world_model.WorldModelWrapper
                           ) -> pd.DataFrame:
    """Calculates RMSE of predicted and actual states for steos amount.

    :param steps: Number of steps to predict
    :type steps: int
    :param world_model_wrapper: Wrapper class for the world model
    :type world_model_wrapper: WorldModelWrapper
    :return: pandas DataFrame with the observations RMSE as one of them
    :rtype: pandas.DataFrame
    """
    # initilize array to save the observations in
    observations = []
    # only a single rollout in this case
    r = 0
    columns = [
        "rollout", "step",
        "source", "state_type",
        "value"
    ]

    plan = pendulum.get_random_plan(steps)

    # get the true states
    true_states = pendulum.run_simulation_plan(plan=plan)
    # use first state as starting state for prediction
    s_0 = true_states[0]
    # predict states
    predicted_states = world_model_wrapper.predict_states(
        initial_state=s_0, plan=plan
    )

    # flatten into observations and calculate RMSE for each step
    step = 1
    for true_state, predicted_state in zip(true_states, predicted_states):
        observations.append(_unpack_state(r,
                                          step,
                                          "simulation",
                                          true_state
                                          ))
        observations.append(_unpack_state(r,
                                          step,
                                          "prediction",
                                          predicted_state
                                          ))
        # add the rmse for each step
        observations.append([[r,
                              step,
                              "RMSE",
                              "RMSE",
                              world_model.single_rmse_loss(predicted_state,
                                                           true_state
                                                           )
                              ]])
        step += 1
    # flatten observations
    observations = [obs for state in observations for obs in state]

    # create the dataframe
    observations_df = pd.DataFrame(data=observations,
                                   columns=columns,
                                   copy=True
                                   )
    return observations_df
