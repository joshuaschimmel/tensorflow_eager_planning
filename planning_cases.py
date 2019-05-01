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
import seaborn as sns
# import helper_functions as hf


def plan_convergence(wmr: world_model.WorldModelWrapper,
                     rollouts: int,
                     steps: int,
                     adaptation_rates: list,
                     visualize: bool = False,
                     ) -> pd.DataFrame:
    """This function tests convergence for different adapation rates.

    :param optimizer: a model for the plan optimizer
    :type optimizer: tf.keras.models.Model
    :return: the gradients in a pandas dataframe
    :rtype: pd.DataFrame
    """
    # TODO visualization
    # TODO expect a world model wrapper and not a direct model
    # TODO expect rollouts, steps and adaptation rates as input
    # setup the scenario
    # do 70 iterations
    iterations = 200
    # iterations = 3
    # use plan length 10
    plan_length = 10
    starting_state = np.array([2, 4])
    env = pendulum.Pendulum(render=False, state=starting_state)
    init_plan = po.get_random_plan(plan_length)
    # iterate through these adaptation rates
    adaptation_rates = [10, 5, 1, 0.1]
    # adaptation_rates = [1]
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
    column_names = ["adaptation_rate",
                    "iteration",
                    "loss",
                    "loss_nr",
                    "grad",
                    "action_nr",
                    ]
    # turn the result into dataframe
    result_df = pd.DataFrame(log_list, columns=column_names)

    return result_df


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
    # TODO make this an atomic function for a single feature set
    # state consists of cos, sin, dot
    state_type = ["cos", "sin", "dot"]
    observations = []
    for s_type, value in zip(state_type, state):
        # create observations
        observation = [rollout, step, source, s_type, value]
        observations.append(observation)

    return observations


def prediction_accuracy(model: world_model.WorldModelWrapper,
                        rollouts: int,
                        steps: int,
                        visualize: bool = False,
                        ) -> pd.DataFrame:
    """Creates df with prediction and truth for a model.

    This case runs the model against the simulation and
    saves each predicted feature and target value in a
    dataframe.
    This can be done for a number of random
    rollouts and consecutive steps per rollout.
    The dataframe is in long-form, with the 
    feature type, the value type (prediction or truth)
    and the value itself as their respective columns.


    :param model: Wrapper for the world model
    :type model: world_model.WorldModelWrapper
    :param rollouts: number of random initialisations
    :type rollouts: int
    :param steps: consecutive steps
    :type steps: int
    :return: dataframe with the values
    :rtype: pd.DataFrame
    """
    # TODO if visualize=True -> visualize)
    column_names = [
        "rollout", "step",
        "value_type", "feature_type",
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
                                   columns=column_names,
                                   copy=True
                                   )
    return observations_df


def eval_model_predictions(world_model_wrapper: world_model.WorldModelWrapper,
                           steps: int,
                           visualize: bool = False,
                           ) -> pd.DataFrame:
    """Calculates RMSE of predicted and actual states for steps amount.

    :param steps: Number of steps to predict
    :type steps: int
    :param world_model_wrapper: Wrapper class for the world model
    :type world_model_wrapper: WorldModelWrapper
    :return: pandas DataFrame with the observations RMSE as one of them
    :rtype: pandas.DataFrame
    """
    # TODO use prediction accuracy function with one rollout
    # TODO visualize
    # TODO rename
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
        initial_state=s_0,
        plan=plan
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


def model_quality_analysis(wmr: world_model.WorldModelWrapper,
                           rollouts: int,
                           steps: int,
                           visualize: bool = False,
                           ) -> pd.DataFrame:
    """Calculates the mean rmse over multiple random instances.

    Calculates the mean of the rmse values for rollouts number of
    runs with steps number of steps. The calculated mean is the mean
    over all test runs. If visualize is true, the mean will be
    plotted with standard deviation error bars and against lines
    for all all values

    :param rollouts: # of random tries
    :param wmr: the model wrapper to be tested
    :param steps: # of steps in a single try
    :param visualize: whether the result should be visualized
    :return: pd.DataFrame
    """
    all_rmse_values = []
    for i in range(rollouts):
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

    # flatten and create dataframe
    data = [datum for sublist in all_rmse_values for datum in sublist]

    df = pd.DataFrame(data, columns=["rollout", "step", "rmse"])

    # calculate the mean rmse with a second dataframe
    df_mean = df.groupby("step").mean()
    df_mean = df_mean.drop(columns="rollout")

    df_mean.reset_index(inplace=True)

    # since the mean is not part of a rollout, we use this categorical
    # column to label it
    df_mean["rollout"] = "mean"

    # append the mean df to the main df
    df_all = pd.concat([df, df_mean], sort=False)

    if visualize:
        sns.set(style="ticks")
        # function for general overview
        sns.relplot(x="step",
                    y="rmse",
                    kind="line",
                    # units="rollout",
                    # hue="rollout",
                    ci="sd",
                    # estimator=None,
                    alpha=1,
                    height=5,
                    aspect=6/2,
                    data=df
                    )
    palette = sns.cubehelix_palette(
        n_colors=len(df["rollout"].unique()),
        start=1,
        rot=-.8,
        hue=1,
        dark=0.4,
        light=0.75
    )

    # facetplot for all rollouts against the mean
    g = sns.relplot(x="step",
                    y="rmse",
                    hue="rollout",
                    palette=palette,
                    alpha=0.5,
                    height=5,
                    aspect=6/2,
                    kind="line",
                    legend=False,
                    data=df
                    )
    sns.lineplot(x="step",
                 y="rmse",
                 linewidth=4,
                 color="cyan",
                 data=df_mean,
                 ax=g.ax
                 )
    plt.show()

    return df_all


def angle_test(wmr: world_model.WorldModelWrapper,
               angles: list,
               speeds: list,
               visualize: bool = False,
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
    # hyperparameters
    _plan_length = 10
    _steps = 50
    _logs = []
    _columns = [
        "init_angle",
        "init_speed",
        "step",
        "theta",
        "theta_dot"
    ]
    # iterate over both lists
    for angle in angles:
        for speed in speeds:
            print(f"current angle and speed: {angle}, {speed}")
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
            _logs.append([angle, speed, 0, *env.get_env_state()])
            # run the rollout
            for step in range(_steps):
                print(f"current step: {step}")
                next_action, _ = plan_optimizer.plan_next_step(current_state)
                current_state = env(next_action)
                _logs.append([angle, speed, step + 1, *env.get_env_state()])
            # close the environment after the rollout
            env.close()
    results = pd.DataFrame(data=_logs, columns=_columns)

    if visualize:
        df = results.copy(deep=True)

        # zip speed and angle together to create an init option identifier
        df["init_id"] = list(zip(df["init_angle"], df["init_speed"]))

        # human readable format
        df["theta"] = df["theta"].apply(np.degrees)

        # plot as relplot
        sns.relplot(
            x="step",
            y="theta",
            hue="init_id",
            kind="line",
            data=df
        )
        # show the plot
        plt.show()

    return results


def environment_performance():
    # cumulative reward for an episode
    # TODO implement
    pass


def best_environment_performance():
    # best '100-episode' performance
    # TODO implement
    pass
