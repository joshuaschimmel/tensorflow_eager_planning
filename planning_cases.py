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
                     plan_iterations: int,
                     plan_length: int,
                     adaptation_rates: list = [10, 5, 1, 0.1],
                     starting_state: np.array = np.array([2, 4]),
                     visualize: bool = False,
                     ) -> pd.DataFrame:
    """This function tests convergence for different adapation rates.

    :param wmr: wrapper for the world model
    :type wmr: world_model.WorldModelWrapper
    :param plan_iterations: number of times the planner should iterate
        over the plan
    :type plan_iterations: int
    :param plan_length: length of the plan, how far the agent sees into
        the future
    :type plan_length: int
    :param adaptation_rates: default adaptation rates to test,
        defaults to [10, 5, 1, 0.1]
    :type adaptation_rates: list, optional
    :param starting_state: optional starting state,
        defaults to np.array([2, 4])
    :type starting_state: np.array, optional
    :param visualize: whether this function should visualize its results,
        defaults to False
    :type visualize: bool, optional
    :return: dataframe with the results
    :rtype: pd.DataFrame
    """
    # setup the scenario
    env = pendulum.Pendulum(render=False, state=starting_state)

    init_plan = po.get_zero_plan(plan_length)

    # save result
    log_list = []
    # iterate through these adaptation rates
    for rate in adaptation_rates:
        print(f"current adaptation_rate: {rate}")
        # create planning object
        plan_optimizer = po.Planner(world_model=wmr.get_model(),
                                    learning_rate=rate,
                                    iterations=plan_iterations,
                                    initial_plan=copy.deepcopy(init_plan),
                                    fill_function=po.get_zero_action,
                                    strategy="first",
                                    return_logs=True
                                    )
        # save the logs created while planning
        _, logs = plan_optimizer.plan_next_step(env.get_state())
        # extract gradient_log and append to full log list
        for iteration_log in logs:
            log_list.append(iteration_log["gradient_log"])

    # flatten results
    log_list = [log for adaptation_r in log_list
                for log in adaptation_r
                ]

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

    # create visualization if setting is on
    figure = None
    if visualize:
        df = result_df.copy(deep=True)
        # calc df with absolut grads

        # adaptation rate - convergence plot for a0
        # overview
        df_a0 = df[df["action_nr"] == 0]
        # adaption rate - convergence plot for a0 again, but with catplot
        g_point = sns.catplot(x="loss_nr",
                              y="grad",
                              row="adaptation_rate",
                              col="action_nr",
                              height=4,
                              # ratio=1,
                              data=df
                              )

        # plot influence on action 0 by loss
        g_box = sns.catplot(x="loss_nr",
                            y="grad",
                            row="adaptation_rate",
                            kind="box",
                            data=df_a0
                            )
        figure = g_box.fig

    return result_df, figure


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


# TODO if time, use this function as a base for single error and model quality
@DeprecationWarning
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
    # TODO if visualize -> visualize)
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

    if visualize:
        pass

    return observations_df


def single_rollout_error(world_model_wrapper: world_model.WorldModelWrapper,
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
    # initilize array to save the observations in
    observations = []
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
        observations.append(_unpack_state(0,
                                          step,
                                          "simulation",
                                          true_state
                                          ))
        observations.append(_unpack_state(0,
                                          step,
                                          "prediction",
                                          predicted_state
                                          ))
        # add the rmse for each step
        observations.append([[0,
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
    figure = None
    if visualize:
        states = observations_df[
            observations_df["source"].str.contains("RMSE") == False
        ]
        errors = observations_df[
            observations_df["source"].str.contains("RMSE")
        ]
        sns.set(style="ticks")
        # confidence interval for all states
        g = sns.relplot(x="step",
                        y="value",
                        hue="state_type",
                        style="source",
                        height=3,
                        aspect=6/2,
                        kind="line",
                        data=states
                        )

        sns.lineplot(x="step",
                     y="value",
                     linewidth=3,
                     color="cyan",
                     legend=False,
                     data=errors,
                     ax=g.ax)
        figure = g.fig
    return observations_df, figure


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
    for r in range(rollouts):
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
                r,
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

    figure = None
    if visualize:
        sns.set(style="ticks")

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
                    data=df,
                    ax=g.ax
                    )
        sns.lineplot(x="step",
                     y="rmse",
                     linewidth=4,
                     color="cyan",
                     data=df_mean,
                     ax=g.ax
                     )
        figure = g.fig

    return df_all, figure


def angle_test(wmr: world_model.WorldModelWrapper,
               angles: list,
               speeds: list,
               plan_length: int = 10,
               steps: int = 50,
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
    _plan_length = plan_length
    _steps = steps
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
            plan = po.get_zero_plan(_plan_length)

            # initialize the plan optimizer
            plan_optimizer = po.Planner(world_model=wmr.get_model(),
                                        learning_rate=2,
                                        iterations=20,
                                        initial_plan=plan,
                                        fill_function=po.get_zero_action,
                                        strategy="first"
                                        )
            current_state = env.get_state()
            # create first entry
            _logs.append([angle, speed, 0, *env.get_env_state()])
            # run the rollout
            for step in range(_steps):
                print(f"current step: {step}")
                next_action, _ = plan_optimizer.plan_next_step(current_state)
                current_state, _ = env(next_action)
                _logs.append([angle, speed, step + 1, *env.get_env_state()])
            # close the environment after the rollout
            env.close()
    results = pd.DataFrame(data=_logs, columns=_columns)
    figure = None
    if visualize:
        sns.set(style="ticks")
        df = results.copy(deep=True)

        # zip speed and angle together to create an init option identifier
        df["init_id"] = list(zip(df["init_angle"], df["init_speed"]))

        # human readable format
        df["theta"] = df["theta"].apply(np.degrees)

        # plot as relplot
        g = sns.relplot(
            x="step",
            y="theta",
            hue="init_id",
            kind="line",
            data=df
        )
        figure = g.fig

    return results, figure


def environment_angle_behavior(visualize: bool = False) -> pd.DataFrame:
    """This functions shows how the environment can be influenced.

    This functions returns state transitions of theta using the max
    axtion towards theta = 0. This shows, at what angles we can
    expect the agent to hold up the pendulum.

    :param visualize: whether results should be visualized,
    defaults to False
    :type visualize: bool, optional
    :return: theta state transitions in rad in a dataframe
    :rtype: np.DataFrame
    """
    # start with action 0
    _action = 0
    # push with maximal force counter clockwise
    _max_left_action = -8
    observations = []
    # start states are in [0, 180]
    start_rads = np.radians(np.arange(0, 181, 1))
    # create the transitions
    for theta_0 in start_rads:
        starting_state = [theta_0, _action]
        env = pendulum.Pendulum(state=starting_state)
        _, _ = env(_max_left_action)
        theta_1, theta_dot_1 = env.get_env_state()
        observations.append([theta_0, theta_1, theta_dot_1])
        env.close()
    # put the data into a dataframe
    results = pd.DataFrame(data=observations,
                           columns=["theta_0", "theta_1", "theta_dot_1"]
                           )
    # visualize if needed
    figure = None
    if visualize:
        sns.set_context(context="paper")
        sns.set(style="whitegrid")
        df = results.copy(deep=True)
        df["delta_theta"] = df["theta_0"] - df["theta_1"]
        df["theta_0_deg"] = df["theta_0"].apply(np.degrees)
        df["theta_1_deg"] = df["theta_1"].apply(np.degrees)
        # plot as relplot
        g = sns.relplot(
            x="theta_0_deg",
            y="delta_theta",
            hue="delta_theta",
            height=5,
            legend=False,
            aspect=3/1,
            data=df
        )
        figure = g.fig
    return results, figure


def environment_performance(planner: po.Planner,
                            steps: int,
                            visualize: bool = False
                            ) -> float:
    """Executes the Planner for a number of steps.

    :param planner: plan optimizer to produce an optimal plan
    :type planner: po.Planner
    :param steps: number of steps to do
    :type steps: int
    :param visualize: whether results should be visualized,
        defaults to False
    :type visualize: bool, optional
    :return: the accumulated reinforcement and list of reinforcements
    :rtype: float, list
    """
    env = pendulum.Pendulum()
    current_state = env.get_state()
    reinforcements = [[0, env.get_reinforcement()]]

    for step in range(steps):
        print(f"step {step}/{steps}")
        next_action, _ = planner.plan_next_step(current_state)
        current_state, reinf = env(next_action)
        print(reinf)
        reinforcements.append([step, reinf])
    accumulated_reinforcement = env.get_accumulated_reinforcement()
    env.close()

    figure = None
    if visualize:
        df = pd.DataFrame(data=reinforcements,
                          columns=["step", "reinforcement"]
                          )
        df["cumsum"] = df["reinforcements"].cumsum()

        sns.set_context(context="paper")
        sns.set(style="whitegrid")
        g = sns.relplot(x="step",
                        y="reinforcement",
                        height=5,
                        legend=False,
                        aspect=3/1,
                        data=df
                        )
        sns.relplot(x="step",
                    y="cumsum",
                    height=5,
                    legend=False,
                    aspect=3/1,
                    ax=g.ax,
                    data=df)
        figure = g.fig

    return ((accumulated_reinforcement, reinforcements),
            figure)


def best_environment_performance():
    # best '100-episode' performance
    # => best performance for 100  rollouts?
    # TODO implement
    pass
