import tensorflow as tf
import pandas as pd
import numpy as np
import copy

import pendulum
import world_model
import plan_optimizer as po
import matplotlib.pyplot as plt
import seaborn as sns


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
    # setup the scenario with given state
    env = pendulum.Pendulum(render=False, state=starting_state)
    init_plan = po.get_zero_plan(plan_length)
    log_list = []

    # iterate through these adaptation rates
    for rate in adaptation_rates:
        print(f"current adaptation_rate: {rate}")
        plan_optimizer = po.Planner(world_model=wmr.get_model(),
                                    adaptation_rate=rate,
                                    iterations=plan_iterations,
                                    initial_plan=copy.deepcopy(init_plan),
                                    fill_function=po.get_zero_action,
                                    strategy="none",
                                    return_logs=True
                                    )
        # save the logs created while planning
        _, logs = plan_optimizer.plan_next_step(env.get_state())
        # extract gradient_log and append to full log list
        for iteration_log in logs:
            log_list.append(iteration_log["gradient_log"])

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
    result_df = pd.DataFrame(log_list, columns=column_names)

    figure = None
    if visualize:
        df = result_df.copy(deep=True)

        # adaptation rate - convergence plot for a0
        # overview
        df_a0 = df[df["action_nr"] == 0]
        # adaption rate - convergence plot for a0 again, but with catplot
        g_point = sns.catplot(x="loss_nr",
                              y="grad",
                              row="adaptation_rate",
                              col="action_nr",
                              height=4,
                              data=df
                              )

        # influence on action 0 by loss
        g_box = sns.catplot(x="loss_nr",
                            y="grad",
                            row="adaptation_rate",
                            kind="box",
                            data=df_a0
                            )
        figure = g_box

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
    # state consists of cos, sin, dot
    state_type = ["cos", "sin", "dot"]
    observations = []
    for s_type, value in zip(state_type, state):
        observation = [rollout, step, source, s_type, value]
        observations.append(observation)

    return observations


def single_rollout_error(world_model_wrapper: world_model.WorldModelWrapper,
                         steps: int,
                         visualize: bool = False,
                         ) -> pd.DataFrame:
    """Calculates RMSE of predicted and actual states for steps amount.

    :param steps: Number of steps to predict
    :type steps: int
    :param world_model_wrapper: Wrapper class for the world model
    :type world_model_wrapper: WorldModelWrapper
    :return: pandas DataFrame with observations
    :rtype: pandas.DataFrame
    """
    observations = []
    # some columns are capitalized for visualizations
    columns = [
        "rollout", "Step",
        "Source", "State",
        "Value"
    ]
    plan = pendulum.get_random_plan(steps)
    true_states = pendulum.run_simulation_plan(plan=plan)

    # use first state as starting state for prediction
    s_0 = true_states[0]

    # predict states
    predicted_states = world_model_wrapper.predict_states(
        initial_state=s_0,
        plan=plan
    )

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
    observations = [obs for state in observations for obs in state]
    observations_df = pd.DataFrame(data=observations,
                                   columns=columns,
                                   copy=True
                                   )
    figure = None
    if visualize:
        states = observations_df[
            observations_df["Source"].str.contains("RMSE") == False
        ]
        errors = observations_df[
            observations_df["Source"].str.contains("RMSE")
        ]
        sns.set(style="ticks")
        # confidence interval for all states
        g = sns.relplot(x="Step",
                        y="Value",
                        hue="State",
                        style="Source",
                        height=3,
                        aspect=6/2,
                        kind="line",
                        data=states
                        )

        sns.lineplot(x="Step",
                     y="Value",
                     linewidth=3,
                     color="cyan",
                     legend=False,
                     data=errors,
                     ax=g.ax)
        figure = g
    return observations_df, figure


def model_quality_analysis(wmr: world_model.WorldModelWrapper,
                           rollouts: int,
                           steps: int,
                           visualize: bool = False,
                           ) -> pd.DataFrame:
    """Calculates the mean rmse over multiple random instances.

    Calculates the mean of the rmse values for a number of rollouts
    with a given amount of conescutive steps.
    The calculated mean is the mean over all test runs.
    If visualize is true, the mean will be
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
        plan = pendulum.get_random_plan(steps)

        # let the simulation run on the plan to create
        # the expected states as well as
        # the starting state s_0
        sim_states = pendulum.run_simulation_plan(plan=plan)
        s_0 = sim_states[0]
        # let the model predict the states
        pred_states = wmr.predict_states(initial_state=s_0, plan=plan)

        current_rmse_values = []
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
        all_rmse_values.append(np.array(current_rmse_values))

    data = [datum for sublist in all_rmse_values for datum in sublist]
    df = pd.DataFrame(data, columns=["rollout", "step", "rmse"])

    # calculate the mean rmse with a second dataframe
    df_mean = df.groupby("step").mean()
    df_mean = df_mean.drop(columns="rollout")
    df_mean.reset_index(inplace=True)

    # since the mean is not part of a rollout, we use this categorical
    # column to label it
    df_mean["rollout"] = "mean"
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
        figure = g

    return df_all, figure


def angle_test(planner: po.Planner,
               angles: list,
               speeds: list,
               steps: int = 50,
               plan_length: int = 10,
               visualize: bool = False,
               ) -> pd.DataFrame:
    """Initializes with speeds and angles, returns true theta and dot values.

    This case tests the planning algorithm for a set of starting angles
    and speeds. The agent then has to solve the task. Observations of
    true theta and thetadot values will be returned as observations in a
    DataFrame.

    :param planner: initialized planer object
    :type planner: po.Planner
    :param angles: List of starting angles in DEGREES
    :type angles: list
    :param speeds: List of speeds in [-8, 8]
    :type speeds: list
    :param steps: numbre of consecutive steps to do
    :type steps: int
    :param plan_length: lenght of the plan to use
    :type plan_length: int
    :param visualize: whether to visualize the result
    :type visualize:
    :return: Results
    :rtype: pd.DataFrame
    """
    _steps = steps
    _logs = []
    _columns = [
        "init_angle",
        "init_speed",
        "step",
        "theta",
        "theta_dot"
    ]

    for angle in angles:
        for speed in speeds:
            print(f"current angle and speed: {angle}, {speed}")
            # initialize the environment with the given parameters
            rad = np.radians(angle)
            env = pendulum.Pendulum(state=[rad, speed])
            plan = po.get_zero_plan(plan_length)
            planner.reset(plan)

            current_state = env.get_state()
            _logs.append([angle, speed, 0, *env.get_env_state()])
            # run the rollout
            for step in range(_steps):
                print(f"current step: {step}")
                next_action, _ = planner.plan_next_step(current_state)
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
        figure = g

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

    results = pd.DataFrame(data=observations,
                           columns=["theta_0", "theta_1", "theta_dot_1"]
                           )

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
        figure = g
    return results, figure


def environment_performance(planner: po.Planner,
                            steps: int,
                            visualize: bool = False
                            ):
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
    env = pendulum.Pendulum(render=True)
    current_state = env.get_state()
    reinforcements = [[0, env.get_reinforcement()]]

    for step in range(1, steps):
        next_action, _ = planner.plan_next_step(current_state)
        current_state, reinf = env(next_action)
        print(f"step {step}/{steps}, reinforcement: {reinf}")
        reinforcements.append([step, reinf])
    accumulated_reinforcement = env.get_accumulated_reinforcement()
    env.close()

    figure = None
    if visualize:
        df = pd.DataFrame(data=reinforcements,
                          columns=["step", "reinforcement"]
                          )
        df["cumsum"] = df["reinforcement"].cumsum()

        sns.set_context(context="paper")
        sns.set(style="whitegrid")
        g = sns.relplot(x="step",
                        y="reinforcement",
                        height=5,
                        legend="brief",
                        kind="line",
                        aspect=3/1,
                        data=df
                        )
        g = sns.relplot(x="step",
                        y="cumsum",
                        height=5,
                        legend=False,
                        aspect=3/1,
                        kind="line",
                        ax=g.ax,
                        data=df)
        figure = g

    return (accumulated_reinforcement,
            reinforcements,
            figure)
