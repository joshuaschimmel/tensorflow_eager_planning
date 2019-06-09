import copy
import time
import numpy as np
import pandas as pd
import tensorflow as tf

import gym
import seaborn as sns
import matplotlib.pyplot as plt

import pendulum
import world_model
import plan_optimizer
import planning_cases

tf.enable_eager_execution()

print(f"TensorFlow version: {tf.__version__}")
print(f"Eager execution: {tf.executing_eagerly()}")


env = pendulum.Pendulum()
wm = world_model.WorldModelWrapper()

# +++ switch comments if you want to train a new model +++
# this WILL OVERWRITE the currently saved model!
wm.load_model()

# wm.build_keras_model(neurons=30, hidden_layers=1, dropout_rate=0)
# wm.print_summary()
# train_l, test_l = wm.train_model(env=env, rollouts=4000, steps=15)
# test_l["test_position"] = [len(train_l.index)] * len(test_l.index)
# ax = test_l.plot(x="test_position", y="test_loss",
#                  color="red", marker="+",
#                  s=50,
#                  # legend=True,
#                  title="World Model Training Error",
#                  kind="scatter"
#                  )
# train_l.plot(x="rollout", y="mean_loss",
#              legend=False,
#              grid=True,
#              title="World Model Training Error",
#              kind="line",
#              ax=ax
#              )
# ax.set_ylabel("RMSE")
# ax.set_xlabel("Iteration")
# ax.set_xticklabels(train_l.index, minor=True)
# plt.show()
wm.print_summary()


def test_performance(wmr: world_model.WorldModelWrapper):
    """Visualizes performance of current model.

    :param wmr: [description]
    :type wmr: world_model.WorldModelWrapper
    """
    _rollouts = 200
    _steps = 100
    # eval behaviour of RMSE for a single rollout
    _, f1 = planning_cases.single_rollout_error(
        steps=_steps,
        world_model_wrapper=wmr,
        visualize=True
    )
    # see RMSE for multiple rollouts
    _, f2 = planning_cases.model_quality_analysis(
        wmr=wmr,
        rollouts=_rollouts,
        steps=_steps,
        visualize=True
    )
    # see whether the plan converges TODO still works?
    _, f3 = planning_cases.plan_convergence(wmr=wmr,
                                            plan_iterations=10,
                                            plan_length=10,
                                            adaptation_rates=[
                                                0.1, 0.5, 1, 2],
                                            visualize=True
                                            )
    # check whether the agent can hold up the pendulum TODO still works?
    angles = [-20, -15, -10, 0, 10, 15, 20]
    speeds = [0]
    _, f4 = planning_cases.angle_test(wmr=wmr,
                                      angles=angles,
                                      speeds=speeds,
                                      steps=50,
                                      plan_length=10,
                                      visualize=True
                                      )
    plt.show()

# test_world_model(wmr=wm)


def angle_test_experiment(wm: world_model.WorldModelWrapper):
    planner = plan_optimizer.Planner(world_model=wm.get_model(),
                                     learning_rate=0.5,
                                     iterations=100,
                                     initial_plan=plan_optimizer.get_zero_plan(
                                         10),
                                     fill_function=plan_optimizer.get_zero_action,
                                     strategy=None
                                     )

    strategies = ["none", "first", "last"]
    angles = np.arange(0, 61, 2) - 30
    #angles = [0]
    steps = 50
    plan_length = 10
    full_df = None
    for i in range(10):
        planner.set_strategy("none")
        none_result, _ = planning_cases.angle_test(planner=planner,
                                                   angles=angles,
                                                   speeds=[0],
                                                   steps=steps,
                                                   plan_length=plan_length,
                                                   visualize=False
                                                   )
        none_result["condition"] = "None"

        planner.set_strategy("first")
        first_result, _ = planning_cases.angle_test(planner=planner,
                                                    angles=angles,
                                                    speeds=[0],
                                                    steps=steps,
                                                    plan_length=plan_length,
                                                    visualize=False
                                                    )
        first_result["condition"] = "First"

        planner.set_strategy("last")
        last_result, _ = planning_cases.angle_test(planner=planner,
                                                   angles=angles,
                                                   speeds=[0],
                                                   steps=steps,
                                                   plan_length=plan_length,
                                                   visualize=False
                                                   )
        last_result["condition"] = "Last"

        df = pd.concat([none_result, first_result, last_result],
                       ignore_index=True)
        df["i"] = i
        df.to_parquet(f"data/angle_test_{i}.parquet",
                      engine="pyarrow")
        if full_df is not None:
            full_df = pd.concat([full_df, df],
                                ignore_index=True
                                )
        else:
            full_df = df

        full_df.to_parquet(f"data/angle_test_condition_{i}.parquet",
                           engine="pyarrow")


def accumulated_reinforcement_experiment(wm: world_model.WorldModelWrapper,
                                         experiment_runs: int = 10
                                         ):
    plan_length = 10
    strategies = ["none", "first", "last"]
    steps = 100
    result_df = None

    planner = plan_optimizer.Planner(
        world_model=wm.get_model(),
        learning_rate=0.1,
        iterations=100,
        initial_plan=plan_optimizer.get_zero_plan(plan_length),
        fill_function=plan_optimizer.get_zero_action,
        strategy=None
    )

    for i in range(experiment_runs):
        condition_dfs = []
        for planning_strat in strategies:
            print(planning_strat)
            planner.set_strategy(planning_strat)
            acc_reinf, reinforcements, _ = planning_cases.\
                environment_performance(
                    planner=planner,
                    steps=steps,
                    visualize=False
                )
            condition_df = pd.DataFrame(data=reinforcements,
                                        columns=["step", "reinf"]
                                        )
            condition_df["strategy"] = planning_strat
            condition_df["acc_reinf"] = acc_reinf
            condition_df["cumsum_reinf"] = condition_df["reinf"].cumsum()
            condition_dfs.append(condition_df)

        # Different interface for the random agent
        print("random")
        acc_reinf, reinfs = pendulum.run_random_agent(steps)
        condition_df = pd.DataFrame(data=reinfs,
                                    columns=["step", "reinf"]
                                    )
        condition_df["strategy"] = "random"
        condition_df["acc_reinf"] = acc_reinf
        condition_df["cumsum_reinf"] = condition_df["reinf"].cumsum()
        condition_dfs.append(condition_df)

        run_df = pd.concat(condition_dfs, ignore_index=True)
        if result_df is None:
            result_df = run_df
        else:
            result_df = pd.concat([result_df, run_df], ignore_index=True)

        # result_df.to_parquet(f"data/acc_reinf_experiment_{i}.parquet",
        #                     engine="pyarrow"
        #                     )


def demo(wm: world_model.WorldModelWrapper,
         strategy: str = "random",
         steps: int = 500
         ):

    strategies = ["none", "first", "last"]
    if strategy in strategies:
        planner = plan_optimizer.Planner(
            world_model=wm.get_model(),
            learning_rate=0.1,
            iterations=100,
            initial_plan=plan_optimizer.get_zero_plan(10),
            fill_function=plan_optimizer.get_zero_action,
            strategy=strategy
        )
        planning_cases.environment_performance(planner=planner,
                                               steps=steps,
                                               visualize=True
                                               )

    elif strategy == "random":
        pendulum.run_random_agent(steps, render=True)
    else:
        error_str = (f"strategy not one of (none, first, last, random), "
                     f"got '{strategy}' instead")
        raise ValueError(error_str)
