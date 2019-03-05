import gym
import csv
import numpy as np
import tensorflow as tf
import forward_model_tf as fm
import matplotlib.pyplot as plt

import helper_functions as hp


tf.enable_eager_execution()


def reward_function(state: np.ndarray) -> float:
    """Returns the reward for a state array

    :param state: state of the pendulum
    :return: reward value
    """
    # -(theta ^ 2 + 0.1 * theta_dt ^ 2 + 0.001 * action ^ 2)
    # action will not be used as part of the reward for a state
    reward = -(np.square(np.arccos(state[0]))
               + 0.1 * np.square(state[2]))
    return reward


def get_action(action_space) -> float:
    """returns an action in the actionspace"""
    return action_space.sample()


def create_training_data(iterations: int = 100,
                         file_path: str = "data/pendulum_data.csv"
                         ) -> None:
    """Creates training data for the pendulum environment by recording
    the states before and after an action taken.
    Writes the data into a csv file.

    Args:
        iterations: # of iterations
        file_path: path to csv file
    Returns:
        None
    """
    data = []
    headers = ["s_0_cos(theta)", "s_0_sin(theta)", "s_0_theta_dot",
               "s_1_cos(theta)", "s_1_sin(theta)", "s_1_theta_dot",
               "action", "reward"
               ]
    env = gym.make('Pendulum-v0')
    # [state_t, state_t+1, action, reward]

    for i in range(iterations):
        # initialize the state
        # theta is in [-pi, pi), thetadot is in [-1,1)
        # thetadot therefore needs to be stretched to [-8,8)
        state_0 = env.reset()

        # get the states from the environment
        theta, thetadot = env.env.state
        # scale thetadot form [-1, 1) to [-8, 8) using min/max norm.
        thetadot = hp.min_max_norm(thetadot,
                                   v_min=-1, v_max=1,
                                   n_min=-8, n_max=8
                                   )
        # reasign the new state
        env.env.state = np.array([theta, thetadot])
        # reasign state_0
        state_0 = env.env._get_obs()

        #env.render(mode = "rgb_array")

        # take a random action
        action = env.action_space.sample()
        state_1, reward, done, info = env.step(action)
        data.append([state_0[0], state_0[1], state_0[2],
                     state_1[0], state_1[1], state_1[2],
                     action[0],
                     reward
                     ])
        print(f"Iteration {i}, action: {action[0]}")

    env.close()

    with open(file_path, "w",
              newline="", encoding="utf-8") as csvfile:
        filewriter = csv.writer(csvfile)
        filewriter.writerow(headers)
        filewriter.writerows(data)
        print("Done writing")


@DeprecationWarning
def run_simulation(steps: int = 10) -> list:
    """Runs the simulation for steps steps.

    :param steps: number of steps
    :return: list of states as numpy arrays
    """
    # initialise simulaiton environtment
    env = gym.make('Pendulum-v0')

    # get first state
    s_0 = env.reset()
    # save first state in the list
    simulation_states = []
    for i in range(steps):
        #env.render(mode="human")
        action = get_action(env.action_space)
        s_0, _, _, _ = env.step(np.array([0]))
        simulation_states.append(s_0)

    env.close()
    return simulation_states


def run_simulation_plan(plan: list) -> list:
    """Runs the simulation for steps steps.

    :param steps: number of steps
    :return: list of states as numpy arrays
    """
    # initialise simulaiton environtment
    env = gym.make('Pendulum-v0')

    # initialize the state
    # theta is in [-pi, pi), thetadot is in [-1,1)
    # thetadot therefore needs to be stretched to [-8,8)
    state_0 = env.reset()

    # get the states from the environment
    theta, thetadot = env.env.state
    # scale thetadot form [-1, 1) to [-8, 8) using min/max norm.
    thetadot = hp.min_max_norm(thetadot,
                               v_min=-1, v_max=1,
                               n_min=-8, n_max=8
                               )
    # reasign the new state
    env.env.state = np.array([theta, thetadot])
    # reasign state_0
    state_0 = env.env._get_obs()


    # save first state in the list
    simulation_states = [state_0]
    for action in plan:
        #env.render(mode="human")
        #action = get_action(env.action_space)
        s_0, _, _, _ = env.step(np.array([action]))
        simulation_states.append(s_0)

    env.close()
    return simulation_states



