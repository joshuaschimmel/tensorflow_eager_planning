import gym
import csv
import numpy as np
import tensorflow as tf
import forward_model_tf as fm
import matplotlib.pyplot as plt

import helper_functions as hp


class Pendulum:

    def __init__(self):
        """A wrapper object for the gym pendulum environment.

        Handels the simulation in the background and accepts
        actions in [-2, 2] as input for the state transistions.
        """
        # save the env in a variable
        self.env = gym.make("Pendulum-v0")
        self.env.render(mode="human")

        # initialize the environment. normally you would save
        # the state, but we will do this later
        self.env.reset()

        # since gym initializes theta in [-pi, pi) and thetadot
        # in [-1, 1), thetadot needs to be stretched to [-8, 8)
        # to allow random initialization over the whole statespace.
        # first, extract the states
        theta, thetadot = self.env.env.state

        # scale thetadot using min/max normalisation
        thetadot = hp.min_max_norm(thetadot,
                                   v_min=-1, v_max=1,
                                   n_min=-8, n_max=8
                                   )

        # reassign the state
        self.env.env.state = np.array([theta, thetadot])

        # finally get the initial state
        self.state = self.env.env._get_obs()

    def __call__(self, action: float) -> np.ndarray:
        """Executes the action and returns the resulting state.

        :param action: float in [-2, 2]
        :return: state of the environment [cos, sin, dot]
        """
        new_state, reward, done, info = self.env.step([action])
        self.state = new_state

        return self.get_state()

    def get_state(self) -> np.ndarray:
        """Returns a copy of the current state as a numpy array.

        The form of the state is [cos(theta), sin(theta), theta_dot].

        :return: state of the environment
        """
        return np.copy(self.state)


def get_action(action_space) -> float:
    """returns a random action from the actionspace"""
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
        env.reset()

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

        # env.render(mode = "rgb_array")

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


def run_simulation_plan(plan: list) -> list:
    """Runs the simulation by stepping through the plan.

    :param plan: list of actions in [-2, 2]
    :return: list of states as numpy arrays
    """
    # initialise simulation environment
    env = gym.make('Pendulum-v0')

    # initialize the state
    # theta is in [-pi, pi), thetadot is in [-1,1)
    # thetadot therefore needs to be stretched to [-8,8)
    env.reset()

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
        # env.render(mode="human")
        # action = get_action(env.action_space)
        s_0, _, _, _ = env.step(np.array([action]))
        simulation_states.append(s_0)

    env.close()
    return simulation_states
