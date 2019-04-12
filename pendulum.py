import gym
import csv
import numpy as np
import pandas as pd
from typing import List

#import helper_functions as hp


class Pendulum:

    def __init__(self,
                 render: bool = True,
                 state: List[float] = None
                 ):
        """A wrapper object for the gym pendulum environment.

        Handels the simulation in the background. The default
        initialization of the pendulum environment is with theta
        in [-pi, pi) and thetadot in [-1, 1).
        The rendering of the Visualization can be turned off for
        performance reasons or if it is not needed.
        Even though thetadot is in [-8, 8),
        the initialization is, accoding to the wiki,
        deliberatly low to increase the difficulty.
        For visualization and testing purposes, this class accepts
        custom initialization values, but this should never be
        used for training.


        :param state: list of size 2 as [theta, thetadot] with theta
            in [-pi, pi) and thetadot in [-8, 8)
        :param render: bool whether the visualization should be renderd
        """
        # save the env in a variable
        self.env = gym.make("Pendulum-v0")

        # set state
        self.state = self.env.reset()

        # if given, set the state
        if state is not None:
            theta, thetadot = state
            self.env.env.state = np.array(theta, thetadot)
            self.state = self.env.env._get_obs()

        # set render
        self.render = render
        # render the environment
        if self.render:
            self.env.render(mode="human")

    def __call__(self, action: float) -> np.ndarray:
        """Executes the action and returns the resulting state.

        :param action: float in [-2, 2]
        :return: state of the environment [cos, sin, dot]
        """
        # do the action
        # return is new_state, reward, done, info
        new_state, _, _, _ = self.env.step([action])
        # return the new state
        self.state = new_state

        # render the environment
        if self.render:
            self.env.render(mode="human")

        return self.get_state()

    def close(self):
        """Closes the environment."""
        self.env.close()

    def get_state(self) -> np.ndarray:
        """Returns a copy of the current state as a numpy array.

        The form of the state is [cos(theta), sin(theta), theta_dot].

        :return: state of the environment
        """
        return np.copy(self.state)

    def get_env_state(self) -> np.ndarray:
        """Gets the state of the backend gym environment object.

        :return: numpy array with [theta, thetadot]
        """
        return self.env.env.state

    def set_env_state(self, state: np.ndarray) -> np.ndarray:
        """Sets the state of the backend gym environment."""
        self.env.env.state = state
        self.state = self.env.env._get_obs()
        return self.state


def get_random_action() -> float:
    """Returns a random float in [-2, 2)"""
    return np.random.uniform(-2, 2)


def run_simulation_plan(plan: list, render: bool = False) -> list:
    """Runs the simulation by iteration through the plan.

    :param plan: list of actions in [-2, 2]
    :return: list of states as numpy arrays
    """
    # initialise simulation environment
    # theta is in [-pi, pi), thetadot is in [-1, 1)
    env = Pendulum(render=render)
    # get the state
    state_0 = env.get_state()
    # save first state in the list
    simulation_states = [state_0]

    for action in plan:
        simulation_states.append(env(action))

    env.close()
    return simulation_states


def get_state_generator(steps: int):
    """Generator function to get state transition.

    This is a generator function which yields a list of state
    transition arrays of ´steps´ amount of consecutive steps.
    A transition has the form [[state_0, action],[state_1]],
    with action being a random value in [-2, 2] neccessary
    to transform state_0 to state_1.
    This generator will continue until the end of time.
    (Or battery power)

    :param rollouts: Number of random initialization
    :type rollouts: int
    :param steps: Number of consecutive steps
    :type steps: int
    """
    try:
        while True:
            # create env and list
            env = Pendulum()
            rollout_transitions = []
            # get the first state
            current_state = env.get_state()
            for _ in range(steps):
                # get an action
                next_action = get_random_action()
                # create the s_0, action tuple
                last_input = np.append(current_state, next_action)
                # get the next state
                current_state = env(next_action)
                # save the tuple
                rollout_transitions.append([last_input, current_state])
            # close the current env
            env.close()
            # yield the states
            yield rollout_transitions
    except KeyboardInterrupt:
        print("Interrupted")


@DeprecationWarning
def create_training_data(iterations: int = 100,
                         file_path: str = "data/training.parquet"
                         ) -> None:
    """Creates training data for the pendulum environment by recording
    the states before and after an action taken.
    Writes the data into a parquet file.

    Args:
        iterations: # of iterations
        file_path: path to parquet file
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
        state_0 = env.reset()

        # env.render(mode = "rgb_array")

        # take a random action
        action = env.action_space.sample()
        state_1, reward, _, _ = env.step(action)
        data.append(np.array(state_0[0], state_0[1], state_0[2],
                             state_1[0], state_1[1], state_1[2],
                             action[0],
                             reward
                             ))
        print(f"Iteration {i}, action: {action[0]}")

    # close the environment
    env.close()

    # save the data in a dataframe for easy saving
    df = pd.DataFrame(data=data, columns=headers)
    if file_path is not None:
        df.to_parquet(file_path, engine="pyarrow")
        print("Done saving")

    return df
