import gym
import csv
import copy
import numpy as np
import pandas as pd
from typing import List

#import helper_functions as hp


class Pendulum:

    def __init__(self,
                 render: bool = False,
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
        self.acc_reinf = 0
        self.reinforcement = 0

        # if given, set the state
        if state is not None:
            theta, thetadot = state
            self.env.env.state = np.array([theta, thetadot])
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
        new_state, new_reinforcement, _, _ = self.env.step([action])
        # return the new state
        self.state = new_state
        self.acc_reinf += new_reinforcement
        self.reinforcement = new_reinforcement

        # render the environment
        if self.render:
            self.env.render(mode="human")

        return (self.get_state(), self.get_reinforcement())

    def close(self):
        """Closes the environment."""
        self.env.close()

    def get_state(self) -> np.ndarray:
        """Returns a copy of the current state as a numpy array.

        The form of the state is [cos(theta), sin(theta), theta_dot].

        :return: state of the environment
        """
        return np.copy(self.state)

    def get_reinforcement(self) -> float:
        """Returns a copy of the reinforcement recieved for the last action.

        :return: last recieved reinforcement
        :rtype: float
        """
        return copy.copy(self.reinforcement)

    def get_accumulated_reinforcement(self) -> float:
        """Returns the accumulated reinforcement

        :return: accumulated reinforcement
        :rtype: float
        """
        return copy.copy(self.acc_reinf)

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

    def reset(self) -> np.array:
        """Resets the pendulum."""
        self.state = self.env.reset()
        self.reinforcement = 0
        self.acc_reinf = 0

        return self.get_state()


def get_random_action() -> float:
    """Returns a random float in [-2, 2)"""
    return np.random.uniform(-2, 2)


def get_zero_action() -> float:
    """Returns 0 as float, implicating that this action does nothing."""
    return 0.0


def get_random_plan(steps: int) -> np.array:
    """Returns an array with steps random actions."""
    return np.array([get_random_action() for _ in range(steps)])


def get_zero_plan(steps: int) -> np.array:
    """Returns an array of length steps with zeros"""
    return np.array([get_zero_action() for _ in range(steps)])


def run_simulation_plan(plan: list, render: bool = False) -> list:
    """Runs the simulation by iteration through the plan.

    :param plan: list of actions in [-2, 2]
    :return: list of states as numpy arrays
    """
    # initialise simulation environment
    # theta is in [-pi, pi), thetadot is in [-1, 1)
    env = Pendulum(render=render)
    # get the state
    state_0 = env.reset()
    # save first state in the list
    simulation_states = [state_0]

    for action in plan:
        state, _ = env(action)
        simulation_states.append(state)

    env.close()
    return simulation_states


def run_random_agent(steps: int = 10,
                     render: bool = False
                     ):
    # initialise simulation environment
    # theta is in [-pi, pi), thetadot is in [-1, 1)
    env = Pendulum(render=render)
    env.reset()
    # save first state in the list
    reinforcements = [[0, env.get_reinforcement()]]

    for step in range(1, steps):
        action = get_random_action()
        _, reinf = env(action)
        reinforcements.append([step, reinf])

    env.close()
    return env.get_accumulated_reinforcement(), reinforcements


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
    while True:
        # create env and list
        env = Pendulum(render=False)
        rollout_feature = []
        rollout_target = []
        # get the first state
        current_state = env.get_state()
        for _ in range(steps):
            # get an action
            next_action = get_random_action()
            # create the s_0, action tuple
            last_input = np.append(current_state, next_action)
            # get the next state
            current_state, _ = env(next_action)
            # save the tuple
            rollout_feature.append(last_input)
            rollout_target.append(current_state)
        # close the current env
        env.close()
        # yield the states
        yield (np.array(rollout_feature, dtype="float32"),
               np.array(rollout_target, dtype="float32")
               )
