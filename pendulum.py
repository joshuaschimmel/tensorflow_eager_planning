import gym
import csv
import numpy as np
import tensorflow as tf
import forward_model_tf as fm
import matplotlib.pyplot as plt


tf.enable_eager_execution()


def example_run(episodes: int = 20) -> None:
    """Runs an example of the pendulum simulation.

    Args:
        episodes: # of episodes
    """
    env = gym.make('Pendulum-v0')
    for episode in range(episodes):
        observation = env.reset()
        for i in range(500):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print(f"Episode finished after {i+1} timesteps")
                break


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
        state_0 = env.reset()
        env.render(mode = "rgb_array")

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
        env.render(mode="human")
        action = get_action(env.action_space)
        s_0, _, _, _ = env.step(np.array([0]))
        simulation_states.append(s_0)

    env.close()
    return simulation_states



