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
                         file_path: str = "pendulum_data.csv"
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
    env = gym.make('Pendulum-v0')
    simulation_states = []

    cos, sin, dot = env.reset()
    for i in range(steps):
        env.render(mode="human")
        action = get_action(env.action_space)
        s_0 = np.array([cos, sin, dot, action])
        s_0 = s_0.reshape(1, 4)
        simulation_states.append(s_0)

        s_1, _, _, _ = env.step(action)
        cos, sin, dot = s_1

    env.close()
    return simulation_states


def predict_simulation(predictor: tf.keras.models.Model,
                       loss,
                       steps: int
                       ) -> list:
    """Uses the predictor to predict the simulation states.

    :param predictor: Model used for prediction.
    :param loss: A function calculation the loss.
    :param steps: # of steps to be done in the simulation.
    :return: List of losses calculated by the loss function.
    """
    # build the simulation
    env = gym.make("Pendulum-v0")
    prediction_losses = []

    # get the initial state
    cos, sin, dot = env.reset()
    for i in range(steps):
        # render the environment
        env.render(mode="human")
        # get a random action
        #action = get_action(env.action_space)

        action = np.array([0])
        # build the model input
        s_0 = np.array([cos, sin, dot, action]).reshape(1, 4)
        # do a step and get the next state
        s_1,_,_,_ = env.step(action)
        # reassign for next state
        cos, sin, dot = s_1
        # reshape s_1 into a label
        s_1 = s_1.reshape(1, 3)
        # compare the models prediction to reality
        prediction_loss = loss(predictor, s_0, s_1)
        print(f"Prediction Loss is: {prediction_loss}")
        prediction_losses.append(prediction_loss)

    env.close()
    return prediction_losses


def predict_plan_effect(model: tf.keras.models.Model,
                        plan: list,
                        loss
                        ) -> list:
    pass