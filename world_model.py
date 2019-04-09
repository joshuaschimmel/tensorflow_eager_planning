import os
from collections import deque
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import json
from typing import List, Tuple

import gym

import pendulum

tf.enable_eager_execution()


def plot_model_performance(training_x, training_y,
                           validation_x, validation_y,
                           test_x, test_y):
    """Prints the metrices for the model

    :param training_x: Training Step in all epoch
    :param training_y: Error during training
    :param validation_x: Validation after each epoch
    :param validation_y: Error during validation
    :param test_x: Values of max epoch
    :param test_y: Test_Error after last epoch
    :return:
    """
    plt.figure()
    # Train loss
    plt.plot(training_x,
             training_y,
             label="Training Loss")

    # Validation loss
    plt.plot(validation_x,
             validation_y,
             "--",
             label="Validation Loss",
             linewidth=2)

    # Test loss
    plt.plot(test_x,
             test_y,
             "r+",
             label="Test Loss",
             markersize=10,
             linewidth=10,
             )

    plt.xlabel("End of Epoch #")
    plt.ylabel("Error Value")
    plt.grid(b=True, alpha=0.25, linestyle="--")
    plt.tick_params(axis="both", which="major", direction="out")
    plt.legend()

    plt.show()


def predict_states(model: tf.keras.models.Model,
                   state_0: list,
                   plan: list
                   ) -> list:
    """Uses the model to predict the state after each step.

    :param model: Model used for prediction
    :param state_0: initial state s_0: [cos, sin, dot]
    :param plan: list of actions [a_1, ..., a_n]
    :return: list of predicted states [s_0, ..., s_n]
    """
    # initial state does not need to be predicted
    predicted_states = [state_0]

    # use deque for efficient deconstruction of the list
    plan = deque(plan)

    # get the current state
    current_state = state_0

    while plan:
        # get the next action
        next_action = plan.popleft()

        # merge it with the current state
        next_input = np.append(current_state, next_action)

        # shape the input for the model (because of expected batching)
        # into the form [[current_state]]
        next_input = np.array(next_input).reshape(1, 4)

        # let the model predict the next state
        prediction = model(next_input).numpy()[0]

        # add the prediction to the return list
        predicted_states.append(prediction)

        # reassign current state to prediction
        current_state = prediction

    return predicted_states


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
        # action = get_action(env.action_space)

        action = np.array([0])
        # build the model input
        s_0 = np.array([cos, sin, dot, action]).reshape(1, 4)
        # do a step and get the next state
        s_1, _, _, _ = env.step(action)
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


def mse_loss(model: tf.keras.Model,
             model_input: tf.Tensor,
             model_target: tf.Tensor
             ):
    """Calculates MSE of the model given output and expected values

    :param model: a model the mse is to be calculated for
    :param model_input: input
    :param model_target: teacher value
    :returns: loss value
    """
    y_ = model(model_input)
    _reduction_string = "weighted_sum_over_batch_size"
    return tf.losses.mean_squared_error(labels=model_target,
                                        predictions=y_,
                                        reduction=_reduction_string
                                        )


def abs_loss(model: tf.keras.Model,
             model_input: tf.Tensor,
             model_target: tf.Tensor):
    """Calculates the absolute difference loss for a model.

    :param model: the model used for the prediction
    :param model_input: input tensor to be used for the prediction
    :param model_target: wanted output tensor of the prediction
    :returns loss float Tensor with the shape of target
    """
    y_ = model(model_input)
    return tf.losses.absolute_difference(labels=model_target,
                                         predictions=y_,
                                         reduction="none"
                                         )


def mae_loss(model: tf.keras.Model,
             model_input: tf.Tensor,
             model_target: tf.Tensor):
    """Calculates the Mean Absolute Error for a model given input and target.

    :param model: the model used for the prediction
    :param model_input: input tensor to be used for the prediction
    :param model_target: wanted output tensor of the prediction
    :returns loss float Tensor with the shape of a scalar
    """
    _y = model(model_input)
    _reduction_string = "weighted_sum_over_batch_size"
    return tf.losses.absolute_difference(labels=model_target,
                                         predictions=_y,
                                         reduction=_reduction_string
                                         )


def rmse_loss(model: tf.keras.Model,
              model_input: tf.Tensor,
              model_target: tf.Tensor):
    """Calculates RMSE of the model given input and teacher values

    :param model: a kerase model for which to calculate the RMSE
    :param model_input: input for the model
    :param model_target: target|teacher values
    :return: RMSE value
    """
    model_output = model(model_input)
    return tf.sqrt(
        tf.losses.mean_squared_error(labels=model_target,
                                     predictions=model_output,
                                     reduction="weighted_sum_over_batch_size"
                                     ))


class WorldModelWrapper:

    def __init__(self):
        """A wrapper class for a world model, adding functionality.

        This class is a wrapper for world models, taking care of
        initializing and training a new keras model, having different
        loss functions at its disposal and being able to save
        the currently held world model.
        """
        self.model = None

    def _raise_none_model(self):
        """Raises a ValueError, indicating that self.model is None."""

        raise ValueError("Model is of type None! Was it not initialized?")

    def get_model(self) -> tf.keras.Model:
        """Returns currently held model."""
        if self.model is None:
            self._raise_none_model()
        return self.model

    def set_model(self, model: tf.keras.Model):
        """Set the model of this object."""
        self.model = model

    def save_model(self,
                   file_path: str = "data/world_model.h5"
                   ):
        """Saves the model to the given path as an HDF5 File.

        Saves the current model as an HDF5-file to the relative folder
        structure, unless otherwise specified

        :param file_path: path to save to,
            defaults to "forward_model.h5"
        :param file_path: str, optional
        """
        self.model.save(file_path)

    def load_model(self,
                   file_path: str = "data/world_model.h5",
                   pre_compile: bool = False
                   ) -> tf.keras.Model:
        """Loads and updates the current model from a HDF5-file.

        Loads a keras model from the given HDF5-file AND sets it as
        the new current model. Also the returns the new current
        world model.

        :param file_path: path to the file,
            defaults to "world_model.h5"
        :type file_path: str, optional
        :param pre_compile: whether the model should be compiled
            before it will be returned,
            defaults to False
        :type pre_compile: bool, optional
        :return: the new current world model
        :rtype: tf.keras.Model
        """
        self.model = tf.keras.models.load_model(file_path, compile=pre_compile)

        return self.get_model()

    def build_keras_model(self,
                          neurons: int = 20,
                          hidden_layers: int = 2,
                          input_shape: tuple = (4,),
                          output_shape: tuple = 3,
                          dropout_rate: float = 0.5
                          ) -> tf.keras.Model:
        """Builds a sequential keras model with the given parameters.

        Using the parameters, this function builds a keras model with
        at least one fully connected Dense input layer and a Dropout layer
        before the output layer.

        :param neurons: number of neurons in each layer, defaults to 20
        :param neurons: int, optional
        :param hidden_layers: number of hidden layer used, defaults to 2
        :param hidden_layers: int, optional
        :param input_shape: shape for the first layer to accept,
            defaults to (4,)
        :param input_shape: tuple, optional
        :param output_shape: number of neurons in the output, defaults to 3
        :param output_shape: int, optional
        :param dropout_rate: dropout rate before the output layer,
            defaults to 0.5
        :param dropout_rate: float, optional
        :return: the current modl
        :rtype: tf.keras.Model
        """
        # number of hidden layers is 0 if hidden_layers is smaller than that
        _layers = 0 if hidden_layers < 0 else hidden_layers

        # prepare model by gettin a simple sequential object
        model = tf.keras.Sequential()
        # add input layer with input shape
        model.add(tf.keras.layers.Dense(neurons,
                                        input_shape=input_shape,
                                        activation=tf.nn.relu
                                        ))

        # add any additional dense layers
        for i in range(_layers):
            model.add(tf.keras.layers.Dense(neurons,
                                            activation=tf.nn.relu
                                            ))
        # add dropout layer
        model.add(tf.keras.layers.Dropout(dropout_rate))

        # add output layer
        # TODO: check if output shape is correct
        model.add(tf.keras.layers.Dense(output_shape))

        self.model = model
        # return the model
        return self.get_model()

    def train_model(self,
                    env: gym.Env,
                    iterations: int = 100,
                    steps: int = 1,
                    epochs: int = 10,
                    batch_size: int = 32,
                    test_split: float = 0.1,
                    validation_split: float = 0.1,
                    learning_rate: float = 0.001,
                    loss_function=rmse_loss,
                    plot_performance: bool = True,
                    ):
        """
        """
        # choose an optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)
        losses = []
        test_losses = []

        # generate new data in each epoch
        for e in range(epochs):

            iteration_data = []

            for ite in range(iterations):
                # generate the data for this epoch

                # initialize the first state and the environment
                current_state = env.reset()
                # get the next action
                next_action = env.action_space.sample()
                # get the resulting state from the action
                next_state, _, _, _ = env.step(next_action)

                model_input = np.hstack([current_state, next_action]
                                        ).reshape(1, 4)
                model_target = next_state.reshape(1, 3)

                iteration_data.append([ite, model_input, model_target])

            # train the model
            for i, data, target in iteration_data:

                # calculate loss in GradientTape
                with tf.GradientTape() as tape:
                    loss_value = loss_function(
                        self.model,
                        data,
                        target
                    )
                # get the gradients
                grads = tape.gradient(
                    loss_value,
                    self.model.variables
                )

                # apply gradients
                optimizer.apply_gradients(
                    zip(grads, self.model.variables),
                    global_step=tf.train.get_or_create_global_step()
                )

                # log loss
                loss = loss_value.numpy()
                losses.append(np.array([e, i, loss]))

                # output status to console
                print(f"Loss {loss} in {e}/{epochs};"
                      f" {i}/{iterations}")

            # TODO Validation loss

        # save model
        self.save_model()

        # generate test data

        # TODO: Add parameter for amount of test data points
        for _ in range(100):
            # reset the environment
            current_state = env.reset()
            next_action = env.action_space.sample()
            # get the next state
            next_state, _, _, _ = env.step(next_action)

            loss_value = loss_function(self.model,
                                       np.hstack(
                                           [current_state, next_action]
                                       ).reshape(1, 4),
                                       next_state.reshape(1, 3)
                                       )
            test_losses.append(loss_value.numpy())

        # return logs
        return losses, test_losses
