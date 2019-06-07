import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from collections import deque
from typing import List, Tuple
import matplotlib.pyplot as plt

import gym
import pendulum

tf.enable_eager_execution()


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
             model_target: tf.Tensor
             ):
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
             model_target: tf.Tensor
             ):
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
    """Calculates root MSE of the model given input and teacher values

    :param model: a kerase model for which to calculate the RMSE
    :param model_input: input for the model
    :param model_target: target|teacher values
    :return: RMSE value
    """
    model_input = model_input.astype("float32")
    model_output = model(model_input)
    return tf.sqrt(
        tf.losses.mean_squared_error(labels=model_target,
                                     predictions=model_output,
                                     reduction="weighted_sum_over_batch_size"
                                     ))


def single_rmse_loss(output: np.array, target: np.array) -> float:
    """Calculates the RMSE for an output/target pair

    :param output: model output
    :type output: np.array
    :param target: model target
    :type target: np.array
    :return: RMSE value
    :rtype: float
    """
    output = output.astype("float32")
    target = target.astype("float32")
    return np.sqrt(
        np.mean(
            np.square(
                np.subtract(output,
                            target))))


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
                   file_path: str = "models/world_model.h5"
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
                   file_path: str = "models/world_model.h5",
                   pre_compile: bool = False
                   ) -> tf.keras.Model:
        """Loads and updates the current model from a HDF5-file.

        Loads a keras model from the given HDF5-file AND sets it as
        the new current model. Also returns the new current
        world model.

        :param file_path: path to the file,
            defaults to "world_model.h5"
        :type file_path: str, optional
        :param pre_compile: whether the model should be compiled
            before it will be returned (unused in this project),
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
        :param dropout_rate: (removed) dropout rate before the output layer,
            defaults to 0.5
        :param dropout_rate: float, optional
        :return: the current model
        :rtype: tf.keras.Model
        """
        # keras adds the first hidden layer implicitly
        _layers = max(0, hidden_layers - 1)

        model = tf.keras.Sequential()
        # add input layer with input shape and first hidden layer
        model.add(tf.keras.layers.Dense(neurons,
                                        input_shape=input_shape,
                                        activation=tf.nn.relu,
                                        dtype="float32"
                                        ))
        for _ in range(_layers):
            model.add(tf.keras.layers.Dense(neurons,
                                            activation=tf.nn.relu
                                            ))
        # add output layer
        model.add(tf.keras.layers.Dense(output_shape))

        self.model = model
        return self.get_model()

    def train_model(self,
                    env: pendulum.Pendulum,
                    rollouts: int = 100,
                    steps: int = 1,
                    learning_rate: float = 0.001,
                    tests: int = 10,
                    loss_function=rmse_loss,
                    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Trains the model on an gym environment.

        Using the initialized gym environment (pendulum-0), this
        function trains the model for a certain amount of iterations.
        The data will be randomly generated per each epoch.
        Additional consecutive steps per random initialization can be
        generated by changing the steps parameter. Returns the losses
        as DataFrames with the columns "epoch", "iteration" and "loss"
        for where the loss was generated. In the df for test losses,
        the epoch and iteration column contain the last value for
        the training cycle so that a following plot function can
        draw the values at the end of the training loss graph.

        :param env: gym environment wrapper
        :type env: pendulum.Pendulum
        :param rollouts: iterations to train,
            defaults to 100
        :type rollouts: int, optional
        :param steps: consecutive steps per rollout (mini-batch),
            defaults to 1
        :type steps: int, optional
        :param learning_rate: learning rate to use,
            defaults to 0.001
        :type learning_rate: float, optional
        :param tests: number of test runs to do,
            defaults to 10
        :type tests: int, optional
        :param loss_function: loss function to use,
            defaults to rmse_loss
        :type loss_function: function, optional
        :return: Tuple of train and test dataframes
        :rtype: Tuple[pd.DataFrame, pd.DataFrame]
        """
        optimizer = tf.train.AdamOptimizer(learning_rate)
        losses = []
        test_losses = []
        rollout_counter = 0

        try:
            # call generator function
            for rollout in pendulum.get_state_generator(steps):

                # break loop if enough rollouts have been made
                if rollout_counter >= rollouts:
                    break
                rollout_counter += 1
                feature, target = rollout

                # calculate loss in GradientTape
                with tf.GradientTape() as tape:
                    loss_value = loss_function(
                        self.model,
                        feature,
                        target
                    )
                # extract the gradients
                grads = tape.gradient(
                    loss_value,
                    self.model.variables
                )
                optimizer.apply_gradients(
                    zip(grads, self.model.variables),
                    global_step=tf.train.get_or_create_global_step()
                )

                loss = loss_value.numpy()
                losses.append(np.array([rollout, loss]))

                print(f"rollout {rollout_counter}/{rollouts}, "
                      f"loss: {loss}\n")

            self.save_model()

            # save the losses in a df for easy visualization
            losses_df = pd.DataFrame(
                losses,
                columns=["rollout", "mean_loss"]
            )

            # run tests
            test_run = 0
            for data, target in pendulum.get_state_generator(1):
                if test_run > tests:
                    break
                # calc the loss value
                loss_value = loss_function(
                    self.model,
                    data,
                    target
                )
                # append loss to the list and keep last iteration
                test_losses.append(np.array([
                    # reuse training variable for plotting
                    test_run,
                    loss_value.numpy()
                ]))
                test_run += 1

            # create dataframe out of test losses
            test_losses_df = pd.DataFrame(
                test_losses,
                columns=["test", "test_loss"]
            )
        except KeyboardInterrupt:
            self.save_model()

        return losses_df, test_losses_df

    def predict_states(self,
                       initial_state: list,
                       plan: list
                       ) -> list:
        """Uses the model to predict the state after each step.

        Starting from a starting state, this function lets the model
        predict each consecutive state after an action from the plan,
        which is a list of actions. Returns a list of the predicted
        states.

        :param initial_state: Initial starting state
        :type initial_state: list-like array
        :param plan: list of actions as floats in [-2, 2]
        :type plan: list
        :return: list of visited states as numpy arrays
        :rtype: list
        """
        state_0 = np.array(initial_state)
        current_state = state_0
        predicted_states = [state_0.reshape(3,)]

        # use deque for efficient deconstruction of the list
        plan = deque(plan)

        while plan:
            # get the next action
            next_action = plan.popleft()
            # merge it with the current state
            next_input = np.append(current_state, next_action)
            # shape the input for the model
            # into the form [[current_state]]
            next_input = next_input.reshape(1, 4).astype("float32")
            # let the model predict the next state
            prediction = self.model(next_input)
            # and cast the returned tensor to an np array
            prediction = prediction.numpy().reshape(3,)
            # add the prediction to the return list
            predicted_states.append(prediction)
            # reassign current state to prediction
            current_state = prediction

        return predicted_states

    def print_summary(self):
        """Prints the model summary."""
        self.model.summary()
