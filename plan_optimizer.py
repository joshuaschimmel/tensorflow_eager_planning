import tensorflow as tf
import numpy as np
import time
import pendulum


class Planner:

    def __init__(self,
                 world_model: tf.keras.models.Model,
                 learning_rate: float,
                 iterations: int,
                 initial_plan: list,
                 fill_function,
                 ):
        """Initializes the Planner object.

        :param world_model: The model used for the prediction
        :param learning_rate: the rate at whitch gradients are applied
            in each iteration
        :param iterations: the # of optimization iterations
        :param initial_plan: the initial plan
        :param fill_function: a function used to fill the plan with
            new actions
        """
        self.model = world_model
        self.plan = initial_plan
        self.adaptation_rate = learning_rate
        self.iterations = iterations
        self.new_action = fill_function
        self.current_state = None
        self.current_action = None

    def plan_next_step(self, next_state: list) -> (float, list):
        """Update the state and returns the next action.

        Updates the state and adds a new plan step after removing the
        action for the old state. Then optimizes the new plan based
        on the new state.
        Returns the next action the agent wants to take.

        :param next_state: a tensor representing the environments
            current state with shape (1, 3)
        :return: np.float32, log_dicts the next action to be taken and
            a list of log dicts
        """
        # reassign to new state
        self.current_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
        # remove the first element
        self.plan = self.plan[1:]
        # add a new action
        self.plan.append(self.new_action())
        # optimize the plan
        logs = self.optimize_plan()
        # save the current action in a field
        self.current_action = self.plan[0]
        # call the function to return the numpy value of the action
        return self.get_numpy_action(), logs

    def get_numpy_action(self):
        """Returns the current action as a numpy array.

        :return: numpy array of the action the agent wants to take
        """
        return tf.convert_to_tensor(self.current_action).numpy()

    def get_numpy_plan(self):
        """Returns the plan as an np array of arrays

        :return: np array
        """
        return np.array([tf.convert_to_tensor(x).numpy() for x in self.plan])

    def set_plan(self, plan: list):
        """Assigns a new plan to the object"""
        self.plan = plan

    def set_adaptation_rate(self, rate: float):
        """Sets a new adaptation rate"""
        self.adaptation_rate = rate

    def set_iterations(self, iterations: int):
        """Sets a new number of iterations"""
        self.iterations = iterations

    def reset(self, plan):
        """Resets the object and gives it a new starting plan

        :param plan: a new plan
        :return: Does not return anything
        """
        self.current_action = None
        self.current_state = None
        self.plan = plan

    def optimize_plan(self) -> list:
        """Optimizes the plan using gradient descent.

        This functions first predicts all future states based on the
        current plan and calculates their corresponding rewards.
        The rewards are then stored in a list with a copy of all
        previous actions for later derivation.
        After all pairs have been made, all derivatives for each
        reward will be calculated. Those are again saved in a list,
        for later summation of all gradients for an action.
        After all gradients have been calculated, the list of
        list of gradients is reduced by summing all inner gradients.
        The resulting list contains a final gradient for each action
        in the plan.
        Lastly, this function applies the gradients to the plan and
        returns nothing.

        :return: a list of dictionaries with log data such as a
            dictionary for measured times and the gradients as an
            numpy array
            [{
             iteration: epsilon
             times: {start: timestamp_1, ..., end: timesamp_n},
             gradients: np.array
            }]
        """
        logs = []
        for e in range(self.iterations):
            #print(f"Iteration {e + 1}")

            # log the starting time for each iteration
            start_time = time.time()

            # set the initial states
            prediction_state = self.current_state
            # list taken actions before the current step
            taken_actions = []
            # gradients for each action
            grads = []
            # list of the step in the plan, the actions previously
            # taken and the resulting loss after the current step
            derivatives = []
            # log the position, loss and gradients for each action
            optimization_log = []

            # collect the rewards and calculations using
            # gradient tape
            with tf.GradientTape(persistent=True) as tape:
                # iterate over all actions and keep track
                # of the actions index for logging
                for step in range(len(self.plan)):
                    action = self.plan[step]
                    # watch the action variable
                    tape.watch(action)
                    action = tf.reshape(action, shape=(1,))
                    # concat the state with the action to get
                    # the model input. for this, squeeze the state
                    # by one axis into a list
                    next_input = tf.concat(
                        [tf.squeeze(prediction_state), action],
                        axis=0
                    )
                    # reshape the input for the model
                    next_input = tf.reshape(next_input, shape=(1, 4))
                    # get the next state prediction
                    prediction_state = self.model(next_input)
                    # update the list of already taken actions
                    taken_actions.append(action)
                    # flatten the state and calculate the loss
                    loss_value = reinforcement(tf.squeeze(prediction_state))
                    # add the loss value together with the actions that
                    # led up to it and add them
                    # to the list of derivatives
                    derivatives.append([step,
                                        taken_actions.copy(),
                                        loss_value
                                        ])

            # Log time after the tape is done
            tape_time = time.time()
            #print(f"Tape Time: {tape_time - start_time}")

            # iterate over all derivative pairs and
            # add the gradients to the the grads list
            for loss_pos, taken_actions, loss in derivatives:
                # init counter to assign derivatives to the action
                taken_action_i = 0

                # calc gradients for all actions taken before this loss
                for action in taken_actions:
                    # calculate the gradients for each action
                    grad = tape.gradient(loss, action)
                    # add the gradient to the position in grads
                    # corresponding to the actions position in plan
                    # a bit of EAFP
                    try:
                        # add the grad to the existing list
                        grads[taken_action_i] += grad
                    except IndexError:
                        # initialize a new one
                        grads.append(grad)

                    # add the log for each action to the whole log list
                    # as a numpy array
                    optimization_log.append(np.array([
                        # objects adaptation rate
                        self.adaptation_rate,
                        # epsilon, current iteration
                        e,
                        # loss for this action
                        loss,
                        # the position of the loss
                        loss_pos,
                        # the gradient
                        grad,
                        # the position of the action
                        taken_action_i,
                    ]))

                    # update counter
                    taken_action_i += 1

            # Log the time when gradients were calculated
            grad_time = time.time()
            #print(f"Grad Time: {grad_time - tape_time}")

            # apply the sums to each action
            for grad, action in zip(grads, self.plan):
                # add gradients weighted with adaptation rate
                action.assign_add(grad * self.adaptation_rate)

            # Log time when the gradients where assigned to the actions
            end_time = time.time()
            #print(f"Assign Time: {end_time - grad_time}")
            #print(f"Iteration {e + 1} Total Time: {end_time - start_time}\n")

            # append data to the log dict
            logs.append({
                "times": {
                    "start": start_time,
                    "tape": tape_time,
                    "grad": grad_time,
                    "end": end_time
                },
                "gradient_log": np.array(optimization_log)
            })
        # return the logs
        return logs


def reinforcement(state: tf.Tensor):
    """Calculates the reinforcement for a state

    This reinforcement is a loss depending on the angle of
    the pendulum and its speed. The loss is minimal, if
    the pendulums angle theta is 0
    (and thus cos(theta) 1 and sin(theta) 0).

    :param state: A Tensor of shape (3,)
    :return: a scalar reinforcement value
    """
    return -(tf.square(tf.subtract(state[0], 1))
             + tf.square(state[1])
             + 0.01 * tf.square(state[2]))


def get_random_plan(steps: int) -> list:
    """Returns a list of random tf Variables for planning.

    :param steps: length of the list
    :return: list with random float32s as tf.Variables
    """
    # initialize a random plan of with _steps steps
    plan = []
    for _ in range(steps):
        plan.append(get_random_action())

    return plan


def get_random_action() -> tf.Variable:
    """Returns a random action as tf Variable in [-2 ,2]

    :return: uniform random tf.Variable in [-2, 2]
    """
    return tf.Variable(pendulum.get_random_action(), dtype=tf.float32)