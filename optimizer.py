import tensorflow as tf
import numpy as np


class Optimizer:

    def __init__(self,
                 world_model: tf.keras.models.Model,
                 learning_rate: float,
                 iterations: int,
                 initial_plan: list,
                 fill_function,
                 ):
        """Initializes the Plan Optimizer object.

        :param world_model: The model used for the prediction
        :param learning_rate: the rate at whitch gradients are applied
            in each iteration
        :param iterations: the # of optimization iterations
        :param initial_plan: the initial plan
        :param fill_function: a function used to fill the plan with
            new actions
        """
        self.model = world_model
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.plan = initial_plan
        self.new_action = fill_function
        self.current_state = None
        self.current_action = None

    def __call__(self, next_state: list) -> float:
        """Update the state and returns the next action.

        Updates the state and adds a new plan step after removing the
        action for the old state. Then optimizes the new plan based
        on the new state.
        Returns the next action the agent wants to take.

        :param next_state: a tensor representing the environments
            current state with shape (1, 3)
        :return: the next action to be taken
        """
        # reassign to new state
        self.current_state = next_state
        # remove the first element
        self.plan = self.plan[1:]
        # add a new action
        self.plan.append(self.new_action())
        # optimize the plan
        self.optimize_plan()
        # save the current action in a field
        self.current_action = self.plan[0]
        # call the function to return the numpy value of the action
        return self.get_numpy_action()

    def optimize_plan(self):
        """Optimizes the plan.

        """
        for e in range(self.iterations):
            print(f"Iteration {e + 1}")

            # set the initial states
            prediction_state = self.current_state
            taken_actions = []
            derivatives = []
            grads = []

            # collect the rewards and calculations using
            # gradient tape
            with tf.GradientTape(persistent=True) as tape:
                # iterate over all actions
                for action in self.plan:
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
                    derivatives.append([taken_actions.copy(), loss_value])

            # now iterate over all derivative pairs and
            # add the gradients to the the grads list
            for actions, loss in derivatives:
                # initialize counter for list accessing
                i = 0

                for action in actions:
                    # calculate the gradients for each action
                    grad = tape.gradient(loss, action)
                    # add the gradient to the position in grads
                    # corresponding to the actions position in plan
                    # a bit of EAFP
                    try:
                        # add the grad to the existing one
                        grads[i].append(grad)
                    except IndexError:
                        # initialize a new one
                        grads.append([grad])

                    # update counter
                    i += 1

            # iterate over grads and fill the lists with tf constants to get
            # an even shape
            # first list is the longes
            max_len = len(grads[0])
            for grad_list in grads:
                while len(grad_list) < max_len:
                    # add zero constants until the lengths are the same
                    grad_list.append(tf.zeros(1))

            # reduce the the grads by summing them up
            sums = tf.reduce_sum(grads, axis=0)

            # apply the sums to each action
            for grad, action in zip(sums, self.plan):
                # add learning rate
                action.assign_add(grad * self.learning_rate)

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

    def reset(self, plan):
        """Resets the object and gives it a new starting plan

        :param plan: a new plan
        :return: Does not return anything
        """
        self.current_action = None
        self.current_state = None
        self.plan = plan


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
             + 0.1 * tf.square(state[2]))
