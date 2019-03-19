import tensorflow as tf
import numpy as np
import time


class Optimizer:

    def __init__(self,
                 world_model: tf.keras.models.Model,
                 learning_rate: float,
                 iterations: int,
                 initial_plan: list,
                 fill_function,
                 use_test_function: bool = False
                 ):
        """Initializes the Plan Optimizer object.

        :param world_model: The model used for the prediction
        :param learning_rate: the rate at whitch gradients are applied
            in each iteration
        :param iterations: the # of optimization iterations
        :param initial_plan: the initial plan
        :param fill_function: a function used to fill the plan with
            new actions
        :param use_test_function: whether to use a test function.
            Used for debugging.
        """
        self.model = world_model
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.plan = initial_plan
        self.new_action = fill_function
        self.current_state = None
        self.current_action = None
        self.use_test = use_test_function

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
        self.current_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
        # remove the first element
        self.plan = self.plan[1:]
        # add a new action
        self.plan.append(self.new_action())
        # optimize the plan
        if self.use_test:
            self.optimize_plan_modified()
        else:
            self.optimize_plan()
        # save the current action in a field
        self.current_action = self.plan[0]
        # call the function to return the numpy value of the action
        return self.get_numpy_action()

    def optimize_plan(self):
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

        :return: Nothing.
        """
        for e in range(self.iterations):
            print(f"Iteration {e + 1}")

            # set the initial states
            prediction_state = self.current_state
            taken_actions = []
            derivatives = []
            grads = []

            # log the starting time for each iteration
            start_time = time.time()

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

            # Log time after the tape is done
            tape_time = time.time()
            print(f"Tape Time: {start_time - tape_time}")

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
                        # add the grad to the existing list
                        grads[i] += grad
                    except IndexError:
                        # initialize a new one
                        grads.append(grad)

                    # update counter
                    i += 1

            # Log the time when gradients were calculated
            grad_time = time.time()
            print(f"Grad Time: {tape_time - grad_time}")

            # iterate over grads and fill the lists with tf constants to get
            # an even shape
            # first list is the longes
            # max_len = len(grads[0])
            # for grad_list in grads:
            #     while len(grad_list) < max_len:
            #         # add zero constants until the lengths are the same
            #         grad_list.append(tf.zeros(1))

            # reduce the the grads by summing them up
            # sums = tf.reduce_sum(grads, axis=0)

            # apply the sums to each action
            for grad, action in zip(grads, self.plan):
                # add learning rate
                action.assign_add(grad * self.learning_rate)

            # Log time when the gradients where assigned to the actions
            end_time = time.time()
            print(f"Assign Time: {grad_time - end_time}")
            print(f"Iteration {e + 1} Total Time: {start_time - end_time}\n")

    def optimize_plan_modified(self):
        """Tries to improve optimize_plan

        :return:
        """
        for e in range(self.iterations):
            print(f"Iteration {e + 1}")

            # set the initial states
            prediction_state = self.current_state
            taken_actions = []
            derivatives = []
            grads = []

            # log the starting time for each iteration
            start_time = time.time()

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

                    i = 0
                    for taken_action in taken_actions:
                        # calculate the gradients for each action
                        grad = tape.gradient(loss_value, taken_action)
                        # add the gradient to the position in grads
                        # corresponding to the actions position in plan
                        # a bit of EAFP
                        try:
                            # add the grad to the existing one
                            grads[i] += grad
                        except IndexError:
                            # initialize a new one
                            grads.append(grad)

                        # update counter
                        i += 1

            # Log Tape/Grad time (both the same here)
            grad_time = time.time()
            print(f"Grad Time: {start_time - grad_time}")

            # iterate over grads and fill the lists with tf constants to get
            # an even shape
            # first list is the longes
            #max_len = len(grads[0])
            #for grad_list in grads:
            #    while len(grad_list) < max_len:
            #        # add zero constants until the lengths are the same
            #        grad_list.append(tf.zeros(1))

            # reduce the the grads by summing them up
            #sums = tf.reduce_sum(grads, axis=0)

            # apply the sums to each action
            for grad, action in zip(grads, self.plan):
                # add learning rate
                action.assign_add(grad * self.learning_rate)

            # Log time when the gradients where assigned to the actions
            end_time = time.time()
            print(f"Assign Time: {grad_time - end_time}")
            print(f"Iteration {e + 1} Total Time: {start_time - end_time}\n")

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
             + 0.01 * tf.square(state[2]))
