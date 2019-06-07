import tensorflow as tf
import numpy as np
import time
import pendulum


class Planner:

    def __init__(self,
                 world_model: tf.keras.models.Model,
                 adaptation_rate: float,
                 iterations: int,
                 initial_plan: list,
                 fill_function,
                 strategy: str = "none",
                 return_logs: bool = False
                 ):
        """Initializes the Planner object.

        :param world_model: The Keras world model used for the prediction
        :param adaptation_rate: the rate at whitch gradients are adapted
            in each iteration
        :param iterations: the # of optimization iterations
        :param initial_plan: an initial plan to start optimizing from
        :param fill_function: a function used to fill the plan with
            new actions
        :param strategy: a weighing strategy to prefer different reinforcments,
            optional, defaults to 'none' (no weighing)
        :param return_logs: whether this planner creates and returns logs
            while planning. If False, will return None (default False)
        """
        self.model = world_model
        self.plan = initial_plan
        self.adaptation_rate = adaptation_rate
        self.iterations = iterations
        self.new_action = fill_function
        self.plan_strategy = strategy
        self.current_state = None
        self.current_action = None
        self.return_logs = return_logs

    def plan_next_step(self, next_state: list) -> (float, list):
        """Plan for the new state and return the next action.

        Starts plan optimization for the new state,
        Returns the next action the agent wants to take.

        :param next_state: a tensor representing the environments
            new current state with shape (1, 3)
        :return: np.float32, log_dicts the next action to be taken and
            a list of log dicts
        """
        self.current_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
        # update the plan
        self.plan = self.plan[1:]
        self.plan.append(self.new_action())
        # optimize the plan
        logs = self.optimize_plan()

        self.current_action = self.plan[0]
        return self.get_numpy_action(), logs

    def get_numpy_action(self):
        """Returns the current action as a numpy value.

        :return: numpy representation of the action the agent wants to take
        """
        return tf.convert_to_tensor(self.current_action).numpy()

    def get_numpy_plan(self):
        """Returns the whole plan as numpy objects.

        :return: plan as numpy array
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

    def set_strategy(self, strategy: str):
        """Sets a new strategy for this planner."""
        self.plan_strategy = strategy

    def reset(self, plan):
        """Resets the planner and updates the plan

        :param plan: a new plan
        """
        self.current_action = None
        self.current_state = None
        self.plan = plan

    def optimize_plan(self) -> list:
        """Optimizes the plan using gradient descent.

        This function is the core of this task and look-ahead planning.
        First, it unrolls the plan using the world model and
        calculates predicted reinforcements from the resulting states.
        The reinforcements are then reduced to a single reinforcement
        energy, with respect to whitch gradients of all actions in the
        plan are calculated.
        If the planner uses either the 'first' or 'last' strategy,
        the gradients are linearly weighed.
        If no weighing is to take place, the planning strategy needs
        to be 'none', else a ValueError is raised.
        Lastly, the gradients are applied to the current plan.

        If the flag was set, the optimizer creates logs of the form
        specified below.

        :return: a list of dictionaries with log data such as a
            dictionary for measured times and the gradients as an
            numpy array
            [{
             iteration: epsilon
             times: {start: timestamp_1, ..., end: timesamp_n},
             gradients: np.array
            }]
        """
        logs = [] if self.return_logs else None
        for e in range(self.iterations):
            global_step = tf.Variable(0)

            # log the starting time for each iteration
            start_time = time.time()

            # initialize variables
            prediction_state = self.current_state
            taken_actions = []
            grads = []
            losses = []

            # collect the rewards and calculations using gradient tape
            with tf.GradientTape() as tape:
                # unroll the plan
                for step in range(len(self.plan)):
                    action = self.plan[step]
                    # tell the tape to track the action
                    tape.watch(action)
                    action = tf.reshape(action, shape=(1,))
                    # concat the current state with the action to get
                    # the model input. for this, squeeze the state
                    # by one axis into a list
                    next_input = tf.concat(
                        [tf.squeeze(prediction_state), action],
                        axis=0
                    )
                    next_input = tf.reshape(next_input, shape=(1, 4))

                    # get the next state prediction
                    prediction_state = self.model(next_input)
                    loss_value = reward(prediction_state, action) * -1

                    losses.append(loss_value)
                    taken_actions.append(action)

                # collapse losses into single unit
                e_reinf = tf.reduce_sum(losses)

            # Log time after the tape is done
            tape_time = time.time()
            taken_action_i = 0
            grads = tape.gradient(e_reinf, taken_actions)

            # check weighing strategy
            if self.plan_strategy == "last":
                temp_grads = []
                for grad in grads:
                    temp_grads.append(grad * ((taken_action_i + 1) /
                                              len(taken_actions)))
                    taken_action_i += 1
                grads = temp_grads

            elif self.plan_strategy == "first":
                temp_grads = []
                for grad in grads:
                    temp_grads.append(grad * (1 - ((taken_action_i + 1) /
                                                   len(taken_actions))))
                    taken_action_i += 1
                grads = temp_grads

            elif self.plan_strategy == "none":
                taken_action_i = len(grads)

            # raise a value error if no valid strategy was given
            else:
                raise ValueError(f"Expected planning strategy, got "
                                 f"{self.plan_strategy} instead.")

            grads = [tf.reshape(x, []) for x in grads]

            if self.return_logs:
                # add the log for each action to the whole log list
                # as a numpy array
                plan_length = len(self.plan)
                adaptation_rates = [self.adaptation_rate] * plan_length
                reinf_energies = np.repeat(
                    np.asscalar(e_reinf.numpy()),
                    plan_length
                )

                optimization_log = np.stack((
                    # objects adaptation rate
                    adaptation_rates,
                    # epsilon, current iteration
                    [e] * plan_length,
                    # loss for this action
                    reinf_energies,
                    # the position of the loss
                    [0] * plan_length,
                    # the gradient
                    [x.numpy() for x in grads],
                    # the position of the action
                    np.arange(plan_length),
                ), -1)

            # Log the time when gradients were calculated
            grad_time = time.time()

            optimizer = tf.train.GradientDescentOptimizer(
                self.adaptation_rate
            )
            # apply the gradients to the actions
            optimizer.apply_gradients(
                zip(grads, self.plan),
                global_step)

            # Log time when the gradients where assigned to the actions
            end_time = time.time()

            # if logging is on, append times and gradients to the log list
            if self.return_logs:
                logs.append({
                    "times": {
                        "start": start_time,
                        "tape": tape_time,
                        "grad": grad_time,
                        "end": end_time
                    },
                    "gradient_log": optimization_log
                })
        return logs


def reward(states: tf.Tensor, taken_actions: tf.Variable):
    """Calculates the reinforcement for a state

    This reinforcement is a loss and its result depending on
    the pendulum's state, speed and the action needed to get
    to the state.
    The loss is minimal, if the pendulum is up (theta 0),
    still (thetadot 0) and no actions were used.

    :param states: Tensor of states
    :type states: tf.Tensor
    :param taken_actions: List of Actions taken to get to the states
    :type taken_actions: tf.Variable
    :return: reinforcement value
    :rtype: tf.constant
    """
    rewards = -(tf.square(states[:, 0] - 1)
                + 0.1 * tf.square(states[:, 2])
                + 0.001 * tf.square(taken_actions)
                )
    return rewards


def get_random_action() -> tf.Variable:
    """Returns a random action as tf Variable in [-2 ,2]"""
    return tf.Variable(pendulum.get_random_action(), dtype=tf.float32)


def get_zero_action() -> tf.Variable:
    """Returns a zero action (do nothing) as a tf Variable."""
    return tf.Variable(pendulum.get_zero_action(), dtype=tf.float32)


def get_random_plan(steps: int) -> list:
    """Returns a list of random tf Variables for planning."""
    return [get_random_action() for _ in range(steps)]


def get_zero_plan(steps: int) -> list:
    """Returns a list of zero tf Variables for planning."""
    return [get_zero_action() for _ in range(steps)]
