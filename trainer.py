import copy
import time
import numpy as np
import tensorflow as tf

import gym
import seaborn as sns
import matplotlib.pyplot as plt

import pendulum
import world_model
import plan_optimizer
import planning_cases
# import helper_functions as hf


tf.enable_eager_execution()

print(f"TensorFlow version: {tf.__version__}")
print(f"Eager execution: {tf.executing_eagerly()}")

# hyperparameters for the world model
_neurons = 40
_hidden_layer = 2
_epochs = 1
_loss_function = world_model.rmse_loss
_drop_rate = 0.5
_load_model = True


_steps = 100
_test_runs = 50

# get the model identifier
drop_text = "nodrop" if _drop_rate == 0 else f"{_drop_rate}drop"
neuron_text = (str(_neurons) + '-') \
    * _hidden_layer \
    + str(_neurons) + "_" \
    + str(_loss_function.__name__)

# load saved model
model_name = f"model_{neuron_text}_{_epochs}e_{drop_text}"
model_path = f"models/{model_name}.h5"


wm = world_model.WorldModelWrapper()
wm.load_model()

# wm.build_keras_model(neurons=30, hidden_layers=1, dropout_rate=0)
# wm.print_summary()
# train_l, test_l = wm.train_model(env=env, rollouts=4000, steps=15)
# ax = train_l.plot(x="rollout", y="mean_loss")
# test_l["test_position"] = [len(train_l.index)] * len(test_l.index)
# test_l.plot(x="test_position", y="test_loss",
#             color="red", marker="+",
#             kind="scatter", ax=ax
#             )
# plt.show()
wm.print_summary()
env = pendulum.Pendulum()


def test_world_model(wmr: world_model.WorldModelWrapper):
    _rollouts = 200
    _steps = 100
    # # eval behaviour of RMSE for a single rollout
    # _, f1 = planning_cases.single_rollout_error(
    #     steps=_steps,
    #     world_model_wrapper=wmr,
    #     visualize=True
    # )
    # # see RMSE for multiple rollouts
    # _, f2 = planning_cases.model_quality_analysis(
    #     wmr=wmr,
    #     rollouts=_rollouts,
    #     steps=_steps,
    #     visualize=True
    # )
    # see whether the plan converges
    # _, f3 = planning_cases.plan_convergence(wmr=wmr,
    #                                         plan_iterations=10,
    #                                         plan_length=10,
    #                                         adaptation_rates=[
    #                                             0.1, 0.5, 1, 2],
    #                                         visualize=True
    #                                         )
    # check whether the agent can hold up the pendulum
    angles = [-20, -15, -10, 0, 10, 15, 20]  # TODO TBD angles
    speeds = [0]  # TODO TBD speeds
    _, f4 = planning_cases.angle_test(wmr=wmr,
                                      angles=angles,
                                      speeds=speeds,
                                      steps=50,
                                      plan_length=10,
                                      visualize=True
                                      )
    plt.show()


test_world_model(wmr=wm)
planner = plan_optimizer.Planner(world_model=wm.get_model(),
                                 learning_rate=2,
                                 iterations=10,
                                 initial_plan=plan_optimizer.get_zero_plan(25),
                                 fill_function=plan_optimizer.get_zero_action,
                                 strategy="first"
                                 )


# planning_cases.environment_performance(planner=planner,
#                                        steps=50,
#                                        visualize=True
#                                        )

# planning_cases.eval_model_predictions(10, wm)

# planning_cases.plan_convergence(wm.get_model())
# planning_cases.model_quality_analysis(test_runs=50,
#                                      wmr=wm,
#                                      steps=50,
#                                      visualize=True
#                                      )

# df = planning_cases.prediction_accuracy(model=wm,
#                                        rollouts = 200,
#                                        steps = 25
#                                        )
# df.to_parquet("data/world_model_prediction.parquet", engine="pyarrow")

# planning_cases.plan_convergence(model)
# wm = world_model.WorldModelWrapper()
# wm.build_keras_model(neurons=_neurons, hidden_layers=_hidden_layer)
# env = gym.make("Pendulum-v0")
# loss, test_loss = wm.train_model(env=env,
#                                  steps=3,
#                                  epochs=2
#                                  )

# print(loss, "\n")
# print(test_loss)
