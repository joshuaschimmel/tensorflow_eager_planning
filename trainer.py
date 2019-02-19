import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import forward_model_tf as fm
import pendulum as pend
import helper_functions as _hf


tf.enable_eager_execution()

print(f"TensorFlow version: {tf.__version__}")
print(f"Eager execution: {tf.executing_eagerly()}")

# hyperparameters
_neurons = 40
_hidden_layer = 1
_epochs = 1
_loss_function = fm.rmse_loss
_drop_rate = 0.5
_load_model = True
_steps = 100
_test_runs = 50


drop_text = "nodrop"
if _drop_rate > 0.0:
    drop_text = f"{_drop_rate}drop"
neuron_text = (str(_neurons) + '-') \
              * _hidden_layer \
              + str(_neurons) + "_" \
              + str(_loss_function.__name__)

# load saved model

model_name = f"model_{neuron_text}_{_epochs}e_{drop_text}"
model_path = f"models/{model_name}.h5"

# TODO check if model exists
if _load_model:
    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )
else:
    model = fm.build_forward_model(epochs=_epochs,
                                   neurons=_neurons,
                                   hidden_layers=_hidden_layer,
                                   loss=_loss_function,
                                   dropout_rate=_drop_rate,
                                   validation_split=0.05,
                                   test_split=0.05,
                                   plot_performance=True)

print(model.summary())
print(f"Using {model_name}")

# save the model if it is new
if not _load_model:
    fm.save_model(model, model_path)
    print("saved")

def model_quality_analysis(test_runs: int,
                           model: tf.keras.Model,
                           steps: int,
                           visualize: bool = True,
                           plot_title: str = ""
                           ):
    """Calculates the mean rmse over multiple random instances.

    Calculates the mean of the rmse values for test_runs number of
    runs with steps number of steps. The calculated mean is the mean
    over all test runs. If visualize is true, the rmse values and
    the mean will be plotted against each other in a single plot.

    :param test_runs: # of random tries
    :param model: the model to be tested
    :param steps: # of steps in a single try
    :param visualize: whether the result should be visualized
    :param plot_title: the title of the plot
    :return:
    """
    all_rmse_values = []
    plot_list = []
    for i in range(test_runs):
        # create a new random plan
        plan = _hf.get_random_plan(steps)

        # let the simulation run on the plan to create
        # the expected states as well as the starting state
        # s_0
        sim_states = pend.run_simulation_plan(plan=plan)
        # get starting state
        s_0 = sim_states[0]
        # let the model predict the states
        pred_states = fm.predict_states(model=model, state_0=s_0, plan=plan)

        current_rmse_values = []
        # calculate the rmse
        for pred_state, sim_state in zip(pred_states, sim_states):
            current_rmse_values.append(
                fm.singleton_rmse_loss(prediction=pred_state,
                                       target=sim_state
                                       )
            )
        # append the values to the list of values as an numpy array
        all_rmse_values.append(np.array(current_rmse_values))
        plot_list.append({
            "label": i+1,
            "values": np.array(current_rmse_values),
            "format": "-"
        })

    mean_dict = {
        "label": "Mean",
        "values": np.mean(all_rmse_values, axis=0),
        "format": ""
    }

    if visualize:
        # initialize figure
        plt.figure()
        # plot title
        plt.suptitle(plot_title)
        # iterate through list and plot all entries
        for plot_data in plot_list:
            plt.plot(plot_data["values"],
                     plot_data["format"],
                     label=plot_data["label"],
                     linewidth=1,
                     alpha=0.5
                     )
        # plot the mean
        mean_plot = plt.plot(mean_dict["values"],
                             mean_dict["format"],
                             label=mean_dict["label"],
                             linewidth=2
                             )


        plt.ylim(0, 10)
        plt.grid(b=True, alpha=0.25, linestyle="--")
        plt.tick_params(axis="both", which="major", direction="out")
        plt.legend(mean_plot, ["Mean"], loc=1)
        plt.show()

plot_title = f"RMSE for {_test_runs} Random Initializations"
model_quality_analysis(test_runs=_test_runs,
                       model=model,
                       steps=_steps,
                       visualize=True,
                       plot_title=plot_title)