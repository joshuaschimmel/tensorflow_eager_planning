import tensorflow as tf
import forward_model_tf as fm
import helper_functions as hf

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

hf.plot_model_prediction(200, model, model_name)

plot_title = f"RMSE for {_test_runs} Random Initializations"
print(hf.model_quality_analysis(test_runs=_test_runs,
                                model=model,
                                steps=_steps,
                                visualize=True,
                                plot_title=plot_title))