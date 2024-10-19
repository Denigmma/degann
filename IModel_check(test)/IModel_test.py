import degann
import os
from degann.networks.imodel import IModel
import numpy as np
import pandas as pd
import time
import tensorflow as tf

class TimeHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        super(TimeHistory, self).__init__()
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time.time() - self.start_time
        self.times.append(elapsed_time)
        print(f"Epoch {epoch + 1} finished in {elapsed_time:.2f} seconds")

    def get_average_time(self):
        return np.mean(self.times) if self.times else 0

base_dir = os.path.dirname(__file__)
csv_path = os.path.join(base_dir, "../experiments/data/ode_train_400.csv")
train_data = pd.read_csv(csv_path, names=["x", "y"])

train_data_x = train_data["x"].values.reshape(-1, 1)
train_data_y = train_data["y"].values.reshape(-1, 1)

shape =[32, 16, 8]
activations = ["swish", "relu"] + ["linear"]
nn_testing = IModel(
    input_size=1,
    block_size=shape,
    output_size=1,
    activation_func=activations,
)

print("Activation functions per layer for nn_testing")
acts = nn_testing.get_activations
for i, act_name in enumerate(activations):
    print(i, act_name)

print(nn_testing)

nn_testing.compile(
    optimizer="Adam",
    loss_func="MaxAbsoluteDeviation",  # max(abs(y_true - y_prediction))
    metrics=[]
)

loss_before_train = nn_testing.evaluate(train_data_x, train_data_y, verbose=0)

time_callback = TimeHistory()

nn_testing.train(train_data_x, train_data_y, epochs=50, verbose=0, callbacks=[time_callback])

loss_after_train = nn_testing.evaluate(train_data_x, train_data_y, verbose=0)

average_time = time_callback.get_average_time()
print(f"Average time per epoch: {average_time:.2f} seconds")

print(f"Loss before training = {loss_before_train}")
print(f"Loss after training = {loss_after_train}")

nn_testing.export_to_file("model")
