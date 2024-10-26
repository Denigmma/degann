import os
from degann.networks.imodel import IModel
import tensorflow as tf
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

base_dir = os.path.dirname(__file__)

# name_dataset="gauss_400_train.csv"
# name_dataset="ode_train_400.csv"
# name_dataset="exp_400_train.csv"
# name_dataset="lin_400_train.csv"
# name_dataset="ode_train_1000.csv"
name_dataset="hardsin_1000_train.csv"

# name_validate="ode_validate_1000.csv"
# name_validate="gauss_400_validate.csv"
name_validate="hardsin_1000_validate.csv"
# name_validate="exp_400_validate.csv"


csv_path = os.path.join(base_dir, "../../experiments/data/"+name_dataset)
val_csv_path = os.path.join(base_dir, "../../experiments/data/"+name_validate)


val_data = pd.read_csv(val_csv_path, names=["x", "y"])

val_data_x = val_data["x"].values.reshape(-1, 1)
val_data_y = val_data["y"].values.reshape(-1, 1)

train_data = pd.read_csv(csv_path, names=["x", "y"])

train_data_x = train_data["x"].values.reshape(-1, 1)
train_data_y = train_data["y"].values.reshape(-1, 1)


# function for creating time sequences
def create_sequences(data_x, data_y, time_steps):
    x, y = [], []
    for i in range(len(data_x) - time_steps):
        x.append(data_x[i:i + time_steps])
        y.append(data_y[i + time_steps])
    return np.array(x), np.array(y)

# count time steps
time_steps = 10

train_data_x, train_data_y = create_sequences(train_data_x, train_data_y, time_steps)

val_data_x, val_data_y = create_sequences(val_data_x, val_data_y, time_steps)

# tracking time for epochs
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

    def get_total_time(self):
        return np.sum(self.times) if self.times else 0


# tracking loss functions
class LossHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        super(LossHistory, self).__init__()
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))




gru_model = IModel(
    input_size=1,
    output_size=1,
    net_type="GRUNet",
    count_size=20,  # count layers
    gru_units=200,  # count units in layers
)


gru_model.compile(
    # optimizer="Adam",
    optimizer="SGD",
    loss_func="MeanSquaredError",
    metrics=[]
)

loss_before_train = gru_model.evaluate(train_data_x, train_data_y, verbose=0)
val_loss_before_train=gru_model.evaluate(val_data_x, val_data_y, verbose=0)

time_callback = TimeHistory()
loss_history = LossHistory()

gru_model.train(train_data_x, train_data_y, validation_data=(val_data_x, val_data_y), epochs=200, verbose=0, callbacks=[time_callback,loss_history])

loss_after_train = gru_model.evaluate(train_data_x, train_data_y, verbose=0)
val_loss_after_train = gru_model.evaluate(val_data_x, val_data_y, verbose=0)


gru_model.export_to_file("gru_madel")



average_time = time_callback.get_average_time()
total_time = time_callback.get_total_time()
print(f"Total training time: {total_time} seconds")
print(f"Average time per epoch: {average_time} seconds")

print(f"Loss before training = {loss_before_train}")
print(f"Loss after training = {loss_after_train}")
print(f"different between Losses ={loss_before_train-loss_after_train}")


print(f"validation loss before training = {val_loss_before_train}")
print(f"validation loss after training = {val_loss_after_train}")
print(f"Difference in validation loss = {val_loss_before_train - val_loss_after_train}")




### plot charts

# Chart loss function
plt.figure(figsize=(10, 6))
plt.plot(loss_history.losses, label='Training Loss', color='blue')
plt.plot(loss_history.val_losses, label='Validation Loss', color='red')
plt.title('Loss Function During Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()


predictions = gru_model.predict(train_data_x)

func_name = name_dataset.split('_')[0]
from experiments.gen_dataset import funcs
true_func = None
for func, name in funcs:
    if name == func_name:
        true_func = func
        break

x_values = train_data_x[:, -1, 0]
true_solution = true_func(x_values)

plt.figure(figsize=(10, 6))

plt.scatter(x_values, train_data_y, label="Training Data", color="blue", alpha=0.5)

plt.plot(x_values, predictions, label="Model Prediction", color="red")

plt.plot(x_values, true_solution, label="True Function: "+func_name, color="green")

plt.title("GRU Model: Training Data, Predictions, and True Function")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()
