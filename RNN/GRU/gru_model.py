import os

import numpy as np
import pandas as pd

# import degan's tools
from degann.networks.imodel import IModel
from experiments.gen_dataset import funcs

# import callbacks Classes
from degann.networks.callbacks import MeasureTrainTime, LossTracking, EarlyStopping, SaveBestModel, VisualizationTS

base_dir = os.path.dirname(__file__)
name_dataset = "hardsin_1000"
csv_path = os.path.join(base_dir, "../../experiments/data/" + name_dataset + "_train.csv")
val_csv_path = os.path.join(base_dir, "../../experiments/data/" + name_dataset + "_validate.csv")

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


count_size = 5  # count layers
gru_units = 30  # count units in layers
shape = [gru_units] * count_size
epochs = 20

GRU_IModel = IModel(
    input_size=1,
    output_size=1,
    net_type="GRUNet",
    block_size=shape,
)

GRU_IModel.compile(
    optimizer="Adam",
    loss_func="RootMeanSquaredError",
    metrics=[]
)

loss_before_train = GRU_IModel.evaluate(train_data_x, train_data_y, verbose=0)
val_loss_before_train = GRU_IModel.evaluate(val_data_x, val_data_y, verbose=0)

MeasureTrainTime = MeasureTrainTime()
Loss_tracking = LossTracking()
Early_stopping = EarlyStopping(patience=100) #customize "patience" to your needs
Save_best_model = SaveBestModel(GRU_IModel)
Visualization = VisualizationTS(train_data_x, train_data_y, val_data_x, val_data_y, name_dataset.split('_')[0], funcs)


GRU_IModel.train(train_data_x, train_data_y, validation_data=(val_data_x, val_data_y), epochs=epochs, verbose=0,
                 callbacks=[MeasureTrainTime, Loss_tracking, Early_stopping, Save_best_model, Visualization])
GRU_IModel.export_to_file("GRU_IModel_full")


loss_after_train = GRU_IModel.evaluate(train_data_x, train_data_y, verbose=0)
val_loss_after_train = GRU_IModel.evaluate(val_data_x, val_data_y, verbose=0)




#### Example of use callbacks

epoch_duration = MeasureTrainTime.model.trained_time["epoch_time"]
loss = Loss_tracking.losses
val_loss = Loss_tracking.val_losses

for i in range(len(loss)):
    print(f"Epoch {i + 1}: finished in: {epoch_duration[i]:.2f} seconds | "
          f"Training Loss: {loss[i]:.4f}, Validation Loss: {val_loss[i]:.4f}")

print(f"Total training time: {sum(epoch_duration):.4f} seconds")
print(f"Average time per epoch: {np.mean(epoch_duration):.6f} seconds")

print(f"Loss before training = {loss_before_train:.6f}")
print(f"Loss after training = {loss_after_train:.6f}")
print(f"different between Losses ={loss_before_train - loss_after_train:.6f}")

print(f"validation loss before training = {val_loss_before_train:.6f}")
print(f"validation loss after training = {val_loss_after_train:.6f}")
print(f"Difference in validation loss = {(val_loss_before_train - val_loss_after_train):.6f}")