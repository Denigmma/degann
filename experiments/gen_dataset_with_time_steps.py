import csv
import random
from random import randint
import numpy as np
import experiments.functions as functions

__all__ = ["funcs", "sizes_of_samples", "generate_size"]

funcs = [
    (functions.hardsin, "hardsin"),
]

# Function to create time sequences
def create_sequences(data_x, data_y, time_steps):
    x, y = [], []
    for i in range(len(data_x) - time_steps):
        x.append(data_x[i:i + time_steps])  # Create sequence of time steps
        y.append(data_y[i + time_steps])  # Target value for the sequence
    return np.array(x), np.array(y)

sizes_of_samples = [1000]
generate_size = 1000

if __name__ == "__main__":
    for func, func_name in funcs:
        # Generate X and Y data
        nn_data_x = np.array([[i / generate_size] for i in range(1, generate_size + 2)])  # X data
        nn_data_y = np.array([[func(*x)] for x in nn_data_x])  # Y data based on the function

        # Adding noise
        sigma = np.std(nn_data_y)
        noise = np.random.normal(0, sigma * 0.1, nn_data_y.shape)
        nn_data_y_noisy = nn_data_y + noise

        # Now, after generating all data, split it into train/validation
        for size in sizes_of_samples:
            # Generate train and validation indices
            train_idx = [randint(0, generate_size) for _ in range(size)]
            train_idx.sort()
            val_idx = [randint(0, generate_size) for _ in range(size // 2)]
            val_idx.sort()

            # Select training and validation data based on indices
            train_data_x = nn_data_x[train_idx, :]
            train_data_y = nn_data_y[train_idx, :]
            train_data_y_noisy = nn_data_y_noisy[train_idx, :]

            val_data_x = nn_data_x[val_idx, :]
            val_data_y = nn_data_y[val_idx, :]

            # Create sequences (after selecting data) for time steps
            time_steps = 10  # Number of time steps per sequence
            train_data_x, train_data_y = create_sequences(train_data_x, train_data_y, time_steps)
            train_data_y_noisy, _ = create_sequences(train_data_x, train_data_y_noisy, time_steps)

            val_data_x, val_data_y = create_sequences(val_data_x, val_data_y, time_steps)

            # Save the train and validation data in CSV format
            with open(f"data/{func_name}_{size}_train_times_steps_{str(time_steps)}.csv", "w", newline="") as file:
                csv_writer = csv.writer(file)
                data = list(zip(*train_data_x.reshape(-1, time_steps).T, *train_data_y.reshape(-1, time_steps).T))
                csv_writer.writerows(data)

            with open(f"data/{func_name}_{size}_validate_times_steps_{str(time_steps)}.csv", "w", newline="") as file:
                csv_writer = csv.writer(file)
                data = list(zip(*val_data_x.reshape(-1, time_steps).T, *val_data_y.reshape(-1, time_steps).T))
                csv_writer.writerows(data)
