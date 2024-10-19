import csv
import numpy as np
import random
from degann.equations.simple_equation import equation_solve, str_eq_to_params

# Differential equation dy/dx = -y represented as a string
equation = "(-y)"
# Number of points to generate for the dataset
generate_size = 1000
# Sample sizes for training and validation datasets
sizes_of_samples = [1000]

if __name__ == "__main__":
    params = {'x': "0, 5, 0.01", 'y': "0, 5, 0.01"}
    axes = str_eq_to_params(params)

    data = equation_solve(equation, axes)

    # Loop through specified sample sizes to create datasets
    for size in sizes_of_samples:
        train_idx = random.sample(range(data.shape[0]), size)
        val_idx = random.sample(range(data.shape[0]), size // 2)

        # Extract training data points
        train_data_x = data[train_idx, 0]  # x values
        train_data_y = data[train_idx, 1]  # y values

        # Extract validation data points
        val_data_x = data[val_idx, 0]
        val_data_y = data[val_idx, 1]

        # Save training dataset to a CSV file
        with open(f"data/ode_train_{size}.csv", "w", newline="") as file:
            csv_writer = csv.writer(file)
            data_to_write = zip(train_data_x, train_data_y)
            csv_writer.writerows(data_to_write)

        # Save validation dataset to a CSV file
        with open(f"data/ode_validate_{size}.csv", "w", newline="") as file:
            csv_writer = csv.writer(file)
            data_to_write = zip(val_data_x, val_data_y)
            csv_writer.writerows(data_to_write)