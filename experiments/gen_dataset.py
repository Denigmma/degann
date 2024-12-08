import csv
import random
from random import randint

import numpy as np

import experiments.functions as functions

__all__ = ["funcs", "sizes_of_samples", "generate_size"]

funcs = [
    # (functions.lin, "lin"),
    # (functions.log, "log"),
    # (functions.sin, "sin"),
    # (functions.exp, "exp"),
    # (functions.gauss, "gauss"),
    # (functions.hyperbol, "hyperbol"),
    # (functions.const, "const"),
    # (functions.sig, "sig"),
    # (functions.multidim, "multidim")
    (functions.hardsin,"hardsin"),
    # (functions.expexp, "expexp"),
]
# sizes_of_samples = [50, 150, 400]
# sizes_of_samples = [400]
sizes_of_samples = [1000]
generate_size = 1000

if __name__ == "__main__":
    for func, func_name in funcs:

        # nn_data_x = np.array(
        #     [
        #         [
        #             random.uniform(1 / generate_size, 1),
        #             random.uniform(1 / generate_size, 1),
        #             random.uniform(1 / generate_size, 1),
        #         ]
        #         for i in range(1, generate_size + 2)
        #     ]
        # )  # X data

        # nn_data_x = np.array(
        #     [[i / generate_size] for i in range(1, generate_size + 2)]
        # )  # X data

        start = 0
        end = 1
        nn_data_x = np.array([[start + i * (end - start) / generate_size] for i in range(generate_size + 1)])


        nn_data_y = np.array([[func(*x)] for x in nn_data_x])

        # #NOISE
        # mu = np.mean(nn_data_y)
        sigma = np.std(nn_data_y)
        noise = np.random.normal(0,sigma*0.1, nn_data_y.shape)
        nn_data_y_noisy = nn_data_y + noise

        for size in sizes_of_samples:
            train_idx = [randint(0, generate_size) for _ in range(size)]
            train_idx.sort()
            val_idx = [randint(0, generate_size) for _ in range(size // 2)]
            val_idx.sort()
            val_data_x = nn_data_x[val_idx, :]  # validation X data
            val_data_y = nn_data_y[val_idx, :]  # validation Y data
            train_data_x = nn_data_x[train_idx, :]  # X data
            train_data_y = nn_data_y[train_idx, :]  # Y data
            train_data_y = nn_data_y_noisy[train_idx, :]  # Y data

            with open(f"data/{func_name}_{size}_train.csv", "w", newline="") as file:
                csv_writer = csv.writer(file)
                data = list(zip(*train_data_x.T, *train_data_y.T))
                csv_writer.writerows(data)

            with open(f"data/{func_name}_{size}_validate.csv", "w", newline="") as file:
                csv_writer = csv.writer(file)
                data = list(zip(*val_data_x.T, *val_data_y.T))
                csv_writer.writerows(data)
