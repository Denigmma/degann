import tensorflow as tf
from tensorflow import keras
from degann.networks import layer_creator, losses, metrics, cpp_utils
from degann.networks import optimizers
from typing import List
import random
import numpy as np

# fix seed
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

class TensorflowGRUNet(tf.keras.Model):
    def __init__(
            self,
            input_size: int = 2,
            block_size: List[int] = None,
            output_size: int = 10,
            activation_func: str = "tanh",  # activation function in the GRU of a new layer
            recurrent_activation: str = "sigmoid",  # recurrent state activation function
            # dropout_rate: float = 0.2,  # level dropout
            weight=keras.initializers.RandomUniform(minval=-1, maxval=1,seed=seed),
            biases=keras.initializers.RandomUniform(minval=-1, maxval=1,seed=seed),
            is_debug: bool = False,
            return_sequences: bool = False,
            **kwargs,
    ):
        # checking |block_size| for data accuracy
        if block_size is None or not all(size == block_size[0] for size in block_size):
            raise ValueError("\n//ERROR//\n"
                f"All layers in Recurrent Neural Networks must have the same number of neurons in each layer "
                f"\nReceived block_size: {block_size}"
            )

        super(TensorflowGRUNet, self).__init__(**kwargs)

        self.input_size = input_size
        self.count_size = block_size[0]
        self.gru_units = len(block_size)
        self.output_size = output_size
        self.activation_func = activation_func
        self.recurrent_activation = recurrent_activation
        # self.dropout_rate = dropout_rate

        self.gru_layers = []  # list GRU layers

        for i in range(len(block_size)):
            self.gru_layers.append(
                keras.layers.GRU(
                    units=block_size[0],
                    activation=activation_func,
                    recurrent_activation=recurrent_activation,
                    return_sequences=True if i < len(block_size) - 1 else return_sequences,
                    kernel_initializer=weight,
                    bias_initializer=biases,
                    name=f"GRULayer{i}"
                )
            )

            # # Dropout
            # self.gru_layers.append(
            #     keras.layers.Dropout(rate=self.dropout_rate, name=f"DropoutLayer{i}")
            # )

        # The output is a fully connected layer for predicting the result
        self.out_layer = keras.layers.Dense(
            output_size,
            activation="linear",  # Usually linear activation for the last regression layer
            kernel_initializer=weight,
            bias_initializer=biases,
            name="OutputLayer"
        )

        self.trained_time = {"train_time": 0.0, "epoch_time": [], "predict_time": 0}

    def call(self, inputs, **kwargs):
        """
        Performing a direct pass through the network
        """
        x = inputs
        for gru_layer in self.gru_layers:
            x = gru_layer(x, **kwargs)
        return self.out_layer(x, **kwargs)

    def custom_compile(
            self,
            rate=1e-2,
            # optimizer="Adam",
            optimizer="SGD",
            loss_func="MeanSquaredError",
            metric_funcs=None,
            run_eagerly=False,
    ):
        """
        Setting up a training model.
        """
        opt = optimizers.get_optimizer(optimizer)(learning_rate=rate)
        loss = losses.get_loss(loss_func)
        m = [metrics.get_metric(metric) for metric in metric_funcs] if metric_funcs else []
        self.compile(
            optimizer=opt,
            loss=loss,
            metrics=m,
            run_eagerly=run_eagerly,
        )

    def set_name(self, new_name):
        self._name = new_name

    def __str__(self):
        res = f"GRUModel {self.name}\n"
        for gru_layer in self.gru_layers:
            res += str(gru_layer)
        res += str(self.out_layer)
        return res

    def to_dict(self, **kwargs):
        """
        Exporting architecture as a dictionary.
        """
        res = {
            "net_type": "GRU",
            "name": self._name,
            "input_size": self.input_size,
            "gru_units": self.gru_layers[0].units,  # The number of units is the same for all layers
            "count_size": len(self.gru_layers),  # Number of layers
            "output_size": self.output_size,
            "out_layer": self.out_layer.get_config(),
        }
        return res

    @classmethod
    def from_layers(
            cls,
            input_size: int,
            count_size: int,
            gru_units: int,
            output_size: int,
            activation_func: str = "tanh",
            recurrent_activation: str = "sigmoid",
            weight=keras.initializers.RandomUniform(minval=-1, maxval=1),
            biases=keras.initializers.RandomUniform(minval=-1, maxval=1),
            return_sequences: bool = False,
            **kwargs,
    ):
        return cls(
            input_size=input_size,
            count_size=count_size,
            gru_units=gru_units,
            output_size=output_size,
            activation_func=activation_func,
            recurrent_activation=recurrent_activation,
            weight=weight,
            biases=biases,
            return_sequences=return_sequences,
            **kwargs,
        )
