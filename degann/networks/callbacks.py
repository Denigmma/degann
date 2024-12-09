import gc
import time

import keras.backend as k
from keras.callbacks import Callback
from keras.callbacks import History

import matplotlib.pyplot as plt


class MemoryCleaner(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        k.clear_session()


class MeasureTrainTime(Callback):
    """
    Callback for measuring time.
    Supports measuring training time,
    measuring the time of each epoch during training,
    and measuring the running time of the predict method
    """

    def __init__(self):
        super(MeasureTrainTime, self).__init__()
        self.start_train_time = 0
        self.end_train_time = 0

        self.start_evaluate_time = 0
        self.end_evaluate_time = 0

        self.start_predict_time = 0
        self.end_predict_time = 0

        self.start_epoch_time = 0
        self.end_epoch_time = 0

    def on_test_begin(self, logs=None):
        self.model.trained_time["predict_time"] = 0
        self.start_evaluate_time = time.perf_counter()

    def on_test_end(self, logs=None):
        self.end_evaluate_time = time.perf_counter()
        self.model.trained_time["predict_time"] = (
            self.end_evaluate_time - self.start_evaluate_time
        )

    def on_predict_begin(self, logs=None):
        self.model.trained_time["predict_time"] = 0
        self.start_predict_time = time.perf_counter()

    def on_predict_end(self, logs=None):
        self.end_predict_time = time.perf_counter()
        self.model.trained_time["predict_time"] = (
            self.end_predict_time - self.start_predict_time
        )

    def on_train_begin(self, logs=None):
        self.model.trained_time["train_time"] = 0.0
        self.model.trained_time["epoch_time"] = []
        self.start_train_time = time.perf_counter()

    def on_epoch_begin(self, epoch, logs=None):
        self.start_epoch_time = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.end_epoch_time = time.perf_counter()
        self.model.trained_time["epoch_time"].append(
            self.end_epoch_time - self.start_epoch_time
        )

    def on_train_end(self, logs=None):
        self.end_train_time = time.perf_counter()
        self.model.trained_time["train_time"] = (
            self.end_train_time - self.start_train_time
        )


class LightHistory(History):
    """
    Class based on Keras.History,
    but which only stores information about the last training epoch,
    not the entire process
    """

    def __init__(self):
        super(History, self).__init__()
        self.history = {}

    def on_train_begin(self, logs=None):
        self.epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch = epoch
        for k, v in logs.items():
            self.history[k] = v

        # Set the history attribute on the model after the epoch ends. This will
        # make sure that the state which is set is the latest one.
        self.model.history = self


class EarlyStopping(Callback):
    """
       Callback for early stopping during training.

       Stops training when validation loss does not improve for a specified number of consecutive epochs (=patience).

       Attributes:
           best_val_loss (float): Tracks the best validation loss observed during training.
           patience (int): Number of epochs to wait for an improvement in validation loss before stopping.
           wait (int): Counter for tracking consecutive epochs without improvement.
       """

    def __init__(self,patience):
        super(EarlyStopping, self).__init__()
        self.best_val_loss = float("inf")
        self.patience = patience  # param for Early Stopping
        self.wait = 0

    def check_stop(self, val_loss, logs=None):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print("//Early stopping triggered after",self.patience,"unsuccessful epoch//")
                self.model.stop_training = True

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        if val_loss is not None:
            self.check_stop(val_loss)


class SaveBestModel(Callback):
    """
        Callback for saving the best model during training.

        Tracks the validation loss at the end of each epoch and saves the model
        if the validation loss improves.

        Attributes:
            best_val_loss (float): Tracks the best validation loss observed during training.
            save_best_model_path (str): Path where the best model is saved.
            model_save: An instance of the model that supports an export_to_file method for saving.
        """

    def __init__(self,model_save):
        super(SaveBestModel, self).__init__()
        self.best_val_loss = float("inf")
        self.save_best_model_path = "best_Model"
        self.model_save = model_save

    def check_save(self, val_loss, logs=None):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.model.save(self.save_best_model_path+".h5")
            self.model_save.export_to_file(self.save_best_model_path)

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        if val_loss is not None:
            self.check_save(val_loss)


class VisualizationTS(Callback):
    """
    Callback for visualizing training progress and model predictions.

    This callback generates two types of plots:
    1. Training and validation loss during training.
    2. Comparison of model predictions, training data, and the true function after training.

    Attributes:
        train_data_x (np.ndarray): Training input data for prediction visualization.
        train_data_y (np.ndarray): Training output data for prediction visualization.
        val_data_x (np.ndarray): Validation input data for loss tracking.
        val_data_y (np.ndarray): Validation output data for loss tracking.
        func_name (str): Name of the true function for comparison.
        funcs (list): List of available functions with their names.
        losses (list): List of training losses for each epoch.
        val_losses (list): List of validation losses for each epoch.
    """

    def __init__(self, train_data_x, train_data_y, val_data_x, val_data_y, func_name, funcs):
        """
        Initializes the VisualizationCallback.

        Args:
            train_data_x (np.ndarray): Training input data.
            train_data_y (np.ndarray): Training output data.
            val_data_x (np.ndarray): Validation input data.
            val_data_y (np.ndarray): Validation output data.
            func_name (str): Name of the true function.
            funcs (list): List of available functions with their names.
        """
        super(VisualizationTS, self).__init__()
        self.train_data_x = train_data_x
        self.train_data_y = train_data_y
        self.val_data_x = val_data_x
        self.val_data_y = val_data_y
        self.func_name = func_name
        self.funcs = funcs
        self.losses = []
        self.val_losses = []
        self.saved_history = None

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        if loss is not None:
            self.losses.append(loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)

    def on_train_end(self, logs=None):
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses, label='Training Loss', color='blue')
        plt.plot(self.val_losses, label='Validation Loss', color='red')
        plt.title('Loss Function During Training')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()

        self.saved_history = self.model.history
        predictions = self.model.predict(self.train_data_x, verbose=0)

        true_func = None
        for func, name in self.funcs:
            if name == self.func_name:
                true_func = func
                break

        x_values = self.train_data_x[:, -1, 0]
        true_solution = true_func(x_values)

        save_dir = "../../experiments/approximation_graphs/gru/"
        plt.savefig(save_dir+f"{self.func_name}_{len(self.losses)}eph_gru_loss_plot.png")

        plt.figure(figsize=(10, 6))
        plt.scatter(x_values, self.train_data_y, label="Training Data", color="blue", alpha=0.5)
        plt.plot(x_values, predictions, label="Model Prediction", color="red")
        plt.plot(x_values, true_solution, label="True Function: " + self.func_name, color="green")
        plt.title("GRU Model: Training Data, Predictions, and True Function")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid()

        plt.savefig(save_dir+f"{self.func_name}_{len(self.losses)}eph_gru_apr_plot.png")

        # plt.show()
        plt.close()

    def get_saved_history(self):
        return self.saved_history

class VisualizationDense(Callback):
    """
    Callback for visualizing training progress and model predictions.

    This callback generates two types of plots:
    1. Training and validation loss during training.
    2. Comparison of model predictions, training data, and the true function after training.

    Attributes:
        train_data_x (np.ndarray): Training input data for prediction visualization.
        train_data_y (np.ndarray): Training output data for prediction visualization.
        val_data_x (np.ndarray): Validation input data for loss tracking.
        val_data_y (np.ndarray): Validation output data for loss tracking.
        func_name (str): Name of the true function for comparison.
        funcs (list): List of available functions with their names.
        losses (list): List of training losses for each epoch.
        val_losses (list): List of validation losses for each epoch.
    """

    def __init__(self, train_data_x, train_data_y, val_data_x, val_data_y, func_name, funcs):
        """
        Initializes the VisualizationDense callback.

        Args:
            train_data_x (np.ndarray): Training input data.
            train_data_y (np.ndarray): Training output data.
            val_data_x (np.ndarray): Validation input data.
            val_data_y (np.ndarray): Validation output data.
            func_name (str): Name of the true function.
            funcs (list): List of available functions with their names.
        """
        super(VisualizationDense, self).__init__()
        self.train_data_x = train_data_x
        self.train_data_y = train_data_y
        self.val_data_x = val_data_x
        self.val_data_y = val_data_y
        self.func_name = func_name
        self.funcs = funcs
        self.losses = []
        self.val_losses = []
        self.saved_history = None

    def on_epoch_end(self, epoch, logs=None):
        """
        Logs training and validation losses at the end of each epoch.
        """
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        if loss is not None:
            self.losses.append(loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)

    def on_train_end(self, logs=None):
        """
        Generates plots for training and validation loss and predictions.
        """
        # Plot training and validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses, label='Training Loss', color='blue')
        plt.plot(self.val_losses, label='Validation Loss', color='red')
        plt.title('Loss Function During Training')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()

        self.saved_history = self.model.history
        predictions = self.model.predict(self.train_data_x, verbose=0)

        # Find the true function
        true_func = None
        for func, name in self.funcs:
            if name == self.func_name:
                true_func = func
                break

        save_dir = "../experiments/approximation_graphs/dense/"
        plt.savefig(save_dir+f"{self.func_name}_{len(self.losses)}eph_dense_loss_plot.png")

        if true_func is not None:
            x_values = self.train_data_x.flatten()
            true_solution = true_func(x_values)

            # Plot predictions vs true function
            plt.figure(figsize=(10, 6))
            plt.scatter(x_values, self.train_data_y, label="Training Data", color="blue", alpha=0.5)
            plt.plot(x_values, predictions, label="Model Prediction", color="red")
            plt.plot(x_values, true_solution, label="True Function: " + self.func_name, color="green")
            plt.title("DenseNet Model: Training Data, Predictions, and True Function")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend()
            plt.grid()

            plt.savefig(save_dir + f"{self.func_name}_{len(self.losses)}eph_dense_apr_plot.png")
        else:
            print("True function not found in the provided function list.")
        # plt.show()
        plt.close()

    def get_saved_history(self):
        return self.saved_history

