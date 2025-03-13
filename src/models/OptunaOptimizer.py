import optuna
from src.models.data_management.cnn_formes import CNNFormes
from models.base_model import BaseModel

class OptunaOptimizer:
    def __init__(self, base_model: BaseModel) -> None:
        """
        Initializes the Optuna optimizer for hyperparameter search.

        Parameters:
        base_model (BaseModel): An instance of the BaseModel class to train the model.
        """
        self.base_model = base_model
        self.artifact_path = None
        self.run_name = None
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        The objective function for Optuna that trains the model with the hyperparameters generated.

        Parameters:
        trial (optuna.Trial): An instance of the Trial class to manage the search.

        Returns:
        float: The validation loss obtained after training.
        """

        # Select hyperparameters to try for this trial
        epochs = trial.suggest_int("epochs", 5, 50)  # Number of epochs (adjust the range as needed)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)  # Learning rate
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])  # Batch size
        loss_function_name = trial.suggest_categorical("loss_function", ["CrossEntropy", "DiceLoss"])  # Loss function
        optimizer_name = trial.suggest_categorical("optimizer", ["adam"])  # Optimizer

        # Train the model with the selected hyperparameters
        self.base_model.load_data(data_source=self.base_model.data, formes_class=CNNFormes, batch_size=batch_size)
        
        # Train the model with these hyperparameters
        self.base_model.train(
            epochs=epochs,
            loss_function_name=loss_function_name,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            artifact_path=self.artifact_path,
            run_name=self.run_name
        )

        # The objective is to minimize the validation loss, so return the final loss
        validation_loss = self.base_model.metrics_validation.get_last_loss()
        
        return validation_loss

    def optimize(self, n_trials: int = 10, artifact_path = None, run_name = None) -> None:
        """
        Optimizes the hyperparameters using Optuna.

        Parameters:
        n_trials (int): The number of trials to perform for hyperparameter search. Default: 10.
        """
        self.artifact_path = artifact_path
        self.run_name = run_name

        study = optuna.create_study(direction="minimize")  # Minimize the loss
        study.optimize(self.objective, n_trials=n_trials)

        # Print the best hyperparameters found
        print("Best hyperparameters: ", study.best_params)
        print("Best validation loss: ", study.best_value)