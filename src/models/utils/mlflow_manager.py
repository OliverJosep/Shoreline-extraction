import mlflow
import mlflow.pytorch
import os
from datetime import datetime
from mlflow.entities import RunStatus
from dotenv import load_dotenv, find_dotenv


class MLflowManager:
    def __init__(self, experiment_name: str = "default_experiment"):
        """
        Initializes the MLflowManager object.

        Parameters:
        experiment_name (str): The name of the experiment to be created in MLflow.
        """
        load_dotenv(find_dotenv(), override=True)
        
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")

        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str = None):
        """
        Starts a new MLflow run.

        Parameters:
        run_name (str): The name of the run. Default is None.

        Returns:
        None
        """
        if run_name is None:
            run_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict):
        """
        Logs the parameters to the MLflow run.

        Parameters:
        params (dict): The dictionary containing the parameters to log.

        Returns:
        None
        """
        mlflow.log_params(params)

    def log_tag(self, key: str, value: str):
        """
        Logs the tag to the MLflow run.

        Parameters:
        key (str): The key of the tag.
        value (str): The value of the tag.

        Returns:
        None
        """
        mlflow.set_tag(key, value)

    def log_metrics(self, metrics: dict, step: int):
        """
        Logs the metrics to the MLflow run.

        Parameters:
        metrics (dict): The dictionary containing the metrics to log.
        step (int): The step number for the metrics (epoch number).

        Returns:
        None
        """
        mlflow.log_metrics(metrics, step=step)

    def log_artifacts(self, artifact_path: str):
        """
        Logs the artifacts to the MLflow run.

        Parameters:
        artifact_path (str): The path to the artifact to log.

        Returns:
        None
        """
        mlflow.log_artifacts(artifact_path)

    def log_model(self, model, model_name: str):
        """
        Logs the model to the MLflow run.

        Parameters:
        model: The model to log.
        model_name (str): The name of the model.

        Returns:
        None
        """
        mlflow.pytorch.log_model(model, model_name)

    def end_run(self, status: str = RunStatus.to_string(RunStatus.FINISHED)):
        """
        Ends the current MLflow run.

        Returns:
        None
        """
        mlflow.end_run(status=status)