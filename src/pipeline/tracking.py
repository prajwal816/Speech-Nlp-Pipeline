import mlflow
import os

class ExperimentTracker:
    def __init__(self, experiment_name="speech_nlp_baseline", tracking_uri="./mlruns"):
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        
    def start_run(self, run_name=None):
        return mlflow.start_run(run_name=run_name)
        
    def log_params(self, params: dict):
        mlflow.log_params(params)
        
    def log_metrics(self, metrics: dict):
        mlflow.log_metrics(metrics)
        
    def log_artifact(self, local_path: str, artifact_path: str = None):
        mlflow.log_artifact(local_path, artifact_path)
