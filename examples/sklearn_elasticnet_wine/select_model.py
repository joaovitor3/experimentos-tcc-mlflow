from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType

import mlflow

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    client = MlflowClient()
    experiment_name = 'exp'
    experiment = client.get_experiment_by_name(experiment_name)

    experiment_ids = [experiment.experiment_id]
    query = "metrics.r2 < 0.125 and metrics.rmse > 0.780"
    runs = client.search_runs(experiment_ids, query, ViewType.ALL)

    r2_low = None
    rmse_high = None
    best_run = None
    for run in runs:
        if (r2_low == None or run.data.metrics['r2'] < r2_low) and \
            (rmse_high == None or run.data.metrics['rmse'] > rmse_high):
            r2_low = run.data.metrics['r2']
            rmse_high = run.data.metrics['rmse']
            best_run = run

    best_run_id = best_run.info.run_id
    print('Run ID: ', best_run_id)
    print('RMSE high: ', rmse_high)
    print('R2 low: ', r2_low)

    best_model_uri = f's3://mlflow/{best_run_id}/artifacts/model'
    print('BEST Model URI: ', best_model_uri)
    print('BEST Model Artifact URI: ', best_run.info.artifact_uri, '/model')

    model = mlflow.pyfunc.load_model(best_model_uri)
    print('Loaded model: ', model)