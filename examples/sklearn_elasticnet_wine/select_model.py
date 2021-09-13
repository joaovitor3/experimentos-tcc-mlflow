from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType

import mlflow

import logging
import time

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
# https://dzlab.github.io/ml/2020/07/12/ml-ci-mlflow/

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
    print('BEST Model Artifact URI: ', best_run.info.artifact_uri + '/model')
    best_model_uri = f'{best_run.info.artifact_uri}/model'

    model = mlflow.pyfunc.load_model(best_model_uri)
    print('Loaded model: ', model)

    prediction = model.predict(
        data=[
            [7.2,0.27,0.46,18.75,0.052,45,255,1,3.04,0.52,8.9]
        ]
    )
    print(f'Prediction: {prediction}')

    max_version = 0
    model_name = 'ElasticnetWineModel'
    for mv in client.search_model_versions(f"name='{model_name}'"):
        current_version = int(dict(mv)['version'])
        if current_version > max_version:
            max_version = current_version
        if dict(mv)['current_stage'] == 'Production':
            version = dict(mv)['version']
            client.transition_model_version_stage(model_name, version, stage='Archived')

    #Create a new version for this model with best metric (accuracy)
    client.create_model_version(model_name, best_model_uri, best_run_id)
    # Check the status of the created model version (it has to be READY)
    status = None
    while status != 'READY':
        for mv in client.search_model_versions(f"run_id='{best_run_id}'"):
            status = mv.status if int(mv.version)==max_version + 1 else status
        time.sleep(5)

    # Promote the model version to production stage
    client.transition_model_version_stage(model_name, max_version + 1, stage='Production')
