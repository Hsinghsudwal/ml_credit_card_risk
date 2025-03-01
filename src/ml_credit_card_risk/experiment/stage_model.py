import joblib
from src.constants import (
    ARTIFACT,
    EXPERIMENT,
    EXPERIMENT_NAME,
    MODEL_NAME,
    STAGE,
    PROJECT_NAME,
    AUTHOR,
)
import os
from src.helper import save_model, save_metadata


class StageModel:

    def __init__(self):
        pass

    def stage_model(self, model, params, metrics):
        try:
            """Stage the model and save it with its metadata"""
            # os.makedirs(staging_dir, exist_ok=True)

            experiment_path = os.path.join(ARTIFACT, EXPERIMENT, str(STAGE))
            os.makedirs(experiment_path, exist_ok=True)
            model_name = MODEL_NAME
            experiment_name = EXPERIMENT_NAME
            # Save the model and its  metadata
            model_path = save_model(model, model_name, experiment_path)
            save_metadata(
                model_name,
                metrics,
                params,
                experiment_name,
                experiment_path,
                PROJECT_NAME,
                AUTHOR,
            )

            print("Stage Staging Completed")
            return model_path

        except Exception as e:
            raise Exception(e)
