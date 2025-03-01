from src.constants import (
    ARTIFACT,
    EXPERIMENT,
    EXPERIMENT_NAME,
    MODEL_NAME,
    PROJECT_NAME,
    AUTHOR,
    ARCHIVED,
    PROD,
)
from src.helper import save_model, save_metadata  # ,upload_to_s3
import os
import shutil

# import boto3


class ProductionModel:

    def __init__(self):
        pass

    def check_if_old_model_exists(self):
        """Check if an old production model exists."""
        production_dir = os.path.join(ARTIFACT, EXPERIMENT, str(PROD))
        model_name = MODEL_NAME
        model_path = os.path.join(production_dir, f"{model_name}.pkl")
        print("Achived production model")
        return os.path.exists(model_path)

    def move_to_production(self, model, params, metrics):
        try:

            production_dir = os.path.join(ARTIFACT, EXPERIMENT, str(PROD))
            os.makedirs(production_dir, exist_ok=True)

            model_name = MODEL_NAME
            experiment_name = EXPERIMENT_NAME
            model_path = save_model(model, model_name, production_dir)
            save_metadata(
                model_name,
                metrics,
                params,
                experiment_name,
                production_dir,
                PROJECT_NAME,
                AUTHOR,
            )

            print("Model moved to production.")
            return production_dir

        except Exception as e:
            raise Exception(f"Error while moving the model to production: {e}")

    def archive_current_production_model(self):
        try:

            production_dir = os.path.join(ARTIFACT, EXPERIMENT, str(PROD))
            archive_dir = os.path.join(ARTIFACT, EXPERIMENT, str(ARCHIVED))
            os.makedirs(archive_dir, exist_ok=True)

            # Get the current model file path
            model_name = MODEL_NAME
            current_model_path = os.path.join(production_dir, f"{model_name}.pkl")

            if os.path.exists(current_model_path):
                archived_model_path = os.path.join(
                    archive_dir, f"{model_name}_archived.pkl"
                )
                shutil.move(current_model_path, archived_model_path)
                print(f"Old production model archived")
                return archived_model_path
            else:
                print("No production model found to archive.")
                return None

        except Exception as e:
            raise Exception(f"Error while archiving the current production model: {e}")
