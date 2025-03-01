import logging
import json
import os
import pickle


# Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Log each step
def log_step(message):
    logging.info(message)


def save_model(model, model_name, model_dir):

    model_path = os.path.join(model_dir, f"{model_name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    return model_path


def load_model(model_path):
    """Load a model from the given path"""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def save_metadata(model_name, metrics, params, experiment_name, path, project, author):
    # def save_metadata(model_name, metrics, params, dir):
    """Save metadata for the model"""
    metadata = {
        "project_name": project,
        "author": author,
        "model_name": model_name,
        "metrics": metrics,
        "parameters": params,
        "experiment": experiment_name,
    }
    metadata_path = os.path.join(path, f"{model_name}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
