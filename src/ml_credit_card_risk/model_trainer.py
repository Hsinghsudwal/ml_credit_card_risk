import os
import joblib
import pandas as pd
from xgboost import XGBClassifier
from src.constants import ARTIFACT, MODEL, CLF
from sklearn.model_selection import GridSearchCV
import prefect
from prefect import task


class ModelTrainer:
    def __init__(self) -> None:
        pass

    @task
    def model_trainer(self, X_train, y_train):
        try:
            model = XGBClassifier()

            # Define the parameter grid
            param_grid = {
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 6, 10],
                "n_estimators": [50, 100, 150],
                "subsample": [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0],
            }

            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=3,
                n_jobs=-1,
                verbose=1,
                scoring="accuracy",
            )

            # Perform Grid Search
            grid_search.fit(X_train, y_train)

            best_params = grid_search.best_params_

            best_model = grid_search.best_estimator_

            model_path = os.path.join(ARTIFACT, MODEL)
            os.makedirs(model_path, exist_ok=True)

            model_filename = os.path.join(model_path, CLF)

            joblib.dump({"model": best_model, "params": best_params}, model_filename)

            return best_model, best_params

        except Exception as e:
            raise e
