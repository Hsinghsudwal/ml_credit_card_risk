import pandas as pd
from src.constants import (
    ARTIFACT,
    TRANSFORM,
    PROCESSOR,
    XTRAIN_TRANS,
    XTEST_TRANS,
    YTRAIN_TRANS,
    YTEST_TRANS,
)
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import joblib
import prefect
from prefect import task


class DataTransformation:

    def __init__(self) -> None:
        pass

    @task
    def data_transformation(self, train_data, test_data):
        try:

            train_data = train_data.drop(
                train_data[train_data["person_age"] > 80].index
            )
            test_data = test_data.drop(test_data[test_data["person_age"] > 80].index)

            xtrain_data = train_data.drop(columns=["loan_status"], axis=1)
            xtest_data = test_data.drop(columns=["loan_status"], axis=1)

            y_train = train_data["loan_status"]
            y_test = test_data["loan_status"]

            num_features = xtrain_data.select_dtypes(
                include=["int64", "float64"]
            ).columns
            cat_features = xtrain_data.select_dtypes(include=["object"]).columns

            num_transformer = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler()),
                ]
            )

            cat_transformer = Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(strategy="constant", fill_value="missing"),
                    ),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num", num_transformer, num_features),
                    ("cat", cat_transformer, cat_features),
                ],
                remainder="passthrough",
            )

            # fit and transform train and test
            xtrain_preprocessed = preprocessor.fit_transform(xtrain_data)
            xtest_preprocessed = preprocessor.transform(xtest_data)

            # Get column names after transformation
            cat_columns = (
                preprocessor.transformers_[1][1]
                .named_steps["encoder"]
                .get_feature_names_out(cat_features)
            )
            new_columns = list(num_features) + list(cat_columns)

            # Convert data to DataFrame with new column names
            X_train = pd.DataFrame(xtrain_preprocessed, columns=new_columns)
            X_test = pd.DataFrame(xtest_preprocessed, columns=new_columns)

            transformer_path = os.path.join(ARTIFACT, TRANSFORM)
            os.makedirs(transformer_path, exist_ok=True)

            joblib.dump(preprocessor, os.path.join(transformer_path, PROCESSOR))

            X_train.to_csv(os.path.join(transformer_path, str(XTRAIN_TRANS)))
            X_test.to_csv(os.path.join(transformer_path, str(XTEST_TRANS)))

            y_train.to_csv(os.path.join(transformer_path, str(YTRAIN_TRANS)))
            y_test.to_csv(os.path.join(transformer_path, str(YTEST_TRANS)))

            return X_train, X_test, y_train, y_test

        except Exception as e:
            raise e
