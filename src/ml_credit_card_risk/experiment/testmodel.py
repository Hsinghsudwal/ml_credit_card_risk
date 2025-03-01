import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib


class TestModel:

    def __init__(self):
        pass

    def test_model(self):
        try:

            data = pd.read_csv("data/credit.csv")

            data = data.drop(data[data["person_age"] > 80].index)

            x = data.drop(columns=["loan_status"], axis=1)
            y = data["loan_status"]

            _, xtest, _, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

            num_features = x.select_dtypes(include=["int64", "float64"]).columns
            cat_features = x.select_dtypes(include=["object"]).columns

            # Preprocessor
            preprocess = ColumnTransformer(
                transformers=[
                    (
                        "categorical",
                        Pipeline(
                            steps=[
                                (
                                    "imputer",
                                    SimpleImputer(
                                        strategy="constant", fill_value="missing"
                                    ),
                                ),
                                ("encoder", OneHotEncoder(handle_unknown="ignore")),
                            ]
                        ),
                        cat_features,
                    ),
                    (
                        "numerical",
                        Pipeline(
                            steps=[
                                ("imputer", SimpleImputer(strategy="mean")),
                                ("scaler", StandardScaler()),
                            ]
                        ),
                        num_features,
                    ),
                ],
                remainder="passthrough",
            )

            xtest_preprocessed = preprocess.fit_transform(xtest)

            # Get the names
            cat_column_names = (
                preprocess.transformers_[0][1]
                .named_steps["encoder"]
                .get_feature_names_out(cat_features)
            )
            num_column_names = num_features

            columns = list(cat_column_names) + list(num_column_names)

            xtest = pd.DataFrame(xtest_preprocessed, columns=columns)

            model_stage = "artifact/experiment/Staging/xgboost.pkl"
            model = joblib.load(model_stage)

            model_columns = model.get_booster().feature_names
            xtest = xtest[model_columns]

            ypred = model.predict(xtest)

            accuracy = accuracy_score(ytest, ypred)
            expected_accuracy = 0.90

            print("Testing stage model completed")

            if accuracy < expected_accuracy:
                print(
                    f"Model accuracy {accuracy} is below the threshold {expected_accuracy}."
                )
                return "Retraining"
            else:
                return "Production"

        except Exception as e:

            raise Exception(f"Error occurred: {e}")
