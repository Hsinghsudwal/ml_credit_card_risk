import pandas as pd
import os
from sklearn.metrics import (
    recall_score,
    f1_score,
    accuracy_score,
    precision_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
import joblib
import json
import seaborn as sns
import matplotlib.pyplot as plt
from src.constants import (
    ARTIFACT,
    EVALUATE,
    CLASS_REPORT,
    METRIC,
    MATRIX,
)
import prefect
from prefect import task


class ModelEvaluation:

    def __init__(self) -> None:
        pass

    @task
    def evaluate_clf(true, predicted):
        accuracy = accuracy_score(true, predicted)
        f1 = f1_score(true, predicted)
        precision = precision_score(true, predicted)
        recall = recall_score(true, predicted)
        roc_auc = roc_auc_score(true, predicted)

        return accuracy, f1, precision, recall, roc_auc

    def model_evaluation(self, model, params, X_test, y_test):
        # def model_evaluation(self, path, X_test, y_test):
        try:

            # modeled = joblib.load(path)
            # model = modeled["model"]
            # params = modeled["params"]
            y_pred = model.predict(X_test)

            accuracy, f1, precision, recall, roc_auc = ModelEvaluation.evaluate_clf(
                y_test, y_pred
            )

            metrics_dict = {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
                "roc_auc": roc_auc,
            }

            evaluate_path = os.path.join(ARTIFACT, EVALUATE)
            os.makedirs(evaluate_path, exist_ok=True)

            evaluate_filename = os.path.join(evaluate_path, METRIC)

            with open(evaluate_filename, "w") as f:
                json.dump(
                    {
                        "metrics": metrics_dict,
                    },
                    f,
                    indent=4,
                )

            class_report = classification_report(y_test, y_pred)

            class_report_txt_path = os.path.join(ARTIFACT, EVALUATE, str(CLASS_REPORT))
            with open(class_report_txt_path, "w") as f:
                f.write(class_report)

            conf_matrix = confusion_matrix(y_test, y_pred)
            # print("Confusion Matrix:",conf_matrix)
            # print(conf_matrix)
            sns.heatmap(conf_matrix, annot=True, fmt="g")
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            cm_path = os.path.join(ARTIFACT, EVALUATE, MATRIX)
            plt.savefig(cm_path, dpi=120)
            plt.close()

            return model, params

        except Exception as e:
            raise e
