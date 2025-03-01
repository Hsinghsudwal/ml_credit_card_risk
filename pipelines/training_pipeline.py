from src.ml_credit_card_risk.data_loader import DataLoader
from src.ml_credit_card_risk.data_validate import DataValidation
from src.ml_credit_card_risk.data_transformer import DataTransformation
from src.ml_credit_card_risk.model_trainer import ModelTrainer
from src.ml_credit_card_risk.model_evaluate import ModelEvaluation

# from src.ml_credit_card_risk.experiment_tracking import ModelMlflowExperiment

from src.helper import log_step


class TrainingPipeline:
    def __init__(self, path):
        self.path = path
        self.outputs = {}

    def run_pipeline(self):
        # Step 1: Data-Loader
        log_step("Starting data loader")
        dataload = DataLoader()
        train, test = dataload.data_load(self.path)
        # self.outputs["train"] = train
        log_step("Data loader completed")

        # Step 2: Data-Validate
        log_step("Starting data validation")
        datavali = DataValidation()
        train_val, test_val = datavali.data_validation(train, test)
        # self.outputs["train-vali"] = train_val
        log_step("Data validation completed")

        # Step 3: Data-Transformation
        log_step("Starting data transformation")
        datatrans = DataTransformation()
        X_train, X_test, y_train, y_test = datatrans.data_transformation(
            train_val, test_val
        )
        # self.outputs["X_train"] = X_train
        log_step("Data transformation completed")

        # Step 4: Model-Trainer
        log_step("Starting model trainer")
        modeltrain = ModelTrainer()
        model_in, param_in = modeltrain.model_trainer(X_train, y_train)
        # self.outputs["params"] = param_in
        log_step("Model trainer completed")

        # Step 5: Model-Evaluation
        log_step("Starting model trainer")

        model_path = "artifact/model/model.joblib"
        modeleval = ModelEvaluation()

        model, params = modeleval.model_evaluation(model_in, param_in, X_test, y_test)
        # model, params = modeleval.model_evaluation(model_path, X_test, y_test)
        self.outputs["model"] = model
        self.outputs["params"] = params
        log_step("Model evaluate completed")

        log_step("Training pipeline executed.")
        return self.outputs, model, params
