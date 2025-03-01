import joblib
import json
from pipelines.training_pipeline import TrainingPipeline
from src.helper import log_step
from src.ml_credit_card_risk.experiment.stage_model import StageModel
from src.ml_credit_card_risk.experiment.testmodel import TestModel
from src.ml_credit_card_risk.experiment.prod_model import ProductionModel


class ExperimentPipeline:

    def __init__(self, model, param):
        self.model = model
        self.param = param

    def experiment_pipeline(self):

        # model, params = TrainingPipeline().run_pipeline()
        # model_path = "artifact/model/model.joblib"
        metric_path = "artifact/evaluate/metric.json"
        # models = joblib.load(model_path)
        # model = models["model"]
        # params = models["params"]

        model = self.model
        params = self.param

        with open(metric_path, "r") as file:
            metric = json.load(file)

        log_step("Starting experiment pipeline")
        stage = StageModel()
        stage.stage_model(model, params, metric)

        log_step("Starting testing stage model")
        testmodel = TestModel()
        results = testmodel.test_model()

        if results == "Retraining":
            log_step("Test results indicate re-training is needed")
            return "Needs Re-training"

        else:

            log_step("Moving to production")
            prodmodel = ProductionModel()
            # check model exist
            if prodmodel.check_if_old_model_exists():

                log_step("Moving old production model to archived")
                prodmodel.archive_current_production_model()

            log_step("Moving stage to production")
            prod_dir = prodmodel.move_to_production(model, params, metric)

            log_step("Experiment pipeline completed")
            return "Model in Production"
