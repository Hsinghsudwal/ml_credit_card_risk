from pipelines.training_pipeline import TrainingPipeline
from pipelines.experiment_training import ExperimentPipeline
from s3_stack import localstack_s3

import prefect
from prefect import task, flow


@task(name="training")
def training_step_1(path: str):

    pipeline = TrainingPipeline(path)
    outputs, model, params = pipeline.run_pipeline()

    return outputs, model, params


@task(name="experiment")
def experiment_step_2(model, params):
    # Running the experiment pipeline
    ex_pipe = ExperimentPipeline(model, params)
    result = ex_pipe.experiment_pipeline()

    return result


@flow(name="flow", log_prints=True)
def main():
    """
    1. train-evaluate-experiment [stage,prod,archived]
    2. re-train
    3. prefect (schedule)
    """

    path = "data/credit.csv"

    outputs, model, params = training_step_1(path)
    print(outputs)

    result = experiment_step_2(model, params)

    if result == "Need Re-training":
        print("Model performance is below threshold. Re-training needed.")

        outputs, model, params = training_step_1(path)
        experiment_step_2(model, params)

    # Upload model and preprocesor
    localstack_s3()


if __name__ == "__main__":
    main()
    # main.serve(name="ml_credit_card_risk", cron="* * * * *")


# cron="minutes hour day-month month day-week"
