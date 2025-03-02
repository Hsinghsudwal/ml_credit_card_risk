# Credit Card Risk
The aim of the project is to build end-to-end mlops pipeline.

## Table of Content
- [Problem Statement](#problem-statement)
- [Installation](#installation)
- [Development](#development)
- [Pipeline](#pipeline)
- [Orchestration](#orchestration)
- [Deployment](#deployment)
- [Monitoring](#monitoring)
- [Best Practices](#best-practices)
- [Next Step](#next-step)

## Problem Statement

To minimize loss from the bank’s perspective, the bank needs a decision rule regarding who to give approval of the loan and who not to. An applicant’s demographic and socio-economic profiles are considered by applicant managers before a decision is taken regarding their loan application. When a bank receives a loan application, based on the applicant’s profile the bank has to make a decision regarding whether to go ahead with the approved or not.
There are two main risks involved in this process:

* Credit Risk: The possibility that the applicant might not be able to repay the loan, which would be a bad credit risk for the bank.
* Financial Gain or Loss: The bank could either benefit from lending to a trustworthy borrower or suffer losses if the applicant is too risky.

**Context:**

In this scenario, financial institutions and lending companies want to figure out how risky it is to lend money to someone. They do this by looking at the borrower’s financial and personal details to predict if they'll be able to pay the loan back or if there's a chance they might default. The goal is to make the loan approval process smoother and smarter, helping lenders reduce risk while offering loans that fit each person’s situation. Lending money is always a bit risky, so these companies want to make sure they’re lending to people who can actually repay. By understanding the borrower’s background and financial history, they can make more informed decisions. This ultimately benefits both the lender, who faces less risk, and the borrower, who gets a loan that works for them.


**The following features is used to predict the loan approval status (loan_status):**

1. person_age: The age of the individual.
2. person_income: The annual income of the individual.
3. person_home_ownership: The home ownership status of the individual (e.g., rent, own).
4. person_emp_length: The number of years the individual has been employed.
5. loan_intent: The type of loan the person is applying for (e.g., personal, education).
6. loan_grade: The credit grade assigned to the loan (e.g., A, B, C, D, etc.).
7. loan_amnt: The loan amount requested by the individual.
8. loan_int_rate: The interest rate on the loan.
9. loan_status: Whether the loan was approved (1) or denied (0).
10. loan_percent_income: The percentage of the individual's income relative to the loan amount.
11. cb_person_default_on_file: Whether the individual has a default record on file (Y for Yes, N for No).
12. cb_person_cred_hist_length: The length of the individual's credit history in years.

13. target: loan_status: Whether the loan was approved (1) or denied (0).

The goal is to build a model that can predict whether a given individual’s loan application will be approved or denied based on these features.


### Dataset
The data source for this analysis is the Credit Risk Dataset available on Kaggl. This dataset contains financial and personal information of loan applicants, which can be used to evaluate credit risk, predict loan defaults, and improve decision-making processes for lending institutions. You can access the dataset [credit-risk](https://www.kaggle.com/datasets/laotse/credit-risk-dataset/data).

**Note:** In order to work local set-up please download the dataset in path = `data/credit.csv`

## Installation

1. Navigate to your project directory: Open your terminal and cmd to your project directory:
```bash
cd your/project
```

2. Create an environment: Use this command to create a new environment named `venv` with Python latest:
```bash
conda create -n "venv" python=3.12 -y
```

3. Activate an environment:
```bash
conda activate venv
```

4. Install the dependencies from requirements.txt:
```bash
pip install -r requirements.txt
```

## Development

1. Navigate to your project directory: Open your terminal and cmd to your notebook directory:
```bash
cd project/notebook
```

For developing this experiment, we use jupyter notebooks to perform EDA, feature engineering, model trainer, model evaluation, hypertunning and testing model performance to evalute in `credit.ipynb`.

**Steps:**
Preparing the Data:

* Filling in Missing Information: We’ll check for any missing values and decide the best way to handle them (either by filling them inwith the average or removing them if needed).
* Converting Categories: Some features (like home ownership or loan intent) are words, so we need to convert them into numbers so themachine can understand them. We’ll use encoding methods for this.
* Scaling the Numbers: Features like income or loan amount are numbersthat can vary a lot, so we need to scale them so they all fitwithin a similar range. This helps the model perform better.

Training the Model:

* Splitting the Data: We'll divide the data into two parts—one for training the model (about 80% of the data) and one for testing how well the model works (about 20% of the data).
* Choosing the Right Model: K-Neighbors, XGBOOST, Lightgbm, Gradient Boosting, Decision Tree,
Logistic Regression, Random Forest, Support Vector Classifier.

* Fine-Tuning the Model: We'll try different settings and options to make sure the model works as well as possible.
* Measuring Success: We’ll evaluate how well the model does by looking at how many correct predictions it makes. We’ll check metrics like accuracy, precision, and recall to get a full picture of its performance.


## Pipeline: 

1. Navigate to your project directory: Open your terminal and cmd to your src directory:
```bash
cd project/src
```

2. This main directory contains `helper.py` is for logging and save metadata and `constant.py` is for config outputs

3. In `src/ml_credit_card_risk` directory: contains files

* `data_loader.py`: This file handles the loading of data, typically from .csv files.
* `data_validate.py`: Ensures that the training and test data are correctly formatted. It checks that the number of features align between training and test sets.
* `data_transformer.py`: This file performs pre-processing tasks, such as data cleaning, feature engineering, and transformations, to prepare the data for machine learning models.
* `model_trainer`: Contains code to train models using the GridSearchCV technique, specifically training the XGBoost model by tuning hyperparameters to find the best configuration.
* `model_evaluate.py`: This file evaluates the trained model's performance by calculating relevant metrics (e.g., accuracy, precision, recall,f1,auc).

4. In `src/ml_credit_card_risk/experiment` directory contains files for experimenting and managing models:
* `stage_model.py`: Used for local experiments, this file stages the model in preparation for testing and deployment.
`testmodel.py`: Takes the staged model and tests it against a predefined threshold. If the model’s performance is below the threshold, it may trigger a re-training process or push the model to production if it meets the requirements.
* `prod_model.py`: Handles production-level testing of the model. It first checks whether the model is already in production, and if necessary, it archives the existing model. It then stages the new model for deployment based on the output from testmodel.py.

**Pipelines:**
1. Navigate to your project directory: Open your terminal and cmd to your pipelines directory:
```bash
cd project/pipelines
```

2. This directory contains:
* `training_pipeline.py`: which outlines the end-to-end pipeline for training the model. It might include steps such as loading the data, transforming features, training the model, and evaluating the performance. The purpose of this file is to automate the training process in a structured sequence, ensuring that each step is executed correctly and efficiently.

* `experiment_pipeline.py`: This is responsible for setting up and managing the training experiments. With logic for testing and staging, testing, archived and production the model.


**Running pipeline:**
Navigate to your main project directory: Open your terminal.
1. cmd to your main directory:
```bash
cd project/
```

2. Run pipeline:

```bash
python main.py
```

## Orchestration:
Orchestrating ML Workflow: Prefect

1. Setting Up Prefect
```bash
pip install prefect
```

2. Create prefect task and flow
```bash
@task
def load_data(filename: str):
    df = pd.read_csv(filename)
    return df

@flow(log_prints=True)
def ml_workflow(filename: str = "data.csv"):
    data = load_data(filename)

if __name__ == "__main__":
    ml_workflow()

test: python main.py
```


3. Deploying the Flow:

with ability to monitor the pipeline,
schedule using CRON, automatically retry in case of failure,
enable logging, receive notifications,
and automated workflows.

**Local prefect server:**
Navigate to your main project directory: Open your terminal.

1. Run prefect server:
```bash
prefect server start
```
2. Run pipelines
```bash
python main.py
```

**Deployment prefect**

1. Set the default work pool name as an environment variable via.

    Window cmd:`set PREFECT_DEFAULT_WORK_POOL_NAME=default`
    Linux cmd:`export PREFECT_DEFAULT_WORK_POOL_NAME=default`

    OR

2. Deployment configuration: Add a call to `flow.deploy` to tell Prefect how to deploy flow.
    ```bash
    flow.deploy(name="credit",
        work_pool_name="default",
        image="my-docker-image:dev",
        push=False)
    ``` 


Run your script to deploy your flow.
```bash
python main.py
```

**Trigger Run**

1. Work-pool: Create a Process work pool:
```bash
prefect work-pool create "default"
```
Verify that the work pool exists: `prefect work-pool ls`


2. Start a worker to poll the work pool:
    ```bash
    prefect worker start --pool "default"
    ```

3. Then, we can trigger a run of our flow using the Prefect CLI:


    ```bash
    prefect deployment run 'flow/my-deployment'
    ```

## Deployment
Once we’ve trained the model, we’ll be able to use it in real-time to predict whether future loan applications should be approved or denied.For deploying credit risk prediction application we built using Flask and hosted in a Docker container. The model is stored in a local S3 bucket managed by LocalStack, which mimics AWS services. The application loads the pre-trained machine learning model from this LocalStack S3 service to make predictions about credit risk based on user inputs.

**Technologies Used:**
1. Flask: A lightweight web framework for building the API and web interface.
2. Scikit-learn: For machine learning, including model prediction and preprocessing.
3. Boto3/LocalStack: AWS SDK and LocalStack (for local development) to interact with S3 storage.

**Navigate to your project directory:** 
1. Open your terminal and cmd to your deployment directory:

    ```bash
    cd project/deployment
    ```

**Dockerizing the Flask Application: Local**
1. build docker image
    ```bash
    docker build -t credit-risk .
    ```
2. run docker container
    ```bash
    docker run -it --rm -p 9696:9696 credit-risk
    ```

**To Run the Application:** Clone the repository containing the `Dockerfile`, `app.py`, and any other files.
Build and start the containers:
```bash
docker-compose up --build
```
Access the application at http://localhost:9696 in your web browser to interact with the Flask app and make predictions.

 
## Monitoring
This Python application monitors the performance of machine learning models and tracks data stability. The app uses Evidently, Prefect, Streamlit, and Plotly to generate and visualize reports that evaluate data drift, and data quality. It allows users to upload their own data or use a default dataset, processes it through various metrics, and provides alerts based on predefined thresholds.

**Navigate to your project directory:** 
1. Open your terminal and cmd to your monitoring directory:

    ```bash
    cd project/monitoring
    ```

**Dockerizing the Flask Application: Local**

1. Build Docker Image:
    ```bash
    docker build -t app .
    ```

2. Run Docker Container:
    ```bash
    docker run -p 8501:8501 app
    ```
**To Run the Application:** Clone the repository containing the `Dockerfile`, `app.py`, and any other files.
Build and start the containers:
```bash
docker-compose up --build
```

This will start the Streamlit app at http://localhost:8501 in web browser to Monitor and check alerts 

## Best Practices

* pytest: Quick test src/ml_credit_card_risk/experiment/testmodel.py

* format: black

## Next Step
1. Enhanced Monitoring: Implement continuous monitoring of model drift over time, along with database storage. This will allow users to download reports and save historical monitoring data whenever the pipeline runs. Additionally, track the deployment stages (e.g., development environment) and integrate test suite results for better insights into the deployment pipeline.

2. Cloud Integration: Add cloud capabilities for better scalability and accessibility. Consider using cloud services to store data, track models, and ensure smooth deployment.

3. CI/CD Pipeline: Integrate a CI/CD pipeline with GitHub Actions for automated deployment and testing. This will streamline the process of pushing updates and maintaining consistent application quality. 

* **Cloud**: AWS
* **Experiment tracking tools**: Mlflow
* **Workflow orchestration**: Kubeflow
* **Monitoring**: Evidently
* **CI/CD**: Github actions
* **Infrastructure as code (IaC)**: Terraform


