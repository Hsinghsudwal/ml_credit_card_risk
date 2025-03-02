## Monitoring Application
With lightweight application monitor the performance of the models and keeps track of the stability of the data over time. By using tools like Evidently, Prefect, Streamlit, and Plotly to create and display insightful reports that highlight data drift and data quality.

With this app, user can easily upload their own data or use a default dataset. It processes your data through various metrics, providing you with real-time feedback and alerts when predefined thresholds are exceeded. This ensures that models and data stay reliable, accurate, and up-to-date.

**Technologies:**
* Pandas: Data manipulation library for handling and processing CSV files.
* Evidently: A powerful library for monitoring machine learning models, focused on detecting data drift, evaluating data quality, and analyzing model performance.
* Prefect: A workflow orchestration tool that manages the execution of monitoring tasks in a controlled manner.
* Streamlit: Web application framework for building interactive dashboards to visualize results.
* Plotly: Visualization tool for generating dynamic and interactive charts, such as gauge charts, to represent drift scores and quality.
* boto3: AWS SDK to interact with S3 for storing and retrieving model files.
* Joblib: Used for loading serialized model and transformer objects.

### Input:
1. CSV File Upload: Users can upload a CSV file containing the features and target variable (e.g., loan_status) for analysis.
2. Reference Data: If no file is uploaded, the app defaults to using a dataset like credit.csv for comparison.
### Output:
1. Metrics:
* Data Drift: Reports the drift score between the reference and current data.
* Data Quality: Measures the quality of the data, focusing on missing values and inconsistencies.

### Visualizations:
1. Gauge Charts: Display drift and quality scores.
2. Performance Drift: Visualizes model performance over time.
* Alerts:
    Alerts are triggered when significant data drift or low data quality is detected.


### Workflow
1. User Uploads Data:

    Users upload their CSV file using the Streamlit file uploader or the system defaults to a sample dataset (credit.csv).

2. Task Execution (Managed by Prefect):

3. The fetch_data task loads the reference and current data.
4. The generate_drift_report task computes drift scores between the datasets.
5. The generate_model_drift_report task computes changes in model predictions.

### Streamlit UI:

The app displays interactive gauge charts and alerts for data and model drift.
The monitoring flow results are displayed in separate tabs (Data Drift, Data Quality, Model Drift).


## How to Run the Application
Local Setup (Docker, Cloud, or Local Environment)

### Docker Setup:

Create a Dockerfile to containerize the app and ensure it can run locally or on the cloud (e.g., AWS, GCP, or Azure).

1. Build Docker Image:
    ```bash
    docker build -t app .
    ```

2. Run Docker Container:
    ```bash
    docker run -p 8501:8501 app
    ```
This will start the Streamlit app at http://localhost:8501.
Cloud Setup:
To deploy the app in the cloud, you can push the Docker image to a container registry (e.g., AWS ECR, Docker Hub) and deploy it on a cloud service like AWS ECS, Google Cloud Run, or Azure App Service.

### Build and Run Locally
Install Dependencies: Ensure all the necessary dependencies are installed:

```bash
pip install -r requirements.txt
```
Run the Application: Start the Streamlit app locally:

```bash
streamlit run app.py
```

### Database Setup
SQLite Database:

The app stores monitoring results (drift scores, quality metrics, etc.) in an SQLite database called monitoring_results.db.

## Key Flow
File Upload → Fetch Data (Reference + Current) → Generate Drift Report → in Streamlit.

Alerts are displayed when significant drift or low-quality data is detected.

Downloadable HTML Report with complete monitoring results.



## Next Step:

The next step is to keep tracks changes in model predictions (model drift), evaluates model performance over time, and ensures data quality through a test suite. The results are saved to an SQLite database and users can download a full HTML report containing detailed metrics and drift analysis.

**Technologies**
* SQLAlchemy: ORM (Object Relational Mapper) used to interact with the SQLite database to store monitoring results and logs.
* SQLite: A lightweight database used to store monitoring results for historical analysis and tracking.

**Outpus**
* Model Drift: Tracks changes in model predictions compared to the reference model.
* Model Performance: Assesses how the model's performance (e.g., AUC, accuracy) has changed.
* Test Suite: Runs tests to ensure the data meets stability criteria and produces alerts if thresholds are exceeded.

**Visualize**
1. Database Storage:
    * Monitoring results are stored in an SQLite database for future analysis and reporting.

2. HTML Report:
    * Users can download a full report containing detailed metrics and drift analysis.

    * Users can download the monitoring report.

### Database:

The results are saved into an SQLite database named monitoring_results.db.

