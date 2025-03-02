import pandas as pd
import joblib
import json
import streamlit as st
from sqlalchemy import create_engine
import plotly.graph_objects as go
from prefect import task, flow

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)
from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset
from evidently.metrics.base_metric import generate_column_metrics

# from evidently.model_profile import ModelProfile

import boto3

s3 = boto3.client(
    "s3",
    endpoint_url="http://localstack:4566",
    aws_access_key_id="test",
    aws_secret_access_key="test",
    region_name="us-east-1",
)

bucket_name = "risk-bucket"
model_file = "s3_risk_model.pkl"
transform_file = "s3_transformer.pkl"


# download model from S3
def download_model_from_s3():
    with open(model_file, "wb") as f:
        s3.download_fileobj(bucket_name, model_file, f)

    return joblib.load(model_file)


def download_transformer_from_s3():
    with open(transform_file, "wb") as f:
        s3.download_fileobj(bucket_name, transform_file, f)

    return joblib.load(transform_file)


def _load_s3():
    model = download_model_from_s3()
    transformer = download_transformer_from_s3()

    return model, transformer


# Database setup
engine = create_engine("sqlite:///monitoring_results.db")


# Helper functions
def create_metric_gauge(value, title):
    """Create a gauge chart to visualize scores."""
    return go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title},
            gauge={
                "axis": {"range": [None, 1]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 0.3], "color": "red"},
                    {"range": [0.3, 0.7], "color": "yellow"},
                    {"range": [0.7, 1], "color": "green"},
                ],
            },
        )
    )


def inspect_columns(reference_data, current_data):
    # two columns
    col1, col2 = st.columns(2)

    with col1:
        st.write("Reference Data Columns:")
        st.write(reference_data.columns)

    with col2:
        st.write("Current Data Columns:")
        st.write(current_data.columns)


def reference_data():
    reference_data = pd.read_csv("../data/credit.csv", index_col=0)
    return reference_data


def current_data(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file, index_col=0)
    else:
        return reference_data().sample(frac=0.2, random_state=42)


@task
def fetch_data(uploaded_file=None):
    reference = reference_data()
    current = current_data(uploaded_file)
    return reference, current


@task
def generate_drift_report(reference_data, current_data):
    column_mapping = ColumnMapping()
    column_mapping.target = "loan_status"

    # Set numerical and categorical features based on data types in reference_data
    column_mapping.numerical_features = reference_data.select_dtypes(
        include=["float64", "int64"]
    ).columns.tolist()
    column_mapping.categorical_features = reference_data.select_dtypes(
        include=["object"]
    ).columns.tolist()

    if "loan_status" in column_mapping.numerical_features:
        column_mapping.numerical_features.remove("loan_status")
    if "loan_status" in column_mapping.categorical_features:
        column_mapping.categorical_features.remove("loan_status")

    report = Report(
        metrics=[
            ColumnDriftMetric(column_name="loan_status"),
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
            DataDriftPreset(),
            TargetDriftPreset(),
        ]
    )

    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )
    return report


@task
def generate_model_drift_report(reference_data, current_data):
    # Download model/ transformer
    model, transformer = _load_s3()

    # Preparing features and target
    x = current_data.drop(columns=["loan_status"], axis=1)
    y = current_data["loan_status"]

    processed = transformer.transform(x)
    predictions = model.predict(processed)

    # Set up column mapping for target and prediction columns
    column_mapping = ColumnMapping()
    column_mapping.target = "loan_status"
    column_mapping.prediction = "prediction"

    # Add predictions to the current data
    current_data["prediction"] = predictions
    # Create model profile with drift and performance sections
    model_profile = profile.ModelProfile(
        sections=[ModelDriftProfileSection(), ModelPerformanceProfileSection()]
    )

    # Calculate the profile with drift and performance analysis
    model_profile.calculate(reference_data, current_data, column_mapping=column_mapping)

    return model_profile


@flow
def monitoring_flow(uploaded_file=None):
    reference_data, current_data = fetch_data(uploaded_file)
    drift_report = generate_drift_report(reference_data, current_data)
    # model_drift_report = generate_model_drift_report(reference_data, current_data)
    model_drift_report = 10
    return drift_report, model_drift_report


def handle_alerts(result, drift_report):
    alerts = []
    if result["metrics"][0]["result"]["drift_score"] > 0.1:
        alerts.append("ALERT: Significant data drift detected!")

    if alerts:
        for alert in alerts:
            st.error(alert)

    if result["metrics"][1]["result"]["dataset_drift"] > 0.2:
        alerts.append("ALERT: Data quality below threshold!")

    if alerts:
        for alert in alerts:
            st.error(alert)


def display_tabs(result, reference_data, current_data, data_drift_report):
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Columns", "Data Drift", "Data Quality", "Model_Drift"]
    )

    with tab1:
        st.subheader("Inspect columns")
        inspect_columns(reference_data, current_data)

    with tab2:
        st.subheader("Data Drift")
        data_drift_score = result["metrics"][0]["result"]["drift_score"]
        st.plotly_chart(
            create_metric_gauge(data_drift_score, "Data Drift Score"),
            use_container_width=True,
        )

    with tab3:
        st.subheader("Data Quality")
        data_quality = result["metrics"][1]["result"]["dataset_drift"]
        status = 1 if data_quality else 0
        st.plotly_chart(
            create_metric_gauge(status, "Data Quality"), use_container_width=True
        )

    with tab4:
        st.subheader("Model Drift")

        mdrift = 0
        st.plotly_chart(
            create_metric_gauge(mdrift, "Model Drift"), use_container_width=True
        )


def main():
    st.set_page_config(page_title="Monitoring Dashboard", layout="wide")
    st.title("Monitoring Dashboard")

    uploaded_file = st.file_uploader("Upload your CSV file to test", type=["csv"])

    if st.button("Run Monitoring Flow"):
        with st.spinner("Running monitoring flow..."):
            drift_report, model_drift_report = monitoring_flow(uploaded_file)
            reference_data, current_data = fetch_data(uploaded_file=uploaded_file)
            result = drift_report.as_dict()

            st.success("Monitoring flow completed!")

        # Handle alerts
        handle_alerts(result, drift_report)

        # Display tabs with the results
        display_tabs(result, reference_data, current_data, model_drift_report)


if __name__ == "__main__":
    main()
