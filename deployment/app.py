import joblib
import pandas as pd
import boto3
from flask import Flask, request, jsonify, render_template


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


# for local transform
def load_preprocessor():
    path = "../artifact/transform/transformer.pkl"
    preprocessor = joblib.load(path)
    joblib.dump(preprocessor, "preprocessor.pkl")
    return preprocessor


# for local model
def load_local_model():
    path = "../artifact/experiment/Production/xgboost.pkl"
    model = joblib.load(path)
    joblib.dump(model, "model.pkl")
    return model


def predictor(data, use_s3=True):
    process_data = pd.DataFrame(data, index=[0])

    # Load model and transformer either from S3 or local based on user input
    if use_s3:
        model, transformer = _load_s3()
    else:
        model = load_local_model()
        transformer = load_preprocessor()

    # Process the data
    processed_data = transformer.transform(process_data)
    preds = model.predict(processed_data)
    return preds[0]


app = Flask("credit-risk-prediction")


@app.route("/", methods=["GET", "POST"])
def index():
    use_s3 = False
    if request.method == "POST":
        use_s3 = request.form.get("use_s3") == "yes"

        user_input = {
            "person_age": float(request.form["person_age"]),
            "person_income": float(request.form["person_income"]),
            "person_home_ownership": str(request.form["person_home_ownership"]),
            "person_emp_length": float(request.form["person_emp_length"]),
            "loan_intent": str(request.form["loan_intent"]),
            "loan_grade": str(request.form["loan_grade"]),
            "loan_amnt": float(request.form["loan_amnt"]),
            "loan_int_rate": float(request.form["loan_int_rate"]),
            "loan_percent_income": float(request.form["loan_percent_income"]),
            "cb_person_default_on_file": str(request.form["cb_person_default_on_file"]),
            "cb_person_cred_hist_length": float(
                request.form["cb_person_cred_hist_length"]
            ),
        }

        pred = predictor(user_input, use_s3)
        print(pred)

        if pred == 1:
            status = "GOOD CREDIT RISK"
        else:
            status = "BAD CREDIT RISK"

        print("status: ", status)

        return render_template("index.html", status=status, use_s3=use_s3)

    return render_template("index.html", status=None, use_s3=use_s3)


@app.route("/predict", methods=["POST"])
def predict_endpoint():

    input_data = request.get_json()
    use_s3 = input_data.get("use_s3", True)
    pred = predictor(input_data, use_s3)

    if pred == 1:
        status = "GOOD CREDIT RISK"
    else:
        status = "BAD CREDIT RISK"

    result = {"result": status}
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
