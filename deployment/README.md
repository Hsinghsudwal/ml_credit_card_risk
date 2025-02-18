## Deploying a model as Flask appication service
For model deploying, first create Flask-based web application designed for credit risk prediction. It utilizes machine learning to assess whether a person presents a "GOOD" or "BAD" credit risk based on various features such as age, income, loan amount, and credit history. The application integrates with AWS S3 (via LocalStack for local development) to download a trained machine learning model and used for making predictions.

**Technologies**:
1. Flask: A lightweight web framework for building the API and web interface.
2. Scikit-learn: For machine learning, including model prediction and preprocessing.
3. Boto3/LocalStack: AWS SDK and LocalStack (for local development) to interact with S3 storage.

### Deployment Steps:
*Setting Up the Environment*
Navigate to the deployment directory:
Open your terminal and go to the deployment folder in your project `cd deployment`

* step 1: `setup environment`
* step 2: `activate environment`
* step 3: `pip install -r requirements.txt or --dev[package]`
* step 4: open your editor




### Create function to download production model
The `download_model_from_s3()` function in the Flask app interacts with AWS S3 (or LocalStack during development) to download the trained machine learning model. This model is then used to make predictions about credit risk. The function uses the boto3 library to connect to S3 and joblib to load the model


## Dockerizing the Flask Application
* build docker image
```bash
docker build -t credit-risk .
```
* run docker container
```bash
docker run -it --rm -p 9696:9696 credit-risk
```
**Access the Flask app:** After running the container, the Flask app will be accessible at `http://localhost:9696`. You can visit this URL in your browser or use tools like `Postman` to interact with the app's `API`