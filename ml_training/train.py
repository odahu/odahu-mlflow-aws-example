# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties.
# In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import argparse
import os

# MLFlow Packages for model storing
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# NumPy, Pandas, SKLearn
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Custom SDK
from odahu_mlflow_aws_sdk.inference import save_inference_logic, create_test_handler, InvalidModelInputException

# Constants
EXPERIMENT_ID = 1  # number of the experiment on the MlFlow Server
MODEL_PREDICTOR_NAME = 'price-prediction/model'  # path inside experiment save model in
MODEL_NAME = 'wine-quality-prediction'  # name of model to be registered

# Data Locations
LOCAL_DATASET_PATH = os.path.join(os.path.expanduser('~'), 'datasets', 'wine-quality.csv')
# Dataset Location (can be customized)
DATASET = LOCAL_DATASET_PATH \
    if os.path.exists(LOCAL_DATASET_PATH) \
    else '/domino/datasets/local/WineQualityPrediction/wine-quality.csv'

# Env configuration (can be customized)
if not os.getenv('MLFLOW_TRACKING_URL'):
    mlflow.set_tracking_uri('http://localhost:5000')


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train_model(alpha, l1_ratio, train_x, train_y, test_x, test_y):
    with mlflow.start_run(experiment_id=str(EXPERIMENT_ID)) as run:

        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print(f"Elasticnet model (alpha={alpha}, l1_ratio={l1_ratio}):")
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")

        # Save hyper parameters values and metric
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(
            lr,
            MODEL_PREDICTOR_NAME,
            signature=infer_signature(test_x, predicted_qualities),
            input_example=test_x,
            registered_model_name=MODEL_NAME
        )

        # Safe inference logic (lambda handler) as a part of this run (scoped)
        save_inference_logic()

        return run


if __name__ == "__main__":
    np.random.seed(40)

    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', default=0.3, type=float)
    parser.add_argument('--l1-ratio', default=0.5, type=float)

    args = parser.parse_args()

    # Read the wine-quality csv file
    data = pd.read_csv(DATASET)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]

    # Train model, get RUN ID
    train_run = train_model(
        alpha=args.alpha,
        l1_ratio=args.l1_ratio,
        train_x=train.drop(["quality"], axis=1),
        train_y=train[["quality"]],
        test_x=test.drop(["quality"], axis=1),
        test_y=test[["quality"]]
    )

    # Testing logic for the model
    # create_test_handler builds a wrapper of the inference code
    # and this can be used to test how your inference code works with a model
    test_handler = create_test_handler(
        run_id=train_run.info.run_id,
        model_name=MODEL_PREDICTOR_NAME,
        inference_code_location='../ml_service'
    )

    # Wine A
    graphql_prediction = test_handler.query_graphl(
        '''
        query {
            prediction(
              fixedAcidity: 6.2, volatileAcidity: 0.32, 
              citricAcid: 0.35, residualSugar: 6.1, 
              chlorides: 0.04, freeSulfurDioxide: 50.0, 
              totalSulfurDioxide: 100.0, density: 0.98,
              pH: 3.10, sulphates: 0.2, alcohol: 10.1
            ) {
                quality
            }
        }
        '''
    )['prediction']['quality']
    print(f'Quality of Wine A predicted as: {graphql_prediction}')

    dict_prediction = test_handler.query(**{
        'fixed acidity': 7.0,
        'volatile acidity': 0.30,
        'citric acid': 0.14,
        'residual sugar': 1.2,
        'chlorides': 0.02,
        'free sulfur dioxide': 10.0,
        'total sulfur dioxide': 120.0,
        'density': 0.97,
        'pH': 3.2,
        'sulphates': 0.3,
        'alcohol': 11.0
    })
    print(f'Quality of Wine B predicted as: {dict_prediction}')

    # Check that density is being validated
    try:
        test_handler.query(**{
            'fixed acidity': 7.0,
            'volatile acidity': 0.30,
            'citric acid': 0.14,
            'residual sugar': 1.2,
            'chlorides': 0.02,
            'free sulfur dioxide': 10.0,
            'total sulfur dioxide': 120.0,
            'density': 1.2,
            'pH': 3.2,
            'sulphates': 0.3,
            'alcohol': 11.0
        })
        raise Exception('Exception has not been raised for density = 1.2')
    except InvalidModelInputException:
        print('Exception has been raised for density = 1.2')
