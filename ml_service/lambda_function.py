#
#    Copyright 2021 EPAM Systems
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
"""
Handler for pre-processing, validation and post-processing of a data
flow is:

Client --[HTTP(s) req.]--> API Gateway + Lambda / ECS / Docker --[HTTP(s) req. or API]--> SageMaker endpoint / Docker
                           ^^^^^ This code covers this ^^^^^^^

Each request is processing in a next way:
1. Parse HTTP Request
2. Detect type (based on content type) - supported JSON/CSV (MLFlow original implementation) & GraphQL
3. Parse request for detected type to pandas DataFrame / Series
4. Run "validation" logic of a handler: Can be customized below in function validate()
5. Transform input using "pre_process" logic of handler: Can be customized below in function pre_process()
6. Invoke model: over HTTP(s), using AWS API (SageMaker invoke) or in memory for testing cases
7. Transform output using "post_process" logic of handler: Can be customized below in function post_process()
8. Encode using JSON and send in HTTP Response
"""
import typing

# SDK - helper for building ready-to-deploy handlers
#  which can be deployed as AWS Lambda / Docker image / started locally
import pandas as pd
from odahu_mlflow_aws_sdk.inference import sdk

# Schema and ColSpec are needed for declaring input & output schema
from mlflow.types import Schema, ColSpec

# Pandas for typing


# Declare a handler, which will do pre- & post- processing of invocations
class CustomModelHandler(sdk.BaseModelHandler):
    """
    This class should be implemented to provide custom logic for:
    - validating(raw input)
    - pre processing(validated input) -> transformed input
    - post processing(prediction result) -> transformed prediction result
    of the model invocation API call

    Parent class provides different invocation options child class can use:
    - CSV (MLFlow Original Implementation)
    - JSON (MLFlow Original Implementation)
    - Tensor (MLFlow Original Implementation)
    - GraphQL query (provided by the odahu_mlflow_aws_sdk)
    """

    # Declare a schema for input data, each incoming request will be parsed and validated against this schema
    # For the GraphQL - name of parameters/arguments will be converted to the camelCase (fixed acidity -> fixedAcidity)
    INPUT_SCHEMA = Schema([
        ColSpec('double', 'fixed acidity'),
        ColSpec('double', 'volatile acidity'),
        ColSpec('double', 'citric acid'),
        ColSpec('double', 'residual sugar'),
        ColSpec('double', 'chlorides'),
        ColSpec('double', 'free sulfur dioxide'),
        ColSpec('double', 'total sulfur dioxide'),
        ColSpec('double', 'density'),
        ColSpec('double', 'pH'),
        ColSpec('double', 'sulphates'),
        ColSpec('double', 'alcohol'),
    ])

    # Declare a schema for output data, results of prediction (for GraphQL) will be encoded using this schema
    OUTPUT_SCHEMA = Schema([
        ColSpec('double', 'quality'),
    ])

    def pre_process(
            self,
            query: typing.Union[pd.DataFrame, pd.Series]
    ) -> None:
        """
        This function can be implemented to provide custom logic how input query should be "mapped" to
        expected for the model set of values

        :param query: model prediction
        :type query: typing.Union[pd.DataFrame, pd.Series]
        :return: post processed data, ready to be sent to model prediction endpoint
        :rtype: typing.Union[pd.DataFrame, pd.Series]
        """
        return query

    def post_process(
            self,
            prediction_response: typing.Union[pd.DataFrame, pd.Series]
    ) -> typing.Union[pd.DataFrame, pd.Series]:
        """
        This function can be implemented to provide custom logic how prediction should be "mapped" to the response

        :param prediction_response: model prediction response
        :type prediction_response: typing.Union[pd.DataFrame, pd.Series]
        :return: post processed data
        :rtype: typing.Union[pd.DataFrame, pd.Series]
        """
        return prediction_response

    def validate(
            self,
            query: typing.Union[pd.DataFrame, pd.Series]
    ) -> None:
        """
        This function can be implemented to validate that input query (before pre_process) is valid,
        e.g. values are in expected ranges

        :param query: Query to be validated
        :type query: typing.Union[pd.DataFrame, pd.Series]
        :return: nothing
        :rtype: None
        """
        if (query.values <= 0).any():
            raise sdk.InvalidModelInputException('Only positive values are supported')
        if (query.density >= 1.0).any():
            raise sdk.InvalidModelInputException('Density should be in range (0;1)')


# For AWS Lambda (API Gateway backend)
def lambda_handler(event, context):
    """
    AWS Lambda handler - provides an entrypoint for the AWS Lambda
    :param event:
    :param context:
    :return:
    """
    return CustomModelHandler.handle_lambda(event, context)


# For local testing
if __name__ == '__main__':
    CustomModelHandler.start_local_service()

# Fow WSGI Serving (in an independent container)
wsgi = CustomModelHandler.wsgi_handler
