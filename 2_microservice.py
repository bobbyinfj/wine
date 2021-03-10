"""
With a model made, create a service which responds to HTTP requests. We only need one endpoint:

`[GET] /rating/prediction.json`

Return a JSON document with the rating prediction. You can assume that the attributes are provided as URL encoded parameters like `/rating/predcition.json?ph=3&brightness=5` and you should expect all 6 parameters above to be provided. If a parameter is missing, return a 400 status code.

"""

from flask import Flask, jsonify
import pandas as pd
import xgboost as xgb

from typing import List, Optional
from flask_parameter_validation import ValidateParameters, Route, Json, Query

app = Flask(__name__)

@app.route('/')
def index():
    return 'Welcome to the Wine Predictor Microservice'

@app.route('/rating/prediction.json', methods=['GET'])
@ValidateParameters() #parameter validation
def predict(
        brightness: float = Query(),
        chlorides: float = Query(),
        ph: float = Query(),
        sugar: float = Query(),
        sulfates: float = Query(),
        acidity: float = Query()
        ):
    input_parameters = {
        'brightness': [brightness],
        'chlorides': [chlorides],
        'ph': [ph],
        'sugar': [sugar],
        'sulfates': [sulfates],
        'acidity': [acidity]
    }
    print(input_parameters)
    query_df = pd.DataFrame.from_dict(input_parameters)
    print(query_df)
    prediction = clf.predict(query_df)
    print(f"Prediction is {prediction}")
    return jsonify({'prediction': int(prediction)})
    
# test query
# http://127.0.0.1:8080/rating/prediction.json?ph=5&brightness=3&chlorides=2&sugar=2&sulfates=15&acidity=12
# http://127.0.0.1:8080/rating/prediction.json?ph=1&brightness=2&chlorides=3&sugar=4&sulfates=5&acidity=6
# http://127.0.0.1:8080/rating/prediction.json?ph=3.5&brightness=2&chlorides=0&sugar=9.1&sulfates=1.5&acidity=15 (yields 4, real is 6)
# http://127.0.0.1:8080/rating/prediction.json?ph=2.4&brightness=1&chlorides=0.02&sugar=0&sulfates=1.5&acidity=8 (predicts 4, real is 5)
# http://127.0.0.1:8080/rating/prediction.json?ph=2&brightness=9&chlorides=0.02&sugar=9.7&sulfates=0.9&acidity=4 (predicts 4, real is 7)

#fix case when bogus parameters are given
#currently, it ignores bogus parameters if the required ones are given

if __name__ == '__main__':
    # load model
    clf = xgb.XGBClassifier()
    clf.load_model('xgb_full_train.json')
    print(clf)
    app.run(port=8080)