from flask import Flask, request, jsonify, make_response
import json
import pickle
import numpy as np
import pandas as pd


app = Flask(__name__)

features_list = []
predictions = []
prospects_names = []
results = {}


@app.route("/predict", methods=["POST"])
def get_predict():
    # get the auth token
    auth_header = request.headers.get('Authorization')
    if auth_header:
        try:
            auth_token = auth_header.split(" ")[1]
        except (IndexError, json.decoder.JSONDecodeError):
            response_object = {
                'status': 'fail',
                'message': 'Bearer token malformed.'
            }
            return make_response(jsonify(response_object)), 401
    else:
        auth_token = ''
    if auth_token == 'sdfghjkloerdtfyguhiopfghjkl;fghjkl':
        predictions.clear()
        prospects_names.clear()
        results.clear()
        data = request.get_json(force=True)
        features = data['features']
        for values in features.values():
            for value in values.values():
                features_list.append(value)
            lab_name = data['lab_name']
            model_path = '../../resource/' + lab_name + '/models/order_prediction/' + data['model']
            model = pickle.load(open(model_path, 'rb'))
            arr = np.array(features_list).reshape(1, 15)
            df = pd.DataFrame(arr, columns=(
                ['visit_month', 'potencial', 'activity', 'speciality', 'sector', 'locality', 'product', 'cumul_ech',
                 'cumul_sale', 'cumul_cmd', 'cumul_visit', 'moy_satisf', 'visit_period', 'frequency', 'recency']))
            prediction = abs(int(model.predict(df)))
            predictions.append(prediction)
            features_list.clear()
        for key in features.keys():
            prospects_names.append('accumulation of orders for ' + key)
        for prospect in range(len(predictions)):
            results[prospects_names[prospect]] = predictions[prospect]
        response_object = {
            'status': 'success',
            'data': results
        }
    else:
        response_object = {
            'status': 'fail',
            'message': 'Provide a valid auth token.'
        }

    return make_response(jsonify(response_object))


@app.route('/commentclean')
def run_script():
    file = open('C:/machine-learning-internship_2022/machine-learning-internship_2022/src/comment_analysis/comments_cleaning2.py').read()

    return exec(file)



@app.route("/comment", methods=["POST"])
def get_comment_score():
    # get the auth token
    auth_header = request.headers.get('Authorization')
    if auth_header:
        try:
            auth_token = auth_header.split(" ")[1]
        except (IndexError, json.decoder.JSONDecodeError):
            response_object = {
                'status': 'fail',
                'message': 'Bearer token malformed.'
            }
            return make_response(jsonify(response_object)), 401
    else:
        auth_token = ''
    if auth_token == 'sdfghjkloerdtfyguhiopfghjkl;fghjkl':
        data = request.get_json(force=True)
        model_name = 'comment_rating2'
        #lab_name = data['lab_name']
        lab_name = 'stoderma'
        model_path = '../../resource/' + lab_name + '/models/comment/' + model_name
        model = pickle.load(open(model_path, 'rb'))
        score = model.predict([str(data['comment'])])[0]
        model_methods = str(model[0]) + ' with ' + str(model[1])
        model_description = "".join([i.lower() for i in model_methods if i not in ')('])
        response_object = {
            'status': 'success',
            'model_description': model_description,
            'score': score

        }
    return make_response(jsonify(response_object))


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
