import sys
import os
import shutil
import time
import traceback
import base64
import json
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.externals import joblib

from features import uima
from features.extractor import FeatureExtraction
from cassis.xmi import load_cas_from_xmi
from io import BytesIO
from pandas.core.frame import DataFrame

try:
    from _thread import allocate_lock as Lock
except:
    from _dummy_thread import allocate_lock as Lock

app = Flask(__name__)

# inputs
training_data = 'data/gold-standard-features.tsv'
include = ['_InitialView-Keyword-Overlap', '_InitialView-Token-Overlap',
           'studentAnswer-Token-Overlap', '_InitialView-Chunk-Overlap',
           'studentAnswer-Chunk-Overlap', '_InitialView-Triple-Overlap',
           'studentAnswer-Triple-Overlap', 'LC_TOKEN-Match', 'LEMMA-Match',
           'SYNONYM-Match', 'Variety', 'Outcome']
dependent_variable = include[-1]

model_directory = 'model'
# model_file_name = '%s/model.pkl' % model_directory

model_default_name = "default"

# UIMA / features stuff
# type system
isaac_ts = uima.load_isaac_ts()
# feature extraction
extraction = FeatureExtraction()
# in-memory feature data
features = {}
lock = Lock()

# These will be populated at training time
model_columns = {}
clf = {} # model objects


def do_prediction(data: DataFrame, model_id: str = None) -> list:
    query = pd.get_dummies(data)
    if not model_id:
        model_id = model_default_name

    # https://github.com/amirziai/sklearnflask/issues/3
    # Thanks to @lorenzori
    query = query.reindex(columns=model_columns[model_id], fill_value=0)

    return list(clf[model_id].predict(query))


@app.route('/predict', methods=['POST'])
def predict():
    if clf:
    #try: # todo: maybe uncomment and try running it again ?
        json_ = request.json
        model_id_ = json_["modelId"]
        base64_cas = base64.b64decode(json_["cas"])

        # TODO: maybe check if modelId is in models, if not, do not proceed? See if the error gets passed on to REST service
        if model_id not in clf:
            return "Model with modelId \"{}\" has not been trained yet. Please train first".format(model_id_)

        print("printing deseralized json cas modelID: ", model_id_)

        # from_cases feature extraction
        cas = load_cas_from_xmi(BytesIO(base64_cas), typesystem=isaac_ts)
        print("loaded the cas...")
        feats = extraction.from_cases([cas])
        print("extracted feats")
        data = pd.DataFrame.from_dict(feats)
        prediction = do_prediction(data, model_id_)

        # Converting to int from int64
        print(jsonify({"prediction": list(map(int, prediction))}))
        return jsonify({"prediction": list(map(int, prediction))})

    else:
        print('train first')
        return 'no model here'


@app.route('/addInstance', methods=['POST'])
def addInstance():
    json_cas = request.json
    model_id = json_cas["modelId"]
    base64_string = base64.b64decode(json_cas["cas"])

    if not model_id: # todo: changed b/c there should always be a modelId
        return 'No model id passed as argument. Please include a modelId', 400

    cas = load_cas_from_xmi(BytesIO(base64_string), typesystem=isaac_ts)
    feats = extraction.from_cases([cas])
    with lock:
        if model_id in features:
            # append new features
            for name, value in feats.items():
                print("Printing value inside addInstance: ", value)
                features[model_id][name].append(value[0])
        else:
            features[model_id] = feats

    return "Successfully added cas to model {}".format(model_id)


@app.route('/trainFromCASes', methods=['GET'])
def trainFromCASes():
    model_id = request.args.get('modelId')
    if not model_id: # todo: changed b/c there should always be a modelId
        # model_id = model_default_name
        return 'No model id passed as argument. Please include a modelId', 400
    if features[model_id]:
        print("type of features[model_id] in trainFromCASes: ", type(features[model_id]))
        data = pd.DataFrame.from_dict(features[model_id])
        print("type of data in trainFromCASes (after DataFrame.from_dict: ", type(data))
        return do_training(data, model_id)
    else:
        print('add CAS instances first')
        return 'No model here with id {}'.format(model_id) + '. Add CAS instances first.'

def do_training(df: DataFrame, model_id: str = None) -> str:
    # using random forest as an example
    # can do the training separately and just update the pickles
    from sklearn.ensemble import RandomForestClassifier as rf
    df_ = df[include]

    categoricals = []  # going to one-hot encode categorical variables

    for col, col_type in df_.dtypes.items():
        if col_type == 'O':
            categoricals.append(col)
        else:
            df_[col].fillna(0, inplace=True)  # fill NA's with 0 for ints/floats, too generic

    # get_dummies effectively creates one-hot encoded variables
    df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)
    x = df_ohe[df_ohe.columns.difference([dependent_variable])]
    y = df_ohe[dependent_variable]

    if not model_id:
        model_id = model_default_name

    # capture a list of columns that will be used for prediction
    model_columns_file_name = '{}/{}_columns.pkl'.format(model_directory, model_id)
    with lock:
        model_columns[model_id] = list(x.columns)
    joblib.dump(model_columns[model_id], model_columns_file_name)

    # build classifier
    with lock:
        clf[model_id] = rf()
        start = time.time()
        clf[model_id].fit(x, y)

    out_file = '{}/{}.pkl'.format(model_directory, model_id)
    joblib.dump(clf[model_id], out_file)

    message1 = 'Trained in %.5f seconds' % (time.time() - start)
    message2 = 'Model training score: %s' % clf[model_id].score(x, y)
    return_message = 'Success. \n{0}. \n{1}.'.format(message1, message2)
    print(return_message)
    return return_message


@app.route('/train', methods=['GET'])
def train():
    print("Training")
    df = pd.read_table(training_data)
    return do_training(df, None)


@app.route('/wipe', methods=['GET'])
def wipe():
    try:
        shutil.rmtree(model_directory)
        os.makedirs(model_directory)
        return 'Models wiped'

    except Exception as e:
        print(str(e))
        return 'Could not remove and recreate the model directory'


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except Exception as e:
        port = 80

    try:
        for f in os.listdir(model_directory):
            if f.endswith(".pkl"):
                if "_columns" in f:
                    model_id = f[:-12]
                    model_columns[model_id] = joblib.load('{}/{}_columns.pkl'.format(model_directory, model_id))
                    print('model columns {} loaded'.format(model_id))
                else:
                    model_id = f[:-4]
                    clf[model_id] = joblib.load('{}/{}.pkl'.format(model_directory, model_id))
                    print('model {} loaded'.format(model_id))

    except Exception as e:
        print('No model here')
        print('Train first')
        print(str(e))

    app.run(host='0.0.0.0', port=port, debug=True)
