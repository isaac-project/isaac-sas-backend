import sys
import os
import shutil
import time
import traceback

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
model_file_name = '%s/model.pkl' % model_directory
model_columns_file_name = '%s/model_columns.pkl' % model_directory

# UIMA / features stuff
# type system
isaac_ts = uima.load_isaac_ts() 
# feature extraction
extraction = FeatureExtraction()
# in-memory feature data
features = {}
lock = Lock()

# These will be populated at training time
model_columns = None
clf = None

def do_prediction(data: DataFrame) -> list:
    query = pd.get_dummies(data)

    # https://github.com/amirziai/sklearnflask/issues/3
    # Thanks to @lorenzori
    query = query.reindex(columns=model_columns, fill_value=0)

    return list(clf.predict(query))

@app.route('/predict', methods=['POST'])
def predict():
    if clf:
        try:
            json_ = request.json
            data = None
            
            if (json_):
                data = pd.DataFrame(json_)
            else:
                # not json, assume XML (CAS XMI)
                # from_cases feature extraction
                cas = load_cas_from_xmi(BytesIO(request.data), typesystem=isaac_ts)
                feats = extraction.from_cases([cas])
                data = pd.DataFrame.from_dict(feats)

            prediction = do_prediction(data)

            # Converting to int from int64
            return jsonify({"prediction": list(map(int, prediction))})

        except Exception as e:

            return jsonify({'error': str(e), 'trace': traceback.format_exc()})
    else:
        print('train first')
        return 'no model here'

@app.route('/addInstance', methods = ['POST'])
def addInstance():
    model_id = request.form['modelId']
    cas = load_cas_from_xmi(BytesIO(request.form['xmi']), typesystem=isaac_ts)
    feats = extraction.from_cases([cas])
    with lock:
        if model_id in features:
            # merge
            for name, value in feats.iteritems():
                features[model_id][name].append(value)
        else:
            features[model_id] = feats

@app.route('/train', methods=['GET'])
def train():
    # using random forest as an example
    # can do the training separately and just update the pickles
    from sklearn.ensemble import RandomForestClassifier as rf

    df = pd.read_table(training_data)
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

    # capture a list of columns that will be used for prediction
    global model_columns
    model_columns = list(x.columns)
    joblib.dump(model_columns, model_columns_file_name)

    global clf
    clf = rf()
    start = time.time()
    clf.fit(x, y)

    joblib.dump(clf, model_file_name)

    message1 = 'Trained in %.5f seconds' % (time.time() - start)
    message2 = 'Model training score: %s' % clf.score(x, y)
    return_message = 'Success. \n{0}. \n{1}.'.format(message1, message2) 
    return return_message


@app.route('/wipe', methods=['GET'])
def wipe():
    try:
        shutil.rmtree('model')
        os.makedirs(model_directory)
        return 'Model wiped'

    except Exception as e:
        print(str(e))
        return 'Could not remove and recreate the model directory'


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except Exception as e:
        port = 80

    try:
        clf = joblib.load(model_file_name)
        print('model loaded')
        model_columns = joblib.load(model_columns_file_name)
        print('model columns loaded')

    except Exception as e:
        print('No model here')
        print('Train first')
        print(str(e))
        clf = None

    app.run(host='0.0.0.0', port=port, debug=True)
