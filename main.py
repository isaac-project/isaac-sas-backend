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
#model_file_name = '%s/model.pkl' % model_directory

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
clf = {}

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
    if not model_id:
        model_id = model_default_name
    cas = load_cas_from_xmi(BytesIO(request.form['xmi']), typesystem=isaac_ts)
    feats = extraction.from_cases([cas])
    with lock:
        if model_id in features:
            # append new features
            for name, value in feats.iteritems():
                features[model_id][name].append(value)
        else:
            features[model_id] = feats

@app.route('/trainFromCASes', methods = ['GET'])
def trainFromCASes():
    model_id = request.args.get('modelId')
    if not model_id:
        model_id = model_default_name
    if features[model_id]:
        data = pd.DataFrame.from_dict(features[model_id])
        return do_training(data, model_id)
    else:
        print('add CAS instances first')
        return 'no model here with id {}'.format(model_id)

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
    model_columns[model_id] = list(x.columns)
    joblib.dump(model_columns[model_id], model_columns_file_name)

    # build classifier
    clf[model_id] = rf()
    start = time.time()
    clf[model_id].fit(x, y)

    out_file = '{}/{}.pkl'.format(model_directory, model_id)
    joblib.dump(clf[model_id], out_file)
    
    message1 = 'Trained in %.5f seconds' % (time.time() - start)
    message2 = 'Model training score: %s' % clf[model_id].score(x, y)
    return_message = 'Success. \n{0}. \n{1}.'.format(message1, message2) 
    return return_message
    
    
@app.route('/train', methods=['GET'])
def train():
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
                    model_columns[model_id] = joblib.load('{}/{}_columns.pkl'.format(model_directory,model_id))
                    print('model columns {} loaded'.format(model_id))
                else:
                    model_id = f[:-4]
                    clf[model_id] = joblib.load('{}/{}.pkl'.format(model_directory,model_id))
                    print('model {} loaded'.format(model_id))
                
    except Exception as e:
        print('No model here')
        print('Train first')
        print(str(e))
        
    app.run(host='0.0.0.0', port=port, debug=True)
