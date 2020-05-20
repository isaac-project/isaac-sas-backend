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

training_data = 'data/gold-standard-features.tsv'

include = ['_InitialView-Keyword-Overlap', '_InitialView-Token-Overlap',
           'studentAnswer-Token-Overlap', '_InitialView-Chunk-Overlap',
           'studentAnswer-Chunk-Overlap', '_InitialView-Triple-Overlap',
           'studentAnswer-Triple-Overlap', 'LC_TOKEN-Match', 'LEMMA-Match',
           'SYNONYM-Match', 'Variety', 'Outcome']
if __name__ == '__main__':

    df = pd.read_table(training_data)
    print(df)
    df_ = df[include]
    print(df_)

    import requests

    print("Testing wipe...")
    res = requests.get('http://localhost:9999/wipe')
    if res.ok:
        print("Wipe testing: successful")
    else:
        print("Wipe testing: UNsuccessful")
    print("Testing /train...")

    res = requests.get('http://localhost:9999/train')
    if res.ok:
        print("Training successful")
    else:
        print("Training UNsuccessful")

    # res = requests.post('http://localhost:9999/train', open('1ET5_7_0.xmi', 'rb'))
    # TESTING PREDICT WITH XMI FILE
    print("Testing predict with an xmi file...")
    res = requests.post('http://localhost:9999/predict', open('1ET5_6_79.xmi','rb'))
    if res.ok:
        print (res.json())
    else:
        print("request failed: ", res)
    print("Finished testing predict with xmi.")

    # ADDING A CAS INSTANCE TO TRAINING DATA
    print("Testing trainFromCases..")
    res = requests.get('http://localhost:9999/trainFromCASes?modelId=bla')
    if res.ok:
        print(res.content)
    else:
        print("request failed ", res)
    #curl -i http://localhost:9999/trainFromCASes

    
    params = {'modelId': 'bla','xmi': open('1ET5_6_79.xmi', 'rb')}
    res = requests.post('http://localhost:9999/addInstance', params)
    print(res)
    if res.ok:
        print(res.json()
              )
