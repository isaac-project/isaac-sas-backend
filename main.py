import os
import shutil
import time
import base64
import numpy as np
import onnxruntime as rt
import pandas as pd

from fastapi import FastAPI
from fastapi import HTTPException
from features import uima
from features.extractor import FeatureExtraction
from cassis.xmi import load_cas_from_xmi
from io import BytesIO
from pandas.core.frame import DataFrame
from pydantic import BaseModel
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.helpers.onnx_helper import load_onnx_model
from typing import Dict

try:
    from _thread import allocate_lock as Lock
except:
    from _dummy_thread import allocate_lock as Lock

app = FastAPI()

# inputs
training_data = "data/gold-standard-features.tsv"
include = [
    "_InitialView-Keyword-Overlap",
    "_InitialView-Token-Overlap",
    "studentAnswer-Token-Overlap",
    "_InitialView-Chunk-Overlap",
    "studentAnswer-Chunk-Overlap",
    "_InitialView-Triple-Overlap",
    "studentAnswer-Triple-Overlap",
    "LC_TOKEN-Match",
    "LEMMA-Match",
    "SYNONYM-Match",
    "Variety",
    "Outcome",
]
dependent_variable = include[-1]

onnx_model_dir = "onnx_models"
model_default_name = "default"

# UIMA / features stuff
# type system
isaac_ts = uima.load_isaac_ts()
# feature extraction
extraction = FeatureExtraction()
# in-memory feature data
features = {}
lock = Lock()

clf = {}  # model objects


class ClassificationInstance(BaseModel):
    model_id: str
    cas: str


class CASPrediction(BaseModel):
    prediction: str
    classProbabilities: Dict[str, float]
    features: Dict


class TrainFromCASRequest(BaseModel):
    model_id: str


def do_prediction(data: DataFrame, model_id: str = None) -> dict:
    # Check that the model object has been stored in clf. If not, store it.
    if model_id not in clf:
        clf[model_id] = load_onnx_model("{}/{}.onnx".format(onnx_model_dir, model_id))

    session = rt.InferenceSession("{}/{}.onnx".format(onnx_model_dir, model_id))

    query = pd.get_dummies(data)
    # The columns in string format are retrieved from the model and converted
    # back to a list.
    model_columns = clf[model_id].metadata_props[1].value.split(" ")
    # https://github.com/amirziai/sklearnflask/issues/3
    # Thanks to @lorenzori
    query = query.reindex(columns=model_columns, fill_value=0)

    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name
    # Prediction takes place here.
    pred = session.run([label_name], {input_name: query.to_numpy(dtype=np.float32)})[0]

    # The model classes are retrieved
    output_classes = clf[model_id].metadata_props[0].value.split(" ")

    # get prediction as class probability distribution and map to classes
    probs = dict(zip(map(str, output_classes), map(float, pred)))

    # prediction is the class with max probability
    return {
        "prediction": max(probs, key=lambda k: probs[k]),
        "classProbabilities": probs,
    }


@app.post("/predict", response_model=CASPrediction)
def predict(req: ClassificationInstance):
    model_id = req.model_id
    base64_cas = base64.b64decode(req.cas)

    # If no model ID is there, use the default model.
    if not model_id:
        model_id = model_default_name

    # Check that the model is stored in a file.
    if model_id not in [model.rstrip(".onnx") for model in os.listdir(onnx_model_dir)]:
        raise HTTPException(
            status_code=422,
            detail='Model with model ID "{}" could not be'
            " found in the ONNX model directory."
            " Please train first.".format(model_id),
        )

    print("printing deseralized json cas modelID: ", model_id)

    # from_cases feature extraction
    cas = load_cas_from_xmi(BytesIO(base64_cas), typesystem=isaac_ts)
    print("loaded the cas...")
    feats = extraction.from_cases([cas])
    print("extracted feats")
    data = pd.DataFrame.from_dict(feats)
    prediction = do_prediction(data, model_id)
    prediction["features"] = {k: v[0] for k, v in feats.items()}
    print(prediction)
    return prediction


@app.post("/addInstance")
def addInstance(req: ClassificationInstance):
    model_id = req.model_id
    base64_string = base64.b64decode(req.cas)

    if not model_id:  # todo: changed b/c there should always be a modelId
        raise HTTPException(
            status_code=400,
            detail="No model ID passed as argument." " Please include a model ID.",
        )

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
    print("Successfully added cas to model {}".format(model_id))
    return "Successfully added cas to model {}".format(model_id)


@app.post("/trainFromCASes")
def trainFromCASes(req: TrainFromCASRequest):
    model_id = req.model_id
    if not model_id:  # todo: changed b/c there should always be a modelId
        # model_id = model_default_name
        raise HTTPException(
            status_code=400,
            detail="No model id passed as argument. " "Please include a modelId",
        )
    if features.get(model_id):
        print(
            "type of features[model_id] in trainFromCASes: ", type(features[model_id])
        )
        data = pd.DataFrame.from_dict(features[model_id])
        print("type of data in trainFromCASes (after DataFrame.from_dict: ", type(data))
        return do_training(data, model_id)
    else:
        raise HTTPException(
            status_code=422,
            detail="No model here with id {}".format(model_id)
            + ". Add CAS instances first.",
        )


def do_training(df: DataFrame, model_id: str = None) -> str:
    # using random forest as an example
    # can do the training separately and just update the pickles
    from sklearn.ensemble import RandomForestClassifier as rf

    df_ = df[include]

    categoricals = []  # going to one-hot encode categorical variables

    for col, col_type in df_.dtypes.items():
        if col_type == "O":
            categoricals.append(col)
        else:
            df_[col].fillna(
                0, inplace=True
            )  # fill NA's with 0 for ints/floats, too generic

    # get_dummies effectively creates one-hot encoded variables
    df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)
    x = df_ohe[df_ohe.columns.difference([dependent_variable])]
    y = df_ohe[dependent_variable]

    if not model_id:
        model_id = model_default_name

    # build classifier
    with lock:
        clf[model_id] = rf()
        start = time.time()
        clf[model_id].fit(x, y)
        # Store the score, model columns and the output classes for later use.
        model_score = clf[model_id].score(x, y)
        output_classes = [str(out_class) for out_class in list(clf[model_id].classes_)]
        model_columns = list(x.columns)

    # The number of features of the model is obtained from the n_features_ attribute.
    # Fixme: Note that this works for the Random Forest Classifier in sklearn
    #        but does not work for all other model types.
    num_features = clf[model_id].n_features_
    initial_type = [("float_input", FloatTensorType([None, num_features]))]
    clf[model_id] = convert_sklearn(clf[model_id], initial_types=initial_type)

    # Manually pass the output classes and model columns to the converted model
    # using the metadata_props attribute.
    for category, metadata in zip(
        ("output_classes", "model_columns"), (output_classes, model_columns)
    ):
        new_meta = clf[model_id].metadata_props.add()
        new_meta.key = category
        # The metadata lists must be converted to a string because the
        # metadata_props attribute only allows sending strings.
        new_meta.value = " ".join(metadata)

    with open("{}/{}.onnx".format(onnx_model_dir, model_id), "wb") as onnx_file:
        onnx_file.write(clf[model_id].SerializeToString())

    message1 = "Trained in %.5f seconds" % (time.time() - start)
    message2 = "Model training score: %s" % model_score
    return_message = "Success. \n{0}. \n{1}.".format(message1, message2)

    return return_message


@app.get("/train")
def train():
    print("Training")
    # Fixme: When running the test on this function I get a depracation warning
    #        for the function read_table. (read_csv is recommended)
    df = pd.read_table(training_data)
    return do_training(df, None)


@app.get("/wipe_models")
def wipe_models():
    try:
        shutil.rmtree(onnx_model_dir)
        os.makedirs(onnx_model_dir)
        return "ONNX Models wiped"

    except Exception as e:
        print(str(e))
        raise HTTPException(
            status_code=400,
            detail="Could not remove and recreate the onnx_models directory",
        )
