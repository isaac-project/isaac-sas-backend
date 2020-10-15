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
from typing import Dict
from typing import List
from typing import Union

try:
    from _thread import allocate_lock as Lock
except:
    from _dummy_thread import allocate_lock as Lock

app = FastAPI()


# These are the standard input features for the two endpoints
# /train and /trainFromCASes.
include_norm = [
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
dependent_variable = include_norm[-1]

onnx_model_dir = "onnx_models"

# UIMA / features stuff
# type system
isaac_ts = uima.load_isaac_ts()
# feature extraction
extraction = FeatureExtraction()
# in-memory feature data
features = {}
lock = Lock()

# Inference session object for predictions.
inf_sessions = {}

# Store all model objects and inference session objects in memory for
# quick access.
for model_file in os.listdir(onnx_model_dir):
    model_id = model_file.rstrip(".onnx")
    if model_id not in inf_sessions:
        inf_sessions[model_id] = rt.InferenceSession(
            os.path.join(onnx_model_dir, model_file)
        )


class ClassificationInstance(BaseModel):
    modelId: str
    cas: str


class CASPrediction(BaseModel):
    prediction: int
    classProbabilities: Dict[Union[str, int], float]
    features: Dict[str, Union[float, int, None]]


class TrainFromCASRequest(BaseModel):
    modelId: str


class TrainingInstance(BaseModel):
    fileName: str
    modelId: str


class ShortAnswerInstance(BaseModel):
    taskId: str
    itemId: str
    itemPrompt: str
    itemTargets: List[str]
    learnerId: str
    answer: str


class TrainFromLanguageDataRequest(BaseModel):
    instances: List[ShortAnswerInstance]
    modelId: str


def do_prediction(data: DataFrame, model_id: str = None) -> dict:

    session = inf_sessions[model_id]

    query = pd.get_dummies(data)
    # The columns in string format are retrieved from the model and converted
    # back to a list.
    model_columns = (
        session.get_modelmeta().custom_metadata_map["model_columns"].split(" ")
    )

    # https://github.com/amirziai/sklearnflask/issues/3
    # Thanks to @lorenzori
    query = query.reindex(columns=model_columns, fill_value=0)

    input_name = session.get_inputs()[0].name
    # The predict_proba function is used because get_outputs() is indexed at 1.
    # If it is indexed at 0, the predict method is used.
    label_name = session.get_outputs()[1].name
    # Prediction takes place here.
    pred = session.run([label_name], {input_name: query.to_numpy(dtype=np.float32)})[0]

    # The Prediction dictionary is stored in a list by ONNX so it can be
    # retrieved by indexing.
    probs = pred[0]
    print(probs)

    # prediction is the class with max probability
    return {
        "prediction": max(probs, key=lambda k: probs[k]),
        "classProbabilities": probs,
    }


@app.post("/predict", response_model=CASPrediction)
def predict(req: ClassificationInstance):
    model_id = req.modelId
    base64_cas = base64.b64decode(req.cas)

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
    model_id = req.modelId
    base64_string = base64.b64decode(req.cas)

    if not model_id:
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

    model_id = req.modelId

    if not model_id:
        raise HTTPException(
            status_code=400,
            detail="No model id passed as argument. " "Please include a model ID",
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


@app.post("/trainFromAnswers")
def trainFromAnswers(req: TrainFromLanguageDataRequest):
    model_id = req.modelId

    df = extract_features(req.instances)
    include = list(df.columns)

    return do_training(df, model_id, include=include, dependent_variable=include[-1])


def extract_features(instances: List[ShortAnswerInstance]) -> DataFrame:

    # This is a dummy feature setup, which will be replaced by the real
    # one later.
    features = []

    # The import should be removed once the dummy is removed.
    from random import randint

    for instance in instances:
        item_eq_answer = []
        if any(target == instance.answer for target in instance.itemTargets):
            item_eq_answer.append(1)
        else:
            item_eq_answer.append(0)

        # Also create dummy values for the outcome variable.
        item_eq_answer.append(randint(0, 1))

        features.append(item_eq_answer)

    columns = ["item_eq_answer", "outcome"]
    # Dummy setup ends here, all this code needs to be replaced.

    return pd.DataFrame(features, columns=columns)


def do_training(
    df: DataFrame,
    model_id: str = None,
    include: List[str] = include_norm,
    dependent_variable: str = dependent_variable,
) -> str:
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

    # build classifier
    with lock:
        clf = rf()
        start = time.time()
        clf.fit(x, y)
        # Store the score, model columns and the output classes for later use.
        model_score = clf.score(x, y)
        model_columns = list(x.columns)

    # The number of features of the model is obtained from the n_features_ attribute.
    # Fixme: Note that this works for the Random Forest Classifier in sklearn
    #        but does not work for all other model types.
    num_features = clf.n_features_
    initial_type = [("float_input", FloatTensorType([None, num_features]))]
    clf_onnx = convert_sklearn(clf, initial_types=initial_type)

    # Manually pass the model columns to the converted model using the
    # metadata_props attribute.
    new_meta = clf_onnx.metadata_props.add()
    new_meta.key = "model_columns"
    # The metadata lists must be converted to a string because the
    # metadata_props attribute only allows sending strings.
    new_meta.value = " ".join(model_columns)

    with open("{}/{}.onnx".format(onnx_model_dir, model_id), "wb") as onnx_file:
        onnx_file.write(clf_onnx.SerializeToString())

    # Store an inference session for this model to be used during prediction.
    inf_sessions[model_id] = rt.InferenceSession(
        "{}/{}.onnx".format(onnx_model_dir, model_id)
    )

    message1 = "Trained in %.5f seconds" % (time.time() - start)
    message2 = "Model training score: %s" % model_score
    return_message = "Success. \n{0}. \n{1}.".format(message1, message2)

    return return_message


@app.post("/train")
def train(req: TrainingInstance):
    model_id = req.modelId
    file_name = req.fileName

    print("Training")
    if not model_id:
        raise HTTPException(
            status_code=400,
            detail="No model id passed as argument. " "Please include a model ID",
        )

    df = pd.read_csv(file_name, delimiter="\t")
    return do_training(df, model_id)


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
