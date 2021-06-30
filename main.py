import json
import os
import shutil
import time
import numpy as np
import onnxruntime as rt
import pandas as pd

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from features.feature_groups import BOWGroupExtractor
from features.feature_groups import SIMGroupExtractor
from features.data import ShortAnswerInstance
from pandas.core.frame import DataFrame
from pydantic import BaseModel
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from typing import Dict
from typing import List
from typing import Union


app = FastAPI()

# TODO: The allow_origins=['*'] variable should probably changed and contain a
#       specified number of origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

onnx_model_dir = "onnx_models"
bow_model_dir = "bow_models"

# in-memory feature data
features = {}

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

# For prediction from ShortAnswerInstances the BOW model belonging to the ML model
# must be loaded for feature extraction.
bow_models = {}
for bow_file in os.listdir(bow_model_dir):
    # Ignore hidden files like .keep
    if bow_file.startswith("."):
        continue
    model_id = bow_file.rstrip(".json")
    if model_id not in bow_models:
        bow_path = os.path.join(bow_model_dir, bow_file)
        with open(bow_path) as bowf:
            state_dict = json.load(bowf)
            # Instances list is passed empty here because bag of words setup has 
            # already been done.
            bow_models[model_id] = BOWGroupExtractor([])
            bow_models[model_id].bag = state_dict["bag"]


class LanguageDataRequest(BaseModel):
    """A request with language data, used for training and predicting."""
    instances: List[ShortAnswerInstance]
    modelId: str


class SinglePrediction(BaseModel):
    """A single prediction result, including probabilities for individual classes."""
    prediction: Union[int, str]
    classProbabilities: Dict[Union[str, int], float]


class PredictFromLanguageDataResponse(BaseModel):
    """A response containing one or more prediction results."""
    predictions: List[SinglePrediction]


class ModelIdResponse(BaseModel):
    """A response containing the IDs of the models currently available."""
    modelIds: List[str]


@app.get("/fetchStoredModels", response_model=ModelIdResponse)
def fetch_stored_models():
    return {"modelIds": list(inf_sessions.keys())}


@app.post("/trainFromAnswers")
def trainFromAnswers(req: LanguageDataRequest):
    model_id = req.modelId
    # All feature extractor objects that should be used, are defined here.
    ft_extractors = [SIMGroupExtractor()]

    df = pd.DataFrame()
    
    # Note that the BOW feature extractor is set up later because it needs a new
    # setup for every new train-test split.
    for ft_extractor in ft_extractors:
        df = pd.concat([df, ft_extractor.extract(req.instances)], axis=1)

    labels = pd.DataFrame([instance.label for instance in req.instances], columns=["labels"])

    best_metrics = init_best_metrics(model_id)
    best_model = None

    n_splits = (10 if df.shape[0] > 1000 else 5) if df.shape[0] > 50 else 2

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2)
    for train_ids, test_ids in skf.split(df, labels):

        # The right indices must be found to extract the BOW features for the correct instances.
        train_instances = [req.instances[idx] for idx in train_ids]
        bow_extractor = BOWGroupExtractor(train_instances)

        x = pd.concat([df, bow_extractor.extract(req.instances)], axis=1)

        # NOTE: If categorical features are included, One-hot should be included here as well.

        start = time.time()

        clf = RandomForestClassifier()

        x_train = x.iloc[train_ids]
        y_train = labels.iloc[train_ids]
        x_test = x.iloc[test_ids]
        y_test = labels.iloc[test_ids]

        clf.fit(x_train, y_train)

        y_pred = clf.predict(x_test)

        end = time.time()

        metrics = classification_report(
            y_test, y_pred, output_dict=True, target_names=["False", "True"]
        )

        accuracy = accuracy_score(y_test, y_pred)
        f1 = metrics["macro avg"]["f1-score"]
        cohens_kappa = cohen_kappa_score(y_test, y_pred)

        # Add accuracy and cohens kappa to the metrics dictionary.
        metrics["accuracy"] = accuracy
        metrics["cohens_kappa"] = cohens_kappa

        best_list = best_metrics[model_id]

        best_acc = best_list["accuracy"]
        best_f1 = best_list["f1"]
        best_ck = best_list["cohens_kappa"]

        for best, current in zip(
            (best_acc, best_f1, best_ck), (accuracy, f1, cohens_kappa)
        ):
            if current > best["value"]:
                best["value"] = current
                best["metrics"] = metrics
                best["model_type"] = clf.__class__.__name__
                bow_models[model_id] = bow_extractor
                bow_path = os.path.join(bow_model_dir, model_id + ".json")
                with open(bow_path, "w") as bowf:
                    json.dump(bow_extractor.__dict__, bowf)

        best_list["train_time"] = end - start

        # TODO: How to determine which model should be stored 
        # (accuracy, f1, cohens kappa)?
        if not best_model or accuracy > best_acc["value"]:
            best_model = clf
            model_columns = list(x.columns)
            num_features = clf.n_features_

    # Write best results metrics to file
    with open("model_metrics/" + model_id + ".json", "w") as score_file:
        json.dump(best_metrics, score_file, indent=4)

    # Store all models (no double storing if same model).
    store_as_onnx(best_model, model_id, model_columns, num_features)

    return best_metrics


def init_best_metrics(model_id):
    # Initialize the best training acc, f1, cohens kappa and their models.
    metrics_out = {
        model_id: {
            "accuracy": {
                "value": 0.0,
                "metrics": None,
                "model_type": None,
            },
            "f1": {
                "value": 0.0,
                "metrics": None,
                "model_type": None,
            },
            "cohens_kappa": {
                "value": 0.0,
                "metrics": None,
                "model_type": None,
            },
        }
    }

    return metrics_out


def store_as_onnx(model, model_id, model_columns, num_features):
    initial_type = [("float_input", FloatTensorType([None, num_features]))]
    clf_onnx = convert_sklearn(model, initial_types=initial_type, target_opset=12)

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


@app.post("/predictFromAnswers", response_model=PredictFromLanguageDataResponse)
def predictFromAnswers(req: LanguageDataRequest):
    model_id = req.modelId

    if model_id not in [model.rstrip(".onnx") for model in os.listdir(onnx_model_dir)]:
        raise HTTPException(
            status_code=422,
            detail='Model with model ID "{}" could not be'
            " found in the ONNX model directory."
            " Please train first.".format(model_id),
        )
    if model_id not in [model.rstrip(".json") for model in os.listdir(bow_model_dir)]:
        raise HTTPException(
            status_code=422,
            detail='BOW Model with model ID "{}" could not be'
            " found in the Bag of words model directory."
            " Please check that the model was trained with training"
            " instances (not with CAS).".format(model_id),
        )

    bow_extractor = bow_models[model_id]
    ft_extractors = [SIMGroupExtractor(), bow_extractor]

    predictions = []

    for instance in req.instances:        
        data = pd.DataFrame()
        for ft_extractor in ft_extractors:
            data = pd.concat([data, ft_extractor.extract([instance])], axis=1)

        predictions.append(do_prediction(data, model_id))

    return {"predictions": predictions}


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

    # prediction is the class with max probability
    return {
        "prediction": max(probs, key=lambda k: probs[k]),
        "classProbabilities": probs,
    }


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
