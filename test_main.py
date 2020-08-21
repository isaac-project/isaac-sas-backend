import base64
import json
import pytest
import main

from fastapi.testclient import TestClient
from main import app


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture()
def xmi_bytes():
    with open("testdata/xmi/1ET5_7_0.xmi", "rb") as in_file:
        xmi_bytes = in_file.read()
    return xmi_bytes


def test_predict(client, xmi_bytes):
    encoded_bytes = base64.b64encode(xmi_bytes)
    instance_dict = {"modelId": "default", "cas": encoded_bytes.decode("ascii")}
    response = client.post("/predict", json=instance_dict)

    assert response.status_code == 200


def test_predict_wrong_model_ID(client, xmi_bytes):
    encoded_bytes = base64.b64encode(xmi_bytes)
    instance_dict = {"modelId": "non-existent", "cas": encoded_bytes.decode("ascii")}
    response = client.post("/predict", json=instance_dict)

    assert response.status_code == 422
    assert (
        json.loads(response.text)["detail"]
        == 'Model with model ID "non-existent" could '
        "not be found in the ONNX model directory. Please train first."
    )


def test_addInstance(client, xmi_bytes):
    encoded_bytes = base64.b64encode(xmi_bytes)
    instance_dict = {"modelId": "default", "cas": encoded_bytes.decode("ascii")}
    response = client.post("/addInstance", json=instance_dict)

    assert response.status_code == 200


def test_addInstance_no_model_ID(client, xmi_bytes):
    encoded_bytes = base64.b64encode(xmi_bytes)
    instance_dict = {"modelId": "", "cas": encoded_bytes.decode("ascii")}
    response = client.post("/addInstance", json=instance_dict)

    assert response.status_code == 400
    assert (
        json.loads(response.text)["detail"] == "No model ID passed as argument."
        " Please include a model ID."
    )


# Todo:If the CAS string model is run, it changes the default model so that
#      Only one output class is returned. Therefore, this test cannot be run.
# def test_train_from_CASes(client):
#     # Todo: Add CAS instance here.
#     instance_dict = {"modelId": "default"}
#     response = client.post("/trainFromCASes", json=instance_dict)
#
#     assert response.status_code == 200


def test_train_from_CASes_missing_CAS_instance(client):
    instance_dict = {"modelId": "default"}
    # Pretend that main.features is empty but store its value to put back in later.
    temp_features = main.features
    main.features = {}
    response = client.post("/trainFromCASes", json=instance_dict)
    # Put the original values back into main.clf.
    main.features = temp_features

    assert response.status_code == 422
    assert (
        json.loads(response.text)["detail"] == "No model here with id"
        " default. Add CAS instances first."
    )


def test_train_from_CASes_no_modelID(client):
    instance_dict = {"modelId": ""}
    response = client.post("/trainFromCASes", json=instance_dict)

    assert response.status_code == 400
    assert (
        json.loads(response.text)["detail"] == "No model id passed as"
        " argument. Please include a modelId"
    )


@pytest.mark.skip(reason="depracated function call: pandas.read_table")
def test_train(client):
    # Todo: This test is not complete yet.
    response = client.get("/train")

    assert response.status_code == 200


# Todo: I haven't written a test for /wipe because it would cause models to be
#       removed everytime the tests are run. If a test is desired, I can
#       maybe find a workaround.
