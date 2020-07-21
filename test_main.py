import base64
import json
import pytest

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


@pytest.mark.skip(reason="Not  possible to send base64 string through json yet.")
def test_predict(client, xmi_bytes):
    encoded_bytes = base64.b64encode(xmi_bytes)
    instance_dict = {"modelId": "default", "cas": encoded_bytes}
    response = client.post("/predict", json=instance_dict)

    assert response.status_code == 200


@pytest.mark.skip(reason="Not  possible to send base64 string through json yet.")
def test_predict_wrong_model_ID(client, xmi_bytes):
    encoded_bytes = base64.b64encode(xmi_bytes)
    instance_dict = {"modelId": "non-existent", "cas": encoded_bytes}
    response = client.post("/predict", json=instance_dict)

    assert response.status_code == 422
    assert (
        json.loads(response.text)["detail"] == "Model with modelId non-existent has "
        "not been trained yet. Please train first"
    )


@pytest.mark.skip(
    reason="This is hard to test because the 'clf' dictionary"
    "is bound directly to main.py and not to a function or class."
)
def test_predict_no_model(client, xmi_bytes):
    encoded_bytes = base64.b64encode(xmi_bytes)
    instance_dict = {"modelId": "", "cas": encoded_bytes}
    response = client.post("/predict", json=instance_dict)

    assert response.status_code == 400
    assert json.loads(response.text)["detail"] == "Train first.\nNo model here."


@pytest.mark.skip(reason="Not possible to send base64 string through json yet.")
def test_addInstance(client, xmi_bytes):
    encoded_bytes = base64.b64encode(xmi_bytes)
    instance_dict = {"modelId": "default", "cas": encoded_bytes}
    response = client.post("/addInstance", json=instance_dict)

    assert response.status_code == 200


@pytest.mark.skip(reason="Not possible to send base64 string through json yet.")
def test_addInstance_no_model_ID(client, xmi_bytes):
    encoded_bytes = base64.b64encode(xmi_bytes)
    instance_dict = {"modelId": "", "cas": encoded_bytes}
    response = client.post("/addInstance", json=instance_dict)

    assert response.status_code == 400
    assert (
        json.loads(response.text)["detail"] == "No model ID passed as argument."
        " Please include a model ID."
    )


@pytest.mark.skip(
    reason="For this test to work a CAS instance must be added,"
    "which also requires sending the base64 string."
)
def test_train_from_CASes(client):
    # Todo: Add CAS instance here.
    instance_dict = {"model_id": "default"}
    response = client.post("/trainFromCASes", json=instance_dict)

    assert response.status_code == 200
    assert (
        json.loads(response.text)["detail"] == "No model here with id"
        " default. Add CAS instances first."
    )


def test_train_from_CASes_missing_CAS_instance(client):
    instance_dict = {"model_id": "default"}
    response = client.post("/trainFromCASes", json=instance_dict)

    assert response.status_code == 422
    assert (
        json.loads(response.text)["detail"] == "No model here with id"
        " default. Add CAS instances first."
    )


def test_train_from_CASes_no_modelID(client):
    instance_dict = {"model_id": ""}
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