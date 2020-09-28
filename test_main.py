import base64
import json
import os
import pytest
import main

from fastapi.testclient import TestClient
from main import app
from main import extract_features


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture()
def xmi_bytes():
    with open("testdata/xmi/1ET5_7_0.xmi", "rb") as in_file:
        xmi_bytes = in_file.read()
    return xmi_bytes


@pytest.fixture()
def mock_instances():
    instance1 = {
        "taskId": "0",
        "itemId": "0",
        "itemPrompt": "mock_prompt",
        "itemTargets": ["one", "two", "three"],
        "learnerId": "0",
        "answer": "two",
    }
    instance2 = {
        "taskId": "1",
        "itemId": "1",
        "itemPrompt": "mock_prompt2",
        "itemTargets": ["four", "five", "six"],
        "learnerId": "1",
        "answer": "two",
    }
    instance3 = {
        "taskId": "2",
        "itemId": "2",
        "itemPrompt": "mock_prompt3",
        "itemTargets": ["four", "five", "six"],
        "learnerId": "2",
        "answer": "five",
    }

    # The dicionaries are used to set up shortAnswerInstances.
    return [instance1, instance2, instance3]


def test_predict(client, xmi_bytes):
    """
    Test the /addInstance endpoint with an example model.

    This test predicts from the default model that sits in the actual onnx
    model directory instead of from a model sitting in the testdata directory.
    That is necessary because the models are loaded from this directory
    to memory while the service is running.
    The default model cannot be taken out of onnx_models directory.
    Otherwise this test will not run anymore.

    :param client: A client for testing.
    :param xmi_bytes: A byte-encoded CAS instance.
    """
    encoded_bytes = base64.b64encode(xmi_bytes)
    instance_dict = {"modelId": "default", "cas": encoded_bytes.decode("ascii")}
    response = client.post("/predict", json=instance_dict)

    assert response.status_code == 200

    # Assert that all values are between 0 and 1.
    assert 0 <= response.json()["prediction"] <= 1

    for cls in response.json()["classProbabilities"]:
        assert 0 <= response.json()["classProbabilities"][cls] <= 1

    for cls in response.json()["features"]:
        assert 0 <= response.json()["features"][cls] <= 1


def test_predict_wrong_model_ID(client, xmi_bytes):
    """
    Test the /predict endpoint with a model ID that is not present in the
    session object dictionary.

    :param client: A client for testing.
    :param xmi_bytes: A byte-encoded CAS instance.
    """
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
    """
    Test the /addInstance endpoint with an example instance.

    :param client: A client for testing.
    :param xmi_bytes: A byte-encoded CAS instance.
    """
    encoded_bytes = base64.b64encode(xmi_bytes)
    instance_dict = {"modelId": "default", "cas": encoded_bytes.decode("ascii")}
    response = client.post("/addInstance", json=instance_dict)

    added_to_features = "default" in main.features

    # Clean the main.features dictionary for future tests.
    main.features = {}

    assert added_to_features
    assert response.status_code == 200


def test_addInstance_no_model_ID(client, xmi_bytes):
    """
    Test the /addInstance endpoint with missing model ID.

    :param client: A client for testing.
    :param xmi_bytes: A byte-encoded CAS instance.
    """
    encoded_bytes = base64.b64encode(xmi_bytes)
    instance_dict = {"modelId": "", "cas": encoded_bytes.decode("ascii")}
    response = client.post("/addInstance", json=instance_dict)

    assert response.status_code == 400
    assert (
        json.loads(response.text)["detail"] == "No model ID passed as argument."
        " Please include a model ID."
    )


def test_trainFromCASes(client, xmi_bytes):
    """
    Test the /train_from_CASes endpoint with test data.

    :param client: A client for testing.
    :param xmi_bytes: A byte-encoded CAS instance.
    """
    # Change the onnx model directory for testing purposes.
    main.onnx_model_dir = "testdata"

    # I am using the addInstance endpoint here to create a CAS instance.
    # This is not optimal because this makes this test depend on this endpoint
    # but this is the most natural way to populate the features dictionary.
    encoded_bytes = base64.b64encode(xmi_bytes)
    instance_dict = {
        "modelId": "default_cas_test",
        "cas": encoded_bytes.decode("ascii"),
    }
    client.post("/addInstance", json=instance_dict)
    # Check that main.features has actually been populated.
    assert "default_cas_test" in main.features

    # The actual test of the endpoint happens here.
    instance_dict = {"modelId": "default_cas_test"}
    response = client.post("/trainFromCASes", json=instance_dict)

    # Store states to check whether the file and session object were created.
    path_exists = os.path.exists(
        os.path.join(main.onnx_model_dir, "default_cas_test.onnx")
    )
    session_stored = "default_cas_test" in main.inf_sessions

    # Change onnx model directory back and delete test file and inference
    # session object.
    if session_stored:
        del main.inf_sessions["default_cas_test"]
    if path_exists:
        os.remove(os.path.join(main.onnx_model_dir, "default_cas_test.onnx"))
    main.onnx_model_dir = "onnx_models"

    assert response.status_code == 200
    assert path_exists
    assert session_stored


def test_trainFromCASes_missing_CAS_instance(client):
    """
    Test the /train_from_CASes endpoint with missing CAS instance.

    :param client: A client for testing.
    """
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


def test_trainFromCASes_no_modelID(client):
    """
    Test the /train_from_CASes endpoint with missing model ID.

    :param client: A client for testing.
    """
    instance_dict = {"modelId": ""}
    response = client.post("/trainFromCASes", json=instance_dict)

    assert response.status_code == 400
    assert (
        json.loads(response.text)["detail"] == "No model id passed as"
        " argument. Please include a model ID"
    )


def test_train(client):
    """
    Test the /train endpoint.

    The test makes use of a randomly generated dataset.
    This way no real dataset must be revealed to git.

    :param client: A client for testing.
    """
    # Change the onnx model directory for testing purposes.
    main.onnx_model_dir = "testdata/train_data"

    instance_dict = {
        "fileName": os.path.join(main.onnx_model_dir, "random_train_data.tsv"),
        "modelId": "random_data",
    }
    response = client.post("/train", json=instance_dict)

    # Store states to check whether the file and session object were created.
    path_exists = os.path.exists(os.path.join(main.onnx_model_dir, "random_data.onnx"))
    session_stored = "random_data" in main.inf_sessions

    # Change onnx model directory back and delete test file and inference
    # session object.
    if session_stored:
        del main.inf_sessions["random_data"]
    if path_exists:
        os.remove(os.path.join(main.onnx_model_dir, "random_data.onnx"))
    main.onnx_model_dir = "onnx_models"

    # The assertions are made after the clean-up process on the basis of the
    # stored states. This ensures that cleaning is done in any case.
    assert response.status_code == 200
    assert path_exists
    assert session_stored


def test_trainFromAnswers(client, mock_instances):
    """
    Test the /trainFromAnswers endpoint.

    :param client: A client for testing.
    :param mock_instances: Mock short answer instances in dictonary
    form.
    """
    # Change the onnx model directory for testing purposes.
    main.onnx_model_dir = "testdata/train_data"

    instance_dict = {
        "instances": mock_instances,
        "modelId": "random_data",
    }
    response = client.post("/trainFromAnswers", json=instance_dict)

    # Store states to check whether the file and session object were created.
    path_exists = os.path.exists(os.path.join(main.onnx_model_dir, "random_data.onnx"))
    session_stored = "random_data" in main.inf_sessions

    # Change onnx model directory back and delete test file and inference
    # session object.
    if session_stored:
        del main.inf_sessions["random_data"]
    if path_exists:
        os.remove(os.path.join(main.onnx_model_dir, "random_data.onnx"))
    main.onnx_model_dir = "onnx_models"

    # The assertions are made after the clean-up process on the basis of the
    # stored states. This ensures that cleaning is done in any case.
    assert response.status_code == 200
    assert path_exists
    assert session_stored


# Todo: This test must be changed once the dummy implementation
#       has been replaced.
def test_extract_features(mock_instances):
    # The instance dictionaries are converted to ShortAnswerInstance objects.
    instances = [main.ShortAnswerInstance(**instance) for instance in mock_instances]

    features = extract_features(instances)

    assert list(features["item_eq_answer"]) == [1, 0, 1]

    # Test that the randomly generated values are either 0 or 1.
    for outcome in features["outcome"]:
        assert outcome in (0, 1)
