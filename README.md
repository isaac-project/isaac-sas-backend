# ISAAC Short Answer Scoring Backend

This component started out as a fork of sklearnflask, intended to be used as ML backend for UIMA-based Short Answer Assessment in the ISAAC project (https://www.uni-tuebingen.de/isaac). It evolved into a fully-fledged backend for Short Answer Scoring that can be used as a service with arbitrary frontends over HTTP.

## HTTP endpoints

The following endpoints currently exist (more detailed documentation at /docs on the running service):

/fetchStoredModels (GET) - fetch IDs of currently stored SAS models
/trainFromAnswers (POST) - train a new model based on submitted answer data
/predictFromAnswers (POST) - predict score of submitted answers
/wipe\_models (GET) - remove all models from service

## Docker setup

The service can be built and deployed as a container using the supplied Dockerfile. Run the following from the repo root folder to build the image:

```
docker build -t isaac/sas-backend ./
```

To run the image in a container:

```
docker run --name isaac-sas -p 80:80 ramonziai/isaac-ml-service
```

## Manual/development setup

### Installing dependencies

```
pip install -r requirements.txt
```

### Running the development server

A possible server for development is ```uvicorn```:
```
pip install uvicorn
``` 

The service can be run with the following command:
```
uvicorn main:app --port 9999 --reload
```
The ```--port``` number can be set to any free port of the developer's choice.  
For development purposes it is convenient to make use of the ```--reload``` 
flag to automatically restart the service after changes in the code.


