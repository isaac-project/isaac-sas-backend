# Machine Learning backend for the ISAAC system

This is a fork of sklearnflask, intended to be used as ML backend for UIMA-based Short Answer Assessment in the ISAAC project (https://www.uni-tuebingen.de/isaac).

### Dependencies
- scikit-learn
- fastapi
- uvicorn
- pandas
- numpy
- dkpro-cassis

```
pip install -r requirements.txt
```

### Running

For development you can run the service with the following command:
```
uvicorn main:app --port 9999 --reload
```
The ```--port``` number can be set to any free port of the developer's choice.  
For development purposes it is convenient to make use of the ```--reload``` 
flag to automatically restart the service after changes in the code.
