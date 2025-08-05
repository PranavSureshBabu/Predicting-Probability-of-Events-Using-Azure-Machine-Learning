import json
import os
import joblib
import numpy as np
import pandas as pd

# Called once when the container starts
def init():
    global model
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"),   # /var/azureml-app/azureml-models/profit-regressor/1
        "model.pkl"
    )
    model = joblib.load(model_path)

# Called for every invocation
def run(raw_data):
    data = pd.read_json(raw_data)
    preds = model.predict(data)
    return json.dumps(preds.tolist())
