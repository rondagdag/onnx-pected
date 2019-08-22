import sys
import os
import json
import numpy as np
import pandas as pd
from azureml.core.model import Model
import onnxruntime

def init():
    global model
    
    try:
        model_path = Model.get_model_path('component_compliance')
        model_file_path = os.path.join(model_path,'component_compliance.onnx')
        print('Loading model from: ', model_file_path)
        
        # Load the ONNX model
        model = onnxruntime.InferenceSession(model_file_path)
        print('Model loaded...')
    except Exception as e:
        print(e)
        
# note you can pass in multiple rows for scoring
def run(raw_data):
    try:
        print("Received input: ", raw_data)
        
        input_data = np.array(json.loads(raw_data)).astype(np.float32)
        
        # Run an ONNX session to classify the input.
        result = model.run(None, {model.get_inputs()[0].name:input_data})[0]
        result = result[0][0].item()
        
        # return just the classification index (0 or 1)
        return result
    except Exception as e:
        error = str(e)
        return error
