from azureml.core import Model
from azureml.contrib.services.aml_request import AMLRequest, rawhttp
from azureml.contrib.services.aml_response import AMLResponse
import joblib
import numpy as np
import json

def init():
    global model
    model_path = Model.get_model_path("heart_attack_model")
    print("Model Path is  ", model_path)
    model = joblib.load(model_path)

@rawhttp
def run(request):
  if request.method == 'GET':
        try:
            data = validate_input_args(request.args)
            array_data = np.expand_dims(np.array(data), axis=0)
            yres = model.predict(array_data)
            response = {'data' : yres.tolist() , 'message' : "Successfully  classified"}
            return AMLResponse(json.dumps(response), 200)
        except ValueError as e:
            print(e)
            error_json = {'error' : 'missing_parameters' ,'detail' : e.args}
            return AMLResponse(json.dumps(error_json), 404)
        except Exception as e:
            print(e)
            return AMLResponse("Server error", 500)
  else:
      return AMLResponse("Unsopported method", 405)

def validate_input_args(args):
    required_params = ['age','sex','cp','trtbps','chol','fbs','restecg','thalachh','exng','oldpeak','slp','caa','thall']
    return_list = []
    missing_params = []
    for key in required_params:
        if key in args.keys():
            return_list.append(float(args[key]))
        else:
            missing_params.append(key)
    if len(missing_params) > 0:
        raise ValueError(missing_params)
    else:
        return return_list