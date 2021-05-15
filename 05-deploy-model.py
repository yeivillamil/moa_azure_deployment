from azureml.core import Environment
from azureml.core.model import InferenceConfig
from azureml.core import Workspace
from azureml.core import Model
from azureml.core.webservice import LocalWebservice
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.webservice import AciWebservice
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()      
    parser.add_argument(
        '--model_name',
        type=str,
        default="moa_prediction_model",
        help='Nombre del modelo en AzureML'
    )
    parser.add_argument(
        '--model_version',
        type=int,
        default=1,
        help='Versión del modelo en AzureML'
    )
    args = parser.parse_args()
    deployment_config = AciWebservice.deploy_configuration(cpu_cores=0.1, 
                                                  memory_gb=0.5, 
                                                  tags={"data": "Hearth",  "method" : "sklearn"}, 
                                                  description='Predict MoA activation with sklearn')

    # Despliegue local mediante docker
    #deployment_config = LocalWebservice.deploy_configuration(port=6789)

    ws = Workspace.from_config()
    model = Model(ws, args.model_name, version=args.model_version)
                      
    print(model.name, model.id, model.version, sep='\t')

    # Creciación de azure constainer con ambiente gestionado mediante conda.
    myenv = Environment(name="myenv")
    # Se habilita docker
    myenv.docker.enabled = True
    # Definición de dependencias de docker.
    myenv.python.conda_dependencies = CondaDependencies.create(conda_packages=['scikit-learn'],
                                                              pip_packages=['azureml-defaults', 'numpy', 'pandas'])
                                                              
    inf_config = InferenceConfig(environment=myenv, source_directory='./src', entry_script='entry.py')
    service = Model.deploy(ws, "MoA-webservice", [model], inf_config, deployment_config)
    service.wait_for_deployment(show_output=True)
    print(service.get_logs())