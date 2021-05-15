from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core import Model
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()  
    parser.add_argument(
        '--criterion',
        type=str,
        default="entropy",
        help='criterio de solución'
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Entero aleatorio'
    )
    parser.add_argument(
        '--class_weight',
        type=int,
        default='balanced',
        help='data balanceada'
    )
    args = parser.parse_args()

    ws = Workspace.from_config()
    experiment = Experiment(workspace=ws, name='cloud-moa-prediction')

    config = ScriptRunConfig(
        source_directory='./src',
        script='remote-train.py',
        compute_target='cpu-cluster',
        arguments=['--criterion', args.criterion,'--random_state', args.random_state, '--class_weight', args.class_weight]
    )
    # set up pytorch environment
    env = Environment.from_conda_specification(
        name='sklearn-remote-env',
        file_path='./azure-config/compute-env-config.yml'
    )
    config.run_config.environment = env

    run = experiment.submit(config)
    aml_url = run.get_portal_url()
    print("Submitted to compute cluster. Click link below")
    print("")
    print(aml_url)
    run.wait_for_completion(show_output=True)
    run.register_model( model_name='moa_prediction_model',
                    model_path='outputs/RandomForestClassifier.pkl', # run outputs path
                    description='Random Forest for MoA prediction',
                    tags={'data-format': 'CSV'},
                    model_framework=Model.Framework.SCIKITLEARN,
                    model_framework_version='0.20.3')



    