from azureml.core import Dataset
from azureml.core import Workspace
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src_path',
        type=str,
        default="./data",
        help='Path to the source data directory'
    )
    parser.add_argument(
        '--target_path',
        type=str,
        default="MoA/proteins",
        help='Path to the target data directory in Azure ML'
    )
    parser.add_argument(
        '--name',
        type=str,
        default="moa_ds",
        help='Azure Dataset name'
    )
    parser.add_argument(
        '--description',
        type=str,
        default="cells and genetics information",
        help='Azure Dataset description'
    )
    args = parser.parse_args()
    print("===== DATA =====")
    print("SCR_PATH: " + args.src_path)
    print("TARGET_PATH: " + args.target_path)

    print("================")

    # get default workspace
    workspace = Workspace.from_config()

    # get the default datastore of the workspace
    datastore = workspace.get_default_datastore()

    # create & register weather_ds version 1 pointing to all files in the folder of week 27
    datastore.upload(src_dir=args.src_path,
                    target_path=args.target_path,
                    overwrite=True)
    datastore_path = [(datastore, args.target_path)]

    dataset = Dataset.Tabular.from_delimited_files(path=datastore_path)

    dataset.register(workspace = workspace,
                      name = args.name,
                      description = args.description,
                      create_new_version = True)