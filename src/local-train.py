import os
import numpy as np
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score
from train_helper import validate_data, split_data, train_model
from sklearn.metrics import log_loss, balanced_accuracy_score
from azureml.core import Run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        default="./data/",
        help='Path to the training data'
    )
    parser.add_argument(
        '--file_name',
        type=str,
        default="dataset.csv",
        help='Filename'
    )
    args = parser.parse_args()
    print("===== DATA =====")
    print("DATA PATH: " + args.data_path)
    print("FILE NAME: " + args.file_name)
    print("LIST FILES IN DATA PATH...")
    print(os.listdir(args.data_path))
    print("================")

    data = pd.read_csv(args.data_path + args.file_name)
    datos = validate_data(data)
    X_train, y_train, X_test, y_test = split_data(datos)
    model = train_model(X_train, y_train, save=True)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    print(f"balanced_accuracy: {balanced_accuracy_score(y_test, y_pred)}")
    print(f"log_loss: {log_loss(y_test, y_prob)}")