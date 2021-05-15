import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, balanced_accuracy_score
from train_helper import validate_data, split_data, train_model
from azureml.core import Run, Dataset


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

    run = Run.get_context()
    ws = run.experiment.workspace

    datastore = ws.get_default_datastore()
    input_ds = Dataset.get_by_name(ws, 'moa_ds')
    data = input_ds.to_pandas_dataframe()

    dataframe = validate_data(data)
    X_train, X_test, y_train, y_test = split_data(dataframe)
    model = train_model(X_train, y_train, save=True, criterion=args.criterion,random_state=args.random_state, class_weight= args.class_weight)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    print(f"balanced_accuracy: {balanced_accuracy_score(y_test, y_pred)}")
    print(f"log_loss: {log_loss(y_test, y_prob)}")
    run.log('balanced_accuracy', balanced_accuracy_score(y_test, y_pred))