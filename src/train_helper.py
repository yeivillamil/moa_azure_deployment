import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.metrics import log_loss, balanced_accuracy_score
import pickle

def validate_data(dataset):
  X = dataset.drop(columns=["nfkb_inhibitor",'sig_id','cp_type','cp_time','cp_dose'], axis=1)
  y = dataset["nfkb_inhibitor"]
  data = pd.concat([X, y], axis=1)
  return data


def split_data(data):
  train_x = data.iloc[:,1:873]
  train_y = data.iloc[:,873:874]
  return iterative_train_test_split(np.array(train_x), np.array(train_y), test_size=0.2)

def train_model(X_train, y_train, save=False, criterion='entropy', class_weight='balanced', random_state=42):
  rf_model = RandomForestClassifier(criterion=criterion, class_weight= class_weight, random_state=random_state)
  rf_model.fit(X_train, np.ravel(y_train))
  if save:
    save_model(rf_model, 'outputs/RandomForestClassifier.pkl')
  return rf_model

def save_model(model, save_path='outputs/model.pkl'):
  with open(save_path, 'wb') as file:  
    pickle.dump(model, file)



