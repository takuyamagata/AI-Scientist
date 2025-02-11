import argparse
import inspect
import json
import math
import pickle
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, \
                            ConfusionMatrixDisplay, \
                            accuracy_score, \
                            f1_score, \
                            recall_score, \
                            precision_score


def get_classifier_grid():
    # Create cross-validation partitions from training
    # This should select the best set of parameters
    cv = StratifiedKFold(n_splits=5, shuffle=False)
    clf = RandomForestClassifier()
    param_grid = {'n_estimators' : [200, 250, 300, 500],
                  'min_samples_leaf': [5, 10, 20]}
    clf_grid = GridSearchCV(clf, 
                            param_grid=param_grid, 
                            cv=cv, 
                            refit=True,
                )
    return clf_grid

def split_train_test(X, y, partition=0):
    # Create train and test partitions
    skf = StratifiedKFold(n_splits=5, shuffle=False)
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        if i == partition:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
    return (X_train, y_train), (X_test, y_test)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_csv_file", 
        type=str, 
        default=os.path.join(
            os.path.dirname(__file__), 
            '../../data/activity_recognition/activity_recognition.csv'
        )
    )
    parser.add_argument("--out_dir", type=str, default="run_0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int)
    config = parser.parse_args()

    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir)

    # Load the dataset
    dataset_name = 'SPHERE_Challenge' # at moment, only SPHERE_Challenge is supported
    if not os.path.exists(config.dataset_csv_file):
        raise FileNotFoundError(
            "Dataset file not found. Please run prepare.py first."
        )
    df = pd.read_csv(config.dataset_csv_file)
    features_id = ['x','y','z','Kitchen_AP', 'Lounge_AP', 'Upstairs_AP', 'Study_AP']
    data = df[features_id].values
    labels = df['target'].values
    (X_train, y_train), (X_test, y_test) = split_train_test(data, labels)

    # get the classifier grid
    clf_grid = get_classifier_grid()
    
    # train the model
    start_time = time.time()
    clf_grid.fit(X_train, y_train)
    end_time = time.time()

    # predict the test data
    start_inf_time = time.time()
    y_pred = clf_grid.predict(X_test)
    end_inf_time = time.time()

    # post-process the prediction results
    labels = [id for id in list(df) if 'a_' in id or 'p_' in id]
    y_pred_str = [labels[int(i)] for i in y_pred]
    y_test_str = [labels[int(i)] for i in y_test]

    # measure the performance
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test_str, y_pred_str, labels=labels)    
    
    # plot confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test_str, y_pred_str, labels=labels
    )
    
    all_results = {}
    all_results[dataset_name] = {
        'accuracy': accuracy,
        'f1': f1,
        'recall': recall,
        'precision': precision,
        'confusion_matrix': cm.tolist(),
        'labels': labels,
        'best_params': clf_grid.best_params_,
    }

    final_infos = {}
    final_infos[dataset_name] = {
            "means": {
                'training_time': end_time - start_time,
                'inferece_time': end_inf_time - start_inf_time,
                'accuracy': accuracy,
                'f1': f1,
                'confusion_matrix': cm.tolist(),
            }
        }
    
    with open(os.path.join(config.out_dir, "final_info.json"), "w") as f:
        json.dump(final_infos, f)

    with open(os.path.join(config.out_dir, "all_results.pkl"), "wb") as f:
        pickle.dump(all_results, f)