import numpy as np
import os
import json

import pandas as pd
import csv

import socceraction.vaep.features as fs
import socceraction.vaep.labels as lab
import xgboost

from sklearn.metrics import roc_auc_score

from utilities import *

# train action value model
# league names: ['England', 'France', 'Germany', 'Italy', 'Spain']
def train_action_value_model(league_name):

    league_name_transform = {'England': 'england', 'France': 'french', 'Germany': 'german', 'Italy': 'italy', 'Spain': 'spain'}

    # load files
    action_file_path = './data-' + league_name_transform[league_name] + '/spadl-' + league_name_transform[league_name] + '.h5'
    features_file_path = './data-' + league_name_transform[league_name] + '/features.h5'
    labels_file_path = './data-' + league_name_transform[league_name] + '/labels.h5'

    matches = pd.read_hdf(action_file_path, "games")
    matches = matches.sort_values(['game_id']).reset_index(drop=True)
    
    # select feature set
    xfns = [
        fs.actiontype,
        fs.actiontype_onehot,
        fs.result,
        fs.result_onehot,
        fs.goalscore,
        fs.startlocation,
        fs.endlocation,
        fs.team,
    ]
    nb_prev_actions = 3

    Xcols = fs.feature_column_names(xfns, nb_prev_actions)
    features = []

    for match_id in matches.game_id:
        current_features = pd.read_hdf(features_file_path, f"game_{match_id}")
        features.append(current_features[Xcols])

    features = pd.concat(features).reset_index(drop=True)

    Ycols = ["scores","concedes"]
    labels = []

    for match_id in matches.game_id:
        current_labels = pd.read_hdf(labels_file_path, f"game_{match_id}")
        labels.append(current_labels[Ycols])

    labels = pd.concat(labels).reset_index(drop=True)

    # train the classifiers
    predicted_labels = pd.DataFrame()
    models = {}

    for col in list(labels.columns):
        model = xgboost.XGBClassifier(n_estimators=50, max_depth=3, n_jobs=-3, verbosity=1, use_label_encoder=False)
        model.fit(features, labels[col])
        models[col] = model

    # evaluate the model
    for col in list(labels.columns):
        predicted_labels[col] = [p[1] for p in models[col].predict_proba(features)]
        accuracy = roc_auc_score(labels[col], predicted_labels[col])
        print(col)
        print(accuracy)

    return models, features, labels

if __name__ == "__main__":

    train_action_value_model('Spain')