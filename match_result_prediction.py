import numpy as np
import os
import json

import pandas as pd
import csv

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from utilities import *

# generate player feature dict
def generate_player_feature_dict():

    all_players_info_embedding = load_json_data('./data-frontend/all_players_info_embedding.json')
    player_features_dict = {}

    for current_player_info in all_players_info_embedding:

        player_features_dict[str(current_player_info['id'])] = {}

        player_features_dict[str(current_player_info['id'])]['market_value'] = current_player_info['market_value']
        player_features_dict[str(current_player_info['id'])]['action_values'] = current_player_info['action_values']
        player_features_dict[str(current_player_info['id'])]['player_emb'] = current_player_info['player_emb']

    print(len(player_features_dict.keys()))

    player_features_dict = json.dumps(player_features_dict)

    f = open('./data-frontend/player_features_dict.json', 'w')
    f.write(player_features_dict)
    f.close()

# train match result prediction model
# league names: ['England', 'France', 'Germany', 'Italy', 'Spain']
def train_match_prediction_model(league_name):

    all_features = []
    all_labels = []

    player_features_dict = load_json_data('./data-frontend/player_features_dict.json')

    file_path = './data/matches_' + league_name + '.json'
    all_matches = load_json_data(file_path)

    for match in all_matches:

        teams_keys = match['teamsData'].keys()

        for team in teams_keys:

            current_feature = []
            current_label = 0

            players_lineup = match['teamsData'][team]['formation']['lineup']
            players_lineup_list = []

            for player in players_lineup:
                players_lineup_list.append(player['playerId'])

            for current_player_id in players_lineup_list:

                if str(current_player_id) not in player_features_dict.keys():
                    # print(current_player_id)
                    current_player_feature = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                else:
                    current_embedding = player_features_dict[str(current_player_id)]['player_emb']
                    current_market_value = player_features_dict[str(current_player_id)]['market_value']
                    current_action_values = player_features_dict[str(current_player_id)]['action_values']

                    current_player_feature = current_embedding.copy()
                    current_player_feature.append(current_market_value)
                    current_player_feature.append(current_action_values)

                current_feature = current_feature + current_player_feature

            if match['winner'] == int(team):
                current_label = 2
            elif match['winner'] == 0:
                current_label = 1

            all_features.append(current_feature)
            all_labels.append(current_label)

    all_features = np.array(all_features)
    all_labels = np.array(all_labels)

    clf = RandomForestClassifier(max_depth=5, random_state=0)
    clf.fit(all_features, all_labels)

    all_predicted_labels = clf.predict(all_features)

    accuracy = clf.score(all_features, all_labels)
    f1_score_value = f1_score(all_labels, all_predicted_labels, average='macro')

    print(accuracy)
    print(f1_score_value)

    return clf

# predict team performance when players join a new team
# input: {"league_name": string, "team_id": int, "player_id": int, "player_role": string}
# output: {"all_players_info": [{"similarity": float, "curr_team_name": string, "team_similarity": float, "winning_rate_increment": float}]
#          "age_range": [int, int], "market_value_range": [float, float], "team_similarity_range": [float, float], "increment_range": [float, float], "action_value_range":[float, float], "appearance_range": [int, int]}
# league names: ['England', 'France', 'Germany', 'Italy', 'Spain']
def get_all_candidate_players_info(frontend_data):

    # load files
    all_players_info = load_json_data('./data-frontend/all_players_info_embedding.json')

    # input and output
    league_name = frontend_data.get('LeagueName')
    team_id = frontend_data.get('TeamId')
    player_id = frontend_data.get('PlayerId')
    player_role = frontend_data.get('PlayerRole')

    filtered_player_info = {}
    filtered_player_info['all_players_info'] = []

    # filter players by roles
    for current_players_info in all_players_info:

        if current_players_info['role_detail'] == player_role and current_players_info['curr_team_id'] != team_id:

            current_players_info_copy = current_players_info.copy()

            current_players_info_copy['similarity'] = 0
            current_players_info_copy['curr_team_name'] = 0
            current_players_info_copy['team_similarity'] = 0
            current_players_info_copy['winning_rate_increment'] = 0

            filtered_player_info['all_players_info'].append(current_players_info_copy)

    print(len(filtered_player_info['all_players_info']))

    # calculate winning rate increment
    clf = train_match_prediction_model(league_name)

    file_path = './data/matches_' + league_name + '.json'
    all_matches = load_json_data(file_path)

    player_features_dict = load_json_data('./data-frontend/player_features_dict.json')

    match_numbers = 0

    for match in all_matches:

        if str(team_id) in match['teamsData'].keys():

            players_lineup = match['teamsData'][str(team_id)]['formation']['lineup']
            players_lineup_list = []

            for player in players_lineup:
                players_lineup_list.append(player['playerId'])

            if player_id in players_lineup_list:

                match_numbers = match_numbers + 1

                current_feature = []

                for current_player_id in players_lineup_list:

                    if str(current_player_id) not in player_features_dict.keys():
                        # print(current_player_id)
                        current_player_feature = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                    else:
                        current_embedding = player_features_dict[str(current_player_id)]['player_emb']
                        current_market_value = player_features_dict[str(current_player_id)]['market_value']
                        current_action_values = player_features_dict[str(current_player_id)]['action_values']

                        current_player_feature = current_embedding.copy()
                        current_player_feature.append(current_market_value)
                        current_player_feature.append(current_action_values)

                    current_feature = current_feature + current_player_feature

                current_feature = np.array(current_feature)

                predict_result_initial = clf.predict_proba([current_feature])
                winning_rate_initial = predict_result_initial[0][2]

                all_simulate_features = []

                for current_players_info in filtered_player_info['all_players_info']:

                    simulate_player_id = current_players_info['id']
                    simulate_feature = []

                    for current_player_id in players_lineup_list:

                        if str(current_player_id) not in player_features_dict.keys():
                            # print(current_player_id)
                            current_player_feature = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
                        elif current_player_id != player_id:
                            current_embedding = player_features_dict[str(current_player_id)]['player_emb']
                            current_market_value = player_features_dict[str(current_player_id)]['market_value']
                            current_action_values = player_features_dict[str(current_player_id)]['action_values']

                            current_player_feature = current_embedding.copy()
                            current_player_feature.append(current_market_value)
                            current_player_feature.append(current_action_values)
                        else:
                            current_embedding = player_features_dict[str(simulate_player_id)]['player_emb']
                            current_market_value = player_features_dict[str(simulate_player_id)]['market_value']
                            current_action_values = player_features_dict[str(simulate_player_id)]['action_values']

                            current_player_feature = current_embedding.copy()
                            current_player_feature.append(current_market_value)
                            current_player_feature.append(current_action_values)

                        simulate_feature = simulate_feature + current_player_feature

                    all_simulate_features.append(simulate_feature)

                all_simulate_features = np.array(all_simulate_features)
                predict_result_simulate = clf.predict_proba(all_simulate_features)

                player_index = 0

                for current_players_info in filtered_player_info['all_players_info']:

                    winning_rate_simulate = predict_result_simulate[player_index][2]
                    winning_rate_increment = winning_rate_simulate - winning_rate_initial

                    current_players_info['winning_rate_increment'] = current_players_info['winning_rate_increment'] + winning_rate_increment

                    player_index = player_index + 1

    for current_players_info in filtered_player_info['all_players_info']:
        current_players_info['winning_rate_increment'] = current_players_info['winning_rate_increment'] / match_numbers

    # calculate player and team similarities and team name
    player_embedding_vector = []
    team_embedding_vector= []

    for current_player_info in all_players_info:
        if current_player_info['id'] == player_id:
            player_embedding_vector = current_player_info['player_emb'].copy()
            team_embedding_vector = current_player_info['team_emb'].copy()
            break

    player_embedding_vector = np.array(player_embedding_vector)
    team_embedding_vector = np.array(team_embedding_vector)

    for current_player_info in filtered_player_info['all_players_info']:
        
        current_player_embedding_vector = current_player_info['player_emb'].copy()
        current_team_embedding_vector = current_player_info['team_emb'].copy()

        current_player_embedding_vector = np.array(current_player_embedding_vector)
        current_team_embedding_vector = np.array(current_team_embedding_vector)

        similarity = np.linalg.norm(current_player_embedding_vector - player_embedding_vector)
        team_similarity = np.linalg.norm(current_team_embedding_vector - team_embedding_vector)

        current_player_info['similarity'] = similarity
        current_player_info['team_similarity'] = team_similarity

        current_team_id = current_player_info['curr_team_id']
        current_team_name = get_team_name_by_id(current_team_id)

        current_player_info['curr_team_name'] = current_team_name

    # calculate ranges
    filtered_player_info['age_range'] = [filtered_player_info['all_players_info'][0]['age'], filtered_player_info['all_players_info'][0]['age']]
    filtered_player_info['market_value_range'] = [filtered_player_info['all_players_info'][0]['market_value'], filtered_player_info['all_players_info'][0]['market_value']]
    filtered_player_info['team_similarity_range'] = [filtered_player_info['all_players_info'][0]['team_similarity'], filtered_player_info['all_players_info'][0]['team_similarity']]
    filtered_player_info['increment_range'] = [filtered_player_info['all_players_info'][0]['winning_rate_increment'], filtered_player_info['all_players_info'][0]['winning_rate_increment']]
    filtered_player_info['action_value_range'] = [filtered_player_info['all_players_info'][0]['action_values_per_90_min'], filtered_player_info['all_players_info'][0]['action_values_per_90_min']]
    filtered_player_info['appearance_range'] = [filtered_player_info['all_players_info'][0]['appearing_time'], filtered_player_info['all_players_info'][0]['appearing_time']]

    for current_player_info in filtered_player_info['all_players_info']:

        filtered_player_info['age_range'][0] = min(filtered_player_info['age_range'][0], current_player_info['age'])
        filtered_player_info['age_range'][1] = max(filtered_player_info['age_range'][1], current_player_info['age'])

        filtered_player_info['market_value_range'][0] = min(filtered_player_info['market_value_range'][0], current_player_info['market_value'])
        filtered_player_info['market_value_range'][1] = max(filtered_player_info['market_value_range'][1], current_player_info['market_value'])

        filtered_player_info['team_similarity_range'][0] = min(filtered_player_info['team_similarity_range'][0], current_player_info['team_similarity'])
        filtered_player_info['team_similarity_range'][1] = max(filtered_player_info['team_similarity_range'][1], current_player_info['team_similarity'])

        filtered_player_info['increment_range'][0] = min(filtered_player_info['increment_range'][0], current_player_info['winning_rate_increment'])
        filtered_player_info['increment_range'][1] = max(filtered_player_info['increment_range'][1], current_player_info['winning_rate_increment'])

        filtered_player_info['action_value_range'][0] = min(filtered_player_info['action_value_range'][0], current_player_info['action_values_per_90_min'])
        filtered_player_info['action_value_range'][1] = max(filtered_player_info['action_value_range'][1], current_player_info['action_values_per_90_min'])

        filtered_player_info['appearance_range'][0] = min(filtered_player_info['appearance_range'][0], current_player_info['appearing_time'])
        filtered_player_info['appearance_range'][1] = max(filtered_player_info['appearance_range'][1], current_player_info['appearing_time'])

    print(filtered_player_info['all_players_info'][0])
    print(filtered_player_info['increment_range'])

    filtered_player_info = json.dumps(filtered_player_info)

    return filtered_player_info

# get team name by team id
def get_team_name_by_id(team_id):

    team_name = 0
    all_teams = load_json_data('./data/teams.json')

    for team in all_teams:
        if team['wyId'] == team_id:
            team_name = team['name']
            break

    return team_name

if __name__ == "__main__":

    frontend_data = {'LeagueName': 'Spain', 'TeamId': 675, 'PlayerId': 3322, 'PlayerRole': 'Centre-Forward'}

    get_all_candidate_players_info(frontend_data)