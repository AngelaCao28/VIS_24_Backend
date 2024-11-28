import numpy as np
import os
import json

import pandas as pd
import csv

from utilities import *
from action_prediction_net_new import *
from action_value_simulation import *

# find actions to simulate
# league names: ['England', 'France', 'Germany', 'Italy', 'Spain']
def find_actions_by_player(league_name, team_id, player_id):

    # load files
    league_name_transform = {'England': 'england', 'France': 'french', 'Germany': 'german', 'Italy': 'italy', 'Spain': 'spain'}

    action_file_path = './data-' + league_name_transform[league_name] + '/vaep_with_tag.json'
    all_actions = load_json_data(action_file_path)
    
    # outputs
    filtered_action_indices = []
    filtered_action_ids = []
    filtered_action_info = []

    for i in range(len(all_actions)):
        if all_actions[i]['player_id'] == player_id and all_actions[i]['team_id'] == team_id and all_actions[i]['original_event_id'] not in filtered_action_ids:
            filtered_action_indices.append(i)
            filtered_action_info.append(all_actions[i])

            if np.isnan(all_actions[i]['original_event_id']) == False:
                filtered_action_ids.append(int(all_actions[i]['original_event_id']))
            else:
                filtered_action_ids.append(0)
    
    return filtered_action_indices, filtered_action_ids, filtered_action_info

# simulate action decisions with a new player
# input: {"league_name": string, "team_id": int, "player_id": int, "player_id_new": int}
# output: {"all_actions_info": [{"id": int, "end_x_new": float, "end_y_new": float, "distance": float, "value_change": float, "value_original": float, "value_new": float}]}
# league names: ['England', 'France', 'Germany', 'Italy', 'Spain']
def simulate_action_decisions(frontend_data):

    # input and output
    league_name = frontend_data.get('LeagueName')
    team_id = frontend_data.get('TeamId')
    player_id = frontend_data.get('PlayerId')
    player_id_new = frontend_data.get('PlayerIdNew')

    league_name_transform = {'England': 'england', 'France': 'french', 'Germany': 'german', 'Italy': 'italy', 'Spain': 'spain'}

    all_simulated_actions_info = {}
    all_simulated_actions_info['all_actions_info'] = []

    action_indices, action_ids, action_info = find_actions_by_player(league_name, team_id, player_id)

    # load files
    df = pd.read_csv("./data-frames/temp4.csv")
    df = df.reset_index()
    idx_all = np.repeat(True,len(df))
    idx_simulate = np.repeat(False,len(df))

    # replace own goal "h" with goal "g"
    idx = df["act"] == "h"
    df.loc[idx,"act"] = "g"

    # set up idx2char and char2idx
    idx2char = {0: '@', 1: '_', 2: 'd', 3: 'g', 4: 'p', 5: 's', 6: 'x'}
    char2idx = {'@': 0, '_': 1, 'd': 2, 'g': 3, 'p': 4, 's': 5, 'x': 6}

    # replace characters with numbers
    df['act'].replace(char2idx,inplace=True)

    # final set of the actions
    df_player_ids = df['PLID_O']
    df_action_ids = df['ID']
    df_action_types = df['act']
    df_action_valid = df['valid_slice_flag']

    simulate_action_ids_set = []

    for i in range(len(idx_all)):
        if df_player_ids[i] == player_id and df_action_ids[i] != 'na' and df_action_types[i] not in [1, 3] and df_action_valid[i] == True and int(df_action_ids[i]) in action_ids:
            idx_simulate[i + 1] = True
            simulate_action_ids_set.append(int(df_action_ids[i]))

    filtered_action_indices = []
    filtered_action_ids = []
    filtered_action_info = []

    for i in range(len(action_ids)):
        if action_ids[i] in simulate_action_ids_set:
            filtered_action_indices.append(action_indices[i])
            filtered_action_ids.append(action_ids[i])
            filtered_action_info.append(action_info[i])

    # simulation of action choices
    simulate_dataset = SoccerDatasetSimulate(idx=idx_simulate, df=df)

    batch_size, num_workers = 1, 2
    simulate_loader = DataLoader(simulate_dataset,shuffle=False,batch_size=batch_size,num_workers=num_workers,drop_last=True)

    model = Soccer_Model_3ze().to(device)
    model.load_state_dict(torch.load('./model-files/MDLstate_2024_03_28-100645', map_location=device))
    model.eval()

    # replace with the new player
    all_players_index = load_json_data('./data-new/players_club_dict.json')
    player_index_new = all_players_index[str(player_id_new)]

    with torch.no_grad():

        for batch, (X, Y, Index, ID) in enumerate(simulate_loader):

            Index[0][1] = player_index_new

            X, Y, Index = X.to(device), Y.to(device), Index.to(device)
            _, _, pred = model(X, Index)

            end_x = float(Y[0][2]) * 105
            end_y = float(Y[0][1]) * 68
            end_x_new = float(pred[0][8]) * 105
            end_y_new = float(pred[0][7]) * 68

            current_action_info = {}

            current_action_info['id'] = int(ID[0])
            current_action_info['end_x_new'] = end_x_new
            current_action_info['end_y_new'] = end_y_new

            max_distance = np.sqrt(105 * 105 + 68 * 68)
            distance = np.sqrt((end_x_new - end_x) ** 2 + (end_y_new - end_y) ** 2) / max_distance

            current_action_info['distance'] = distance

            all_simulated_actions_info['all_actions_info'].append(current_action_info)

    # match action id to index
    action_id_to_index = {}
    action_id_to_tactic_info = {}

    offensive_category_keys = {'0': 'pass', '1': 'cross', '2': 'freekick', '3': 'corner'}
    defensive_category_keys = {'0': 'foul', '1': 'tackle', '2': 'interception'}

    for i in range(len(filtered_action_ids)):

        action_id_to_index[str(filtered_action_ids[i])] = filtered_action_indices[i]
        action_id_to_tactic_info[str(filtered_action_ids[i])] = {}

        if filtered_action_info[i]['tactic_type'] == -1 and filtered_action_info[i]['defensive_type'] == -1:
            action_id_to_tactic_info[str(filtered_action_ids[i])]['tactic_category'] = 'none'
            action_id_to_tactic_info[str(filtered_action_ids[i])]['tactic_id'] = -1
        elif filtered_action_info[i]['defensive_type'] == -1:
            if filtered_action_info[i]['tactic_type'] in [0, 1, 2, 3]:
                action_id_to_tactic_info[str(filtered_action_ids[i])]['tactic_category'] = offensive_category_keys[str(filtered_action_info[i]['tactic_type'])]
                action_id_to_tactic_info[str(filtered_action_ids[i])]['tactic_id'] = filtered_action_info[i]['tactic_cluster']
            else:
                action_id_to_tactic_info[str(filtered_action_ids[i])]['tactic_category'] = 'others'
                action_id_to_tactic_info[str(filtered_action_ids[i])]['tactic_id'] = -1
        else:
            if filtered_action_info[i]['defensive_type'] in [0, 1, 2]:
                action_id_to_tactic_info[str(filtered_action_ids[i])]['tactic_category'] = defensive_category_keys[str(filtered_action_info[i]['defensive_type'])]
                action_id_to_tactic_info[str(filtered_action_ids[i])]['tactic_id'] = filtered_action_info[i]['defensive_id']
            else:
                action_id_to_tactic_info[str(filtered_action_ids[i])]['tactic_category'] = 'others'
                action_id_to_tactic_info[str(filtered_action_ids[i])]['tactic_id'] = -1

    # load action value simulation model
    models, features, labels = train_action_value_model(league_name)

    # features extraction
    indices_to_extract = []
    simulate_x = []
    simulate_y = []

    for current_action_info in all_simulated_actions_info['all_actions_info']:
        indices_to_extract.append(action_id_to_index[str(current_action_info['id'])])
        simulate_x.append(current_action_info['end_x_new'])
        simulate_y.append((68 - current_action_info['end_y_new']))

    original_features = features.iloc[indices_to_extract]
    original_x_start = list(original_features['start_x_a0'])
    original_y_start = list(original_features['start_y_a0'])
    original_x = list(original_features['end_x_a0'])
    original_y = list(original_features['end_y_a0'])

    simulate_x = np.array(simulate_x)
    simulate_y = np.array(simulate_y)

    simulate_features = original_features.copy()
    simulate_features['end_x_a0'] = simulate_x
    simulate_features['end_y_a0'] = simulate_y

    # action value simulation
    original_score_prob = models['scores'].predict_proba(original_features)
    original_concede_prob = models['concedes'].predict_proba(original_features)
    simulate_score_prob = models['scores'].predict_proba(simulate_features)
    simulate_concede_prob = models['concedes'].predict_proba(simulate_features)

    # add to action info
    index = 0

    for current_action_info in all_simulated_actions_info['all_actions_info']:

        score_prob_change = simulate_score_prob[index][1] - original_score_prob[index][1]
        concede_prob_change = simulate_concede_prob[index][1] - original_concede_prob[index][1]
        value_change = score_prob_change - concede_prob_change

        current_action_info['value_original'] = float(original_score_prob[index][1] - original_concede_prob[index][1])
        current_action_info['value_new'] = float(simulate_score_prob[index][1] - simulate_concede_prob[index][1])
        current_action_info['value_change'] = float(value_change)
        current_action_info['tactic_category'] = action_id_to_tactic_info[str(current_action_info['id'])]['tactic_category']
        current_action_info['tactic_id'] = action_id_to_tactic_info[str(current_action_info['id'])]['tactic_id']

        current_action_info['start_x'] = original_x_start[index]
        current_action_info['start_y'] = 68 - original_y_start[index]
        current_action_info['end_x_original'] = original_x[index]
        current_action_info['end_y_original'] = 68 - original_y[index]

        index = index + 1

    print(all_simulated_actions_info['all_actions_info'][0])

    all_simulated_actions_info = json.dumps(all_simulated_actions_info)

    return all_simulated_actions_info

if __name__ == "__main__":

    # change C. Ronaldo to H. Kane
    frontend_data = {'LeagueName': 'Spain', 'TeamId': 675, 'PlayerId': 3322, 'PlayerIdNew': 8717}

    simulate_action_decisions(frontend_data)