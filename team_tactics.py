import numpy as np
import os
import json

import pandas as pd
import csv

from utilities import *

# get tactic information of a certain team
# input: {"league_name": string, "team_id": int, "tactic_type": string, "tactic_category": string}
# output:{"all_tactics_info": [{"category": string, "id": int, "seq": [int], "success_rate": float, "use_rate": float, "score_by_region": [float]}]}
# league names: ['England', 'France', 'Germany', 'Italy', 'Spain']
# tactic types: ['offensive', 'defensive']
# tactic categories: offensive: ['pass', 'freekick', 'corner'], defensive: ['foul', 'tackle', 'interception']
def get_all_tactics_info(frontend_data):

    # input and output
    league_name = frontend_data.get('LeagueName')
    team_id = frontend_data.get('TeamId')
    tactic_type = frontend_data.get('TacticType')
    tactic_category = frontend_data.get('TacticCategory')

    league_name_transform = {'England': 'england', 'France': 'french', 'Germany': 'german', 'Italy': 'italy', 'Spain': 'spain'}

    all_tactics_info_by_category = {}
    all_tactics_info_by_category['all_tactics_info'] = []

    # load files
    tactic_file_path = './data-' + league_name_transform[league_name] + '/tactics_' + tactic_type + '.json'
    action_file_path = './data-' + league_name_transform[league_name] + '/vaep_with_tag.csv'
    match_file_path = './data/matches_' + league_name + '.json'

    all_tactics = load_json_data(tactic_file_path)
    all_actions = pd.read_csv(action_file_path)
    all_matches = load_json_data(match_file_path)

    # get tactics info
    all_team_tactics = all_tactics[str(team_id)]
    category_keys = []

    if tactic_category == 'pass':
        category_keys = ['pass', 'cross']
    elif tactic_category == 'corner':
        category_keys = ['cornor']
    else:
        category_keys = [str(tactic_category)]

    for category in all_team_tactics.keys():

        if category in category_keys:

            current_category_tactics = all_team_tactics[category]

            for current_tactic_id in current_category_tactics.keys():

                current_tactic_info = current_category_tactics[current_tactic_id]
                
                if current_tactic_id in ['success_rate', 'use_rate']:
                    continue
                
                current_tactic_info_additional = current_tactic_info.copy()

                current_tactic_info_additional['category'] = category
                current_tactic_info_additional['id'] = int(current_tactic_id)
                current_tactic_info_additional['score_by_region'] = [0,0,0,0,0,0,0,0,0]

                all_tactics_info_by_category['all_tactics_info'].append(current_tactic_info_additional)

    # aggregate action values by regions
    offensive_category_keys = {'pass': 0, 'cross': 1, 'freekick': 2, 'cornor': 3}
    defensive_category_keys = {'foul': 0, 'tackle': 1, 'interception': 2}

    tactic_categories = []
    tactic_ids = []

    if tactic_type == 'offensive':
        tactic_categories = all_actions['tactic_type']
        tactic_ids = all_actions['tactic_cluster']
    else:
        tactic_categories = all_actions['defensive_type']
        tactic_ids = all_actions['defensive_id']

    match_ids = all_actions['game_id']
    team_ids = all_actions['team_id']
    action_values = all_actions['vaep_value']
    start_x = all_actions['start_x']
    start_y = all_actions['start_y']

    for i in range(len(team_ids)):

        if team_ids[i] == team_id:

            for current_tactic_info in all_tactics_info_by_category['all_tactics_info']:

                current_tactic_category = current_tactic_info['category']
                current_tactic_id = current_tactic_info['id']

                if tactic_type == 'offensive':
                    tactic_category_key = offensive_category_keys[current_tactic_category]
                else:
                    tactic_category_key = defensive_category_keys[current_tactic_category]

                if tactic_categories[i] == tactic_category_key and tactic_ids[i] == current_tactic_id:

                    for match in all_matches:
                        if match['wyId'] == match_ids[i] and match['teamsData'][str(team_id)]['side'] == 'home':
                            is_home = 1
                            break
                        elif match['wyId'] == match_ids[i] and match['teamsData'][str(team_id)]['side'] == 'away':
                            is_home = 0
                            break

                    if is_home == 1:
                        action_region_index = pitch_division(start_x[i], start_y[i])
                    else:
                        action_region_index = pitch_division(105 - start_x[i], 68 - start_y[i])

                    current_tactic_info['score_by_region'][action_region_index - 1] = current_tactic_info['score_by_region'][action_region_index - 1] + action_values[i]

    print(all_tactics_info_by_category['all_tactics_info'])

    all_tactics_info_by_category = json.dumps(all_tactics_info_by_category)

    return all_tactics_info_by_category

if __name__ == "__main__":

    frontend_data = {'LeagueName': 'Spain', 'TeamId': 675, 'TacticType': 'defensive', 'TacticCategory': 'tackle'}

    get_all_tactics_info(frontend_data)