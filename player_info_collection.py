import numpy as np
import os
import json

import pandas as pd
import csv

from utilities import *

# generate basic player info
# id, age, height, weight, current team, market value
def generate_player_info_basic():

    all_players_index_file = load_json_data('./data-new/players_club_dict.json')
    all_players_index = all_players_index_file.keys()

    all_players_info = load_json_data('./data/players.json')
    all_players_info_new = load_json_data('./data-new/players_with_position.json')

    all_players_info_basic = []

    for player_index in all_players_index:

        current_player_info_basic = {}
        current_player_info_basic['id'] = int(player_index)

        for current_player_info in all_players_info:

            if current_player_info['wyId'] == int(player_index):

                current_player_info_basic['name'] = current_player_info['shortName']
                current_player_info_basic['role'] = current_player_info['role']['name']
                current_player_info_basic['height'] = current_player_info['height']
                current_player_info_basic['weight'] = current_player_info['weight']
                current_player_info_basic['curr_team_id'] = current_player_info['currentTeamId']

                birth_date = current_player_info['birthDate'].split('-')
                birth_year = int(birth_date[0])
                age = 2018 - birth_year

                current_player_info_basic['birth_date'] = current_player_info['birthDate']
                current_player_info_basic['age'] = age

        current_player_info_basic['role_detail'] = 'null'
        current_player_info_basic['market_value'] = 0

        for current_player_info_new in all_players_info_new:

            if current_player_info_new['wyId'] == int(player_index):

                current_player_info_basic['role_detail'] = current_player_info_new['position']
                market_value_str = current_player_info_new['value']

                if market_value_str != '-':
                    if market_value_str[len(market_value_str) - 1] == 'm':
                        market_value = float(market_value_str.split('m')[0])
                    else:
                        market_value = float(market_value_str.split('k')[0]) / 1000

                    current_player_info_basic['market_value'] = market_value

        all_players_info_basic.append(current_player_info_basic)

    print(len(all_players_info_basic))

    all_players_info_basic = json.dumps(all_players_info_basic)

    f = open('./data-frontend/all_players_info_basic.json', 'w')
    f.write(all_players_info_basic)
    f.close()

# generate player historical performance
# action numbers, total action values
def generate_player_info_performance():

    all_players_index_file = load_json_data('./data-new/players_club_dict.json')
    all_players_index = all_players_index_file.keys()

    all_players_action_info = {}

    league_names = ['england', 'french', 'german', 'italy', 'spain']

    for league_name in league_names:

        file_path = './data-' + league_name + '/vaep_with_tag.csv'
        df = pd.read_csv(file_path)

        player_ids = df['player_id']
        action_values = df['vaep_value']

        for i in range(len(player_ids)):

            current_player_id = player_ids[i]
            current_action_value = action_values[i]

            if str(current_player_id) in all_players_index:

                if str(current_player_id) not in all_players_action_info.keys():
                    all_players_action_info[str(current_player_id)] = {'numbers': 0, 'values': 0}

                all_players_action_info[str(current_player_id)]['numbers'] = all_players_action_info[str(current_player_id)]['numbers'] + 1
                all_players_action_info[str(current_player_id)]['values'] = all_players_action_info[str(current_player_id)]['values'] + current_action_value

    print(len(all_players_action_info.keys()))

    all_players_action_info = json.dumps(all_players_action_info)

    f = open('./data-frontend/all_players_action_info.json', 'w')
    f.write(all_players_action_info)
    f.close()

# generate player historical performance
# appearing time
def generate_player_info_time():

    all_players_index_file = load_json_data('./data-new/players_club_dict.json')
    all_players_index = all_players_index_file.keys()

    all_players_time_info = {}

    league_names = ['England', 'France', 'Germany', 'Italy', 'Spain']

    for league_name in league_names:

        file_path = './data/matches_' + league_name + '.json'
        all_matches = load_json_data(file_path)

        for match in all_matches:

            teams_keys = match['teamsData'].keys()

            for team in teams_keys:

                players_lineup = match['teamsData'][team]['formation']['lineup']
                players_substitutions = match['teamsData'][team]['formation']['substitutions']

                if players_substitutions == 'null':
                    players_substitutions = []

                players_lineup_list = []
                players_in_list = []
                players_out_list = []

                for player in players_lineup:
                    players_lineup_list.append(player['playerId'])

                for player_change in players_substitutions:
                    players_in_list.append(player_change['playerIn'])
                    players_out_list.append(player_change['playerOut'])

                for current_player_id in players_lineup_list:

                    if str(current_player_id) in all_players_index:

                        if str(current_player_id) not in all_players_time_info.keys():
                            all_players_time_info[str(current_player_id)] = {'time': 0}

                        if current_player_id not in players_out_list:
                            all_players_time_info[str(current_player_id)]['time'] = all_players_time_info[str(current_player_id)]['time'] + 90
                        else:
                            out_time = 0
                            for player_change in players_substitutions:
                                if player_change['playerOut'] == current_player_id:
                                    out_time = player_change['minute']
                                    break
                            all_players_time_info[str(current_player_id)]['time'] = all_players_time_info[str(current_player_id)]['time'] + out_time

                for current_player_id in players_in_list:

                    if str(current_player_id) in all_players_index:

                        if str(current_player_id) not in all_players_time_info.keys():
                            all_players_time_info[str(current_player_id)] = {'time': 0}

                        in_time = 0
                        for player_change in players_substitutions:
                            if player_change['playerIn'] == current_player_id:
                                in_time = player_change['minute']
                                break

                        if in_time > 90:
                            in_time = 90

                        all_players_time_info[str(current_player_id)]['time'] = all_players_time_info[str(current_player_id)]['time'] + (90 - in_time)

    print(len(all_players_time_info.keys()))
    
    all_players_time_info = json.dumps(all_players_time_info)

    f = open('./data-frontend/all_players_time_info.json', 'w')
    f.write(all_players_time_info)
    f.close()

# merge player info
def merge_player_info():

    all_players_info_basic = load_json_data('./data-frontend/all_players_info_basic.json')
    all_players_action_info = load_json_data('./data-frontend/all_players_action_info.json')
    all_players_time_info = load_json_data('./data-frontend/all_players_time_info.json')

    all_players_info = []

    for current_player_info_basic in all_players_info_basic:

        current_player_id = current_player_info_basic['id']

        current_player_info = current_player_info_basic.copy()

        action_numbers = all_players_action_info[str(current_player_id)]['numbers']
        action_values = all_players_action_info[str(current_player_id)]['values']

        current_player_info['action_numbers'] = action_numbers
        current_player_info['action_values'] = action_values

        if str(current_player_id) in all_players_time_info.keys():
            appearing_time = all_players_time_info[str(current_player_id)]['time']
        else:
            appearing_time = 0

        if appearing_time != 0:
            current_player_info['appearing_time'] = appearing_time
            current_player_info['action_numbers_per_90_min'] = (action_numbers / appearing_time) * 90
            current_player_info['action_values_per_90_min'] = (action_values / appearing_time) * 90
        else:
            current_player_info['appearing_time'] = 1
            current_player_info['action_numbers_per_90_min'] = action_numbers
            current_player_info['action_values_per_90_min'] = action_values

        if current_player_id != 0:
            all_players_info.append(current_player_info)

    print(len(all_players_info))

    all_players_info = json.dumps(all_players_info)

    f = open('./data-frontend/all_players_info.json', 'w')
    f.write(all_players_info)
    f.close()

# generate player current team
def generate_player_team():

    all_players_index_file = load_json_data('./data-new/players_club_dict.json')
    all_players_index = all_players_index_file.keys()

    all_players_team = {}

    league_names = ['england', 'french', 'german', 'italy', 'spain']

    for league_name in league_names:

        file_path = './data-' + league_name + '/vaep_with_tag.csv'
        df = pd.read_csv(file_path)

        player_ids = df['player_id']
        team_ids = df['team_id']

        for i in range(len(player_ids)):

            current_player_id = player_ids[i]
            current_team_id = team_ids[i]

            if str(current_player_id) in all_players_index:

                if str(current_player_id) not in all_players_team.keys():
                    all_players_team[str(current_player_id)] = int(current_team_id)

    print(len(all_players_team.keys()))

    all_players_team = json.dumps(all_players_team)

    f = open('./data-frontend/all_players_team.json', 'w')
    f.write(all_players_team)
    f.close()

# revise player team id
def revise_player_team_id():

    all_players_info_basic = load_json_data('./data-frontend/all_players_info_basic.json')
    all_players_info = load_json_data('./data-frontend/all_players_info.json')
    all_players_team = load_json_data('./data-frontend/all_players_team.json')

    for current_player_info_basic in all_players_info_basic:
        current_player_info_basic['curr_team_id'] = all_players_team[str(current_player_info_basic['id'])]

    for current_player_info in all_players_info:
        current_player_info['curr_team_id'] = all_players_team[str(current_player_info['id'])]

    print(len(all_players_info_basic))
    print(len(all_players_info))

    all_players_info_basic = json.dumps(all_players_info_basic)

    f = open('./data-frontend/all_players_info_basic.json', 'w')
    f.write(all_players_info_basic)
    f.close()

    all_players_info = json.dumps(all_players_info)

    f = open('./data-frontend/all_players_info.json', 'w')
    f.write(all_players_info)
    f.close()

# merge info with embeddings
def merge_embedding_info():

    all_players_info = load_json_data('./data-frontend/all_players_info.json')
    player_embedding_info = load_json_data('./data-frontend/all_players_embedding_vectors.json')
    team_embedding_info = load_json_data('./data-frontend/all_teams_embedding_vectors.json')

    all_players_index = load_json_data('./data-new/players_club_dict.json')
    all_teams_index = load_json_data('./data-new/teams_club_dict.json')

    all_players_info_embedding = []

    for current_player_info in all_players_info:

        current_player_id = current_player_info['id']
        current_player_team_id = current_player_info['curr_team_id']

        current_player_embedding = player_embedding_info[str(all_players_index[str(current_player_id)])].copy()
        current_team_embedding = team_embedding_info[str(all_teams_index[str(current_player_team_id)])].copy()

        current_player_info_embedding = current_player_info.copy()
        current_player_info_embedding['player_emb'] = current_player_embedding
        current_player_info_embedding['team_emb'] = current_team_embedding

        all_players_info_embedding.append(current_player_info_embedding)

    print(len(all_players_info_embedding))

    all_players_info_embedding = json.dumps(all_players_info_embedding)

    f = open('./data-frontend/all_players_info_embedding.json', 'w')
    f.write(all_players_info_embedding)
    f.close()

if __name__ == "__main__":
    
    merge_embedding_info()