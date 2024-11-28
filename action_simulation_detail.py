import numpy as np
import os
import json

import pandas as pd
import csv

from utilities import *

# visualize detailed simulated action information
# input: {'tactic_category': string, 'tactic_id': int, 'start_region': int, 'all_actions_info':[]}
# output: {'details_by_region': [[{'original_actions':[{'value': float}], 'new_actions':[{'value': float, 'value_change': float, 'distance': float}]}]]}
def get_detailed_action_info(frontend_data):

    # input and output
    tactic_category = frontend_data.get('TacticCategory')
    tactic_id = frontend_data.get('TacticId')
    start_region = frontend_data.get('StartRegion')
    all_actions_info = frontend_data.get('AllActionsInfo')

    all_detailed_action_info = {}
    all_detailed_action_info['values_by_region'] = []

    # blank output
    x_division = 12
    y_division = 8

    x_division_length = 105 / x_division
    y_division_length = 68 / y_division

    for i in range(x_division):

        all_detailed_action_info['values_by_region'].append([])

        for j in range(y_division):
            
            all_detailed_action_info['values_by_region'][i].append({})

            all_detailed_action_info['values_by_region'][i][j]['original_actions'] = []
            all_detailed_action_info['values_by_region'][i][j]['new_actions'] = []

    for current_action_info in all_actions_info:

        current_action_region = pitch_division_transformed(current_action_info['start_x'], current_action_info['start_y'])

        if current_action_info['tactic_category'] == tactic_category and current_action_info['tactic_id'] == tactic_id and (current_action_region == start_region or start_region == 0):

            end_region_x = int(current_action_info['end_x_new'] / x_division_length)
            end_region_y = int(current_action_info['end_y_new'] / y_division_length)

            if end_region_x >= x_division:
                end_region_x = x_division - 1

            if end_region_y >= y_division:
                end_region_y = y_division - 1

            end_region_x_original = int(current_action_info['end_x_original'] / x_division_length)
            end_region_y_original = int(current_action_info['end_y_original'] / y_division_length)

            if end_region_x_original >= x_division:
                end_region_x_original = x_division - 1

            if end_region_y_original >= y_division:
                end_region_y_original = y_division - 1

            original_info = {'value': current_action_info['value_original']}
            new_info = {'value': current_action_info['value_new'], 'value_change': current_action_info['value_change']}

            all_detailed_action_info['values_by_region'][end_region_x_original][end_region_y_original]['original_actions'].append(original_info)

            max_distance = np.sqrt(105 * 105 + 68 * 68)
            distance = current_action_info['distance'] * max_distance

            new_info['distance'] = distance

            all_detailed_action_info['values_by_region'][end_region_x][end_region_y]['new_actions'].append(new_info)

    print(all_detailed_action_info['values_by_region'][8][3])

    all_detailed_action_info = json.dumps(all_detailed_action_info)

    return all_detailed_action_info

if __name__ == "__main__":

    all_actions_info = load_json_data('./data-test/test_data.json')
    frontend_data = {'TacticCategory': 'pass', 'TacticId': 8, 'StartRegion': 6, 'AllActionsInfo': all_actions_info}

    get_detailed_action_info(frontend_data)