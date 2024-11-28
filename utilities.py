import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.utils.data as data

import json
import numpy as np

import pandas as pd
import csv

# load json data
def load_json_data(file_path):

    f = open(file_path, 'r')
    all_data = json.load(f)
    f.close()

    return all_data

# discrete the locations to regions
def pitch_division(loc_x, loc_y):

    region_index = 0

    if (loc_x >= 0 and loc_x < 35) and (loc_y >= 0 and loc_y < 15):
        region_index = 7

    if (loc_x >= 35 and loc_x < 70) and (loc_y >= 0 and loc_y < 15):
        region_index = 8

    if (loc_x >= 70 and loc_x <= 105) and (loc_y >= 0 and loc_y < 15):
        region_index = 9
    
    if (loc_x >= 0 and loc_x < 35) and (loc_y >= 15 and loc_y < 53):
        region_index = 4

    if (loc_x >= 35 and loc_x < 70) and (loc_y >= 15 and loc_y < 53):
        region_index = 5

    if (loc_x >= 70 and loc_x <= 105) and (loc_y >= 15 and loc_y < 53):
        region_index = 6

    if (loc_x >= 0 and loc_x < 35) and (loc_y >= 53 and loc_y <= 68):
        region_index = 1

    if (loc_x >= 35 and loc_x < 70) and (loc_y >= 53 and loc_y <= 68):
        region_index = 2

    if (loc_x >= 70 and loc_x <= 105) and (loc_y >= 53 and loc_y <= 68):
        region_index = 3

    return region_index

# discrete the locations to regions (transformed)
def pitch_division_transformed(loc_x, loc_y):

    region_index = 0

    if (loc_x >= 0 and loc_x < 35) and (loc_y >= 0 and loc_y < 15):
        region_index = 1

    if (loc_x >= 35 and loc_x < 70) and (loc_y >= 0 and loc_y < 15):
        region_index = 2

    if (loc_x >= 70 and loc_x <= 105) and (loc_y >= 0 and loc_y < 15):
        region_index = 3
    
    if (loc_x >= 0 and loc_x < 35) and (loc_y >= 15 and loc_y < 53):
        region_index = 4

    if (loc_x >= 35 and loc_x < 70) and (loc_y >= 15 and loc_y < 53):
        region_index = 5

    if (loc_x >= 70 and loc_x <= 105) and (loc_y >= 15 and loc_y < 53):
        region_index = 6

    if (loc_x >= 0 and loc_x < 35) and (loc_y >= 53 and loc_y <= 68):
        region_index = 7

    if (loc_x >= 35 and loc_x < 70) and (loc_y >= 53 and loc_y <= 68):
        region_index = 8

    if (loc_x >= 70 and loc_x <= 105) and (loc_y >= 53 and loc_y <= 68):
        region_index = 9

    return region_index

# transform df to json
def transform_csv_to_json(file_path, dest_path):

    df = pd.read_csv(file_path)
    json_data = df.to_dict(orient='records')

    json_data = json.dumps(json_data)

    f = open(dest_path, 'w')
    f.write(json_data)
    f.close()

if __name__ == "__main__":

    transform_csv_to_json('./data-spain/vaep_with_tag.csv', './data-spain/vaep_with_tag.json')