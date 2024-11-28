import numpy as np
import os
import json

from utilities import *
from action_prediction_net_new import *

# get embedding vectors for players and teams
def generate_embedding_vectors():

    all_players_embedding_vectors = {}
    all_teams_embedding_vectors = {}

    df = pd.read_csv("./data-frames/temp4.csv")
    df = df.reset_index()
    idx_all = np.repeat(True,len(df))

    # replace own goal "h" with goal "g"
    idx = df["act"] == "h"
    df.loc[idx,"act"] = "g"

    # set up idx2char and char2idx
    idx2char = {0: '@', 1: '_', 2: 'd', 3: 'g', 4: 'p', 5: 's', 6: 'x'}
    char2idx = {'@': 0, '_': 1, 'd': 2, 'g': 3, 'p': 4, 's': 5, 'x': 6}

    # replace characters with numbers
    df['act'].replace(char2idx,inplace=True)

    all_dataset = SoccerDataset(idx=idx_all, df=df)

    batch_size, num_workers = 1, 2
    all_loader = DataLoader(all_dataset,shuffle=False,batch_size=batch_size,num_workers=num_workers,drop_last=True)

    model = Soccer_Model_3ze().to(device)
    model.load_state_dict(torch.load('./model-files/MDLstate_2024_03_28-100645', map_location=device))
    model.eval()

    with torch.no_grad():

        for batch, (X, Y, Index) in enumerate(all_loader):

            X, Y, Index = X.to(device), Y.to(device), Index.to(device)

            current_player_index = int(Index[0][1])
            current_team_index = int(Index[0][0])

            if str(current_player_index) in all_players_embedding_vectors.keys() and str(current_team_index) in all_teams_embedding_vectors.keys():
                continue
            else:

                player_emb_vect, team_emb_vect, pred = model(X, Index)
                player_emb_vect_temp = player_emb_vect[0].numpy().tolist()
                team_emb_vect_temp = team_emb_vect[0].numpy().tolist()

                all_players_embedding_vectors[str(current_player_index)] = player_emb_vect_temp
                all_teams_embedding_vectors[str(current_team_index)] = team_emb_vect_temp

                print('embedding vectors calculated')

    print(len(all_players_embedding_vectors.keys()))
    print(len(all_teams_embedding_vectors.keys()))

    all_players_embedding_vectors = json.dumps(all_players_embedding_vectors)

    f = open('./data-frontend/all_players_embedding_vectors.json', 'w')
    f.write(all_players_embedding_vectors)
    f.close()

    all_teams_embedding_vectors = json.dumps(all_teams_embedding_vectors)

    f = open('./data-frontend/all_teams_embedding_vectors.json', 'w')
    f.write(all_teams_embedding_vectors)
    f.close()

if __name__ == "__main__":
    
    generate_embedding_vectors()