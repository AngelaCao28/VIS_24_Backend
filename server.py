import json

from flask import Flask, request
from flask_cors import *

from match_result_prediction import *
from team_tactics import *
from action_simulation import *
from action_simulation_detail import *

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# 前端请求的路由格式
@app.route('/playerInfo',methods=['POST'])
@cross_origin()
# 具体接口内容
def handle_candidate_player_info():
    frontend_data = request.get_json()
    return get_all_candidate_players_info(frontend_data)

@app.route('/tacticInfo',methods=['POST'])
@cross_origin()
def handle_all_tactic_info():
    frontend_data_2 = request.get_json()
    return get_all_tactics_info(frontend_data_2)

@app.route('/actionInfo',methods=['POST'])
@cross_origin()
def handle_simulate_action_info():
    frontend_data_3 = request.get_json()
    return simulate_action_decisions(frontend_data_3)

@app.route('/actionInfoDetail',methods=['POST'])
@cross_origin()
def handle_simulate_action_info_detail():
    frontend_data_4 = request.get_json()
    return get_detailed_action_info(frontend_data_4)

if __name__ == '__main__':

    os.environ.setdefault('DJANGO_SETTINGS MODULE', 'backend.settings')

    app.run(host='0.0.0.0', port=5050, debug=True)