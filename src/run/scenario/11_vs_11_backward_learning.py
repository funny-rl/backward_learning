# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import json
import numpy as np
from gfootball.scenarios import *

def build_scenario(builder):
  
  level_file_path = os.path.join(os.getcwd(), "run/level", \
    f"{os.path.basename(os.path.abspath(__file__)).split('.')[0]}.json")
  
  with open(level_file_path, "r") as f:
    level = json.load(f)
  
  if not level["eval"]:
    total_step = np.array(level["total_step"])
    init_step = np.array(level["init_step"])
    sampling_distribution = np.exp(init_step / total_step) / np.sum(np.exp(init_step / total_step))
    
    np.random.seed(None)
    chosen_epi_id = np.random.choice(len(total_step), p=sampling_distribution) 
    scenario = level[str(chosen_epi_id)][str(init_step[chosen_epi_id])]
    ball_pos = scenario["ball"]
    left_team_pos = np.array(scenario["left_team"])
    right_team_pos = np.array(scenario["right_team"])
    builder.config().game_duration = 500
    builder.config().right_team_difficulty = 0.95
    builder.config().deterministic = False
    builder.config().end_episode_on_score = True
    builder.SetBallPosition(ball_pos[0], ball_pos[1])

    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(left_team_pos[0][0], left_team_pos[0][1], e_PlayerRole_GK, controllable=False)
    builder.AddPlayer(left_team_pos[1][0], left_team_pos[1][1], e_PlayerRole_RM, controllable=True)
    builder.AddPlayer(left_team_pos[2][0], left_team_pos[2][1], e_PlayerRole_CF, controllable=True)
    builder.AddPlayer(left_team_pos[3][0], left_team_pos[3][1], e_PlayerRole_LB, controllable=True)
    builder.AddPlayer(left_team_pos[4][0], left_team_pos[4][1], e_PlayerRole_CB, controllable=True)
    builder.AddPlayer(left_team_pos[5][0], left_team_pos[5][1], e_PlayerRole_CB, controllable=True)
    builder.AddPlayer(left_team_pos[6][0], left_team_pos[6][1], e_PlayerRole_RB, controllable=True)
    builder.AddPlayer(left_team_pos[7][0], left_team_pos[7][1], e_PlayerRole_CM, controllable=True)
    builder.AddPlayer(left_team_pos[8][0], left_team_pos[8][1], e_PlayerRole_CM, controllable=True)
    builder.AddPlayer(left_team_pos[9][0], left_team_pos[9][1], e_PlayerRole_CM, controllable=True)
    builder.AddPlayer(left_team_pos[10][0], left_team_pos[10][1], e_PlayerRole_LM, controllable=True)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(right_team_pos[0][0], right_team_pos[0][1], e_PlayerRole_GK, controllable=False)
    builder.AddPlayer(right_team_pos[1][0], right_team_pos[1][1], e_PlayerRole_RM, controllable=False)
    builder.AddPlayer(right_team_pos[2][0], right_team_pos[2][1], e_PlayerRole_CF, controllable=False)
    builder.AddPlayer(right_team_pos[3][0], right_team_pos[3][1], e_PlayerRole_LB, controllable=False)
    builder.AddPlayer(right_team_pos[4][0], right_team_pos[4][1], e_PlayerRole_CB, controllable=False)
    builder.AddPlayer(right_team_pos[5][0], right_team_pos[5][1], e_PlayerRole_CB, controllable=False)
    builder.AddPlayer(right_team_pos[6][0], right_team_pos[6][1], e_PlayerRole_RB, controllable=False)
    builder.AddPlayer(right_team_pos[7][0], right_team_pos[7][1], e_PlayerRole_CM, controllable=False)
    builder.AddPlayer(right_team_pos[8][0], right_team_pos[8][1], e_PlayerRole_CM, controllable=False)
    builder.AddPlayer(right_team_pos[9][0], right_team_pos[9][1], e_PlayerRole_CM, controllable=False)
    builder.AddPlayer(right_team_pos[10][0], right_team_pos[10][1], e_PlayerRole_LM, controllable=False)
  else:
    builder.config().game_duration = 500
    builder.config().right_team_difficulty = 0.95
    builder.config().deterministic = False
    builder.config().end_episode_on_score = True
    builder.SetBallPosition(0.0, 0.0)
    
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK, controllable=False)
    builder.AddPlayer(0.000000,  0.020000, e_PlayerRole_RM, controllable=True)
    builder.AddPlayer(0.000000, -0.020000, e_PlayerRole_CF, controllable=True)
    builder.AddPlayer(-0.422000, -0.19576, e_PlayerRole_LB, controllable=True)
    builder.AddPlayer(-0.500000, -0.06356, e_PlayerRole_CB, controllable=True)
    builder.AddPlayer(-0.500000, 0.063559, e_PlayerRole_CB, controllable=True)
    builder.AddPlayer(-0.422000, 0.195760, e_PlayerRole_RB, controllable=True)
    builder.AddPlayer(-0.184212, -0.10568, e_PlayerRole_CM, controllable=True)
    builder.AddPlayer(-0.267574, 0.000000, e_PlayerRole_CM, controllable=True)
    builder.AddPlayer(-0.184212, 0.105680, e_PlayerRole_CM, controllable=True)
    builder.AddPlayer(-0.010000, -0.21610, e_PlayerRole_LM, controllable=True)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK, controllable=False)
    builder.AddPlayer(-0.050000, 0.000000, e_PlayerRole_RM, controllable=False)
    builder.AddPlayer(-0.010000, 0.216102, e_PlayerRole_CF, controllable=False)
    builder.AddPlayer(-0.422000, -0.19576, e_PlayerRole_LB, controllable=False)
    builder.AddPlayer(-0.500000, -0.06356, e_PlayerRole_CB, controllable=False)
    builder.AddPlayer(-0.500000, 0.063559, e_PlayerRole_CB, controllable=False)
    builder.AddPlayer(-0.422000, 0.195760, e_PlayerRole_RB, controllable=False)
    builder.AddPlayer(-0.184212, -0.10568, e_PlayerRole_CM, controllable=False)
    builder.AddPlayer(-0.267574, 0.000000, e_PlayerRole_CM, controllable=False)
    builder.AddPlayer(-0.184212, 0.105680, e_PlayerRole_CM, controllable=False)
    builder.AddPlayer(-0.010000, -0.21610, e_PlayerRole_LM, controllable=False)