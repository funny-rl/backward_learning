import time
import numpy as np 
from numba import jit

MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
PENALTY_Y, END_Y = 0.27, 0.42

class FeatureModel:
    def __init__(self, n_agents, n_enemies, obs_ally_feature_dim, obs_enemy_feature_dim, state_agent_feature_dim, state_enemy_feature_dim):
        self.n_agents = n_agents # exclouding goalkeeper
        self.n_allies = self.n_agents # including goalkeeper
        self.n_enemies = n_enemies
        
        self.obs_ally_feature_dim = obs_ally_feature_dim
        self.obs_enemy_feature_dim = obs_enemy_feature_dim
        self.state_agent_feature_dim = state_agent_feature_dim
        self.state_enemy_feature_dim = state_enemy_feature_dim

    def encode(self, obs):
        """
        obs: dictionary
            'active' / 'game_mode'
            
            'left_team' / 'left_team_direction' / 'left_team_roles' / 'left_team_yellow_card' / 'left_team_tired_factor' / 'left_team_active'

            'right_team' / 'right_team_direction' / 'right_team_roles' / 'right_team_yellow_card' / 'right_team_tired_factor' / 'right_team_active'

            'ball' / 'ball_direction' / 'ball_rotation' / 'ball_owned_team' / 'ball_owned_player'
            
            'score' / 'steps_left' / 'designated' / 'sticky_actions'
        """
        player_num = obs['active'] # int
        player_pos_x, player_pos_y = obs["left_team"][player_num] # vector 2
        player_position = obs["left_team"][player_num]
        player_direction = np.array(obs["left_team_direction"][player_num]) # vector 2
        player_speed = np.linalg.norm(player_direction) # float
        player_tired = obs["left_team_tired_factor"][player_num] # float
        is_dribbling = obs["sticky_actions"][9]
        is_sprinting = obs["sticky_actions"][8]
        
        touch_line = END_Y - abs(player_pos_y)
        goal_line = END_X - abs(player_pos_x)  
        
        ball_x, ball_y, _ = obs["ball"]
        ball_x_relative = ball_x - player_pos_x
        ball_y_relative = ball_y - player_pos_y
        ball_x_speed, ball_y_speed, _ = obs["ball_direction"]
        ball_distance = np.linalg.norm([ball_x_relative, ball_y_relative])
        ball_speed = np.linalg.norm([ball_x_speed, ball_y_speed])
        ball_owned = 0.0
        
        if obs["ball_owned_team"] == -1:
            ball_owned = 0.0
        else:
            ball_owned = 1.0
        ball_owned_by_us = 0.0
        if obs["ball_owned_team"] == 0:
            ball_owned_by_us = 1.0
        elif obs["ball_owned_team"] == 1:
            ball_owned_by_us = 0.0
        else:
            ball_owned_by_us = 0.0

        ball_which_zone = self._encode_ball_which_zone(ball_x, ball_y)

        if ball_distance > 0.03:
            ball_far = 1.0
        else:
            ball_far = 0.0
            
        player_obs = np.concatenate(
            (
                player_position,
                player_direction * 100,
                [player_speed * 100],
                [touch_line, goal_line],
                [ball_far, player_tired, is_dribbling, is_sprinting],
            )
        )
        ball_obs = np.concatenate(
            (
                np.array(obs["ball"]),
                np.array(ball_which_zone),
                np.array([ball_x_relative, ball_y_relative]),
                np.array(obs["ball_direction"]) * 20,
                np.array(
                    [ball_speed * 20, ball_distance, ball_owned, ball_owned_by_us]
                ),
            )
        )
        
        # start_time = time.time()
        left_team_relative = np.delete(obs["left_team"], player_num, axis=0) - player_position
        obs_left_team_direction = np.delete(obs["left_team_direction"], player_num, axis=0) - player_direction
        left_team_distance = np.linalg.norm(left_team_relative - player_position, axis=1, keepdims=True)
        left_team_tired = np.delete(obs["left_team_tired_factor"], player_num, axis=0).reshape(-1, 1)

        left_team_obs = ally_info(left_team_relative, obs_left_team_direction, left_team_distance, \
            left_team_tired, self.n_allies, self.obs_ally_feature_dim)
        
        obs_right_team = np.array(obs["right_team"]) - player_position
        obs_right_team_direction = np.array(obs["right_team_direction"]) - player_direction
        right_team_distance = np.linalg.norm(obs_right_team - player_position, axis=1, keepdims=True)
        right_team_tired = np.array(obs["right_team_tired_factor"]).reshape(-1, 1)
        
        right_team_obs = enemy_info(obs_right_team, obs_right_team_direction, right_team_distance, \
            right_team_tired, self.n_enemies, self.obs_enemy_feature_dim)
        
        avail = _get_avail(
            ball_distance, 
            obs["game_mode"],
            obs["ball_owned_team"],
            obs["sticky_actions"],
            obs["ball"],
        )

        # end_time = time.time()
        # execution_time = end_time - start_time
        # print(f"Execution Time: {execution_time:.6f} seconds")
        obs_dict = {
            "player": player_obs,
            "ball": ball_obs,
            "left_team": left_team_obs,
            "right_team": right_team_obs,
        }
        return obs_dict, avail
    
    def encode_state(self, obs):
        
        ball_xy_pos = obs["ball"][:2]
        
        left_team_state = state_agent_info(obs["left_team"], obs["left_team_direction"], \
            obs["left_team_tired_factor"], obs["sticky_actions"], ball_xy_pos, self.n_agents, self.state_agent_feature_dim)
        
        right_team_state = state_enemy_info(obs["right_team"], obs["right_team_direction"], \
            obs["right_team_tired_factor"], ball_xy_pos, self.n_enemies, self.state_enemy_feature_dim)
        
        ball_x, ball_y, _ = obs["ball"]
        ball_which_zone = self._encode_ball_which_zone(ball_x, ball_y)
        ball_x_speed, ball_y_speed, _ = obs["ball_direction"]
        ball_speed = np.linalg.norm([ball_x_speed, ball_y_speed])
        
        ball_owned = 1.0 if obs["ball_owned_team"] != -1 else 0.0
        ball_owned_by_us = 1.0 if obs["ball_owned_team"] == 0 else 0.0

        ball_state = np.concatenate(
            (
                np.array(obs["ball"]),
                np.array(ball_which_zone),
                np.array(obs["ball_direction"]) * 20,
                np.array(
                    [ball_speed * 20, ball_owned, ball_owned_by_us]
                ),
            )
        )
        return np.expand_dims(np.concatenate((left_team_state, right_team_state, ball_state)), axis=0)

    def _encode_ball_which_zone(self, ball_x, ball_y):
        if (-END_X <= ball_x and ball_x < -PENALTY_X) and (
            -PENALTY_Y < ball_y and ball_y < PENALTY_Y
        ):
            return [1.0, 0, 0, 0, 0, 0]
        elif (-END_X <= ball_x and ball_x < -MIDDLE_X) and (
            -END_Y < ball_y and ball_y < END_Y
        ):
            return [0, 1.0, 0, 0, 0, 0]
        elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (
            -END_Y < ball_y and ball_y < END_Y
        ):
            return [0, 0, 1.0, 0, 0, 0]
        elif (PENALTY_X < ball_x and ball_x <= END_X) and (
            -PENALTY_Y < ball_y and ball_y < PENALTY_Y
        ):
            return [0, 0, 0, 1.0, 0, 0]
        elif (MIDDLE_X < ball_x and ball_x <= END_X) and (
            -END_Y < ball_y and ball_y < END_Y
        ):
            return [0, 0, 0, 0, 1.0, 0]
        else:
            return [0, 0, 0, 0, 0, 1.0]

@jit(nopython=True) 
def ally_info(left_team_relative, obs_left_team_direction, left_team_distance, left_team_tired, n_allies, ally_feature_dim):
    ally_feature = np.zeros((n_allies, ally_feature_dim))
    for idx in range(n_allies):
        ally_feature[idx, 0 : 2] = left_team_relative[idx]
        ally_feature[idx, 2 : 4] = obs_left_team_direction[idx]
        ally_feature[idx, 4] = left_team_distance[idx][0]
        ally_feature[idx, 5] = left_team_tired[idx][0]
    return ally_feature

@jit(nopython=True) 
def enemy_info(obs_right_team, obs_right_team_direction, right_team_distance, right_team_tired, n_enemies, enemy_feature_dim):
    ally_feature = np.zeros((n_enemies, enemy_feature_dim))
    for idx in range(n_enemies):
        ally_feature[idx, 0 : 2] = obs_right_team[idx]
        ally_feature[idx, 2 : 4] = obs_right_team_direction[idx]
        ally_feature[idx, 4] = right_team_distance[idx][0]
        ally_feature[idx, 5] = right_team_tired[idx][0]
    return ally_feature

@jit(nopython=True) 
def l2_norm(vec):
    norm = 0.0
    for v in vec:
        norm += v * v 
    return norm ** 0.5  

@jit(nopython=True) 
def state_agent_info(left_team, left_team_direction, left_team_tired_factor, sticky_actions, ball_xy_pos, n_agents, state_agent_feature_dim):
    agent_features = np.zeros((n_agents + 1) * state_agent_feature_dim)
    for idx in range(n_agents + 1):
        index = idx * state_agent_feature_dim
        player_xy_pos = left_team[idx]
        left_player_direction = left_team_direction[idx]
        agent_features[index : index + 2] = player_xy_pos
        agent_features[index + 2 : index + 4] = left_player_direction
        agent_features[index + 4] = l2_norm(left_player_direction)
        agent_features[index + 5] = left_team_tired_factor[idx]
        agent_features[index + 6] = sticky_actions[idx]
        agent_features[index + 7] = l2_norm(player_xy_pos - ball_xy_pos)
    return agent_features

@jit(nopython=True) 
def state_enemy_info(right_team, right_team_direction, right_team_tired_factor, ball_xy_pos, n_enemies, state_enemy_feature_dim):
    enemies_feature = np.zeros(n_enemies * state_enemy_feature_dim)
    for idx in range(n_enemies):
        index = idx * state_enemy_feature_dim
        player_xy_pos = right_team[idx]
        right_player_direction = right_team_direction[idx]
        enemies_feature[index : index + 2] = player_xy_pos
        enemies_feature[index + 2 : index + 4] = right_player_direction
        enemies_feature[index + 4] = l2_norm(right_player_direction)
        enemies_feature[index + 5] = right_team_tired_factor[idx]
        enemies_feature[index + 6] = l2_norm(player_xy_pos - ball_xy_pos)
    return enemies_feature

@jit(nopython=True) 
def _get_avail(ball_distance, game_mode, ball_owned_team, sticky_actions, ball):
    avail = np.ones(19, dtype=np.int32)
    (
        NO_OP,
        LEFT,
        TOP_LEFT,
        TOP,
        TOP_RIGHT,
        RIGHT,
        BOTTOM_RIGHT,
        BOTTOM,
        BOTTOM_LEFT,
        LONG_PASS,
        HIGH_PASS,
        SHORT_PASS,
        SHOT,
        SPRINT,
        RELEASE_MOVE,
        RELEASE_SPRINT,
        SLIDE,
        DRIBBLE,
        RELEASE_DRIBBLE,
    ) = range(19)

    if ball_owned_team == 1:  # opponents owning ball
        (
            avail[LONG_PASS],
            avail[HIGH_PASS],
            avail[SHORT_PASS],
            avail[SHOT],
            avail[DRIBBLE],
        ) = (0, 0, 0, 0, 0)
        if ball_distance > 0.03:
            avail[SLIDE] = 0
    elif (
        ball_owned_team == -1
        and ball_distance > 0.03
        and game_mode == 0
    ):  # Ground ball  and far from me
        (
            avail[LONG_PASS],
            avail[HIGH_PASS],
            avail[SHORT_PASS],
            avail[SHOT],
            avail[DRIBBLE],
            avail[SLIDE],
        ) = (0, 0, 0, 0, 0, 0)
    else:  # my team owning ball
        avail[SLIDE] = 0
        if ball_distance > 0.03:
            (
                avail[LONG_PASS],
                avail[HIGH_PASS],
                avail[SHORT_PASS],
                avail[SHOT],
                avail[DRIBBLE],
            ) = (0, 0, 0, 0, 0)

    # Dealing with sticky actions
    if sticky_actions[8] == 0:  # sprinting
        avail[RELEASE_SPRINT] = 0

    if sticky_actions[9] == 1:  # dribbling
        avail[SLIDE] = 0
    else:
        avail[RELEASE_DRIBBLE] = 0

    if np.sum(sticky_actions[:8]) == 0:
        avail[RELEASE_MOVE] = 0

    # if too far, no shot
    ball_x, ball_y, _ = ball
    if ball_x < 0.64 or ball_y < -0.27 or 0.27 < ball_y:
        avail[SHOT] = 0
    elif (0.64 <= ball_x and ball_x <= 1.0) and (
        -0.27 <= ball_y and ball_y <= 0.27
    ):
        avail[HIGH_PASS], avail[LONG_PASS] = 0, 0

    if game_mode == 2 and ball_x < -0.7:  # Our GoalKick
        avail = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
        avail[LONG_PASS], avail[HIGH_PASS], avail[SHORT_PASS] = 1, 1, 1
        return avail

    elif game_mode == 4 and ball_x > 0.9:  # Our CornerKick
        avail = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
        avail[LONG_PASS], avail[HIGH_PASS], avail[SHORT_PASS] = 1, 1, 1
        return avail

    elif game_mode == 6 and ball_x > 0.6:  # Our PenaltyKick
        avail = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
        avail[SHOT] = 1
        return avail
    
    return avail




class RewardModel:
    def __init__(self, episode_limit) -> None:
        self.episode_limit = episode_limit
        
    def calc_reward(self, prev_obs, obs, done, actions):
        
        prev_ball_owned_player = prev_obs["ball_owned_player"]
      
        prev_ball_owned_team = prev_obs["ball_owned_team"]
        ball_owned_team = obs["ball_owned_team"]
        
        own_changing_r = ownership_changing_reward(prev_ball_owned_team, ball_owned_team)
        oob_r = out_of_boundary_reward(prev_obs["game_mode"], obs["game_mode"], obs["left_team"])
        
        pass_r = pass_reward(
            prev_gamemode = prev_obs["game_mode"], 
            gamemode = obs["game_mode"],
            ball_owned_team = ball_owned_team,
            prev_ball_owned_player = prev_ball_owned_player,
            ball_owned_player = obs["ball_owned_player"],
            action = actions[prev_ball_owned_player - 1],
        )
            
        yellow_r = yellow_reward(
            prev_yellow = prev_obs["left_team_yellow_card"],
            yellow = obs["left_team_yellow_card"],
        )
        ball_position_r = ball_position_reward(obs["ball"])
        score_r = score_reward(
            score = obs["score"],
            done = done,
        )
        return own_changing_r, oob_r, pass_r, yellow_r, ball_position_r, score_r

@jit(nopython=True) 
def ownership_changing_reward(prev_ball_owned_team, ball_owned_team):
    if prev_ball_owned_team == 1 and ball_owned_team == 0 or \
        prev_ball_owned_team == -1 and ball_owned_team == 0: # us -> them & None -> us
        return 0.002
    elif prev_ball_owned_team == 0 and ball_owned_team == 1 or \
        prev_ball_owned_team == -1 and ball_owned_team == 1: # them -> us & None -> them
        return -0.003
    else:
        return 0.0
    
@jit(nopython=True) 
def out_of_boundary_reward(prev_game_mode, game_mode, left_team_position):
    oob_player = 0.0
    if prev_game_mode == 0 and game_mode == 0:
        for x_pos, y_pos in left_team_position:
            if x_pos < -1.0 or x_pos > 1.0 or y_pos < -0.42 or y_pos > 0.42:
                oob_player += 1.0
    return -0.0001 * oob_player

@jit(nopython=True) 
def pass_reward(prev_gamemode, gamemode, ball_owned_team, \
    prev_ball_owned_player, ball_owned_player, action):   
    if prev_gamemode == gamemode == 0:
        if prev_ball_owned_player not in (-1, 0) and action in (9, 10, 11):
            if ball_owned_team == 0 and prev_ball_owned_player != ball_owned_player:
                return 0.1
    return 0.0

@jit(nopython=True)
def yellow_reward(prev_yellow, yellow):
    left_yellow = np.sum(yellow) - np.sum(prev_yellow)
    return -0.1 * left_yellow

@jit(nopython=True)
def ball_position_reward(ball_position):
    ball_x, ball_y, _ = ball_position

    
    ball_position_r = 0.0
    if (-END_X <= ball_x and ball_x < -PENALTY_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = -0.0005
    elif (-END_X <= ball_x and ball_x < -MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = -0.0001
    elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 0.0
    elif (MIDDLE_X < ball_x and ball_x <= END_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 0.0001
    elif (PENALTY_X < ball_x and ball_x <= END_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = 0.0005

    else:
        ball_position_r = 0.0

    return ball_position_r

@jit(nopython=True)
def score_reward(score, done):
    my_score, opponent_score = score
    if my_score > opponent_score:
        return 10.0
    elif done or my_score < opponent_score:
        return -1.0
    else:
        return 0.0