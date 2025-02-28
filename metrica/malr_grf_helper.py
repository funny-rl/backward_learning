import math
import re
import os
import sys
import torch

from numba import jit

MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
PENALTY_Y, END_Y = 0.27, 0.42

obs_own_feature_dims = 11
obs_ball_feature_dims = 18
obs_ally_feature_dim = 6
obs_enemy_feature_dim = 6

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
from tqdm import tqdm


class MALRHelper:
    def __init__(self, data_paths=None):
        self.data_paths = data_paths

        self.feature_types = ["_x", "_y"]
        self.n_features = len(self.feature_types)

        self.team_size = 11
        self.n_agents = self.team_size - 1
        self.n_allies = self.n_agents
        self.n_enemies = self.team_size

        self.normalize = True

        self.ps = (108, 72)
        self.halfline_x = 0.0

    def _encode_ball_which_zone(self, ball_x, ball_y):
        if (-END_X <= ball_x and ball_x < -PENALTY_X) and (-PENALTY_Y < ball_y and ball_y < PENALTY_Y):
            return [1.0, 0, 0, 0, 0, 0]
        elif (-END_X <= ball_x and ball_x < -MIDDLE_X) and (-END_Y < ball_y and ball_y < END_Y):
            return [0, 1.0, 0, 0, 0, 0]
        elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (-END_Y < ball_y and ball_y < END_Y):
            return [0, 0, 1.0, 0, 0, 0]
        elif (PENALTY_X < ball_x and ball_x <= END_X) and (-PENALTY_Y < ball_y and ball_y < PENALTY_Y):
            return [0, 0, 0, 1.0, 0, 0]
        elif (MIDDLE_X < ball_x and ball_x <= END_X) and (-END_Y < ball_y and ball_y < END_Y):
            return [0, 0, 0, 0, 1.0, 0]
        else:
            return [0, 0, 0, 0, 0, 1.0]

    def encode(self, obs):
        player_num = obs["active"]  # int
        player_pos_x, player_pos_y = obs["left_team"][player_num]  # vector 2
        player_position = obs["left_team"][player_num]
        player_direction = np.array(obs["left_team_direction"][player_num])  # vector 2
        player_speed = np.linalg.norm(player_direction)  # float
        player_tired = obs["left_team_tired_factor"][player_num]  # float
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
                np.array([ball_speed * 20, ball_distance, ball_owned, ball_owned_by_us]),
            )
        )

        # start_time = time.time()
        left_team_relative = np.delete(obs["left_team"], player_num, axis=0) - player_position
        obs_left_team_direction = np.delete(obs["left_team_direction"], player_num, axis=0) - player_direction
        left_team_distance = np.linalg.norm(left_team_relative - player_position, axis=1, keepdims=True)
        left_team_tired = np.delete(obs["left_team_tired_factor"], player_num, axis=0).reshape(-1, 1)

        left_team_obs = ally_info(
            left_team_relative,
            obs_left_team_direction,
            left_team_distance,
            left_team_tired,
            self.n_allies,
            obs_ally_feature_dim,
        )

        obs_right_team = np.array(obs["right_team"]) - player_position
        obs_right_team_direction = np.array(obs["right_team_direction"]) - player_direction
        right_team_distance = np.linalg.norm(obs_right_team - player_position, axis=1, keepdims=True)
        right_team_tired = np.array(obs["right_team_tired_factor"]).reshape(-1, 1)

        right_team_obs = enemy_info(
            obs_right_team,
            obs_right_team_direction,
            right_team_distance,
            right_team_tired,
            self.n_enemies,
            obs_enemy_feature_dim,
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
        return obs_dict

    def obs_encoder(self, dataset="dataset"):

        _data_ = []
        _label_ = []

        for file_path in self.data_paths:
            match_traces = pd.read_csv(file_path, header=0)  # 기존 csv에서 데이터를 읽어옴\

            # 기존 데이터의 "_x, _y"에 해당하는 좌표 columns를 가져옴. (좌우 선수, 공 포함)
            pos_x = [c for c in match_traces.columns if c.endswith("_x")]
            pos_y = [c for c in match_traces.columns if c.endswith("_y")]

            # x, y 좌표계 변환
            match_traces[pos_x] = match_traces[pos_x] / self.ps[0] * 2 - 1
            match_traces[pos_y] = match_traces[pos_y] / self.ps[1] * 0.84 - 0.42

            # 속도 계산을 위한 위치 데이터 추출

            prev_frame_traces = None
            for frame in tqdm(match_traces["frame"].unique(), desc="Processing frames"):
                obs = {}

                if frame == 1:
                    continue
                prev_frame_traces = match_traces[match_traces["frame"] == frame - 1].dropna(axis=1)
                frame_traces = match_traces[match_traces["frame"] == frame].dropna(axis=1)

                prev_pos_x = [c for c in prev_frame_traces.columns if c.endswith("_x")]
                prev_pos_y = [c for c in prev_frame_traces.columns if c.endswith("_y")]

                pos_x = [c for c in frame_traces.columns if c.endswith("_x")]
                pos_y = [c for c in frame_traces.columns if c.endswith("_y")]

                _left_pos_x = [c for c in pos_x if c.startswith("A")]
                _left_pos_y = [c for c in pos_y if c.startswith("A")]

                _right_pos_x = [c for c in pos_x if c.startswith("B")]
                _right_pos_y = [c for c in pos_y if c.startswith("B")]

                _ball_pos_x = [c for c in pos_x if c.startswith("ball")]
                _ball_pos_y = [c for c in pos_y if c.startswith("ball")]

                _prev_left_pos_x = [c for c in prev_pos_x if c.startswith("A")]
                _prev_right_pos_x = [c for c in prev_pos_x if c.startswith("B")]

                if set(_prev_left_pos_x) != set(_left_pos_x) or set(_prev_right_pos_x) != set(_right_pos_x):
                    continue

                A_actions = one_hot_action(
                    frame_traces[[c for c in frame_traces.columns if c.startswith("A") and c.endswith("_action")]]
                )
                if A_actions.empty:
                    continue
                # B_actions = one_hot_action(
                #     frame_traces[
                #         [
                #             c for c in frame_traces.columns if c.startswith("B") and c.endswith("_action")
                #         ]
                #     ]
                # )

                left_pos_x = np.array(frame_traces[_left_pos_x].values)[0]
                left_pos_y = np.array(frame_traces[_left_pos_y].values)[0]
                right_pos_x = np.array(frame_traces[_right_pos_x].values)[0]
                right_pos_y = np.array(frame_traces[_right_pos_y].values)[0]

                left_pos = np.vstack((left_pos_x, left_pos_y)).T
                right_pos = np.vstack((right_pos_x, right_pos_y)).T

                prev_left_pos_x = np.array(prev_frame_traces[_left_pos_x].values)[0]
                prev_left_pos_y = np.array(prev_frame_traces[_left_pos_y].values)[0]
                prev_right_pos_x = np.array(prev_frame_traces[_right_pos_x].values)[0]
                prev_right_pos_y = np.array(prev_frame_traces[_right_pos_y].values)[0]

                prev_left_pos = np.vstack((prev_left_pos_x, prev_left_pos_y)).T
                prev_right_pos = np.vstack((prev_right_pos_x, prev_right_pos_y)).T

                ball_pos_x = np.array(frame_traces[_ball_pos_x].values)[0]
                ball_pos_y = np.array(frame_traces[_ball_pos_y].values)[0]
                ball_pos = np.zeros((3,))
                ball_pos[:2] = np.array([ball_pos_x, ball_pos_y]).reshape(2)
                prev_ball_pos_x = np.array(prev_frame_traces[_ball_pos_x].values)[0]
                prev_ball_pos_y = np.array(prev_frame_traces[_ball_pos_y].values)[0]
                prev_ball_pos = np.zeros((3,))
                prev_ball_pos[:2] = np.array([prev_ball_pos_x, prev_ball_pos_y]).reshape(2)

                left_direction = left_pos - prev_left_pos
                right_direction = right_pos - prev_right_pos
                ball_direction = ball_pos - prev_ball_pos

                left_team_tired_factor = np.random.uniform(0.5, 1, 11)
                right_team_tired_factor = np.random.uniform(0.5, 1, 11)

                left_distances = np.linalg.norm(left_pos - ball_pos[:2], axis=1)
                right_distances = np.linalg.norm(right_pos - ball_pos[:2], axis=1)

                closest_left_idx = np.argmin(left_distances)
                closest_right_idx = np.argmin(right_distances)

                closest_left_dist = left_distances[closest_left_idx]
                closest_right_dist = right_distances[closest_right_idx]

                sticky_actions = np.ones((10,))

                if closest_left_dist < closest_right_dist and closest_left_dist <= 0.03:
                    ball_owned_team = 0
                elif closest_left_dist > closest_right_dist and closest_left_dist <= 0.03:
                    ball_owned_team = 1
                else:
                    ball_owned_team = -1

                obs["left_team"] = left_pos
                obs["right_team"] = right_pos
                obs["left_team_direction"] = left_direction
                obs["right_team_direction"] = right_direction
                obs["ball"] = ball_pos
                obs["ball_direction"] = ball_direction
                obs["left_team_tired_factor"] = left_team_tired_factor
                obs["right_team_tired_factor"] = right_team_tired_factor
                obs["ball_owned_team"] = ball_owned_team
                obs["sticky_actions"] = sticky_actions
                _obs = []
                for idx in range(1, self.n_agents + 1):
                    obs_dict = obs
                    obs_dict["active"] = idx
                    obs_array = self.encode(obs_dict)
                    obs_cat = np.hstack([np.array(obs_array[k], dtype=np.float32).flatten() for k in obs_array])
                    _obs.append(obs_cat)
                _obs = np.array(_obs)

                data = _obs
                action = np.array(A_actions.values)[0].reshape(-1, 1)[1:]

                _data_.append(data)
                _label_.append(action)

                if np.concatenate(_data_, axis=0).shape[0] != np.concatenate(_label_, axis=0).shape[0]:
                    print(
                        f"Data and label size match!, {np.concatenate(_data_, axis=0).shape}, {np.concatenate(_label_, axis=0).shape}"
                    )

        _dataset = torch.tensor(np.concatenate(_data_, axis=0))
        _labelset = torch.tensor(np.concatenate(_label_, axis=0))

        torch.save({"data": _dataset, "action": _labelset}, f"./{dataset}.pt")

    def processing_action_labels(self):
        for f in tqdm(self.data_paths, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"):
            match_traces = pd.read_csv(f, header=0)

            if self.normalize:
                x_cols = [c for c in match_traces.columns if c.endswith("_x")]
                y_cols = [c for c in match_traces.columns if c.endswith("_y")]

            player_cols = [c for c in match_traces.columns if c[0] in ["A", "B"] and c[3:] in self.feature_types]

            for phase in match_traces["phase"].unique():
                phase_traces = match_traces[match_traces["phase"] == phase]

                team1_gk, team2_gk = MALRHelper.detect_goalkeepers(phase_traces, self.halfline_x)
                team1_code, team2_code = team1_gk[0], team2_gk[0]

                input_cols = phase_traces[player_cols].dropna(axis=1).columns
                team1_cols = [c for c in input_cols if c[0] == team1_code]
                team2_cols = [c for c in input_cols if c[0] == team2_code]
                if min(len(team1_cols), len(team2_cols)) < self.n_features * self.team_size:
                    continue

                # Reorder teams so that the left team comes first
                input_cols = team1_cols + team2_cols

                episodes = [e for e in phase_traces["episode"].unique() if e > 0]
                for e in episodes:
                    ep_traces = match_traces[match_traces["episode"] == e]
                    ep_size = len(ep_traces)

                    ep_pdata = ep_traces[input_cols]
                    ep_bdata = ep_traces[["ball_x", "ball_y"]].values

                    delta_pdata = ep_pdata.diff(axis=0).values
                    delta_xy = delta_pdata.reshape(ep_size, 22, 2)[1:]

                    vec_get_action = np.vectorize(MALRHelper.compute_movement_actions)
                    actions, angles = vec_get_action(delta_xy[..., 0], delta_xy[..., 1], metric="angle")

                    player_action = [p[:3] + "_action" for p in input_cols if p.endswith("_x")]
                    match_traces.loc[ep_traces.index[:-1], player_action] = actions
                    # The movement of the last frame in each episode is assumed to be 'Idle'.
                    match_traces.loc[ep_traces.index[-1], player_action] = ["Idle"] * 22

                    if angles is not None:
                        player_angle = [p[:3] + "_angle" for p in input_cols if p.endswith("_x")]
                        match_traces.loc[ep_traces.index[:-1], player_angle] = angles
                        match_traces.loc[ep_traces.index[-1], player_angle] = 0

            # Save processed file
            dir_path, f_name = os.path.split(f)
            save_f_name = os.path.join(dir_path, f_name.replace(".csv", "_action.csv"))
            match_traces.to_csv(save_f_name, index=False)

    @staticmethod
    def detect_goalkeepers(traces: pd.DataFrame, halfline_x=54):
        a_x_cols = [c for c in traces.columns if c.startswith("A") and c.endswith("_x")]
        b_x_cols = [c for c in traces.columns if c.startswith("B") and c.endswith("_x")]

        a_gk = (traces[a_x_cols].mean() - halfline_x).abs().idxmax()[:3]
        b_gk = (traces[b_x_cols].mean() - halfline_x).abs().idxmax()[:3]

        a_gk_mean_x = traces[f"{a_gk}_x"].mean()
        b_gk_mean_y = traces[f"{b_gk}_x"].mean()

        return (a_gk, b_gk) if a_gk_mean_x < b_gk_mean_y else (b_gk, a_gk)

    @staticmethod
    def compute_movement_actions(delta_x, delta_y, metric="angle"):
        if metric == "displacement":
            # Define directions based on the signs of delta_x and delta_y
            if delta_x == 0 and delta_y == 0:
                return "Idle", None
            elif delta_x < 0 and delta_y == 0:
                return "Left", None
            elif delta_x < 0 and delta_y > 0:
                return "Top-left", None
            elif delta_x == 0 and delta_y > 0:
                return "Top", None
            elif delta_x > 0 and delta_y > 0:
                return "Top-right", None
            elif delta_x > 0 and delta_y == 0:
                return "Right", None
            elif delta_x > 0 and delta_y < 0:
                return "Bottom-right", None
            elif delta_x == 0 and delta_y < 0:
                return "Bottom", None
            elif delta_x < 0 and delta_y < 0:
                return "Bottom-left", None
            else:
                return "unknown", None
        else:
            # Calculate the angle in degrees from radians
            angle_radians = math.atan2(delta_y, delta_x)
            angle_degrees = math.degrees(angle_radians)

            # Normalize the angle to be between 0 and 360 degrees
            if angle_degrees < 0:
                angle_degrees += 360

            # Determine direction based on angle ranges
            if delta_x == 0 and delta_y == 0:
                return "Idle", angle_degrees
            elif -22.5 <= angle_degrees < 22.5 or angle_degrees >= 337.5:
                return "Right", angle_degrees
            elif 22.5 <= angle_degrees < 67.5:
                return "Top-right", angle_degrees
            elif 67.5 <= angle_degrees < 112.5:
                return "Top", angle_degrees
            elif 112.5 <= angle_degrees < 157.5:
                return "Top-left", angle_degrees
            elif 157.5 <= angle_degrees < 202.5:
                return "Left", angle_degrees
            elif 202.5 <= angle_degrees < 247.5:
                return "Bottom-left", angle_degrees
            elif 247.5 <= angle_degrees < 292.5:
                return "Bottom", angle_degrees
            elif 292.5 <= angle_degrees < 337.5:
                return "Bottom-right", angle_degrees
            else:
                return "unknown", angle_degrees


@jit(nopython=True)
def ally_info(
    left_team_relative, obs_left_team_direction, left_team_distance, left_team_tired, n_allies, ally_feature_dim
):
    ally_feature = np.zeros((n_allies, ally_feature_dim))
    for idx in range(n_allies):
        ally_feature[idx, 0:2] = left_team_relative[idx]
        ally_feature[idx, 2:4] = obs_left_team_direction[idx]
        ally_feature[idx, 4] = left_team_distance[idx][0]
        ally_feature[idx, 5] = left_team_tired[idx][0]
    return ally_feature


@jit(nopython=True)
def enemy_info(
    obs_right_team, obs_right_team_direction, right_team_distance, right_team_tired, n_enemies, enemy_feature_dim
):
    ally_feature = np.zeros((n_enemies, enemy_feature_dim))
    for idx in range(n_enemies):
        ally_feature[idx, 0:2] = obs_right_team[idx]
        ally_feature[idx, 2:4] = obs_right_team_direction[idx]
        ally_feature[idx, 4] = right_team_distance[idx][0]
        ally_feature[idx, 5] = right_team_tired[idx][0]
    return ally_feature


def one_hot_action(actions):
    action_dict = {
        "Idle": 0,
        "Left": 1,
        "Top-left": 2,
        "Top": 3,
        "Top-right": 4,
        "Right": 5,
        "Bottom-right": 6,
        "Bottom": 7,
        "Bottom-left": 8,
    }
    actions = actions.applymap(lambda x: action_dict[x])
    return actions
