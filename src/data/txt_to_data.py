import os
import re
import json 
import numpy as np
from tqdm import tqdm 

os.makedirs("score_data", exist_ok=True)

score_data_dir = './score_data'
episodes = [f for f in os.listdir("txt") if f.endswith('.txt') and 'lost' not in f]

if not os.path.exists(score_data_dir):
    os.makedirs(score_data_dir)

for id, episode in enumerate(tqdm(episodes, desc="Processing episodes"), start=1):
    with open(os.path.join('./txt', episode), 'r', encoding='utf-8') as file:
        episode_data = eval(file.read(), {"array": np.array})
        
    episode_data_chunk = {}
        
    for idx, step_data in enumerate(episode_data):
        """
        Sticky actions are not trackable and demo data is not needed as it is for initialize. 
        """
        step_observations = step_data["observation"]
        for key, value in step_observations.items():
            if isinstance(value, np.ndarray):
                step_observations[key] = value.tolist()
        step_observations["left_agent_sticky_actions"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        episode_data_chunk[f"{idx}"] = step_observations

    episode_filename = f"episode_{id}.json"
    json_filepath = os.path.join(score_data_dir, episode_filename)

    with open(json_filepath, 'w', encoding='utf-8') as json_file:
        json.dump(episode_data_chunk, json_file, ensure_ascii=False, indent=4)