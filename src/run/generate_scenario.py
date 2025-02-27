import os 
import json
import random
import shutil
import atexit
import gfootball

import numpy as np
from run.scenario import *

def load_env(env_name, algo_name, unique_token, scenario_file_path):
    
    new_env_name = f"{env_name}_{algo_name}_{unique_token}.py"
    env_file_path = os.path.join(os.path.dirname(__file__), "scenario", f"{env_name}.py")
    if not os.path.exists(env_file_path):
        raise ImportError(f"Scenario module {env_name}.py not found in run/scenario/")
    
    new_file_path = f"{scenario_file_path}/{new_env_name}"    
    os.makedirs(scenario_file_path, exist_ok=True)
    shutil.copy(env_file_path, new_file_path)
    
    def del_scenario_file():
        if os.path.exists(new_file_path):
                os.remove(new_file_path)
    atexit.register(del_scenario_file)
    
    return new_env_name.split(".")[0]

class Scenario_Manager():
    
    def __init__(self, args):
        self.args = args
        self.new_env_name = load_env(
            env_name = self.args.env_args['env_name'],
            algo_name = self.args.name,
            unique_token = self.args.unique_token, 
            scenario_file_path = os.path.join(os.path.dirname(gfootball.__file__), "scenarios")
        )
        self.batch_size_run = self.args.batch_size_run
        
        self.backward_complete = False
        self.end_backward_step = 0
        
        
        if self.args.backward_learning:
            self.decay_rate = self.args.decay_rate
            self.ewma_win_rate = [0.0 for _ in range(self.args.num_backward_episode)]
        
    def bulid_backward_file(self):
        self.level_file_path = os.path.join(os.path.dirname(__file__), "level", f"{self.new_env_name}.json")
        
        episode_data_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/score_data"))
        
        if not os.path.exists(episode_data_dir):  
            raise ValueError(f"Episode data directory {episode_data_dir} not found.")  
        
        sampled_epi_set = sorted(random.sample(os.listdir(episode_data_dir), self.args.num_backward_episode))
        
        self.total_idx_list = []
        epi_data_set = {}
        for idx, epi_file_name in enumerate(sampled_epi_set):
            data_path = os.path.join(episode_data_dir, epi_file_name)
            with open(data_path, "r") as f:
                data = json.load(f)
            self.total_idx_list.append(max(map(int, data)) - 3) 
            epi_data_set[idx] = data
        epi_data_set["total_step"] = self.total_idx_list
        epi_data_set["init_step"] = self.total_idx_list
        epi_data_set["eval"] = True
        
        with open(self.level_file_path, "w") as f:
            json.dump(epi_data_set, f, indent=4)
            
        def del_level_file():
            if os.path.exists(self.level_file_path):
                    os.remove(self.level_file_path)
        atexit.register(del_level_file)

        with open(self.level_file_path, "r") as f:
            self.epi_backward_data = json.load(f)
    
        return self.level_file_path 

    def episode_recorder(self, episode_record, selected_episode):
        for idx, result in enumerate(episode_record):
            epi_id = selected_episode[idx]
            with open(self.level_file_path, "r") as f:
                data = json.load(f)
            if result:
                self.ewma_win_rate[epi_id] = self.decay_rate * self.ewma_win_rate[epi_id] + (1 - self.decay_rate) * 1.0
                if self.ewma_win_rate[epi_id] > 0.5:
                    while True:
                        data["init_step"][epi_id] = max(0, data["init_step"][epi_id] - 1)
                        if data[f"{epi_id}"][f"{data['init_step'][epi_id]}"]["ball"][0] >= 0.0:
                            break
                    with open(self.level_file_path, "w") as f:
                        json.dump(data, f, indent=4)
            else:
                self.ewma_win_rate[epi_id] = self.decay_rate * self.ewma_win_rate[epi_id] + (1 - self.decay_rate) * 0.0
        ewma_record = self.ewma_win_rate        
        for idx, result in enumerate(episode_record):
            epi_id = selected_episode[idx]
            if result and self.ewma_win_rate[epi_id] > 0.5:
                self.ewma_win_rate[epi_id] = 0.0     
        
        print(f"Eval mode: {data['eval']}")
        print(f"Init step: {data['init_step']}")
        print(f"EWMA win rate: {ewma_record}")
        return np.array(ewma_record), np.array(data["init_step"], dtype=np.float32)
    
    def activate_train_mode(self):
        with open(self.level_file_path, "r") as f:
            data = json.load(f)
        data["eval"] = False
        with open(self.level_file_path, "w") as f:
            json.dump(data, f, indent=4)    

    def activate_eval_mode(self):
        with open(self.level_file_path, "r") as f:
            data = json.load(f)
        data["eval"] = True
        with open(self.level_file_path, "w") as f:
            json.dump(data, f, indent=4)    
    
    def should_stop_backward_learning(self):
        with open(self.level_file_path, "r") as f:
            data = json.load(f)
        init_steps = data["init_step"]
        zero_count = sum(1 for step in init_steps if step < 1)
        if zero_count  >= 2:
            self.backward_complete = True
        