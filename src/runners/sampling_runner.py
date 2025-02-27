import os 
import glob
from envs import REGISTRY as env_REGISTRY

class SamplingRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.env.reset()

    def run(self):
        self.reset()
        terminated = False
        while not terminated:
            terminated = self.env.step([])
        
        
 