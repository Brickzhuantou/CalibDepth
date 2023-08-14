import torch
import numpy as np
import functools

def cat(list_of_tensors, dim=0):
    """
    Concatenate a list of tensors.
    """
    return functools.reduce(lambda x, y: torch.cat([x, y], dim=dim), list_of_tensors)


def catcat(list_of_lists_of_tensors, dim_outer=0, dim_inner=0):
    """
    Recursively concatenate a list of tensors.
    """
    return cat([cat(inner_list, dim_inner) for inner_list in list_of_lists_of_tensors], dim_outer)

class Buffer:
    """replay buffer, to generate trajectories
    """
    
    def __init__(self):
        self.count = 0
        self.sources = []
        self.targets = []
        self.target_depth = []
        self.expert_actions = []
        
    def __len__(self):
        return self.count

    def start_trajectory(self):
        self.count += 1
        self.sources += [[]]
        self.targets += [[]]
        self.target_depth += [[]]
        self.expert_actions += [[]]
        
    def log_step(self, observation, expert_action):
        self.sources[-1].append(observation[0].detach())
        self.targets[-1].append(observation[1].detach())
        self.target_depth[-1].append(observation[2].detach())
        self.expert_actions[-1].append(expert_action.detach())

    def get_samples(self):
        samples = [self.sources, self.targets, self.target_depth, self.expert_actions]
        return [catcat(sample) for sample in samples]
    
    def clear(self):
        self.count = 0
        self.source.clear()
        self.target.clear()
        self.target_depth.clear()
        self.expert_actions.clear()
        

