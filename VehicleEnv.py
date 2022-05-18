# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 08:12:32 2021

@author: ahmed
"""
from stable_baselines3.common import callbacks
import tensorflow as tf
import gym
import torch as th
import torch.nn as nn
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.distributions import CategoricalDistribution
from collections import OrderedDict
import threading
from stable_baselines3.common.logger import configure

class TensorboardCallback(BaseCallback):
    #Custom callback for plotting additional values in tensorboard.
 
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
       
        self.num_timesteps
        if (self.num_timesteps % 500 == 0):
            print("dump at ",str(self.num_timesteps))
            self.logger.dump(self.num_timesteps)
        return True
    
class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key=="snap":
                # Apply CNN 
                ### CNN
                n_input_channels=subspace.shape[0]
                
                extractors[key] = nn.Sequential(
                                        nn.Conv2d(n_input_channels, 16, kernel_size=2, stride=1, padding=0),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=(1,2)),
                                        # nn.Conv2d(16,32, kernel_size=2, stride=(1,2), padding=0),
                                        # nn.ReLU(),
                                        # nn.MaxPool2d(kernel_size=(1,2)),
                                        nn.Flatten(),
                                        )
                
                # Compute shape by doing one forward pass
                with th.no_grad():
                    sample=[observation_space.sample()["snap"]]
                    
                    tensor=th.as_tensor(sample,dtype=th.float)
                    # print(tensor.dtype,tensor.shape)
                    n_dim = extractors[key](tensor).shape[1]
                    total_concat_size +=n_dim
                
            if key=="speed":
                extractors[key] = nn.Sequential(nn.Flatten())
                total_concat_size += 3 

            if key=="info":
                extractors[key] = nn.Sequential(nn.Flatten())
                total_concat_size += 2

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            # print(key,observations[key])
            ele=observations[key]
            encoded_tensor_list.append(extractor(th.as_tensor(ele,dtype=th.float)))
            # encoded_tensor_list.append(extractor(ele))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
       


        return th.cat(encoded_tensor_list, dim=1)
        
class TrainModel():
    def __init__(self,env,name,layers=[256,128],lr=5e-3,update_interval=300,gamma=0.85,eval=False):
        self.env = env
        self.model=None
        self.name=name
        self.layers=layers
        self.lr=lr
        self.layers=layers
        self.update_interval=update_interval
        self.gamma=gamma
        self.eval=eval


    def init(self):
        policy_kwargs =dict(
                    features_extractor_class=CustomCombinedExtractor,
                    activation_fn=th.nn.ReLU,
                    # net_arch=[250,100]
                    net_arch=self.layers
                    )


        path="log/"
        new_logger=configure(path, ["stdout", "tensorboard"])
        
        # model=PPO("MlpPolicy",self.env,verbose=1,tensorboard_log=path)
        self.model=DQN("MultiInputPolicy",self.env,policy_kwargs=policy_kwargs,
                        learning_rate=self.lr,
                        buffer_size=15000,
                        learning_starts=200,
                        batch_size=32,
                        gamma=self.gamma,
                        train_freq=4,
                        target_update_interval=self.update_interval,
                        verbose=1,
                        exploration_fraction=0.8,
                        tensorboard_log=path+self.name)
        # self.model.set_logger(new_logger)
    def learn(self,ts=2e4):
        if not self.eval:
            self.model.learn(total_timesteps=ts,callback=TensorboardCallback())
        else:
            self.model.learn(total_timesteps=ts,callback=TensorboardCallback(),eval_env=self.env,eval_freq=500,n_eval_episodes=8)
        # self.env.road.
        

