import gym
import highway_env
import random
from stable_baselines3 import DQN
from highway_env import utils
import time
# print(utils.near_split(16,1))

env = gym.make('my-env-v0')

env.config["action"]["veh_class"]="highway_env.vehicle.controller.MDPVehicle"
env.reset()

model = DQN('MlpPolicy', env,
              policy_kwargs=dict(net_arch=[50, 40]),
              learning_rate=5e-4,
              buffer_size=15000,
              learning_starts=200,
              batch_size=32,
              gamma=0.8,
              train_freq=1,
              target_update_interval=50,
              verbose=1,)
            #  tensorboard_log="highway_dqn/")

# model.learn(int(2e4))
# model.save("highway_dqn/model")


# model=DQN.load("highway_dqn_HPC/model_LC-exp.35-50k")


# for i in range(1,30):
#     obs=env.reset()
#     done = False
#     while not done:
#         # action =random.randint(0,2)
#         action,st=model.predict(obs,deterministic=True)
#         print("act",action)
#         obs, reward, done, info = env.step(action)
#         env.render()
        

#     print("eps",i)


