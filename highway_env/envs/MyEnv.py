import random
from numpy.random.mtrand import rand
from highway_env import utils, vehicle
from highway_env.envs.common.abstract import Observation
from highway_env.envs.highway_env import AbstractEnv
from highway_env.envs.common.action import Action
import numpy as np
from gym.envs.registration import register
from typing import Optional, Tuple
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle,KraussVehicle
# from highway_env.road.lane import LineType, StraightLane, AbstractLane
from highway_env.vehicle.controller import ControlledVehicle, HamVehicle, MDPVehicle
from highway_env.vehicle.kinematics import Vehicle

from highway_env.envs.common.observation import OccupancyGridObservationExt
# from highway_env.utils import


class MyEnv(AbstractEnv):

    def __init__(self, config: dict = None,train=True) -> None:
        self.crashes_counter=0
        super().__init__(config=config)
        self.count_LC=0
        self.total_dev_speed=0
        self.isLC=False

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "OccupancyGrid",
                "features": ["presence"], ##, "x", "y", "vx", "vy", "cos_h", "sin_h"],
                # "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
                # "grid_step": [5, 5],
            },
            "action": {
                "type": "DiscreteMetaAction",
                 "longitudinal": False,
                 "lateral": True,
                 #"veh_class":"highway_env.vehicle.behavior.HamVehicle"
            },
            "lanes_count":3,
            "vehicles_count":10,
            "controlled_vehicles": 1,
            "other_vehicles_type": "highway_env.vehicle.behavior.KraussVehicle",
            "initial_lane_id": None,
            "duration": 0.3*60,  # [s]
            "ego_spacing": 1,
            "vehicles_density": 1.5,
            "collision_reward": -1,    # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "reward_speed_range": [20, 30],
            "offroad_terminal": False,

            "simulation_frequency": 15,  # [Hz]
            "policy_frequency": 1,  # [Hz]
            "disable_collision_checks": True,
            ## added configuration
            "lane_change_penalty":0.5 ,
            "min_speed":8.3, ## 30 km/h
            "max_speed":16.6, ## 60 km/h
            "min_speed_c":12.5,## 45 km/h
            "max_speed_c":16.6, ## 60 km/h
            "enable_lane_change":True,
            "c":1,
            "ints": None,

            ## IDM eval
            "IDM_type":"agg"


        })
        return config

    def reset(self) -> Observation:
        """
        Reset the environment to it's initial configuration

        :return: the observation of the reset state
        """

        ##update Evaluation variables
        self.count_LC=0
        self.total_dev_speed=0
        # self.crashes_counter=0## set to zero


        self.update_metadata()
        self.define_spaces()  # First, to set the controlled vehicle class depending on action space
        self.time = self.steps = 0
        self.done = False
        self._reset()

        if type(self.observation_type)==OccupancyGridObservationExt:
            shape=self.observation_type.grid.shape[1:]
            ## decide the shape of Occupancy Grid
            for ve in self.controlled_vehicles:
                ve.initSnapShot(shape)

        self.define_spaces()  # Second, to link the obs and actions to the vehicles once the scene is created

        ## make two moves (avoid having empty sanpshots)
        self.step(1)
        self.step(1)
        self.steps-=2

        return self.observation_type.observe()
    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminal, info)
        """
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        self.steps += 1

       
        ##check if LC is occuring
       
        self._simulate(action)

        # print(self.controlled_vehicles[0].speed)
        
        
        obs = self.observation_type.observe()
        
        reward = self._reward(action)
        terminal = self._is_terminal()
        info = self._info(obs, action)

        return obs, reward, terminal, info
 

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"], speed_limit=80),
                         np_random=self.np_random, record_history=self.config["show_trajectories"],ints=self.config["ints"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = utils.near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])


        ## change max/min speed for MDP vehicle
        MDPVehicle.SPEED_MIN=self.config["min_speed"]
        MDPVehicle.SPEED_MAX=self.config["max_speed"]

        self.controlled_vehicles = []
        for others in other_per_controlled:
            controlled_vehicle_class = self.action_type.vehicle_class
            a,b=self.config["min_speed"],self.config["max_speed"]
            controlled_vehicle=controlled_vehicle_class.create_random(
                self.road,
                speed=self.config["min_speed"],
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            controlled_vehicle.target_speed=np.random.uniform(self.config["min_speed_c"],self.config["max_speed_c"])
            # controlled_vehicle.target_speed=np.random.uniform(self.config["max_speed"],self.config["max_speed"])
            # controlled_vehicle.target_speed=np.random.uniform(36, 38.9)
            # controlled_vehicle.target_speed=15

            ## assigned the weather conds to the vehicle if Krauss
            if type(controlled_vehicle)== HamVehicle or type(controlled_vehicle)== IDMVehicle:
                controlled_vehicle.setWeatherCond(depth=self.road.depth,intensity=self.road.intensity)

            ## if IDM , then set the IDM type
            # if self.config["action"]["veh_class"]=="highway_env.vehicle.behavior.IDMVehicle":
            

            # controlled_vehicle.target_speed=np.random.uniform(self.config["min_speed_c"],self.config["max_speed_c"])
            self.controlled_vehicles.append(controlled_vehicle)
            self.road.vehicles.append(controlled_vehicle)

            if type(controlled_vehicle)==IDMVehicle:
                controlled_vehicle.control = True
                controlled_vehicle.setType(self.config["IDM_type"])
                controlled_vehicle.randomize_behavior()

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, 
                                                            ## Density
                                                            spacing=1 / self.config["vehicles_density"],
                                                            ## Distribution??
                                                            speed=self.config["min_speed"],
                                                            )
                                                            # speed=random.randint(self.config["min_speed"],self.config["max_speed"])
                vehicle.enable_lane_change=self.config["enable_lane_change"]
                # vehicle.randomize_behavior()## method defined in IDM
                # vehicle.initModelParameters(self.config["min_speed"],self.config["max_speed"])
                vehicle.target_speed = np.random.uniform(self.config["min_speed"] , self.config["max_speed"])
                self.road.vehicles.append(vehicle)
                if type(vehicle)==KraussVehicle:
                    vehicle.setWeatherCond(depth=self.road.depth,intensity=self.road.intensity)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        # neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)

        reward=0
        c=self.config['c']


        if self.vehicle.crashed:   
            return 0  ## terminal state must always return 0

        ## actual-target speed(error)
        # dv=abs(self.vehicle.target_speed-self.vehicle.speed) 
        ## assume speed <= target , thence maximum reward is achieved at 0
        dv=self.vehicle.speed-self.vehicle.target_speed
        self.total_dev_speed+=abs(dv)

        ratio=(abs(dv)/(self.config["max_speed"]-self.config["min_speed"]))
        reward+=1-ratio
        
        # print(dv)
        ## LC reward
        
        rewardlc=0
        if type(self.controlled_vehicles[0])==HamVehicle:
            rewardlc= -self.config["lane_change_penalty"] if self.controlled_vehicles[0].isLC else 0
            if self.controlled_vehicles[0].isLC:
                # print("LC")
                self.controlled_vehicles[0].isLC=False
        

        return c*reward+rewardlc

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        if self.vehicle.crashed:
            self.crashes_counter+=1


        return self.vehicle.crashed or \
            self.steps >= self.config["duration"] or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)

    def TotalTravelDistance(self):
        total=0
        for v in self.road.vehicles:
            total+=v.totalTravelDistance()
        return total

    def getControledSpeed(self):
        return self.controlled_vehicles[0].speed
      
        


register(
    id='my-env-v0',
    entry_point='highway_env.envs:MyEnv',
)