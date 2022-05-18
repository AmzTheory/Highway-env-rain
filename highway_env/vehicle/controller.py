from typing import List, Tuple, Union

import numpy as np
import copy
from highway_env import utils
from highway_env.road.road import Road, LaneIndex, Route
from highway_env.utils import Vector
from highway_env.vehicle.kinematics import Vehicle


class ControlledVehicle(Vehicle):
    """
    A vehicle piloted by two low-level controller, allowing high-level actions such as cruise control and lane changes.

    - The longitudinal controller is a speed controller;
    - The lateral controller is a heading controller cascaded with a lateral position controller.
    """

    target_speed: float
    """ Desired velocity."""

    """Characteristic time"""
    TAU_ACC = 0.6  # [s]
    TAU_HEADING = 0.2  # [s]
    TAU_LATERAL = 0.6  # [s]

    TAU_PURSUIT = 0.5 * TAU_HEADING  # [s]
    KP_A = 1 / TAU_ACC
    KP_HEADING = 1 / TAU_HEADING
    KP_LATERAL = 1 / TAU_LATERAL  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 3  # [rad]
    DELTA_SPEED = 5  # [m/s]

    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: LaneIndex = None,
                 target_speed: float = None,
                 route: Route = None):
        super().__init__(road, position, heading, speed)
        self.target_lane_index = target_lane_index or self.lane_index
        self.target_speed = target_speed or self.speed
        self.route = route

    @classmethod
    def create_from(cls, vehicle: "ControlledVehicle") -> "ControlledVehicle":
        """
        Create a new vehicle from an existing one.

        The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.road, vehicle.position, heading=vehicle.heading, speed=vehicle.speed,
                target_lane_index=vehicle.target_lane_index, target_speed=vehicle.target_speed,
                route=vehicle.route)
        return v

    def plan_route_to(self, destination: str) -> "ControlledVehicle":
        """
        Plan a route to a destination in the road network

        :param destination: a node in the road network
        """
        try:
            path = self.road.network.shortest_path(self.lane_index[1], destination)
        except KeyError:
            path = []
        if path:
            self.route = [self.lane_index] + [(path[i], path[i + 1], None) for i in range(len(path) - 1)]
        else:
            self.route = [self.lane_index]
        return self

    def act(self, action: Union[dict, str] = None) -> None:
        """
        Perform a high-level action to change the desired lane or speed.

        - If a high-level action is provided, update the target speed and lane;
        - then, perform longitudinal and lateral control.

        :param action: a high-level action
        """

        
        self.follow_road()
        if action == "FASTER":
            self.target_speed += self.DELTA_SPEED
        elif action == "SLOWER":
            self.target_speed -= self.DELTA_SPEED
        elif action == "LANE_RIGHT":
            _from, _to, _id = self.target_lane_index
            target_lane_index = _from, _to, np.clip(_id + 1, 0, len(self.road.network.graph[_from][_to]) - 1)
            if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                self.target_lane_index = target_lane_index
                
        elif action == "LANE_LEFT":
            _from, _to, _id = self.target_lane_index
            target_lane_index = _from, _to, np.clip(_id - 1, 0, len(self.road.network.graph[_from][_to]) - 1)
            if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
                self.target_lane_index = target_lane_index
                
        acc=self.speed_control(self.target_speed)
        action = {"steering": self.steering_control(self.target_lane_index),
                  "acceleration": acc}

        action['steering'] = np.clip(action['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        super().act(action)

    def follow_road(self) -> None:
        """At the end of a lane, automatically switch to a next one."""
        if self.road.network.get_lane(self.target_lane_index).after_end(self.position):
            self.target_lane_index = self.road.network.next_lane(self.target_lane_index,
                                                                 route=self.route,
                                                                 position=self.position,
                                                                 np_random=self.road.np_random)

    def steering_control(self, target_lane_index: LaneIndex) -> float:
        """
        Steer the vehicle to follow the center of an given lane.

        1. Lateral position is controlled by a proportional controller yielding a lateral speed command
        2. Lateral speed command is converted to a heading reference
        3. Heading is controlled by a proportional controller yielding a heading rate command
        4. Heading rate command is converted to a steering angle

        :param target_lane_index: index of the lane to follow
        :return: a steering wheel angle command [rad]
        """
        target_lane = self.road.network.get_lane(target_lane_index)
        lane_coords = target_lane.local_coordinates(self.position)
        lane_next_coords = lane_coords[0] + self.speed * self.TAU_PURSUIT
        lane_future_heading = target_lane.heading_at(lane_next_coords)



        # (Position) Lateral position control
        lateral_speed_command = - self.KP_LATERAL * lane_coords[1]
        # Lateral speed to heading
        heading_command = np.arcsin(np.clip(lateral_speed_command / utils.not_zero(self.speed), -1, 1))


        # Heading Control
        heading_ref = lane_future_heading + np.clip(heading_command, -np.pi/4, np.pi/4)
        # Heading control
        heading_rate_command = self.KP_HEADING * utils.wrap_to_pi(heading_ref - self.heading)
        # Heading rate to steering angle
        steering_angle = np.arcsin(np.clip(self.LENGTH / 2 / utils.not_zero(self.speed) * heading_rate_command,
                                           -1, 1))
        steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        return float(steering_angle)

    def speed_control(self, target_speed: float) -> float:
        """
        Control the speed of the vehicle.

        Using a simple proportional controller.

        :param target_speed: the desired speed
        :return: an acceleration command [m/s2]
        """
        return self.KP_A * (target_speed - self.speed)

    def get_routes_at_intersection(self) -> List[Route]:
        """Get the list of routes that can be followed at the next intersection."""
        if not self.route:
            return []
        for index in range(min(len(self.route), 3)):
            try:
                next_destinations = self.road.network.graph[self.route[index][1]]
            except KeyError:
                continue
            if len(next_destinations) >= 2:
                break
        else:
            return [self.route]
        next_destinations_from = list(next_destinations.keys())
        routes = [self.route[0:index+1] + [(self.route[index][1], destination, self.route[index][2])]
                  for destination in next_destinations_from]
        return routes

    def set_route_at_intersection(self, _to: int) -> None:
        """
        Set the road to be followed at the next intersection.

        Erase current planned route.

        :param _to: index of the road to follow at next intersection, in the road network
        """

        routes = self.get_routes_at_intersection()
        if routes:
            if _to == "random":
                _to = self.road.np_random.randint(len(routes))
            self.route = routes[_to % len(routes)]

    def predict_trajectory_constant_speed(self, times: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        """
        Predict the future positions of the vehicle along its planned route, under constant speed

        :param times: timesteps of prediction
        :return: positions, headings
        """
        coordinates = self.lane.local_coordinates(self.position)
        route = self.route or [self.lane_index]
        return tuple(zip(*[self.road.network.position_heading_along_route(route, coordinates[0] + self.speed * t, 0)
                     for t in times]))


class MDPVehicle(ControlledVehicle):

    """A controlled vehicle with a specified discrete range of allowed target speeds."""

    SPEED_COUNT: int = 3  # []
    SPEED_MIN: float = 20  # [m/s]
    SPEED_MAX: float = 30  # [m/s]

    def __init__(self,
                 road: Road,
                 position: List[float],
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: LaneIndex = None,
                 target_speed: float = None,
                 route: Route = None) -> None:
        super().__init__(road, position, heading, speed, target_lane_index, target_speed, route)
        self.speed_index = self.speed_to_index(self.target_speed)
        self.target_speed = self.index_to_speed(self.speed_index)

    def act(self, action: Union[dict, str] = None) -> None:
        """
        Perform a high-level action.

        - If the action is a speed change, choose speed from the allowed discrete range.
        - Else, forward action to the ControlledVehicle handler.

        :param action: a high-level action
        """
        if action == "FASTER":
            self.speed_index = self.speed_to_index(self.speed) + 1
        elif action == "SLOWER":
            self.speed_index = self.speed_to_index(self.speed) - 1
        else:
            super().act(action)
            return

        self.speed_index = int(np.clip(self.speed_index, 0, self.SPEED_COUNT - 1))
        self.target_speed = self.index_to_speed(self.speed_index)
        super().act()

    def index_to_speed(self, index: int) -> float:
        """
        Convert an index among allowed speeds to its corresponding speed

        :param index: the speed index []
        :return: the corresponding speed [m/s]
        """
        if self.SPEED_COUNT > 1:
            return self.SPEED_MIN + index * (self.SPEED_MAX - self.SPEED_MIN) / (self.SPEED_COUNT - 1)
        else:
            return self.SPEED_MIN

    def speed_to_index(self, speed: float) -> int:
        """
        Find the index of the closest speed allowed to a given speed.

        :param speed: an input speed [m/s]
        :return: the index of the closest speed allowed []
        """
        x = (speed - self.SPEED_MIN) / (self.SPEED_MAX - self.SPEED_MIN)
        return np.int(np.clip(np.round(x * (self.SPEED_COUNT - 1)), 0, self.SPEED_COUNT - 1))

    @classmethod
    def speed_to_index_default(cls, speed: float) -> int:
        """
        Find the index of the closest speed allowed to a given speed.

        :param speed: an input speed [m/s]
        :return: the index of the closest speed allowed []
        """
        x = (speed - cls.SPEED_MIN) / (cls.SPEED_MAX - cls.SPEED_MIN)
        return np.int(np.clip(np.round(x * (cls.SPEED_COUNT - 1)), 0, cls.SPEED_COUNT - 1))

    @classmethod
    def get_speed_index(cls, vehicle: Vehicle) -> int:
        return getattr(vehicle, "speed_index", cls.speed_to_index_default(vehicle.speed))

    def predict_trajectory(self, actions: List, action_duration: float, trajectory_timestep: float, dt: float) \
            -> List[ControlledVehicle]:
        """
        Predict the future trajectory of the vehicle given a sequence of actions.

        :param actions: a sequence of future actions.
        :param action_duration: the duration of each action.
        :param trajectory_timestep: the duration between each save of the vehicle state.
        :param dt: the timestep of the simulation
        :return: the sequence of future states
        """
        states = []
        v = copy.deepcopy(self)
        t = 0
        for action in actions:
            v.act(action)  # High-level decision
            for _ in range(int(action_duration / dt)):
                t += 1
                v.act()  # Low-level control action
                v.step(dt)
                if (t % int(trajectory_timestep / dt)) == 0:
                    states.append(copy.deepcopy(v))
        return states


class HamVehicle(MDPVehicle):
     # Longitudinal policy parameters
    # ACC_MAX = 6.0  # [m/s2]
    # """Maximum acceleration."""
    # COMFORT_ACC_MIN = -5.0  # [m/s2]
    # """Desired maximum deceleration."""
    # # Lateral policy parameters
    # POLITENESS = 0.  # in [0, 1]
    # LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    # LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    # LANE_CHANGE_DELAY = 1  # [s]

    # Longitudinal policy parameters
    ACC_MAX = 6.0  # [m/s2]
    """Maximum acceleration."""

    COMFORT_ACC_MAX = 3  # [m/s2]
    """Desired maximum acceleration."""

    COMFORT_ACC_MIN = -3  # [m/s2]
    """Desired maximum deceleration."""

    DISTANCE_WANTED =2     #1+ ControlledVehicle.LENGTH  # [m]
    """Desired jam distance to the front vehicle."""

    TIME_WANTED = 0.5  # [s]
    """Desired time gap to the front vehicle."""

    # ACC_MAX = 6.0  # [m/s2]
    # """Maximum acceleration."""

    # COMFORT_ACC_MAX = 0.7  # [m/s2]
    # """Desired maximum acceleration."""

    # COMFORT_ACC_MIN = -1.7  # [m/s2]
    # """Desired maximum deceleration."""

    # DISTANCE_WANTED =2     #1+ ControlledVehicle.LENGTH  # [m]
    # """Desired jam distance to the front vehicle."""

    # TIME_WANTED = 1.6  # [s]
    """Desired time gap to the front vehicle."""

    DELTA = 4.0  # []
    """Exponent of the velocity term."""

    DELTA_RANGE = [3.5, 4.5]
    """Range of delta when chosen randomly."""

    def __init__(self, road: Road, position: List[float], heading: float = 0, speed: float = 0, target_lane_index: LaneIndex = None, target_speed: float = None, route: Route = None) -> None:
        super().__init__(road, position, heading=heading, speed=speed, target_lane_index=target_lane_index, target_speed=target_speed, route=route)

        ## init V & M  [0,-1,-2]
        self.V=np.array([[0,0,0]])
        self.isLC=False
        # self.R=np.array([[50,50,0,0]])
        self.mass=94 # in slugs  == 1360kg
        self.text=0.5 # texture depth in mm

    def initSnapShot(self,shape):
        shape=np.asarray((shape),dtype=np.int)
        self.M=np.array(np.zeros((3,shape[0],shape[1])))

    def speed_control(self, target_speed: float) -> float:
        # Longitudinal: Krauss
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self, self.lane_index)
        acc = self.acceleration(ego_vehicle=self,
                                front_vehicle=front_vehicle,
                                rear_vehicle=rear_vehicle)
        # When changing lane, check both current and target lanes
        if self.lane_index != self.target_lane_index:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self, self.target_lane_index)
            target_idm_acceleration = self.acceleration(ego_vehicle=self,
                                                        front_vehicle=front_vehicle,
                                                        rear_vehicle=rear_vehicle)
            
            acc= min(acc, target_idm_acceleration)

        ## consider v_rain
        # v_r=self.v_rain()
        # v_acc=v_r-self.speed
        
        # acc=min(acc,v_acc)
        acc = np.clip(acc, -self.ACC_MAX, self.ACC_MAX)



        ## consider friction
        kgmass=SlugsToKg(self.mass)
        Facc=(kgmass)*acc
        maxacc=self.maxAcc()/kgmass
        minacc=self.minAcc()
        roll=self.rollRes()
        aero=self.AeroRes()

        acc= (Facc- roll - aero)/(kgmass)

        # if accelerating: acc<maxacc otherwise set maximum acc
        acc = min(acc,maxacc)
        # else if: acc >  minacc  otherweise set minimum acc
        acc = max(acc,minacc)
        


        return acc

     ## set the weather conditions 
    def setWeatherCond(self,depth=0,intensity=0):
        
        self.intensity=intensity
        self.depth=depth
        self.fric=self.computeFriction()

        # print("intentsity:",self.intensity,"--depth:",self.depth,"--friction:",self.fric)
        
    def computeCofficients(self):
        a=2.9323*(self.text**2) - 7.2792*(self.text) + 0.7305
        b=50.64*(self.text**(0.0986))
        return a,b
    def computeFriction(self):
        ##TODO: include the skid resistance in the rain-intensity paper
        a,b = self.computeCofficients()

        f=a*(np.log(self.depth))+b
        return f/100  ## convert Bpn value to friction cofficient(skid resistance)

    def acceleration(self,
                     ego_vehicle: ControlledVehicle,
                     front_vehicle: Vehicle = None,
                     rear_vehicle: Vehicle = None) -> float:
        """
        Compute an acceleration command with the Intelligent Driver Model.

        The acceleration is chosen so as to:
        - reach a target speed;
        - maintain a minimum safety distance (and safety time) w.r.t the front vehicle.

        :param ego_vehicle: the vehicle whose desired acceleration is to be computed. It does not have to be an
                            IDM vehicle, which is why this method is a class method. This allows an IDM vehicle to
                            reason about other vehicles behaviors even though they may not IDMs.
        :param front_vehicle: the vehicle preceding the ego-vehicle
        :param rear_vehicle: the vehicle following the ego-vehicle
        :return: the acceleration command for the ego-vehicle [m/s2]
        """
        if not ego_vehicle or not isinstance(ego_vehicle, Vehicle):
            return 0
        ego_target_speed = abs(utils.not_zero(getattr(ego_vehicle, "target_speed", 0)))
        acceleration = self.COMFORT_ACC_MAX * (
                1 - np.power(max(ego_vehicle.speed, 0) / ego_target_speed, self.DELTA))

        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle)
            
            acceleration -= self.COMFORT_ACC_MAX * \
                np.power(self.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2)
        return acceleration

    def desired_gap(self, ego_vehicle: Vehicle, front_vehicle: Vehicle = None, projected: bool = True) -> float:
        """
        Compute the desired distance between a vehicle and its leading vehicle.

        :param ego_vehicle: the vehicle being controlled
        :param front_vehicle: its leading vehicle
        :param projected: project 2D velocities in 1D space
        :return: the desired distance between the two [m]
        """
        d0 = self.DISTANCE_WANTED
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction) if projected \
            else ego_vehicle.speed - front_vehicle.speed
        d_star = d0 + ego_vehicle.speed * tau + ego_vehicle.speed * dv / (2 * np.sqrt(ab))
        return d_star
   
    
    def v_rain(self,quad=True):
        v=0
        if quad:
            v=(0.033*(self.depth**2)) - (0.016*(self.intensity**2)) -(1.595*self.depth)-(1.067*self.intensity) + 51.013
        else:
            v=-(-0.615*self.depth)-(1.248*self.intensity)+46.6


        ## convert km/s to m/s
        v=v/3.6

        # v=np.random.normal(v,0.8)

        return v
    def rollingcofficient(self):
        v=msTofs(self.speed)
        return 0.01 * (1+v/147)
    def rollRes(self):
        # W=g=32.17   g in ft/s^2
        W=32.17*self.mass
        # rolling cofficient (using v)
        cof = self.rollingcofficient()

        ## roll resistance in Pounds 
        resP=cof * W

        ## convert to Newton
        resN=poundsToNewton(resP)

        return resN

    def AeroRes(self):
        ## height of the vehicle
        h=1.5
        ## cofficient of drag
        c=0.3  
        ## air density(in kg/m^3)  resonable values??
        ad=1.3 # (assuming the vehicle driving on sea surface based on table in the book)
        ## frontal Area (h*w)
        area=h* self.WIDTH 

        ## compute the Aerodynamics resistance(in N)
        aero=(ad/2)*c*area*(self.speed**2)

        return aero

    def maxAcc(self):

    
        # center of gravity computation (current assumptions)
        h=0.537
        l=3
        lr=(l/2) ## equally distributed 

        # compute max acceleration
        w=32.17 * self.mass
        mu=self.fric
        cof=self.rollingcofficient()

        t1=(w * mu)*(lr+(cof*h))
        accF=(t1/l)/(1+(mu*h/l))
        acc=poundsToNewton(accF)

        return acc
    def minAcc(self):
        mu=self.fric
        minAcc=-mu*9.81 #in N
        return minAcc

    
def kgToSlugs(val):
    return val/14.594

def poundsToNewton(val):
    return val*4.448
    
def msTofs(val):
    return val*3.28

def SlugsToKg(val):
    return val*14.594
    