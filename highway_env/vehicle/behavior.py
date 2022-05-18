from typing import Tuple, Union

import numpy as np

from highway_env.road.road import Road, Route, LaneIndex
from highway_env.utils import Vector
from highway_env.vehicle.controller import ControlledVehicle, HamVehicle
from highway_env import utils
from highway_env.vehicle.kinematics import Vehicle


class IDMVehicle(ControlledVehicle):
    """
    A vehicle using both a longitudinal and a lateral decision policies.

    - Longitudinal: the IDM model computes an acceleration given the preceding vehicle's distance and speed.
    - Lateral: the MOBIL model decides when to change lane by maximizing the acceleration of nearby vehicles.
    """


    # Longitudinal policy parameters
    ACC_MAX = 6.0  # [m/s2]
    """Maximum acceleration."""

    # COMFORT_ACC_MAX = 3.0  # [m/s2]
    # """Desired maximum acceleration."""

    COMFORT_ACC_MIN = -5.0  # [m/s2]
    """Desired maximum deceleration."""

    DISTANCE_WANTED = 5.0 + ControlledVehicle.LENGTH  # [m]
    """Desired jam distance to the front vehicle."""

    TIME_WANTED = 1.5  # [s]
    """Desired time gap to the front vehicle."""

    DELTA = 4.0  # []
    """Exponent of the velocity term."""

    DELTA_RANGE = [3.5, 4.5]
    """Range of delta when chosen randomly."""

    # Lateral policy parameters
    POLITENESS = 0.  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]

    


    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: int = None,
                 target_speed: float = None,
                 route: Route = None,
                 enable_lane_change: bool = True,
                 timer: float = None,
                 control: bool= False ):
        super().__init__(road, position, heading, speed, target_lane_index, target_speed, route)
        self.enable_lane_change = enable_lane_change
        self.timer = timer or (np.sum(self.position)*np.pi) % self.LANE_CHANGE_DELAY
        ## those parameters are only needed when evaluating IDM ego-vehicle
        self.control=control
        self.lcr = 0
        self.lcl = 0
        # self.initModelParameters()
        ## rain needed
        self.mass=94 # in slugs  == 1360kg
        self.text=0.5 # texture depth in mm


    ## introduce uncertainty in the HV, by defining the model parameter from set of distributions
    ## More details on distribution used  from https://zachary.sunberg.net/thesis.pdf   (Table 3.1)
    ## Currently use scenario-1 

    ##Future work 
    ## experiment with scenario-2

    def initModelParameters(self,minspeed,maxspeed):

        ##IDM parameters
        self.target_speed=np.random.uniform(27.8, 38.9)
        self.TIME_WANTED=np.random.uniform(1,2)
        self.DISTANCE_WANTED=np.random.uniform(0,4)
        self.COMFORT_ACC_MAX=np.random.uniform(0.8,2)
        self.COMFORT_ACC_MIN=-np.random.uniform(1,3)
        
        #MOBIL parmaeters 
        # self.POLITENESS=np.random.uniform(0,1)
        self.POLITENESS=np.random.uniform(0,0)
        self.LANE_CHANGE_MAX_BRAKING_IMPOSED= np.random.uniform(1,3)
        self.LANE_CHANGE_MIN_ACC_GAIN=np.random.uniform(0,0.2)
        self.LANE_CHANGE_DELAY=np.random.uniform(1,4)
    
    def setType(self,type):
        if type=="agg":
            # self.TIME_WANTED=1.0
            # self.DISTANCE_WANTED=0.0
            # self.COMFORT_ACC_MAX=2.0
            # self.COMFORT_ACC_MIN=-3.0
            # self.POLITENESS=0.0
            # self.LANE_CHANGE_MAX_BRAKING_IMPOSED=3
            # self.LANE_CHANGE_MIN_ACC_GAIN=0.1
            self.TIME_WANTED=0.5
            self.DISTANCE_WANTED=2
            self.COMFORT_ACC_MAX=3
            self.COMFORT_ACC_MIN=-3
            self.POLITENESS=0.0
            self.LANE_CHANGE_MAX_BRAKING_IMPOSED=2
            self.LANE_CHANGE_MIN_ACC_GAIN=0.1
        elif type=="nor":
            self.TIME_WANTED=1.5
            self.DISTANCE_WANTED=2
            self.COMFORT_ACC_MAX=1.4
            self.COMFORT_ACC_MIN=-2
            self.POLITENESS=0.5
            self.LANE_CHANGE_MAX_BRAKING_IMPOSED=2
            self.LANE_CHANGE_MIN_ACC_GAIN=0.1
        elif type=="tim":
            self.TIME_WANTED=2
            self.DISTANCE_WANTED=4
            self.COMFORT_ACC_MAX=0.8
            self.COMFORT_ACC_MIN=-1
            self.POLITENESS=1
            self.LANE_CHANGE_MAX_BRAKING_IMPOSED=1
            self.LANE_CHANGE_MIN_ACC_GAIN=0.2


    def randomize_behavior(self):
        self.DELTA = self.road.np_random.uniform(low=self.DELTA_RANGE[0], high=self.DELTA_RANGE[1])

    @classmethod
    def create_from(cls, vehicle: ControlledVehicle) -> "IDMVehicle":
        """
        Create a new vehicle from an existing one.

        The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.road, vehicle.position, heading=vehicle.heading, speed=vehicle.speed,
                target_lane_index=vehicle.target_lane_index, target_speed=vehicle.target_speed,
                route=vehicle.route, timer=getattr(vehicle, 'timer', None))
        return v

    def act(self, action: Union[dict, str] = None):
        """
        Execute an action.

        For now, no action is supported because the vehicle takes all decisions
        of acceleration and lane changes on its own, based on the IDM and MOBIL models.

        :param action: the action
        """
        if self.crashed:
            return
        action = {}
        # Lateral: MOBIL
        self.follow_road()

        ## lc policy
        if self.enable_lane_change:
            self.change_lane_policy()

        action['steering'] = self.steering_control(self.target_lane_index)
        action['steering'] = np.clip(action['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        # Longitudinal: IDM
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self, self.lane_index)
        action['acceleration'] = self.acceleration(ego_vehicle=self,
                                                   front_vehicle=front_vehicle,
                                                   rear_vehicle=rear_vehicle)
        # When changing lane, check both current and target lanes
        if self.lane_index != self.target_lane_index:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self, self.target_lane_index)
            target_idm_acceleration = self.acceleration(ego_vehicle=self,
                                                        front_vehicle=front_vehicle,
                                                        rear_vehicle=rear_vehicle)
            action['acceleration'] = min(action['acceleration'], target_idm_acceleration)
        # action['acceleration'] = self.recover_from_stop(action['acceleration'])
        action['acceleration'] = np.clip(action['acceleration'], -self.ACC_MAX, self.ACC_MAX)

        acc=action['acceleration']
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
        
        action['acceleration']=acc
        Vehicle.act(self, action)  # Skip ControlledVehicle.act(), or the command will be overriden.

    def step(self, dt: float):
        """
        Step the simulation.

        Increases a timer used for decision policies, and step the vehicle dynamics.

        :param dt: timestep
        """
        self.timer += dt
        super().step(dt)

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

        ## add noise (The logic included in Vehicle class)
        

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
        dsq=ego_vehicle.speed * dv / (2 * np.sqrt(ab))
        d_star = d0 + ego_vehicle.speed * tau + ego_vehicle.speed * dv / (2 * np.sqrt(ab))
        return d_star

    def change_lane_policy(self) -> None:
        """
        Decide when to change lane.

        Based on:
        - frequency;
        - closeness of the target lane;
        - MOBIL model.
        """
        # If a lane change already ongoing
        if self.lane_index != self.target_lane_index:
            # If we are on correct route but bad lane: abort it if someone else is already changing into the same lane
            if self.lane_index[:2] == self.target_lane_index[:2]:
                for v in self.road.vehicles:
                    if v is not self \
                            and v.lane_index != self.target_lane_index \
                            and isinstance(v, ControlledVehicle) \
                            and v.target_lane_index == self.target_lane_index:
                        d = self.lane_distance_to(v)
                        d_star = self.desired_gap(self, v)
                        if 0 < d < d_star:
                            self.target_lane_index = self.lane_index
                            break
            return

        # else, at a given frequency,
        if not utils.do_every(self.LANE_CHANGE_DELAY, self.timer):
            return
        self.timer = 0

        # decide to make a lane change
        for lane_index in self.road.network.side_lanes(self.lane_index):
            # Is the candidate lane close enough?
            if not self.road.network.get_lane(lane_index).is_reachable_from(self.position):
                continue
            # Does the MOBIL model recommend a lane change?
            if self.mobil(lane_index):
                if self.control:
                   if lane_index[2] < self.lane_index[2]: ## LEFT
                       self.lcl += 1
                   elif lane_index[2] > self.lane_index[2]: ## RIGHT
                       self.lcr += 1
                else:
                    self.controlact = 1 #NO LC
                self.target_lane_index = lane_index
                
                    

    def mobil(self, lane_index: LaneIndex) -> bool:
        """
        MOBIL lane change model: Minimizing Overall Braking Induced by a Lane change

            The vehicle should change lane only if:
            - after changing it (and/or following vehicles) can accelerate more;
            - it doesn't impose an unsafe braking on its new following vehicle.

        :param lane_index: the candidate lane for the change
        :return: whether the lane change should be performed
        """
        # Is the maneuver unsafe for the new following vehicle?
        new_preceding, new_following = self.road.neighbour_vehicles(self, lane_index)
        new_following_a = self.acceleration(ego_vehicle=new_following, front_vehicle=new_preceding)
        new_following_pred_a = self.acceleration(ego_vehicle=new_following, front_vehicle=self)
        if new_following_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return False

        # Do I have a planned route for a specific lane which is safe for me to access?
        old_preceding, old_following = self.road.neighbour_vehicles(self)
        self_pred_a = self.acceleration(ego_vehicle=self, front_vehicle=new_preceding)
        if self.route and self.route[0][2]:
            # Wrong direction
            if np.sign(lane_index[2] - self.target_lane_index[2]) != np.sign(self.route[0][2] - self.target_lane_index[2]):
                return False
            # Unsafe braking required
            elif self_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
                return False

        # Is there an acceleration advantage for me and/or my followers to change lane?
        else:
            self_a = self.acceleration(ego_vehicle=self, front_vehicle=old_preceding)
            old_following_a = self.acceleration(ego_vehicle=old_following, front_vehicle=self)
            old_following_pred_a = self.acceleration(ego_vehicle=old_following, front_vehicle=old_preceding)
            jerk = self_pred_a - self_a + self.POLITENESS * (new_following_pred_a - new_following_a
                                                             + old_following_pred_a - old_following_a)
            if jerk < self.LANE_CHANGE_MIN_ACC_GAIN:
                return False

        # All clear, let's go!
        return True

    def recover_from_stop(self, acceleration: float) -> float:
        """
        If stopped on the wrong lane, try a reversing maneuver.

        :param acceleration: desired acceleration from IDM
        :return: suggested acceleration to recover from being stuck
        """
        stopped_speed = 5
        safe_distance = 200
        # Is the vehicle stopped on the wrong lane?
        if self.target_lane_index != self.lane_index and self.speed < stopped_speed:
            _, rear = self.road.neighbour_vehicles(self)
            _, new_rear = self.road.neighbour_vehicles(self, self.road.network.get_lane(self.target_lane_index))
            # Check for free room behind on both lanes
            if (not rear or rear.lane_distance_to(self) > safe_distance) and \
                    (not new_rear or new_rear.lane_distance_to(self) > safe_distance):
                # Reverse
                return -self.COMFORT_ACC_MAX / 2
        return acceleration

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

class LinearVehicle(IDMVehicle):

    """A Vehicle whose longitudinal and lateral controllers are linear with respect to parameters."""

    ACCELERATION_PARAMETERS = [0.3, 0.3, 2.0]
    STEERING_PARAMETERS = [ControlledVehicle.KP_HEADING, ControlledVehicle.KP_HEADING * ControlledVehicle.KP_LATERAL]

    ACCELERATION_RANGE = np.array([0.5*np.array(ACCELERATION_PARAMETERS), 1.5*np.array(ACCELERATION_PARAMETERS)])
    STEERING_RANGE = np.array([np.array(STEERING_PARAMETERS) - np.array([0.07, 1.5]),
                               np.array(STEERING_PARAMETERS) + np.array([0.07, 1.5])])

    TIME_WANTED = 2.5

    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: int = None,
                 target_speed: float = None,
                 route: Route = None,
                 enable_lane_change: bool = True,
                 timer: float = None,
                 data: dict = None):
        super().__init__(road, position, heading, speed, target_lane_index, target_speed, route,
                         enable_lane_change, timer)
        self.data = data if data is not None else {}
        self.collecting_data = True

    def act(self, action: Union[dict, str] = None):
        if self.collecting_data:
            self.collect_data()
        super().act(action)

    def randomize_behavior(self):
        ua = self.road.np_random.uniform(size=np.shape(self.ACCELERATION_PARAMETERS))
        self.ACCELERATION_PARAMETERS = self.ACCELERATION_RANGE[0] + ua*(self.ACCELERATION_RANGE[1] -
                                                                        self.ACCELERATION_RANGE[0])
        ub = self.road.np_random.uniform(size=np.shape(self.STEERING_PARAMETERS))
        self.STEERING_PARAMETERS = self.STEERING_RANGE[0] + ub*(self.STEERING_RANGE[1] - self.STEERING_RANGE[0])

    def acceleration(self,
                     ego_vehicle: ControlledVehicle,
                     front_vehicle: Vehicle = None,
                     rear_vehicle: Vehicle = None) -> float:
        """
        Compute an acceleration command with a Linear Model.

        The acceleration is chosen so as to:
        - reach a target speed;
        - reach the speed of the leading (resp following) vehicle, if it is lower (resp higher) than ego's;
        - maintain a minimum safety distance w.r.t the leading vehicle.

        :param ego_vehicle: the vehicle whose desired acceleration is to be computed. It does not have to be an
                            Linear vehicle, which is why this method is a class method. This allows a Linear vehicle to
                            reason about other vehicles behaviors even though they may not Linear.
        :param front_vehicle: the vehicle preceding the ego-vehicle
        :param rear_vehicle: the vehicle following the ego-vehicle
        :return: the acceleration command for the ego-vehicle [m/s2]
        """
        return float(np.dot(self.ACCELERATION_PARAMETERS,
                            self.acceleration_features(ego_vehicle, front_vehicle, rear_vehicle)))

    def acceleration_features(self, ego_vehicle: ControlledVehicle,
                              front_vehicle: Vehicle = None,
                              rear_vehicle: Vehicle = None) -> np.ndarray:
        vt, dv, dp = 0, 0, 0
        if ego_vehicle:
            vt = ego_vehicle.target_speed - ego_vehicle.speed
            d_safe = self.DISTANCE_WANTED + np.maximum(ego_vehicle.speed, 0) * self.TIME_WANTED
            if front_vehicle:
                d = ego_vehicle.lane_distance_to(front_vehicle)
                dv = min(front_vehicle.speed - ego_vehicle.speed, 0)
                dp = min(d - d_safe, 0)
        return np.array([vt, dv, dp])

    def steering_control(self, target_lane_index: LaneIndex) -> float:
        """
        Linear controller with respect to parameters.

        Overrides the non-linear controller ControlledVehicle.steering_control()

        :param target_lane_index: index of the lane to follow
        :return: a steering wheel angle command [rad]
        """
        return float(np.dot(np.array(self.STEERING_PARAMETERS), self.steering_features(target_lane_index)))

    def steering_features(self, target_lane_index: LaneIndex) -> np.ndarray:
        """
        A collection of features used to follow a lane

        :param target_lane_index: index of the lane to follow
        :return: a array of features
        """
        lane = self.road.network.get_lane(target_lane_index)
        lane_coords = lane.local_coordinates(self.position)
        lane_next_coords = lane_coords[0] + self.speed * self.TAU_PURSUIT
        lane_future_heading = lane.heading_at(lane_next_coords)
        features = np.array([utils.wrap_to_pi(lane_future_heading - self.heading) *
                             self.LENGTH / utils.not_zero(self.speed),
                             -lane_coords[1] * self.LENGTH / (utils.not_zero(self.speed) ** 2)])
        return features

    def longitudinal_structure(self):
        # Nominal dynamics: integrate speed
        A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        # Target speed dynamics
        phi0 = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, -1]
        ])
        # Front speed control
        phi1 = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, -1, 1],
            [0, 0, 0, 0]
        ])
        # Front position control
        phi2 = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [-1, 1, -self.TIME_WANTED, 0],
            [0, 0, 0, 0]
        ])
        # Disable speed control
        front_vehicle, _ = self.road.neighbour_vehicles(self)
        if not front_vehicle or self.speed < front_vehicle.speed:
            phi1 *= 0

        # Disable front position control
        if front_vehicle:
            d = self.lane_distance_to(front_vehicle)
            if d != self.DISTANCE_WANTED + self.TIME_WANTED * self.speed:
                phi2 *= 0
        else:
            phi2 *= 0

        phi = np.array([phi0, phi1, phi2])
        return A, phi

    def lateral_structure(self):
        A = np.array([
            [0, 1],
            [0, 0]
        ])
        phi0 = np.array([
            [0, 0],
            [0, -1]
        ])
        phi1 = np.array([
            [0, 0],
            [-1, 0]
        ])
        phi = np.array([phi0, phi1])
        return A, phi

    def collect_data(self):
        """Store features and outputs for parameter regression."""
        self.add_features(self.data, self.target_lane_index)

    def add_features(self, data, lane_index, output_lane=None):

        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)
        features = self.acceleration_features(self, front_vehicle, rear_vehicle)
        output = np.dot(self.ACCELERATION_PARAMETERS, features)
        if "longitudinal" not in data:
            data["longitudinal"] = {"features": [], "outputs": []}
        data["longitudinal"]["features"].append(features)
        data["longitudinal"]["outputs"].append(output)

        if output_lane is None:
            output_lane = lane_index
        features = self.steering_features(lane_index)
        out_features = self.steering_features(output_lane)
        output = np.dot(self.STEERING_PARAMETERS, out_features)
        if "lateral" not in data:
            data["lateral"] = {"features": [], "outputs": []}
        data["lateral"]["features"].append(features)
        data["lateral"]["outputs"].append(output)


class AggressiveVehicle(LinearVehicle):
    LANE_CHANGE_MIN_ACC_GAIN = 1.0  # [m/s2]
    MERGE_ACC_GAIN = 0.8
    MERGE_VEL_RATIO = 0.75
    MERGE_TARGET_VEL = 30
    ACCELERATION_PARAMETERS = [MERGE_ACC_GAIN / ((1 - MERGE_VEL_RATIO) * MERGE_TARGET_VEL),
                               MERGE_ACC_GAIN / (MERGE_VEL_RATIO * MERGE_TARGET_VEL),
                               0.5]


class DefensiveVehicle(LinearVehicle):
    LANE_CHANGE_MIN_ACC_GAIN = 1.0  # [m/s2]
    MERGE_ACC_GAIN = 1.2
    MERGE_VEL_RATIO = 0.75
    MERGE_TARGET_VEL = 30
    ACCELERATION_PARAMETERS = [MERGE_ACC_GAIN / ((1 - MERGE_VEL_RATIO) * MERGE_TARGET_VEL),
                               MERGE_ACC_GAIN / (MERGE_VEL_RATIO * MERGE_TARGET_VEL),
                               2.0]


class KraussVehicle(ControlledVehicle):

    # Longitudinal policy parameters
    ACC_MAX = 6.0  # [m/s2]
    """Maximum acceleration."""
    COMFORT_ACC_MIN = -5.0  # [m/s2]
    """Desired maximum deceleration."""
    # Lateral policy parameters
    POLITENESS = 0 # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.3  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 3  # [s]


    def __init__(self,
                 road: Road,
                 position: Vector,
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: int = None,
                 target_speed: float = None,
                 route: Route = None,
                 enable_lane_change: bool = True,
                 timer: float = None,
                 control: bool= False):
        super().__init__(road, position, heading, speed, target_lane_index, target_speed, route)
        self.enable_lane_change = enable_lane_change
        self.timer = timer or (np.sum(self.position)*np.pi) % self.LANE_CHANGE_DELAY
        ## those parameters are only needed when evaluating IDM ego-vehicle
        self.control=control
        self.lcr = 0
        self.lcl = 0
        self.LANE_CHANGE_MIN_ACC_GAIN=np.random.uniform(0.1,0.3)
        self.LANE_CHANGE_MAX_BRAKING_IMPOSED= np.random.uniform(1,3)
        self.LANE_CHANGE_DELAY=np.random.uniform(1,4)
        


    ## set the weather conditions 
    def setWeatherCond(self,depth=0,intensity=0):
        self.depth=depth/10 ## convert to cm
        self.intensity=intensity

    def act(self, action: Union[dict, str] = None):
        """
        Execute an action.

        For now, no action is supported because the vehicle takes all decisions
        of acceleration and lane changes on its own, based on the IDM and MOBIL models.

        :param action: the action
        """
        if self.crashed:
            return
        action = {}
        # Lateral: MOBIL
        self.follow_road()

        if self.enable_lane_change:
            self.change_lane_policy()

        action['steering'] = self.steering_control(self.target_lane_index)
        action['steering'] = np.clip(action['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        # Longitudinal: Krauss
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self, self.lane_index)
        action['acceleration'] = self.acceleration(ego_vehicle=self,
                                                   front_vehicle=front_vehicle,
                                                   rear_vehicle=rear_vehicle)

        # When changing lane, check both current and target lanes
        if self.lane_index != self.target_lane_index:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self, self.target_lane_index)
            target_Krauss_acceleration = self.acceleration(ego_vehicle=self,
                                                        front_vehicle=front_vehicle,
                                                        rear_vehicle=rear_vehicle)
            action['acceleration'] = min(action['acceleration'], target_Krauss_acceleration)
        # action['acceleration'] = self.recover_from_stop(action['acceleration'])


          ## add noise (The logic included in Vehicle class)
        ext=self.COMFORT_ACC_MAX/2
        action['acceleration']= action['acceleration'] + np.random.triangular(-ext,0,ext)
        action['acceleration'] = np.clip(action['acceleration'], -self.ACC_MAX, self.ACC_MAX)
        Vehicle.act(self, action)  # Skip ControlledVehicle.act(), or the command will be overriden.

    def step(self, dt: float):
        """
        Step the simulation.

        Increases a timer used for decision policies, and step the vehicle dynamics.

        :param dt: timestep
        """
        self.timer += dt
        super().step(dt)
        
    def acceleration(self,
                     ego_vehicle: ControlledVehicle,
                     front_vehicle: Vehicle = None,
                     rear_vehicle: Vehicle = None) -> float:
        """
        Compute an acceleration command with Krauss Car following model.

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

        
        ## shal we use it??
        # acceleration = self.COMFORT_ACC_MAX * (
        #         1 - np.power(max(ego_vehicle.speed, 0) / ego_target_speed, self.DELTA))

        ##assume reaction time of 1 sec
        
        vel_nat=ego_vehicle.speed+ego_vehicle.COMFORT_ACC_MAX 
        vel_safe=vel_nat
        v_r= self.v_rain()
        
        if front_vehicle:
            vel_safe=self.vSafe(ego_vehicle,front_vehicle)

        velocity= min([vel_safe,vel_nat,ego_target_speed,v_r])
        # eps= np.random.uniform(0,1)
        # velocity= np.random.uniform(velocity-(eps*ego_vehicle.COMFORT_ACC_MAX), velocity)
        velocity=max(0,velocity)

        

        ## compute acc
        acceleration= velocity-self.speed


      

        return acceleration
    
    def vSafe(self,ego_vehicle,front_vehicle):

        '''
            v_safe = a/b
            a      = front_vehicle.v + front_vehicle.gap - (front_vehicle.v * reaction_time)
            b      = (vehicle.speed / vehicle.dacc) + reaction time 

            assume reaction time of 1 second
        '''
        v=ego_vehicle.speed
        vl=front_vehicle.speed
        gap = ego_vehicle.lane_distance_to(front_vehicle)
        a= vl + gap - (vl*1)
        b= (v/abs(self.COMFORT_ACC_MIN))+1

        return a/b
    
    def v_rain(self,quad=True):
        v=0
        if quad:
            v=(0.033*(self.depth**2)) - (0.016*(self.intensity**2)) -(1.595*self.depth)-(1.067*self.intensity) + 51.013
        else:
            v=-(-0.615*self.depth)-(1.248*self.intensity)+46.6


        ## convert km/s to m/s
        v=v/3.6
        ## gauss (v_rain,1.39m/s)
        # v=np.random.normal(v,0.8)
        return v



    def change_lane_policy(self) -> None:
        """
        Decide when to change lane.

        Based on:
        - frequency;
        - closeness of the target lane;
        - MOBIL model.
        """
        # If a lane change already ongoing
        if self.lane_index != self.target_lane_index:
            # If we are on correct route but bad lane: abort it if someone else is already changing into the same lane
            if self.lane_index[:2] == self.target_lane_index[:2]:
                for v in self.road.vehicles:
                    if v is not self \
                            and v.lane_index != self.target_lane_index \
                            and isinstance(v, ControlledVehicle) \
                            and v.target_lane_index == self.target_lane_index:
                        d = self.lane_distance_to(v)
                        # d_star = self.desired_gap(self, v)
                        # if 0 < d < d_star:
                        if 0 < d:
                            self.target_lane_index = self.lane_index
                            break
            return

        # else, at a given frequency,
        if not utils.do_every(self.LANE_CHANGE_DELAY, self.timer):
            return
        self.timer = 0

        # decide to make a lane change
        for lane_index in self.road.network.side_lanes(self.lane_index):
            # Is the candidate lane close enough?
            if not self.road.network.get_lane(lane_index).is_reachable_from(self.position):
                continue
            # Does the MOBIL model recommend a lane change?
            if self.mobil(lane_index):
                if self.control:
                   if lane_index[2] < self.lane_index[2]: ## LEFT
                       self.lcl += 1
                   elif lane_index[2] > self.lane_index[2]: ## RIGHT
                       self.lcr += 1
                else:
                    self.controlact = 1 #NO LC
                self.target_lane_index = lane_index
                
                    

    def mobil(self, lane_index: LaneIndex) -> bool:
        """
        MOBIL lane change model: Minimizing Overall Braking Induced by a Lane change

            The vehicle should change lane only if:
            - after changing it (and/or following vehicles) can accelerate more;
            - it doesn't impose an unsafe braking on its new following vehicle.

        :param lane_index: the candidate lane for the change
        :return: whether the lane change should be performed
        """
        # Is the maneuver unsafe for the new following vehicle?
        new_preceding, new_following = self.road.neighbour_vehicles(self, lane_index)
        new_following_a = self.acceleration(ego_vehicle=new_following, front_vehicle=new_preceding)
        new_following_pred_a = self.acceleration(ego_vehicle=new_following, front_vehicle=self)
        
        if new_following_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return False

        # Do I have a planned route for a specific lane which is safe for me to access?
        old_preceding, old_following = self.road.neighbour_vehicles(self)
    
        self_pred_a = self.acceleration(ego_vehicle=self, front_vehicle=new_preceding)
        if self.route and self.route[0][2]:
            # Wrong direction
            if np.sign(lane_index[2] - self.target_lane_index[2]) != np.sign(self.route[0][2] - self.target_lane_index[2]):
                return False
            # Unsafe braking required
            elif self_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
                return False

        # Is there an acceleration advantage for me and/or my followers to change lane?
        else:
            self_a = self.acceleration(ego_vehicle=self, front_vehicle=old_preceding)
            old_following_a = self.acceleration(ego_vehicle=old_following, front_vehicle=self)
            old_following_pred_a = self.acceleration(ego_vehicle=old_following, front_vehicle=old_preceding)
            jerk = self_pred_a - self_a + self.POLITENESS * (new_following_pred_a - new_following_a
                                                             + old_following_pred_a - old_following_a)
            
            if  type(old_preceding)==HamVehicle or type(old_preceding)==IDMVehicle:
                return False

            if jerk < self.LANE_CHANGE_MIN_ACC_GAIN:
                return False

            
            

        # All clear, let's go!
        return True

def kgToSlugs(val):
    return val/14.594

def poundsToNewton(val):
    return val*4.448
    
def msTofs(val):
    return val*3.28

def SlugsToKg(val):
    return val*14.594