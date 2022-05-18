from copy import deepcopy
import gurobipy as gp
from gurobipy import GRB
from highway_env.vehicle.controller import ControlledVehicle



# def ILPvehicle(ControlledVehicle):

#     def predict(self,state):
#         pass

def execute(env,lc,dir,rec):
    envr=deepcopy(env)
    #  ACTIONS_LAT = {
    #     0: 'LANE_LEFT',
    #     1: 'IDLE',
    #     2: 'LANE_RIGHT'
    # }

    ## modify in case of lc (according to ILP definition)
    act = 1 if dir == 0 else (0 if dir==1 else dir)

    obs, reward, terminal, info=envr.step(act)

    acc=1 if terminal else 0
    err=obs["speed"][2]

    return err,acc



def solve(env):
    try:
        m=gp.Model()


        ## constants
        pen=0.4
        rec=1
        
        
        # Create Variables
        lc = m.addVar(vtype=GRB.INTEGER, name="lc")
        dir = m.addVar(vtype=GRB.INTEGER, name="dir")

        # Set the Objective
        m.setObjective(()  - pen*lc, GRB.MAXIMIZE)

        # set Constraints
        m.addConstr(x+ 2*y + 3*z <=4,"c0")
        m.addConstr(x+ y >=1,"c1")


        ## optimise model
        m.optimize()

        for v in m.getVars():
            print(v)

        print("Obj=",m.objVal)


    except gp.Gurobi as e:
        print("Error 1")
    except AttributeError:
        print("Error 2")
    
    
    
