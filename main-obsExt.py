
from tkinter.tix import Tree
import gym
from matplotlib.pyplot import get, step
import pandas as pd
import pickle as p
import os

from sympy import LC, true
# from torch._C import TreeView
import highway_env
import random
from stable_baselines3 import DQN
from highway_env import utils
from VehicleEnv import TrainModel,CustomCombinedExtractor
import numpy as np
from gym.wrappers import Monitor
import itertools as it
from highway_env.envs.common.action import action_factory
import time
from datetime import datetime
from tensorflow.python.summary.summary_iterator import summary_iterator



def getEnv(dur=None,man=False,LC=True,pen=1,IDM=False,IDM_type="agg",eval=False,den=1,speed=60,n_vehicles=8,c=0.75,depth=20,ints=2):
    env = gym.make('my-env-v0')

    if IDM:
        env.config["action"]["veh_class"]="highway_env.vehicle.behavior.IDMVehicle"
        env.config["IDM_type"]=IDM_type
    else:
        env.config["observation"]["type"]="OccupancyGridExt"
        env.config["observation"]["grid_size"]=[[-50,50],[-6,6]]
        env.config["observation"]["grid_step"]=[(100/depth),4]
        env.config["action"]["veh_class"]="highway_env.vehicle.controller.HamVehicle"
    ## grid depth x 3 (# cells)
    
    
    
    
    env.config['manual_control']= man
    env.config['lane_change_penalty']=pen
    env.config['vehicles_density']=den
    env.config['vehicles_count']=n_vehicles
    env.config['c']=c

    if eval:
        env.config["min_speed_c"]=speed
        env.config["max_speed_c"]=speed
        env.config["ints"]=ints


    if dur:
        env.config["duration"]=dur

    # if eval:
    #     env.config["min_speed_c"]=speed
    #     env.config["max_speed_c"]=speed
    # else:
    #     env.config["min_speed_c"]=32
    #     env.config["max_speed_c"]=38.9

    ## 10 * 3=30

    env.config["other_vehicles_type"]="highway_env.vehicle.behavior.KraussVehicle"
    env.config['enable_lane_change']=LC

    return env

def mean(ls):
    return round(sum(ls)/len(ls),2)

def eval(env,avg=5,eps=10,model=None,render=False,IDM=False,env_config=dict(),LK=False):
    
    crashes=np.zeros(avg)
    nLCs=np.zeros(avg)
    nLCls=np.zeros(avg)
    nLCrs=np.zeros(avg)
    stepsls=np.zeros(avg)
    speeddev=np.zeros(avg)
    avgspeeds=np.zeros(avg)
    totaldistance=np.zeros(avg)
    alltotaldistance=np.zeros(avg)
    for r in range(avg):
        # print("run",r)
        lcs=[]
        lcrs=[]
        lcls=[]
        stepss=[]
        devs=[]
        dists=[]
        totaldists=[]
        avgspeed=[]
        # tot=0
        env.crashes_counter=0
        for ie in range(eps):
            obs=env.reset()
            prev_crash=env.crashes_counter
            done = False
            lc=0
            lcr=0
            lcl=0
            steps=0
            speed=0
            # print("eps",i)
            while not done:
                if LK:
                    action = 1
                elif not model: ## random decision
                    action =random.randint(0,2)
                else:
                    action,st=model.predict(obs,deterministic=True)
                        
                
                # if action == 0 or action == 2: ## LC is occuring
                #     lc+=1

                obs, reward, done, info = env.step(action)
                if IDM:
                    lcr=env.controlled_vehicles[0].lcr 
                    lcl=env.controlled_vehicles[0].lcl
                    lc=lcr+lcl
                elif action == 0:  ## LC is occuring
                    lc+=1
                    lcl+=1
                elif action == 2:
                    lc+=1
                    lcr+=1

                if render:
                     env.render()
                steps+=1
                speed+=env.getControledSpeed()
                # tot+=reward

            if prev_crash==env.crashes_counter:
                devs.append(env.total_dev_speed/steps)
                avgspeed.append(speed/steps)
                dists.append(env.controlled_vehicles[0].totalTravelDistance())
                totaldists.append(env.TotalTravelDistance())
                lcs.append(lc)
                lcls.append(lcl)
                lcrs.append(lcr)
            
            stepss.append(steps)

        ##average single run results (/eps)
        crashes[r]=env.crashes_counter  ## per run  
        nLCs[r]=mean(lcs)
        nLCls[r]=mean(lcls) 
        nLCrs[r]=mean(lcrs)
        stepsls[r]=mean(stepss)
        speeddev[r]=mean(devs)
        totaldistance[r]=mean(dists)
        alltotaldistance[r]=mean(totaldists)
        avgspeeds[r]=mean(avgspeed)


    # Average the results of r runs
    data=dict()
    data["Crashes"]=crashes.mean()
    data["LC"]=round(nLCs.mean(),2)
    data["LCL"]=round(nLCls.mean(),2)
    data["LCR"]=round(nLCrs.mean(),2)
    data["speed_dev"]=round(speeddev.mean(),2)
    data["Avg_speed"]=round(avgspeeds.mean(),2)
    data["eps_len"]=round(stepsls.mean(),2)
    data["ego_distance"]=round(totaldistance.mean(),2)
    data["total_distance"]=round(alltotaldistance.mean(),2)

    print("Result of ",avg," runs for current config")
    for k,v in data.items():
        print(k," : ",v)

    return data


def train(env,obj,steps=2e4,loc="local",param=list(),layers=[128,64],gamma=0.85,lr=5e-3,interval=300,eval=False,save=True):
    obs=env.reset()

    param=",".join([str(j) for j in param])
    ts=str((int(steps/1e3)))
    name=obj+"-"+loc+"-"+(param+"-" if param!="" else "")+str(ts)+"k"
    # print(name)
    
    T=TrainModel(env,name,layers,lr,interval,gamma,eval=eval)
    T.init()
    T.learn(int(steps))

   
    if save:
        T.model.save("Models/"+name)


def genVideo(model,dir,nexp=5):
    # env=getEnv(dur=50,n_vehicles=5)   
    env=getEnv(dur=50,LC=True,n_vehicles=5,pen=3,c=3,depth=20,speed=16.6,ints=12,eval=True)             
    # model = DQN.load("Models/"+model, env=env)
    env = Monitor(env, directory=dir, video_callable=lambda e: True)
    env.set_monitor(env)
    env.configure({"simulation_frequency": 15})  # Higher FPS for rendering

    for videos in range(nexp):
        done = False
        obs = env.reset()
        while not done:
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, info = env.step(action)
            # Render
            env.render()
    
def simulateEnv(env,model=None,runs=100):
    for i in range(runs):
        obs=env.reset()
        done = False        
        while not done:
            if model:
                action,_=model.predict(obs,deterministic=True)
            else:
                action=random.randint(0,2)
            
            # print(obs["snap"][0])
            # if action==0 or action==2:
            #     print("stop")
        
            obs, reward, done, info=env.step(action)
            
            env.render() 
            # print(obs["snap"][0])
            


def optimiseTrain(envparam,rates,intervals,layers,gammas,loc="local"):
    for r,inte,lay,gamma in it.product(rates,intervals,layers,gammas):
        env=getEnv(envparam["dur"],pen=envparam["pen"],n_vehicles=envparam["n_vehicles"],c=envparam["c"],depth=envparam["depth"])
        params=[r,inte,gamma]
        for l in lay:
            params.append(l)

        train(env,"OptModel",steps=10e4,loc=loc,param=params,layers=lay,gamma=gamma,lr=r,interval=inte)

def getFiles(obj):
    files=[]
    for f in  os.listdir("Models\\opt"):
        if f.startswith(obj):
            files.append(f)
    
    return files
           
def evalObjective(envparam,obj,avg=5,eps=15,fname=None):
    models=getFiles(obj)
    df=pd.DataFrame(columns=["lr","int","gam","lay","LC","speed_dev","eps_len"])
    for m in models:
        params=m.split("-")[2].split(",")
        # print(params)
        lr=float(params[0])
        interval=int(params[1])
        gam=params[2]
        lay=[int(l) for l in params[3:]]
        print(lr,interval,gam,lay)

        
        # model=DQN.load(models[(pe,ci,d)])
        model=DQN.load("Models\\"+m)
    #   speed used 36, 38.9
        env=getEnv(envparam["dur"],pen=envparam["pen"],n_vehicles=envparam["n_vehicles"],c=envparam["c"],depth=envparam["depth"])
        data=eval(env,avg=avg,eps=eps,model=model)
        data["lr"]=lr
        data["int"]=interval
        data["gam"]=gam
        data["lay"]=lay
       
        df=df.append(data,ignore_index=True)
    print(df)
    if fname:
        p.dump(df, open("Dataframes\\"+fname, 'wb'))


def evalIDM(envparam,fname=None,avg=5,eps=20):
    df=pd.DataFrame(columns=["LC","speed_dev","eps_len"])
    # for ty in ["agg"]: #,"nor","tim"]:
    env=getEnv(dur=envparam["dur"],IDM=True,IDM_type="agg",n_vehicles=envparam["n_vehicles"],eval=True,speed=envparam["speed"],ints=envparam["ints"])
    data=eval(env,avg=avg,eps=eps,IDM=True)
    # data["type"]="agg"

    df=df.append(data,ignore_index=True)

    # print(df)
    if fname:
        p.dump(df, open("Dataframes\\"+fname, 'wb'))
    return df


def selectModel(params,obj):
    files=getFiles(obj)
    for f in files:
        print(f)
        ps=f.split("-")[2].split(",")
        # print(params)
        lr=float(ps[0])
        interval=int(ps[1])
        gam=ps[2]
        lay=[int(l) for l in ps[3:]]
        if lr==params["lr"] and interval==params["int"] and gam==params["gam"] and lay==params["lay"]:
            print("found")

def optimseAgent(envparams,obj,pens,cs,depths,loc="local"):
    for p,c,d in it.product(pens,cs,depths):
        env=getEnv(dur=envparams["dur"],pen=p,n_vehicles=envparams["n_vehicles"],c=c,depth=d)
        train(env,obj,steps=10e4,param=[p,c,d],layers=[500,100])

def evalAgent(envparam,obj,avg=5,eps=15,speed=38.9,dep=4,fname=None):
    models=getFiles(obj)
    df=pd.DataFrame(columns=["pen","c","depth","LC","speed_dev","eps_len"])
    for i,m in enumerate(models):
        params=m.split("-")[2].split(",")

    
        # print(params)
        pen=float(params[0])
        c=int(params[1])
        depth=int(params[2])
        # if(depth!=dep):
        #     continue ## don't evaluate if its of depth of 4
        print(i,pen,c,depth)

    
        model=DQN.load("Models\\"+m)
        env=getEnv(envparam["dur"],pen=pen,n_vehicles=envparam["n_vehicles"],c=c,depth=depth,eval=True,speed=speed)
        data=eval(env,avg=avg,eps=eps,model=model)
        data["pen"]=pen
        data["c"]=c
        data["depth"]=depth

       
        df=df.append(data,ignore_index=True)
    print(df)
    if fname:
        p.dump(df, open("Dataframes\\"+fname, 'wb'))

def evalIDMRain(eps,fn,envparam):
    df=pd.DataFrame(columns=["ints","speed","LC","speed_dev","eps_len"])
    for i,s in it.product([2,6,12],[12.5,14.44,16.6]):
        # env=getEnv(dur=90,LC=True,n_vehicles=15,pen=3,c=3,depth=20,IDM=True)
        envparam["ints"]=i
        envparam["speed"]=s
        fname="EVAL_IDM_"+str(i)+"-"+str(s)
        print(fname)
        data=evalIDM(envparam,avg=1,eps=eps)
        data["ints"]=i
        data["speed"]=s
        ## EVAL_IDM_ints_speed
        df=df.append(data,ignore_index=True)
    p.dump(df, open("Dataframes\\"+fn, 'wb'))
def evalModel(m,eps,fn,LK=False):
    df=pd.DataFrame(columns=["ints","speed","LC","speed_dev","eps_len"])
    for i,s in it.product([2,6,12],[12.5,14.44,16.6]):
        # env=getEnv(dur=90,LC=True,n_vehicles=15,pen=3,c=3,depth=20,IDM=True)
        envparam["ints"]=i
        envparam["speed"]=s
        fname="EVAL_"+str(i)+"-"+str(s)
        print(fname)
        env=getEnv(dur=90,LC=True,n_vehicles=10,pen=3,c=3,depth=20,eval=True,speed=s,ints=i)
        data=eval(env,avg=1,eps=eps,model=m,LK=LK)
        data["ints"]=i
        data["speed"]=s
        ## EVAL_ints_speed
        df=df.append(data,ignore_index=True)
    p.dump(df, open("Dataframes\\"+fn, 'wb'))


if __name__ == "__main__":

    # envparam={
    #     "dur":90,
    #     "n_vehicles":10
    # }

    # env=getEnv(dur=90,IDM=True,IDM_type="agg",n_vehicles=envparam["n_vehicles"],eval=True,speed=16.6,ints=12)
    # simulateEnv(env)
    # eps=300
    
    #IDM
    # evalIDMRain(eps,"df_IDM_"+str(eps),envparam)

    ##models
    # m1=DQN.load("Models\\exp1=p3=c3=90-local-100k")
    # evalModel(m1,eps,"df_e1_"+str(eps))
    # m2=DQN.load("Models\\exp2=p3=c3=90-local-100k")
    # evalModel(m2,eps,"df_e2_"+str(eps))
    # m3=DQN.load("Models\\exp3=p3=c3=90-local-100k")
    # evalModel(m3,eps,"df_e3_"+str(eps))
    
    m4=DQN.load("Models\\exp4=p3=c3=90-local-100k")
    genVideo(m4,dir="video\\r12",nexp=10)
    # evalModel(m4,eps,"df_e4_"+str(eps))
    # m4=DQN.load("Models\\exp4=p3=c3=90-local-100k")
    # evalModel(m4,eps,"df_LK_"+str(eps),LK=True)

    