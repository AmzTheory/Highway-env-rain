
from itertools import count
from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorboard.backend.event_processing import event_accumulator
import tensorflow as tf
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# tf.logging.set_verbosity(tf.logging.ERROR)

basedir = "/path/to/log/directory/"

def load_tf(dirname):
    # prefix = basedir + "tboard/VisibleSwimmer-v2/"
    # dirname = prefix + dirname
    # dirname = glob.glob(dirname + '/*')[0]
    
    ea = event_accumulator.EventAccumulator(dirname, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    dframes = {}
    mnames = ea.Tags()['scalars']
    # ea.
    # print(mnames)
    # n="rollout/ep_rew_mean"
    # print(pd.DataFrame(ea.Scalars(n), columns=["wall_time", "epoch", n.replace('val/', '')]))
    # return mnames
    res=dict()
    for n in mnames:
        ni=n.replace("train/","").replace("rollout/","").replace("eval/","").replace("time/","")
        df = pd.DataFrame(ea.Scalars(n), columns=["wall_time", "epoch", ni])
        df.drop("wall_time", axis=1, inplace=True)
        df = df.set_index("epoch")
        # print(n,dframes[n].dropna().index)
        res[ni]=df[ni]
    # return pd.concat([v for k,v in dframes.items()], axis=1)
    return res

def getData(data,size):
    y=list()
    ymin=list()
    ymax=list()
    # print(data[0][0])
    for i in range(size):
        ls=[d[i] for d in data]
        ymin.append((min(ls)))
        ymax.append(max(ls))
        y.append(sum(ls)/len(ls))


    return y,ymin,ymax

def plotR(metric,data,color,la,half=False):
    mindIndex=np.argmin([len(d[metric].dropna().index.tolist()) for d in data])
    x=data[mindIndex][metric].dropna().index.tolist()
    y,ymin,ymax=getData([d[metric].dropna().tolist() for d in data],len(x))

    if half:
        m=int(len(y)/2)
        l=len(y)
        y =y[m:l]
        ymin =ymin[m:l]
        ymax =ymax[m:l]
        x= x[m:l]

    plt.fill_between(x, y, ymin, alpha=.5, linewidth=1,color=color,)
    plt.fill_between(x, y, ymax, alpha=.5, linewidth=1,color=color)
    plt.plot(x, y, linewidth=2,color=color,label=la)
    plt.ylim((80,220))

    # plt.show()
def computeDepth(x):
    f = 4e-6 ## 4m the lane in km
    g = 9.81
    vis =1e-6  ## water viscosity
    p = 0.95  ## run-off cofficient
    vx = 2  # lateral slope
    vy = 2  # longitudinal slope
    ints = x ## mm/hr

    t1 =(0.3164*(vis**(1/4)))
    t2 = (16.67*p*f*ints)**(7/4)
    t3 = ( (vx**(2)) + (vy**(2)) ) ** (3/8)
    t = t1 * t2 * t3
    b = 8*g*((vx)**(7/4))

    h = (t/b)**(1/3)  ## in m

    return h*1000 ## in mm

def computeCofficients(text=0.5):
        a=2.9323*(text**2) - 7.2792*(text) + 0.7305
        b=50.64*(text**(0.0986))
        return a,b
def computeFriction(depth):
        a,b =computeCofficients()

        f=a*(np.log(depth))+b
        return f/100  ## convert Bpn value to friction cofficient(skid resistance)

def plotWFT(x=[0.5,1,1.5,2,2.5,3,4,5,6,10,15]):
    y=[computeDepth(i) for i in x]
    plt.plot(x, y, linewidth=2,label="WFT Value")

def plotFric(x=[0.5,1,1.5,2,2.5,3,4,5,6,10,15]):
    y=[computeDepth(i) for i in x]
    y=[computeFriction(i) for i in y]
    plt.plot(x, y, linewidth=2,label="Skid Resistance Cofficient")

    


'''
Keys are 'ep_len_mean', 'ep_rew_mean', 'exploration rate', 'fps', 
        'learning_rate', 'loss', 'mean_ep_length', 'mean_reward'
'''

att="mean_reward"
## grid 20 experiments  (60 cells)
df1 = load_tf("log\\exp1=p3=c3=90-local-100k\DQN_1\events.out.tfevents.1650298548.LAPTOP-Q6OB1PVU.12752.1")
df2 = load_tf("log\\exp2=p3=c3=90-local-100k\DQN_1\events.out.tfevents.1650328831.LAPTOP-Q6OB1PVU.24020.1")
df3 = load_tf("log\\exp3=p3=c3=90-local-100k\DQN_1\events.out.tfevents.1650346556.LAPTOP-Q6OB1PVU.9156.1")
df4 = load_tf("log\\exp4=p3=c3=90-local-100k\DQN_1\events.out.tfevents.1650351854.cn-09-12.1868059.1")

plotR(att,[df1,df2,df3,df4],"red","60 cells")

## grid 8 experiments  (24 cells)
df1 = load_tf("log\\exp1=g8=p3=c3=90-local-100k\DQN_1\events.out.tfevents.1650389911.LAPTOP-Q6OB1PVU.18820.1")
df2 = load_tf("log\\exp2=g8=p3=c3=90-local-100k\DQN_1\events.out.tfevents.1650415693.LAPTOP-Q6OB1PVU.25468.1")
df3 = load_tf("log\\exp4=g8=p3=c3=90-local-100k\DQN_1\events.out.tfevents.1650421428.cn-09-11.2655234.1")
df4 = load_tf("log\\exp4=g8=p3=c3=90-local-100k\DQN_2\events.out.tfevents.1650427813.cn-07-21.3253304.1")

plotR(att,[df1,df2,df3,df4],"blue","24 cells")

## grid 4 experiments  (12 cells)
df1 = load_tf("log\\exp1=g4=p3=c3=90-local-100k\DQN_1\events.out.tfevents.1650363209.LAPTOP-Q6OB1PVU.23524.1")
df2 = load_tf("log\\exp2=g4=p3=c3=90-local-100k\DQN_1\events.out.tfevents.1650375401.LAPTOP-Q6OB1PVU.19948.1")
df3 = load_tf("log\\exp3=g4=p3=c3=90-local-100k\DQN_1\events.out.tfevents.1650375260.cn-09-12.1471904.1")
df4 = load_tf("log\\exp4=g4=p3=c3=90-local-100k\DQN_1\events.out.tfevents.1650380240.cn-05-44.3540341.1")



plotR(att,[df1,df2,df3,df4],"green","12 cells")


## plot the WFT graph
# plotWFT()
# plotFric()
# plt.xlabel('Rainfall intensity(mm/hr)')
# plt.ylabel("WFT(mm)")
plt.xlabel("Training timestep")
plt.ylabel("Evaluation Reward")

plt.legend()
plt.show()
# st="log\\eval2=p3=c3=90-local-20k\DQN_1\events.out.tfevents.1650269892.LAPTOP-Q6OB1PVU.22840.1"
# for summary in summary_iterator(st):
#     print(summary)

