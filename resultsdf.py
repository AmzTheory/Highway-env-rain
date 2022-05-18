import pickle as p
import pandas as pd
import matplotlib.pyplot as plt 

print("MOBIL")
dfIDM=p.load(open("Dataframes//df_IDM_300","rb"))
dfIDi=100-(p.load(open("Dataframes//df_IDM_300","rb"))["Crashes"]/300)*100
print(dfIDM)
# print(dfIDi)


for i in range(1,5):
    i=str(i)
    print("Model-"+i)
    dfMod=100-(p.load(open("Dataframes//df_e"+i+"_300","rb"))["Crashes"]/300)*100
    print(p.load(open("Dataframes//df_e"+i+"_300","rb")))

# print("LK")
dfLK=p.load(open("Dataframes//df_LK_300","rb"))
# print(dfLK)

def plotHistogram(df,metrics,param):
    fig, ax = plt.subplots()
    pos=range(len(metrics))
    w=0.3
    c=0
    barsize=w*len(metrics)
    buffer=barsize  #+0.22
    for m in param:
        lb=c*w
        ind=[lb+(buffer*p) for p in pos]
        row=df[df["sol"]==m]
        # print([float(row[m]) for m in metrics])
        ax.bar(ind,[float(row[m])for m in metrics],w,label=m)
        c+=1
        
    ax.set_xticks([buffer*p+(barsize/2)-0.5 for p in pos])
    ax.set_xticklabels(metrics)
    ax.set_yticks(range(0,35,2))
    ax.set(xlabel="Metrics",title="Evaluation Results")
    ax.legend(param)
    plt.show()

def plotGraph(sols,m):
    # fig, ax = plt.subplots(nrows=1,ncols=4)
    x=[2,6,12]
    speeds=[12.50,14.44,16.60]
    ac=0
    # for m in metrics:
    for s in speeds:
        for ms,df in sols.items():
            y=df[df["speed"]==s][m]
            plt.plot(x,y,label=str(s)+"-"+ms,alpha=0.5)
            plt.scatter(x,y)
        # ax[ac].legend()
        plt.xlabel("Rainfall Intensity(mm/hr)")
        plt.ylabel("Total Travel Distance(TTD)")
        plt.xticks(x)
        ac+=1
        plt.legend()

    plt.legend()
    plt.show()
def addColumn(df):
    df["TTD"]=df["total_distance"]-df["ego_distance"]
    return df
dfSol=p.load(open("Dataframes//df_e4_300","rb"))
# dfs={"Always LK":dfLK}
# plotGraph(dfs,"Avg_speed")
# print(dfSol)
# print(dfSol["ego_distance"]-dfIDM["ego_distance"])
# print(dfSol["total_distance"]-dfIDM["total_distance"])



dfIDM=addColumn(dfIDM)
dfSol=addColumn(dfSol)

dfs={"MOBIL":dfIDM,"ps":dfSol}
plotGraph(dfs,"TTD")