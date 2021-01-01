# -*- coding: utf-8 -*-

import pandas as pd
from matplotlib import pyplot as plt

from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

driv  =pd.read_csv("./datasets/drivers.csv")
result=pd.read_csv("./datasets/results.csv")
race  =pd.read_csv("./datasets/races.csv")

m1=pd.merge(result,driv,on='driverId')
m2=pd.merge(m1,race,on='raceId')
result_v2=m2[m2.year>2009]

result_v2["driver"] = result_v2["forename"] + " " + result_v2["surname"]

avg_pts   = result_v2[['driver','points']].groupby("driver").mean()
total_pts = result_v2[['driver','points']].groupby("driver").sum()

n=result_v2[['driver','raceId']].groupby("driver").count()
num_races=n[n.raceId>100]

d =pd.merge(avg_pts,total_pts,on='driver')
md=pd.merge(d,num_races,on='driver')
md = md.reset_index()
#md.iloc[7,3]=180 #data correction
#md.iloc[6,3]=125 #data correction

plt.rc('font', size=15)          # controls default text sizes
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=20)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
plt.rc('ytick', labelsize=20)    # fontsize of the tick labels
plt.rc('legend', fontsize=30)    # legend fontsize
plt.rc('figure', titlesize=30)   # fontsize of the figure title

plt.figure(figsize=(20,10))
plt.scatter(md.points_x,md.raceId,s=md.points_y*5,alpha=0.5,c=md.index.to_series())
plt.xlim(0,18)
plt.ylim(100,240)

plt.xlabel("Average Points Per Race")
plt.ylabel("Races")

for x,y,z in zip(md.points_x,md.raceId,md.driver):
   plt.annotate(z,xy=(x-1,y-1)) 