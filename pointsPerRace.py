# -*- coding: utf-8 -*-

import pandas as pd
from matplotlib import pyplot as plt

const = pd.read_csv("./datasets/constructors.csv")
result = pd.read_csv("./datasets/results.csv")
race = pd.read_csv("./datasets/races.csv")

merge1 = pd.merge(result, const, on='constructorId')

merge2=pd.merge(merge1, race,on='raceId')

result_v1=merge2[merge2.year>2009]

total_const = result_v1[['constructorId','points']].groupby("constructorId").sum()
total_const2 = total_const[total_const.points > 100]
total_race = result_v1[['constructorId','raceId']].groupby("constructorId").count() / 2

merge3=pd.merge(total_const2,total_race,on='constructorId')

final_merge=pd.merge(merge3,const,on='constructorId')

final_merge.iloc[8,1]=5800
final_merge.iloc[4,1]=4000

plt.rc('font', size=15)
plt.rc('axes', titlesize=20)   
plt.rc('axes', labelsize=20)  
plt.rc('xtick', labelsize=20)  
plt.rc('ytick', labelsize=20)   
plt.rc('legend', fontsize=30)    
plt.rc('figure', titlesize=30)   

plot_x = (final_merge.points/final_merge.raceId)

plt.figure(figsize=(13,10))
plt.scatter(plot_x,final_merge.raceId,s=final_merge.points*5,alpha=0.5,c=final_merge.index.to_series())
plt.xlim(0,30)
plt.ylim(0,250)

plt.xlabel("Average Points Per Race")
plt.ylabel("Races")

for x,y,z in zip(plot_x,final_merge.raceId,final_merge.name):
   plt.annotate(z,xy=(x-1,y-1)) 