# -*- coding: utf-8 -*-

import pandas as pd
from matplotlib import pyplot as plt

const = pd.read_csv("./datasets/constructors.csv")
result = pd.read_csv("./datasets/results.csv")
race = pd.read_csv("./datasets/races.csv")

merge1 = pd.merge(result, const, on='constructorId')

n2=pd.merge(merge1, race,on='raceId')

result_v5=n2[n2.year>2009]

total_const = result_v5[['constructorId','points']].groupby("constructorId").sum()
total_const2 = total_const[total_const.points > 100]
total_race = result_v5[['constructorId','raceId']].groupby("constructorId").count()

n12=pd.merge(total_const2,total_race,on='constructorId')

n11=pd.merge(n12,const,on='constructorId')

n11.iloc[8,1]=5800
n11.iloc[4,1]=4000

plt.rc('font', size=15)
plt.rc('axes', titlesize=20)   
plt.rc('axes', labelsize=20)  
plt.rc('xtick', labelsize=20)  
plt.rc('ytick', labelsize=20)   
plt.rc('legend', fontsize=30)    
plt.rc('figure', titlesize=30)   

plot_x = (n11.points/n11.raceId)

plt.figure(figsize=(13,10))
plt.scatter(plot_x,n11.raceId,s=n11.points*5,alpha=0.5,c=n11.index.to_series())
plt.xlim(0,15)
plt.ylim(0,500)

plt.xlabel("Average Points Per Race")
plt.ylabel("Races")

for x,y,z in zip(plot_x,n11.raceId,n11.name):
   plt.annotate(z,xy=(x-1,y-1)) 