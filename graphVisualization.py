# -*- coding: utf-8 -*-
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

 

def read_csv(name, **kwargs):
    df = pd.read_csv(f'./datasets/{name}', na_values=r'\N', **kwargs)
    return df

 

def races_subset(df, races_index):
    df = df[df.raceId.isin(races_index)].copy()
    df['round'] = df.raceId.map(races['round'])
    df['round'] -= df['round'].min()
    return df.set_index('round').sort_index()

 

circuits = read_csv('circuits.csv', index_col=0)
constructorResults = read_csv('constructor_results.csv', index_col=0)
constructors = read_csv('constructors.csv', index_col=0)
constructorStandings = read_csv('constructor_standings.csv', index_col=0)
drivers = read_csv('drivers.csv', index_col=0)
driverStandings = read_csv('driver_standings.csv', index_col=0)
lapTimes = read_csv('lap_times.csv')
pitStops = read_csv('pit_stops.csv')
qualifying = read_csv('qualifying.csv', index_col=0)
races = read_csv('races.csv', index_col=0)
results = read_csv('results.csv', index_col=0)
seasons = read_csv('seasons.csv', index_col=0)
status = read_csv('status.csv', index_col=0)

 

drivers['display'] = drivers.surname
drivers['Driver'] = drivers['forename'] + " " + drivers['surname']
drivers['Driver'] = drivers.apply(lambda r: '<a href="{url}">{Driver}</a>'.format(**r), 1)
constructors['label'] = constructors['name']
constructors['name'] = constructors.apply(lambda r: '<a href="{url}">{name}</a>'.format(**r), 1)

 

results['status'] = results.statusId.map(status.status)
results['Team'] = results.constructorId.map(constructors.name)
results['score'] = results.points>0
results['podium'] = results.position<=3

 

YEAR = 2020

 

races = races.query('year==@YEAR').sort_values('round').copy()
results = results[results.raceId.isin(races.index)].copy()
lapTimes = lapTimes[lapTimes.raceId.isin(races.index)].copy()
driverStandings = races_subset(driverStandings, races.index)
constructorStandings = races_subset(constructorStandings, races.index)

 


lapTimes = lapTimes.merge(results[['raceId', 'driverId', 'positionOrder']], on=['raceId', 'driverId'])
lapTimes['seconds'] = lapTimes.pop('milliseconds') / 1000

 

def format_standings(df, key):
    df = df.sort_values('position')
    gb = results.groupby(key)
    df['Position'] = df.positionText
    df['points'] = df.points.astype(int)
    df['scores'] = gb.score.sum().astype(int)
    df['podiums'] = gb.podium.sum().astype(int)
    for c in [ 'scores', 'points', 'podiums', 'wins' ]:
        df.loc[df[c] <= 0, c] = ''
    return df

 

def drivers_standings(df):
    df = df.join(drivers)
    df = format_standings(df, 'driverId')
    df['Team'] = results.groupby('driverId').Team.last()
    use = ['Position', 'Driver',  'Team', 'points', 'wins', 'podiums', 'scores', 'nationality' ]
    df = df[use].set_index('Position').fillna('')
    df.columns = df.columns.str.capitalize()
    return df

 

def constructors_standings(df):
    df = df.join(constructors)
    df = format_standings(df, 'constructorId')
    
    tmp = results.join(drivers.drop('number', 1), on='driverId')
    df = df.join(tmp.groupby('constructorId').Driver.unique().str.join(', ').to_frame('Drivers'))

 

    use = ['Position', 'name', 'points', 'wins', 'podiums', 'scores', 'nationality', 'Drivers' ]
    df = df[use].set_index('Position').fillna('')
    df.columns = df.columns.str.capitalize()
    return df

 

def format_results(df):
    df['Team'] = df.constructorId.map(constructors.name)
    df['Position'] = df.positionOrder
    df['number'] = df.number.map(int)
    df['points'] = df.points.map(int)
    df.loc[df.points <= 0, 'points'] = ''
    use = ['Driver', 'Team', 'number', 'grid', 'Position', 'points', 'laps', 'time', 'status' ]
    df = df[use].set_index('Position').fillna('')
    df.columns = df.columns.str.capitalize()
    return df

 

plt.rc("figure", figsize=(16, 12))
plt.rc("font", size=(16))
plt.rc("axes", xmargin=0)

 

champ = driverStandings.groupby("driverId").position.last().to_frame("Pos")
champ = champ.join(drivers)
order = np.argsort(champ.Pos)

 

DRIVER_C = {1:"#800000",8:"#00CACA",20:"#191919",154:"#FF0000",807:"#FF69B4",815:"#FF69B4",817:"#00CACA",822:"#B4B400",825:"#7F7F7F",826:"#191919",830:"#FE7F00",832:"#0000B0",839:"#007FFE",840:"#191919",841:"#800000",842:"#7F7F7F",844:"#FF0000",846:"#007FFE",847:"#FE7F00",848:"#007FFE",849:"#0000B0",850:"#FF69B4",851:"#B4B400"} 
DRIVER_LS = {1:0,8:0,20:1,154:0,807:2,815:0,817:0,822:1,825:1,826:1,830:0,832:0,839:1,840:1,841:1,842:0,844:0,846:1,847:0,848:1,849:1,850:2,851:2}
LINESTYLES = ['-', '-.', '--', ':', '-', '-']

 

color = [DRIVER_C[d] for d in champ.index[order]]
style = [LINESTYLES[DRIVER_LS[d]] for d in champ.index[order]]
labels = champ.Pos.astype(str) + ". " + champ.display

 

chart = driverStandings.pivot("raceId", "driverId", "points")
chart.index = races.reindex(chart.index).name.str.replace("Grand Prix", "GP").rename("Race")
chart.columns = labels

 

chart.iloc[:, order].plot(title=f"F1 Drivers\' World Championship — {YEAR}", color=color, style=style)
plt.xticks(range(chart.shape[0]), chart.index, rotation=45)
plt.grid(axis="x", linestyle="--")
plt.ylabel("Points")
legend_opts = dict(bbox_to_anchor=(1.02, 0, 0.2, 1),
                   loc="upper right",
                   ncol=1,
                   shadow=True,
                   edgecolor="black",
                   mode="expand",
                   borderaxespad=0.)
plt.legend(**legend_opts)
plt.tight_layout()
plt.show()

 

display(drivers_standings(driverStandings.loc[driverStandings.index.max()].set_index("driverId")).style)

 


champ = constructorStandings.groupby("constructorId").position.last().to_frame("Pos")
champ = champ.join(constructors)
order = np.argsort(champ.Pos)

 

TEAM_C = {1:"#FE7F00",3:"#007FFE",4:"#B4B400",6:"#FF0000",9:"#0000B0",51:"#800000",131:"#00CACA",210:"#191919",211:"#FF69B4",213:"#7F7F7F"}

 

color = [TEAM_C[c] for c in champ.index[order]]
labels = champ.Pos.astype(str) + ". " + champ.label

 

chart = constructorStandings.pivot("raceId", "constructorId", "points")
chart.index = races.reindex(chart.index).name.str.replace("Grand Prix", "GP").rename("Race")
chart.columns = labels

 

chart.iloc[:, order].plot(title=f"F1 Constructors\' World Championship — {YEAR}", color=color)
plt.xticks(range(chart.shape[0]), chart.index, rotation=45)
plt.grid(axis="x", linestyle="--")
plt.ylabel("Points")
plt.legend(**legend_opts)
plt.tight_layout()
plt.show()

 

display(constructors_standings(constructorStandings.loc[constructorStandings.index.max()].set_index("constructorId")).style)

 


for rid, times in lapTimes.groupby("raceId"):

 

    race = races.loc[rid]
    circuit = circuits.loc[race.circuitId]
    title = "Round {round} — F1 {name} — {year}".format(**race)
    qstr = race["name"].replace(" ", "+")
    
    res = results.query("raceId==@rid").set_index("driverId")
    res = res.join(drivers.drop("number", 1))

 

    chart = times.pivot_table("seconds", "lap", "driverId")

 

    basis = chart.median(1).cumsum()

 

    labels = res.loc[chart.columns].apply(lambda r: "{positionOrder:2.0f}. {display}".format(**r), 1)
    order = np.argsort(labels)
    show = chart.iloc[:, order]
    
    color = [DRIVER_C[d] for d in show.columns]
    style = [LINESTYLES[DRIVER_LS[d]] for d in show.columns]

 

    show = (basis - show.cumsum().T).T
    show.columns = labels.values[order]

 

    show[show>1000] = np.nan
    
    xticks = np.arange(0, len(chart)+1, 2)
    if len(chart) % 2: 
        xticks[-1] += 1

 

    show.plot(title=title, style=style, color=color)
    if show.min().min() < -180:
        plt.ylim(-180, show.max().max()+3)
    plt.ylabel("Time Delta (s)")
    plt.xticks(xticks, xticks)
    plt.grid(linestyle="--")
    plt.legend(bbox_to_anchor=(0, -0.2, 1, 1),
               loc=(0, 0),
               ncol=6,
               shadow=True,
               edgecolor="black",
               mode="expand",
               borderaxespad=0.)
    plt.tight_layout()
    plt.show()
    
    display(format_results(res).style)