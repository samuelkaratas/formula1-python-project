# -*- coding: utf-8 -*-
import pandas as pd
from plotly.offline import plot
import plotly.express as px

races = pd.read_csv('./datasets/races.csv')
drivers = pd.read_csv('./datasets/drivers.csv')
results = pd.read_csv('./datasets/results.csv')

colors = {
    'Alain Prost': '#800006', 
    'Ayrton Senna': '#ffffff', 
    'Michael Schumacher': '#f71120',
    'Sebastian Vettel': '#10428e',
    'Lewis Hamilton': '#34ebe8'
}

concat_driver_name = lambda x: f"{x.forename} {x.surname}" 

drivers['driver'] = drivers.apply(concat_driver_name, axis=1)

results_copy = results.set_index('raceId').copy()
races_copy = races.set_index('raceId').copy()

results_copy = results_copy.query("position == '1'")
results_copy['position'] = 1

results_copy = results_copy[['driverId', 'position']]
races_copy = races_copy[['date']]
drivers_copy = drivers[['driver', 'driverId']]

f1_victories = results_copy.join(races_copy)
f1_victories = f1_victories.merge(drivers_copy, on='driverId', how='left')

f1_victories = f1_victories.sort_values(by='date')

f1_victories['victories'] = f1_victories.groupby(['driverId']).cumsum()   
f1_biggest_winners = f1_victories.groupby('driverId').victories.nlargest(1).sort_values(ascending=False).head(5)

f1 = drivers.merge(f1_biggest_winners, on='driverId')

f1['color'] = f1.driver.map(colors)

f1_biggest_winners_ids = [driver for driver, race in f1_biggest_winners.index]
f1_victories_biggest_winners = f1_victories.query(f"driverId == {f1_biggest_winners_ids}")

winner_drivers_ids = f1_victories_biggest_winners[['driverId', 'driver']].drop_duplicates()
winner_drivers_map = {}

for _, row in winner_drivers_ids.iterrows():
    winner_drivers_map[row['driverId']] = row['driver'] 
    
f1_biggest_winners_poles = results.query(f"driverId == {f1_biggest_winners_ids} & grid == 1")[['driverId', 'grid']]

f1_biggest_winners_poles['driver'] = f1_biggest_winners_poles.driverId.map(winner_drivers_map)
f1_biggest_winners_poles['color'] = f1_biggest_winners_poles.driver.map(colors)

f1_biggest_winners_poles['total_poles'] = f1_biggest_winners_poles.groupby(['driverId']).cumsum()   

f1_biggest_winners_total_poles = f1_biggest_winners_poles.groupby('driver').total_poles.nlargest(1).sort_values(ascending=False).head(5)
f1_biggest_winners_total_poles = pd.DataFrame(f1_biggest_winners_total_poles).reset_index()

f1_biggest_winners_total_poles['color'] = f1_biggest_winners_total_poles.driver.map(colors)

fig = px.bar(
    f1, 
    x='driver', 
    y='victories',
    color='driver',
    color_discrete_sequence=f1.color
)

fig.update_layout(title_text="Race wins between top 5 race winners")

fig2 = px.bar(
    f1_biggest_winners_total_poles, 
    x='driver', 
    y='total_poles',
    color='driver',
    color_discrete_sequence=f1_biggest_winners_total_poles.color
)

fig2.update_layout(title_text="Pole positions between the top 5 race winners")

#To run the pole position visualization run the second comment below. (plot(fig2))
plot(fig)
#plot(fig2)