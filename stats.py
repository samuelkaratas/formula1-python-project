# -*- coding: utf-8 -*-

# Imports
import time
import datetime

import numpy as np 
import pandas as pd
import scipy.stats as sp

import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots

# Loading datasets
races = pd.read_csv('./datasets/races.csv')
status = pd.read_csv('./datasets/status.csv')
drivers = pd.read_csv('./datasets/drivers.csv')
results = pd.read_csv('./datasets/results.csv')
constructors = pd.read_csv('./datasets/constructors.csv')

# Drivers name concat
concat_driver_name = lambda x: f"{x.forename} {x.surname}" 

drivers['driver'] = drivers.apply(concat_driver_name, axis=1)

# Preparing F1 history victories dataset
results_copy = results.set_index('raceId').copy()
races_copy = races.set_index('raceId').copy()

results_copy = results_copy.query("position == '1'")
results_copy['position'] = 1 # casting position 1 to int 

results_cols = ['driverId', 'position']
races_cols = ['date']
drivers_cols = ['driver', 'driverId']

results_copy = results_copy[results_cols]
races_copy = races_copy[races_cols]
drivers_copy = drivers[drivers_cols]

f1_victories = results_copy.join(races_copy)
f1_victories = f1_victories.merge(drivers_copy, on='driverId', how='left')

# Victories cumulative sum
f1_victories = f1_victories.sort_values(by='date')

f1_victories['victories'] = f1_victories.groupby(['driverId']).cumsum()   

# Getting the top five f1 biggest winners drivers id
f1_biggest_winners = f1_victories.groupby('driverId').victories.nlargest(1).sort_values(ascending=False).head(5)
f1_biggest_winners_ids = [driver for driver, race in f1_biggest_winners.index]

# Dataset ready
f1_victories_biggest_winners = f1_victories.query(f"driverId == {f1_biggest_winners_ids}")

# Prepare dataset to plot

cols = ['date', 'driver', 'victories']
winner_drivers = f1_victories_biggest_winners.driver.unique()

colors = {
    'Alain Prost': '#d80005', 
    'Ayrton Senna': '#ffffff', 
    'Michael Schumacher': '#f71120',
    'Sebastian Vettel': '#10428e',
    'Lewis Hamilton': '#e6e6e6'
}

winners_history = pd.DataFrame()

# Including other drivers races date (like a cross join matrix, 
# but cosidering column "victories" in a shift operation) 
for driver in winner_drivers:
    # Current driver victories
    driver_history = f1_victories_biggest_winners.query(f"driver == '{driver}'")[cols]
    
    # Other drivers list
    other_drivers = winner_drivers[winner_drivers != driver]
    other_drivers = list(other_drivers)
    
    # Other drivers victories
    other_driver_history = f1_victories_biggest_winners.query(f"driver == {other_drivers}")[cols]
    
    # Renaming other drivers victories to current driver
    other_driver_history['driver'] = driver
    
    # This isn't current driver victory, so receive zero to "shift" operation
    other_driver_history['victories'] = 0    
    
    driver_history = pd.concat([driver_history, other_driver_history])

    driver_history['color'] = colors[driver]
    
    # Sorting by date to correct "shift" operation
    driver_history.sort_values(by='date', inplace=True)
    
    # Reset index to get the last row (index-1) when necessary
    driver_history.reset_index(inplace=True)
    
    # Iterating each row for remain current driver victory when 
    # race date isn't the current driver victory
    for index, row in driver_history.iterrows():
        if not row['victories'] and index-1 > 0:
            driver_history.loc[index, 'victories'] = driver_history.loc[index-1, 'victories']
        
    # Plot dataset ready
    winners_history = pd.concat([winners_history, driver_history])
    
# Plots the F1 race wons animated chart 
fig = go.Figure()

fig = px.bar(
    winners_history, 
    x='victories', 
    y='driver',
    color='driver',
    color_discrete_sequence=winners_history.color.unique(),
    orientation='h',
    animation_frame="date",
    animation_group="driver",
)

# Bar border line color
fig.update_traces(dict(marker_line_width=1, marker_line_color="black"))

# X axis range
fig.update_layout(xaxis=dict(range=[0, 100]))

# Setting title
fig.update_layout(title_text="Race wins in F1 history between the top 5 winners drivers")

# Animation: Buttons labels and animation duration speed
fig.update_layout(
    updatemenus = [
        {
            "buttons": [
                # Play
                {
                    "args": [
                        None, 
                        {
                            "frame": {
                                "duration": 100, 
                                 "redraw": False
                            }, 
                            "fromcurrent": True,
                            "transition": {
                                "duration": 100, 
                                "easing": "linear"
                            }
                        }
                    ],
                    "label": "Play",
                    "method": "animate"
                },
                # Pause
                {
                    "args": [
                        [None], 
                        {
                            "frame": {
                                "duration": 0, 
                                "redraw": False
                            },
                            "mode": "immediate",
                            "transition": {
                                "duration": 0
                            }
                        }
                    ],
                    "label": "Pause",
                    "method": "animate"
                }
            ]
        }
    ]
)

fig.show()