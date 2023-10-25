import pandas as pd

import utils.imports as inp
import utils.visuals as visu
import plotly.graph_objects as go

flights = inp.read_flights_dataset("data/flightsJan.csv")
airports = inp.read_flights_dataset("data/airports.csv")

print(flights.head())

# plot top 20 airports with most flights departed
flights_count_origin = flights["ORIGIN"].value_counts()
flights_count_dest = flights["DEST"].value_counts()

F = flights_count_origin.to_frame()
F = F.rename(columns={'count': 'count_origin'})
new_airports = airports.set_index('AIRPORT').join(F)
new_airports = new_airports.reset_index()

F = flights_count_dest.to_frame()
F = F.rename(columns={'count': 'count_destination'})
new_airports = new_airports.set_index('AIRPORT').join(F)
new_airports = new_airports.reset_index()

new_airports["count_total"] = new_airports["count_origin"] + new_airports["count_destination"]
new_airports = new_airports.sort_values(by='count_total', ascending=False)
visu.plot_airports_geoscatter(new_airports.head(n=15))

# From flights, extract rows corresponding to top 20 airports only
top_airports = new_airports.head(n=15)
flights_top_origin = flights.loc[flights["ORIGIN"].isin(top_airports["AIRPORT"])]
flights_top_dest = flights.loc[flights["DEST"].isin(top_airports["AIRPORT"])]
flights_top = flights_top_origin.merge(flights_top_dest, how='outer')

idx_filtered = flights_top[flights_top["CANCELLED"] == 0].index
flights_top.drop(idx_filtered, inplace=True)

# Use GeoScatter to show the airports with more cancelled flights
flights_count_origin = flights_top["ORIGIN"].value_counts()
flights_count_dest = flights_top["DEST"].value_counts()

F = flights_count_origin.to_frame()
F = F.rename(columns={'count': 'count_origin'})
new_airports = airports.set_index('AIRPORT').join(F)
new_airports = new_airports.reset_index()

F = flights_count_dest.to_frame()
F = F.rename(columns={'count': 'count_destination'})
new_airports = new_airports.set_index('AIRPORT').join(F)
new_airports = new_airports.reset_index()

new_airports["count_total"] = new_airports["count_origin"] + new_airports["count_destination"]
new_airports = new_airports.sort_values(by='count_total', ascending=False)
visu.plot_airports_geoscatter(new_airports.head(n=15))

visu.save_histogram(flights_top, "DAY_OF_MONTH")
visu.save_histogram(flights_top, "TAIL_NUM")
visu.save_histogram(flights_top, "OP_CARRIER")
visu.save_histogram(flights_top, "DEST")
visu.save_histogram(flights_top, "DEP_DELAY")
visu.save_histogram(flights_top, "TAXI_OUT")
visu.save_histogram(flights_top, "TAXI_IN")
visu.save_histogram(flights_top, "CANCELLED")
visu.save_histogram(flights_top, "DIVERTED")
