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
new_airports = airports.set_index('AIRPORT').join(F)
new_airports = new_airports.reset_index()
visu.plot_airports_geoscatter(new_airports)

visu.save_histogram(flights, "DAY_OF_MONTH")
visu.save_histogram(flights, "TAIL_NUM")
visu.save_histogram(flights, "OP_CARRIER")
visu.save_histogram(flights, "DEST")
visu.save_histogram(flights, "DEP_DELAY")
visu.save_histogram(flights, "TAXI_OUT")
visu.save_histogram(flights, "TAXI_IN")
visu.save_histogram(flights, "CANCELLED")
visu.save_histogram(flights, "DIVERTED")
