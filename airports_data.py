import utils.imports as inp
import utils.visuals as visu
import plotly.graph_objects as go

flights = inp.read_flights_dataset("data/flightsJan.csv")
airports = inp.read_flights_dataset("data/airports.csv")
visu.plot_airports_geoscatter(airports)
print(flights.head())
visu.save_histogram(flights, "DAY_OF_MONTH")
visu.save_histogram(flights, "TAIL_NUM")
visu.save_histogram(flights, "OP_CARRIER")
visu.save_histogram(flights, "DEST")
visu.save_histogram(flights, "DEP_DELAY")
visu.save_histogram(flights, "TAXI_OUT")
visu.save_histogram(flights, "TAXI_IN")
visu.save_histogram(flights, "CANCELLED")
visu.save_histogram(flights, "DIVERTED")

# plot top 20 airports with most flights departed

