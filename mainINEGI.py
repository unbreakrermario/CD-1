import utils.imports as inp
import utils.visuals as visu
import utils.processing as proc
import pandas as pd

data = inp.read_natalidad_dataset_2022("data\conjunto_de_datos_natalidad\conjunto_de_datos\conjunto_de_datos_natalidad_2022.csv")
print("archivo txt cargado")
data_edades_padres = data[["edad_madn", "edad_padn"]]
# quitar datos donde edad >=99
idx_filtered = data_edades_padres[data_edades_padres["edad_madn"] == 99].index
data_edades_padres.drop(idx_filtered, inplace=True)
idx_filtered = data_edades_padres[data_edades_padres["edad_padn"] == 99].index
data_edades_padres.drop(idx_filtered, inplace=True)
visu.save_histogram_edades_padres(data_edades_padres)
print("grafica de padres completa")
# pendiente implementar plotly 2d histogram