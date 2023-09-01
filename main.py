import utils.imports as inp
import utils.visuals as visu

data = inp.read_diabetes_dataset("data/diabetes.tab.txt")
print("archivo txt cargado")

for col in data.columns:
    visu.save_histogram(data, col)

visu.save_scatter_plots(data)

visu.save_histogram_correlations(data)
