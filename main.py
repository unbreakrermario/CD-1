import utils.imports as input
import utils.visuals as visu

data = input.read_diabetes_dataset("data/diabetes.tab.txt")
print("archivo txt cargado")

# for col in data.columns:
#     visu.save_histogram(data, col)

visu.save_scatter_plots(data)

# visu.save_histogram_correlations(data)
