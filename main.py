import utils.imports as inp
import utils.visuals as visu

data = inp.read_diabetes_dataset("data/diabetes.tab.txt")
print("archivo txt cargado")
# x = data.corr()
# x.to_csv("output/correlations.csv")
visu.save_histograms(data)

visu.save_scatter_plots(data)

visu.save_histogram_correlations(data)
