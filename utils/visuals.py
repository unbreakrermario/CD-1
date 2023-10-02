import seaborn as sns
import matplotlib.pyplot as plt
import utils.processing as proc
from sklearn import metrics
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd


def save_histogram(data, column):
    proc.check_output_folder("output/histograms")
    fig = sns.displot(data[column], kde=True)
    fig.savefig("output/histograms/histogram_"+column+".png")
    plt.close()


def save_histograms(data):
    for col in data.columns:
        save_histogram(data, col)


def save_scatter_plot(data, var1, var2, corr_value):
    proc.check_output_folder("output/scatterplots")
    new_fig = plt.figure()
    sns.scatterplot(data=data, x=var1, y=var2).set(title=var1+" vs "+var2+" r = "+corr_value)
    plt.savefig("output/scatterplots/scatter_"+var1+"_"+var2+".png")
    plt.close(new_fig)


def save_scatter_plots(data, correlations):
    columns_pending = data.columns
    for var1 in data.columns:
        columns_pending = columns_pending.drop(var1)
        for var2 in columns_pending:
            corr_value = str(correlations.loc[var1][var2])
            save_scatter_plot(data, var1, var2, corr_value)


def save_histogram_correlations(data):
    proc.check_output_folder("output")
    fig = sns.pairplot(data, hue="SEX")
    fig.savefig("output/All_histograms.png")
    plt.close()


def save_correlations_heatmap(data):
    proc.check_output_folder("output")
    new_fig = plt.figure()
    sns.heatmap(data.corr(), annot=True, fmt='.2f').set(title="Correlations for Diabetes Dataset")
    plt.savefig("output/correlations_heatmap.png")
    plt.close(new_fig)


def save_confusion_matrix(confusion):
    proc.check_output_folder("output")
    new_fig = plt.figure()
    sns.heatmap(confusion, annot=True, cmap='Blues')
    plt.savefig("output/confusion_matrix.png")
    plt.close(new_fig)


def save_roc_curve(diabetes_y_test, diabetes_y_pred):
    proc.check_output_folder("output")
    new_fig = plt.figure()
    metrics.RocCurveDisplay.from_predictions(diabetes_y_test, diabetes_y_pred)
    plt.savefig("output/curve_ROC.png")
    plt.close(new_fig)


def save_edades_padres_heatmap(data):
    proc.check_output_folder("output")
    new_fig = plt.figure()
    sns.heatmap(data, fmt='.2f').set(title="Edades Padres")
    plt.savefig("output/edades_heatmap.png")
    plt.close(new_fig)


def save_histogram_edades_padres(data):
    proc.check_output_folder("output")
    fig = sns.displot(data, kde=True)
    fig.savefig("output/hsitograma_edades_padres.png")
    plt.close()


def save_heatmap_edades(data):
    proc.check_output_folder("output")
    fig = go.Figure(px.density_heatmap(data, x="edad_padn", y="edad_madn",
                                       color_continuous_scale="Rainbow",
                                       range_color=[0, 50],
                                       title="Edades padres recien nacidos Queretaro 2022"))
    fig.show()
    fig.write_image("output/heatmap_edades_padres.svg")


def save_heatmap_edades_logaritmico(data):
    proc.check_output_folder("output")
    z = np.histogram2d(data["edad_padn"], data["edad_madn"], bins=[75, 50],
                       range=[[0, 75], [0, 50]])
    data = np.delete(z[0], range(9), 0)
    data = np.delete(data, range(9), 1)
    data = pd.DataFrame(data, columns=range(10, 51), index=range(10, 76))
    fig = go.Figure(go.Heatmap(
        z=data,
        x=data.columns,
        y=data.index,
        colorscale=[
            [0, 'rgb(0, 0, 80)'],  # 0
            [1. / 10000, 'rgb(0, 0, 230)'],  # 10
            [1. / 1000, 'rgb(230, 230, 230)'],  # 100
            [1. / 100, 'rgb(230, 0, 0)'],  # 1000
            [1. / 10, 'rgb(80, 0, 0)'],  # 10000
            [1., 'rgb(0, 0, 0)'],   # 100000
            ],
        colorbar=dict(
            tick0=0,
            tickmode='array',
            tickvals=[0, 100, 1000, 10000, 100000]
        )
    ))

    fig.show()
    # fig.write_image("output/heatmap_edades_padres.svg")
