import seaborn as sns
import matplotlib.pyplot as plt


def save_histogram(data, column):
    fig = sns.displot(data[column], kde=True)
    fig.savefig("output/histogram_"+column+".png")


def save_scatter_plots(data):
    sns.scatterplot(data=data, x="AGE", y="BMI")
    plt.savefig("output/scatter_AGE_BMI.png")


def save_histogram_correlations(data):
    fig = sns.pairplot(data, hue="SEX")
    fig.savefig("output/All_histograms.png")
