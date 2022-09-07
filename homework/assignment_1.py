# importing required libraries
# import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler

# import plotly.figure_factory as ff

# reading in UCI iris datasets into a pandas dataframe
# naming columns to be more reader friendly
csv_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ["sepal_length", "sepal_width", "pedal_length", "pedal_width", "species"]
iris = pd.read_csv(csv_url, names=column_names)

# checking data
print("Checking the data ...")
print(iris.head(5))
print("\n")

# some simple statisics of the data
# using describe() for a quick output
# can also use mean(), min(), max(), std()
print("Dataset stats ...")
print(iris.describe())
print("\n")

# plotting different classes against eachother
# plot 1 - scatter plot
df = px.data.iris()  # iris is a pandas DataFrame
scatter = px.scatter(
    df,
    x="sepal_width",
    y="sepal_length",
    color="species",
    size="petal_length",
    hover_data=["petal_width"],
    title="Scatter plot of iris data set",
)
scatter.show()

# plot 2 - violin plot
df = px.data.iris()
violin = px.violin(
    df,
    y="sepal_width",
    color="species",
    violinmode="overlay",
    title="Violin plot of iris data set",  # draw violins on top of each other
    # default violinmode is 'group' as in example above
    hover_data=df.columns,
)
violin.show()

# plot 3 - scatter matrix
df = px.data.iris()
scatter_matrix = px.scatter_matrix(
    df,
    dimensions=["sepal_width", "sepal_length", "petal_width", "petal_length"],
    color="species",
    symbol="species",
    title="Scatter matrix of iris data set",
    labels={col: col.replace("_", " ") for col in df.columns},
)  # remove underscore
scatter_matrix.update_traces(diagonal_visible=False)
scatter_matrix.show()

# plot 4 - distribution plot
df = px.data.iris()
hist = px.histogram(
    df,
    x="sepal_width",
    y="sepal_length",
    color="species",
    marginal="violin",
    hover_data=df.columns,
    title="Distribution plot of iris data set",
)
hist.show()

# plot 5 - boxplot
df = px.data.iris()
box = px.box(df, y="sepal_length", color="species", title="Boxplot of iris data set")
box.show()

# Using Standard Scaler
data = iris
scaler = StandardScaler()
print(scaler.fit(data))
