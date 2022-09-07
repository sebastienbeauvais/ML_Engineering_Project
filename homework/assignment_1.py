# importing required libraries
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn import datasets
from sklearn.model_selection import train_test_split
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
# print(iris.describe())

# or use mean(), min(), max(), std()
# Sepal Length
print("Sepal length stats:")
print("Mean of sepal length: ", iris["sepal_length"].mean())
print("Max of sepal length:", iris["sepal_length"].max())
print("Min of sepal length:", iris["sepal_length"].min())
print("Quartiles of sepal length", iris["sepal_length"].quantile([0.25, 0.5, 0.75]))
print("\n")

# Sepal width
print("Sepal width stats:")
print("Mean of sepal width: ", iris["sepal_width"].mean())
print("Max of sepal width:", iris["sepal_width"].max())
print("Min of sepal width:", iris["sepal_width"].min())
print("Quartiles of sepal width", iris["sepal_width"].quantile([0.25, 0.5, 0.75]))
print("\n")

# Pedal length
print("Pedal length stats:")
print("Mean of pedal length: ", iris["pedal_length"].mean())
print("Max of pedal length:", iris["pedal_length"].max())
print("Min of pedal length:", iris["pedal_length"].min())
print("Quartiles of pedal length", iris["pedal_length"].quantile([0.25, 0.5, 0.75]))
print("\n")

# Pedal width
print("Pedal width stats:")
print("Mean of pedal width: ", iris["pedal_width"].mean())
print("Max of pedal width:", iris["pedal_width"].max())
print("Min of pedal width:", iris["pedal_width"].min())
print("Quartiles of pedal width", iris["pedal_width"].quantile([0.25, 0.5, 0.75]))
print("\n")

print(np.mean(iris["sepal_length"]))


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
# intializing data (starting with fresh set incase of any alterations)
data = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, t_test = train_test_split(X, y, test_size=0.33)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

print(X_train[0:5])
print(X_train_std[0:5])
print(X_test[0:5])
print(X_test_std[0:5])

# print(scaler.fit(data))
