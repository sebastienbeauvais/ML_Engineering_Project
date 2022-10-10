# libs
import sys
from io import StringIO

import pandas as pd
import pydot

# using diabetes as initial test data
# code ripped from slides atm
# rename stats to sm
# import statsmodels.api
# from pandas import DataFrame
# from plotly import express as px
# from plotly import graph_objects as go
# from sklearn import datasets
# from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz


#######################################
# model building code                 #
#######################################
def print_heading(title):
    print("*" * 80)
    print(title)
    print("*" * 80)
    return


def plot_decision_tree(decision_tree, feature_names, class_names, file_out):
    with StringIO() as dot_data:
        export_graphviz(
            decision_tree,
            feature_names=feature_names,
            class_names=class_names,
            out_file=dot_data,
            filled=True,
        )
        graph = pydot.graph_from_dot_data(dot_data.getvalue())
        graph[0].write_pdf(file_out + ".pdf")  # must access graphs first element
        graph[0].write_png(file_out + ".png")  # must access graphs first element


# main
def main():
    #######################################
    # model building code
    #######################################
    # Increase pandas print viewport (so we see more on the screen)
    pd.set_option("display.max_rows", 60)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1_000)

    # Load the famous fishers iris data set
    fishers_iris_df = pd.read_csv(
        "https://teaching.mrsharky.com/data/iris.data", header=None
    )

    # renaming columns for better readability
    fishers_iris_df = fishers_iris_df.rename(
        columns={
            0: "sepal_width",
            1: "sepal_length",
            2: "petal_width",
            3: "petal_length",
            4: "class",
        }
    )

    # drop rows with missing vals
    fishers_iris_df = fishers_iris_df.dropna()

    # checking data frame
    print(fishers_iris_df.head())

    # printing column names
    for col in fishers_iris_df.columns:
        print(col)

    print_heading("Original Dataset")
    print(fishers_iris_df)

    # Continuous Features
    continuous_features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X = fishers_iris_df[continuous_features].values

    # Response
    y = fishers_iris_df["class"].values

    # Decision Tree Classifier
    max_tree_depth = 7
    tree_random_state = 0  # Always set a seed
    decision_tree = DecisionTreeClassifier(
        max_depth=max_tree_depth, random_state=tree_random_state
    )
    decision_tree.fit(X, y)

    # Plot the decision tree
    plot_decision_tree(
        decision_tree=decision_tree,
        feature_names=continuous_features,
        class_names="classification",
        file_out="./plots/hw_iris_tree_full",
    )
    # breaking df into response(dependent) and predictors(independent)
    response = fishers_iris_df.iloc[:, -1]  # last column
    predictors = fishers_iris_df.iloc[:, :-1]  # all but last column
    print(response.name)
    print(predictors.columns)

    # loop through each predictor and check continuous or boolean
    for (columnName, columnData) in predictors.iteritems():
        print("Column Name : ", columnName)
        print("Unique Values : ", columnData.nunique())
        if columnData.nunique() > 4:
            print("Continuous\n")
        else:
            print("Boolean\n")

    # Count unique values of response to categorize as boolean or continuous
    print("Column Name : ", response.name)
    print("Unique Values : ", response.nunique())
    if (
        response.nunique() > 4
    ):  # we just need to check for 2 but in this example we have 3
        print("Continuous")
    else:
        print("Boolean")


if __name__ == "__main__":
    sys.exit(main())
