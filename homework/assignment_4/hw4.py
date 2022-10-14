# libraries
import sys
from io import StringIO

import pandas as pd
import plotly.express as px
import pydot

# import plotly.graph_objects as go
import statsmodels.api as sm

# from pandas import DataFrame
# from plotly import graph_objects as go
# from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz


# function to for printing headings
def print_heading(title):
    print("*" * 80)
    print(title)
    print("*" * 80)
    return


# function to load any sklearn dataset in as pandas dataframe
def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df["target"] = pd.Series(sklearn_dataset.target)
    return df


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
        graph[0].write_pdf(file_out + ".pdf")  # must access graph's first element
        graph[0].write_png(file_out + ".png")  # must access graph's first element


# first pass testing on titanic
def main():
    df_titanic = pd.read_csv("./datasets/titanic.csv")
    df_titanic = df_titanic.dropna()

    # split for titanic
    column_to_move = df_titanic.pop("Survived")  # remove survived - dependent variable
    df_titanic[
        "Survived"
    ] = column_to_move  # add survived (DV) to end of df to split easier

    # removing useless columns
    df_titanic.pop("Ticket")
    df_titanic.pop("Cabin")
    df_titanic.pop("PassengerId")
    df_titanic.pop("Name")

    X = df_titanic.iloc[:, :-1]
    y = df_titanic.iloc[:, -1]

    # categorizing each independent (predictor) variable
    print_heading("Predictors")
    catg = pd.DataFrame()
    cont = pd.DataFrame()
    for (columnName, columnData) in X.iteritems():
        print("Column Name : ", columnName)
        print("Unique Values : ", columnData.nunique())
        if columnData.nunique() <= 3:
            catg[columnName] = columnData
            print("Boolean\n")
        elif columnData.dtype == "object" and columnData.nunique() <= 3:
            catg[columnName] = columnData
            print("Boolean\n")
        else:
            cont[columnName] = columnData
            print("Continuous\n")

    # checking catg/cont
    print(catg.head())
    print(cont.head())

    # categorizing dependent variable
    print_heading("Dependent Variable")
    print("Column Name : ", y.name)
    print("Unique Values : ", y.nunique())
    if y.nunique() > 2:  # we just need to check for 2
        print("Continuous")
    else:
        print("Boolean")

    # Linear model for continuous features
    headers = list(cont.columns)

    print(headers)
    print("columns: ", cont.columns[0])
    for key in cont:
        titanic_features = key
        col = cont[key]
        predictor = sm.add_constant(col)
        lm = sm.OLS(y, predictor)
        lm_fitted = lm.fit()
        print(f"Variable: {titanic_features}")
        print(lm_fitted.summary())

        # Get the p/t stats
        t_value = round(lm_fitted.tvalues[1], 6)
        p_value = "{:.6e}".format(lm_fitted.pvalues[1])

        # Plot the figure
        fig = px.scatter(x=y, y=col, trendline="ols")
        fig.update_layout(
            title=f"Variable: {titanic_features}: (t-value={t_value}) (p-value={p_value})",
            xaxis_title=f"y: {y.name}",
            yaxis_title=f"Variable: {titanic_features}",
        )
        # fig.show()

    # checking relationship between variables
    print(X.corr())

    # plotting each predictor against dependent
    for key in X:
        fig = px.density_heatmap(df_titanic, x=key, y=y, height=500, width=500)
        # fig.show()

    # converts object types to dummies for random forest classifier
    dummies_df = pd.get_dummies(X)

    print(dummies_df.head())

    cont_features = list(dummies_df.columns)
    dummies_X = dummies_df[cont_features].values
    dummies_y = y

    # decision tree classifier
    max_tree_depth = 7
    tree_random_state = 0  # setting a seed for reproduction
    decision_tree = DecisionTreeClassifier(
        max_depth=max_tree_depth, random_state=tree_random_state
    )
    decision_tree.fit(dummies_X, dummies_y)

    # Plot the decision tree
    plot_decision_tree(
        decision_tree=decision_tree,
        feature_names=cont_features,
        class_names="classification",
        file_out="titanic_tree_full",
    )


if __name__ == "__main__":
    sys.exit(main())
