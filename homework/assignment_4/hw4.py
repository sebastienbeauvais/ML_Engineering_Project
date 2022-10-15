# libraries
import sys
from io import StringIO

import pandas as pd
import plotly.express as px
import pydot
import statsmodels.api as sm
from pandas import DataFrame
from plotly import graph_objects as go
from sklearn.model_selection import GridSearchCV
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


def main():
    # loading dataset from csv
    df_titanic = pd.read_csv("./datasets/titanic.csv")
    # to load a dataset from sklearn use the sklearn_to_df function

    # dropping nulls
    df_titanic = df_titanic.dropna()
    print(df_titanic.dtypes)

    # initiLizing out final output table
    output_table = pd.DataFrame(
        columns=[
            "Response",
            "Predictor",
            "Plot",
            "t_score",
            "p_value",
            "RF_VarImp",
            "MWR_Unweighted",
            "MWR_Weighted",
        ]
    )

    #####################################################################
    # this code is specific for the titanic dataset
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
    #####################################################################

    X = df_titanic.iloc[:, :-1]
    y = df_titanic.iloc[:, -1]

    # categorizing each independent (predictor) variable as categorical or continuous
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

        # writing to our output table
        temp_row = {
            "Response": y.name,
            "Predictor": key,
            "Plot": "Link to plot",
            "t_score": t_value,
            "p_value": p_value,
        }
        output_table = output_table.append(temp_row, ignore_index=True)

        # Plot the figure
        cont_scatter = px.scatter(x=y, y=col, trendline="ols")
        cont_scatter.update_layout(
            title=f"Variable: {titanic_features}: (t-value={t_value}) (p-value={p_value})",
            xaxis_title=f"y: {y.name}",
            yaxis_title=f"Variable: {titanic_features}",
        )
        cont_scatter.show()

    # checking relationship between variables
    print(df_titanic.corr())

    ################################################
    # heatmap
    ################################################
    for key in X:
        titanic_heatmap = px.density_heatmap(
            df_titanic, x=key, y=y, height=500, width=500
        )
        titanic_heatmap.show()

    ################################################
    # dummies
    ################################################

    # converts object types to dummies for random forest classifier
    dummies_df = pd.get_dummies(X)

    print(dummies_df.head())

    cont_features = list(dummies_df.columns)
    dummies_X = dummies_df[cont_features].values
    dummies_y = y

    ################################################
    # decision tree classifier
    ################################################
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

    # Find an optimal tree via cross-validation
    parameters = {
        "max_depth": range(1, max_tree_depth),
        "criterion": ["gini", "entropy"],
    }
    decision_tree_grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=tree_random_state), parameters, n_jobs=4
    )
    decision_tree_grid_search.fit(X=dummies_X, y=dummies_y)

    cv_results = DataFrame(decision_tree_grid_search.cv_results_["params"])
    cv_results["score"] = decision_tree_grid_search.cv_results_["mean_test_score"]
    print_heading("Cross validation results")
    print(cv_results)
    print_heading("Cross validation results - HTML table")
    print(cv_results.to_html())

    # Plot these cross_val results
    gini_results = cv_results.loc[cv_results["criterion"] == "gini"]
    entropy_results = cv_results.loc[cv_results["criterion"] == "entropy"]
    data = [
        go.Scatter(
            x=gini_results["max_depth"].values,
            y=gini_results["score"].values,
            name="gini",
            mode="lines",
        ),
        go.Scatter(
            x=entropy_results["max_depth"].values,
            y=entropy_results["score"].values,
            name="entropy",
            mode="lines",
        ),
    ]

    layout = go.Layout(
        title="Dataset Cross Validation",
        xaxis_title="Tree Depth",
        yaxis_title="Score",
    )

    fig = go.Figure(data=data, layout=layout)
    fig.show()
    fig.write_html(
        file="titanic_cross_val.html",
        include_plotlyjs="cdn",
    )

    # Get the "best" model
    best_tree_model = decision_tree_grid_search.best_estimator_

    # Plot this "best" decision tree
    plot_decision_tree(
        decision_tree=best_tree_model,
        feature_names=cont_features,
        class_names="classification",
        file_out="titanic_tree_cross_val",
    )
    # ranking each variable
    # looking at feature importance
    feat_importance = decision_tree.tree_.compute_feature_importances(normalize=False)

    print(output_table)
    print(feat_importance)

    return


if __name__ == "__main__":
    sys.exit(main())
