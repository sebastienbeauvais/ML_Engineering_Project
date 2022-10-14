# libraries
import sys

import pandas as pd
import plotly.express as px

# import plotly.graph_objects as go
import statsmodels.api as sm


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

        # Get the stats
        t_value = round(lm_fitted.tvalues[1], 6)
        p_value = "{:.6e}".format(lm_fitted.pvalues[1])

        # Plot the figure
        fig = px.scatter(x=col, y=y, trendline="ols")
        fig.update_layout(
            title=f"Variable: {titanic_features}: (t-value={t_value}) (p-value={p_value})",
            xaxis_title=f"Variable: {titanic_features}",
            yaxis_title="y",
        )
        fig.show()

    # getting p and t values
    # t_value = round(lm_fitted.tvalues[1], 6)
    # p_value = "{:.6e}".format(lm_fitted.pvalues[1])

    # checking relationship between variables
    print(X.corr())

    # converts object types to dummies
    """for key in X:
        print("key: ", key)
        if X[key].dtype == 'object':
            X = pd.get_dummies(X, columns=[key])"""

    # plotting each predictor against dependent
    for key in X:
        fig = px.density_heatmap(df_titanic, x=key, y=y, height=500, width=500)
        # fig.show()


if __name__ == "__main__":
    sys.exit(main())
