import sys

import pandas as pd
import statsmodels.api as sm

# from plotly import express as px
# from sklearn import datasets


# function to printing  headings
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
    # diabetes = datasets.load_diabetes()
    df_titanic = pd.read_csv("./datasets/titanic.csv")
    # df_diabetes = sklearn_to_df(datasets.load_titanic())
    print(df_titanic.head())

    # this split is for most datasets where last column is target
    """
    # splitting features and target
    X = df_diabetes.iloc[:, :-1]  # all but last column
    y = df_diabetes.iloc[:, -1]  # last column
    # X = diabetes.data
    # y = diabetes.target
    """

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

    # splitting predictors into categorical/continuous
    # print(X.dtypes)
    # catg = X.select_dtypes("object")
    # cont = X.select_dtypes("number")

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

    print(catg.head())
    print(cont.head())

    # categorizing dependent variable
    print_heading("Dependent Variable")
    print("Column Name : ", y.name)
    print("Unique Values : ", y.nunique())
    if y.nunique() > 2:  # we just need to check for 2 but in this example we have 3
        print("Continuous")
    else:
        print("Boolean")

    # this works for a single predictor
    feature_name = list(X.columns)
    predictor = sm.add_constant(X.iloc[:, 0])
    linear_regression_model = sm.OLS(y, predictor)
    linear_regression_model_fitted = linear_regression_model.fit()
    print(f"Variable: {feature_name}")
    print(linear_regression_model_fitted.summary())

    # get list of column headers to iterate over
    column_headers = list(X.columns)

    for index, col in enumerate(X.T):
        feature_name = column_headers[index]
        print(feature_name)
        predictor = sm.add_constant(X.iloc[:, col])
        """lm = sm.OLS(y, predictor)
        lm_fitted = lm.fit()
        print(f"Variable: {feature_name}")
        print(lm_fitted.summary())"""

    """
    for idx, column in enumerate(X.T):
        feature_name = diabetes.feature_names[idx]
        print(feature_name)
        predictor = statsmodels.api.add_constant(column)
        linear_regression_model = statsmodels.api.OLS(y, predictor)
        linear_regression_model_fitted = linear_regression_model.fit()
        print(f"Variable: {feature_name}")
        print(linear_regression_model_fitted.summary())

        # Get the stats
        t_value = round(linear_regression_model_fitted.tvalues[1], 6)
        p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])

        # Plot the figure
        fig = px.scatter(x=column, y=y, trendline="ols")
        fig.update_layout(
            title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",
            xaxis_title=f"Variable: {feature_name}",
            yaxis_title="y",
        )
        fig.show()

    return
    """


if __name__ == "__main__":
    sys.exit(main())
