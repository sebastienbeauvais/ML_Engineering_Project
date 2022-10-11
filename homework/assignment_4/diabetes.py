import sys

import pandas as pd
import statsmodels.api as sm

# from plotly import express as px
from sklearn import datasets


def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df["target"] = pd.Series(sklearn_dataset.target)
    return df


def main():
    # diabetes = datasets.load_diabetes()
    df_diabetes = sklearn_to_df(datasets.load_diabetes())
    print(df_diabetes.head())

    # splitting features and target
    X = df_diabetes.iloc[:, :-1]  # all but last column
    y = df_diabetes.iloc[:, -1]  # last column
    # X = diabetes.data
    # y = diabetes.target

    # categorizing each variable
    for (columnName, columnData) in df_diabetes.iteritems():
        print("Column Name : ", columnName)
        print("Unique Values : ", columnData.nunique())
        if columnData.nunique() > 4:
            print("Continuous\n")
        else:
            print("Boolean\n")

    # this works for a single predictor
    feature_name = list(X.columns)
    print(feature_name)  # inlcudes target. must remove
    predictor = sm.add_constant(X.iloc[:, 0])
    linear_regression_model = sm.OLS(y, predictor)
    linear_regression_model_fitted = linear_regression_model.fit()
    print(f"Variable: {feature_name}")
    print(linear_regression_model_fitted.summary())

    # working on all predictors in any dataframe
    column_headers = list(X.columns)  # get list of column headers

    for index, col in enumerate(X.T):
        feature_name = column_headers[index]
        predictor = sm.add_constant(X.iloc[:, col])
        lm = sm.OLS(y, predictor)
        lm_fitted = lm.fit()
        print(f"Variable: {feature_name}")
        print(lm_fitted.summary())

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
