import itertools
import random
import sys
from typing import List

# import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import seaborn
from sklearn import datasets

TITANIC_PREDICTORS = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "embarked",
    "parch",
    "fare",
    "who",
    "adult_male",
    "deck",
    "embark_town",
    "alone",
    "class",
]


# make links clickable
def make_clickable(val):
    return '<a href="{}">{}</a>'.format(val, val)


# dataset loader
def get_test_data_set(data_set_name: str = None) -> (pandas.DataFrame, List[str], str):
    """Function to load a few test data sets

    :param:
    data_set_name : string, optional
        Data set to load

    :return:
    data_set : :class:`pandas.DataFrame`
        Tabular data, possibly with some preprocessing applied.
    predictors :list[str]
        List of predictor variables
    response: str
        Response variable
    """
    seaborn_data_sets = ["mpg", "tips", "titanic", "titanic_2"]
    sklearn_data_sets = ["boston", "diabetes", "breast_cancer"]
    all_data_sets = seaborn_data_sets + sklearn_data_sets

    if data_set_name is None:
        data_set_name = random.choice(all_data_sets)
    else:
        if data_set_name not in all_data_sets:
            raise Exception(f"Data set choice not valid: {data_set_name}")

    if data_set_name in seaborn_data_sets:
        if data_set_name == "mpg":
            data_set = seaborn.load_dataset(name="mpg").dropna().reset_index()
            predictors = [
                "cylinders",
                "displacement",
                "horsepower",
                "weight",
                "acceleration",
                "origin",
                "name",
            ]
            response = "mpg"
        elif data_set_name == "tips":
            data_set = seaborn.load_dataset(name="tips").dropna().reset_index()
            predictors = [
                "total_bill",
                "sex",
                "smoker",
                "day",
                "time",
                "size",
            ]
            response = "tip"
        elif data_set_name == "titanic":
            data_set = seaborn.load_dataset(name="titanic").dropna()
            predictors = TITANIC_PREDICTORS
            response = "survived"
        elif data_set_name == "titanic_2":
            data_set = seaborn.load_dataset(name="titanic").dropna()
            predictors = TITANIC_PREDICTORS
            response = "alive"
    elif data_set_name in sklearn_data_sets:
        if data_set_name == "boston":
            data = datasets.load_boston()
            data_set = pandas.DataFrame(data.data, columns=data.feature_names)
            data_set["CHAS"] = data_set["CHAS"].astype(str)
        elif data_set_name == "diabetes":
            data = datasets.load_diabetes()
            data_set = pandas.DataFrame(data.data, columns=data.feature_names)
            data_set["gender"] = ["1" if i > 0 else "0" for i in data_set["sex"]]
        elif data_set_name == "breast_cancer":
            data = datasets.load_breast_cancer()
            data_set = pandas.DataFrame(data.data, columns=data.feature_names)

        data_set["target"] = data.target
        predictors = data.feature_names
        response = "target"

    print(f"Data set selected: {data_set_name}")
    return data_set, predictors, response


def main():
    # loading a test dataset
    test_data, predictors, response = get_test_data_set(data_set_name="titanic")

    # splitting dataset predictors into categorical and continuous
    df_continuous = test_data.select_dtypes(include="float")
    df_categorical = test_data.select_dtypes(exclude="float")
    print(df_continuous.head(5))
    print(df_categorical.head(5))

    #####################################################################
    # CONT/CONT OUTPUT TABLE
    #####################################################################

    # continuous/continuous Predictor Pairs
    pearson_corr = df_continuous.corr()
    # print(pearson_corr)
    col_1 = pearson_corr.iloc[:, 0][1:]  # remove age to be first column
    col_1 = list(col_1[0:])  # remove line?
    abs_pearson = abs(pearson_corr)
    col_2 = abs_pearson.iloc[:, 0][1:]
    col_2 = list(col_2[0:])  # remove line?

    # table to populate
    cont_cont_output_table = pd.DataFrame(
        columns=[
            "Predictors",
            "Pearson's r",
            "Abs Value of Pearson",
            "Linear Regression Plot",
        ]
    )

    # defining variable to get column combinations
    cols = []

    # loop through column names to get combinations
    for L in range(len(pearson_corr) + 1):
        for subset in itertools.combinations(pearson_corr, L):
            cols = list(subset)

    # getting combinations of columns
    col_combs = ["/".join(map(str, comb)) for comb in itertools.combinations(cols, 2)]

    # heatmap plot for cont/cont variables
    mask = np.triu(np.ones_like(pearson_corr, dtype=bool))
    rLT = pearson_corr.mask(mask)
    abs_rLT = abs(rLT)

    # needed rLT for list of corr values
    # getting them into a list for output table
    whole_list = []
    for row in rLT.values.tolist():
        partial_list = []
        for column in row:
            if pd.notna(column):
                partial_list.append(column)
        whole_list.append(partial_list)

    # flattened list of correlation values
    pearsons_r = [item for sublist in whole_list for item in sublist]

    # doing the same for absolute value of pearson
    whole_list = []
    for row in abs_rLT.values.tolist():
        partial_list = []
        for column in row:
            if pd.notna(column):
                partial_list.append(column)
        whole_list.append(partial_list)

    # flattened list of correlation values
    abs_pearson = [item for sublist in whole_list for item in sublist]

    # defining heatmap
    heat = go.Heatmap(
        z=rLT,
        x=rLT.columns.values,
        y=rLT.columns.values,
        zmin=-0.25,
        zmax=1,
        xgap=1,
        ygap=1,
    )

    # formatting heatmap
    title = "Cont/Cont Predictor Correlation Matrix"
    layout = go.Layout(
        title_text=title,
        title_x=0.5,
        width=600,
        height=600,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        yaxis_autorange="reversed",
    )

    fig = go.Figure(data=[heat], layout=layout)

    # writing heatmap to html for linking in table
    fig.write_html(file="cont_cont_corr_matrix.html")
    # fig.show()

    # adding variables to cont/cont output table
    cont_cont_output_table["Predictors"] = col_combs
    cont_cont_output_table["Pearson's r"] = pearsons_r
    cont_cont_output_table["Abs Value of Pearson"] = abs_pearson

    # Linear regression for each cont/cont predictors
    i = 0
    for column_x in df_continuous:
        for column_y in df_continuous:
            if column_x != column_y and i <= len(cont_cont_output_table):
                fig = px.scatter(df_continuous, x=column_x, y=column_y, trendline="ols")
                results = px.get_trendline_results(fig)
                # results = results.iloc[0]["px_fit_results"].summary()
                t_val = results.iloc[0]["px_fit_results"].tvalues
                t_val = round(t_val[1], 6)
                p_val = results.iloc[0]["px_fit_results"].pvalues
                p_val = p_val[1]
                fig.update_layout(
                    title=f"{column_x}/{column_y}: (t-value={t_val} p-value={p_val})",
                )
                fig.write_html(f"{column_x}_{column_y}_linear_model.html")
                # still  need to add link
                cont_cont_output_table["Linear Regression Plot"][
                    i
                ] = f"{column_x}_{column_y}_linear_model"
                cont_cont_output_table.style.format(
                    {"Linear Regression Plot": make_clickable}
                )
                # fig.show()
                i += 1
            elif column_x == column_y:
                i = i
                continue
            else:
                break

    print(cont_cont_output_table)

    # cont/cont Brute force table
    cont_cont_brute_force = pd.DataFrame(
        columns=[
            "Predictors",
            "Difference of Mean Response",
            "Weighted Difference of Mean Response",
            "Bin Plot",
            "Residual Plot",
        ]
    )

    # calculate mean response
    mse = []
    w_mse = []
    i = 0
    for column_x in df_continuous:

        for column_y in df_continuous:
            if column_x != column_y and i < len(cont_cont_output_table):
                a = np.array(
                    [list(df_continuous[column_x]), list(df_continuous[column_y])]
                )
                # formula for standard error
                mse.append(np.std(a, ddof=1) / np.sqrt(np.size(a)))
                weight = sum(df_continuous[column_x] * df_continuous[column_y]) / sum(
                    df_continuous[column_y]
                )
                sample_size = np.count_nonzero(a)
                formula = np.sqrt(
                    np.sum(
                        df_continuous[column_y]
                        * (df_continuous[column_x] - weight) ** 2
                    )
                    / np.sum(df_continuous[column_y] * (sample_size - 1) / sample_size)
                )
                w_mse.append(formula)
                # added binned plots later
                i += 1
            elif column_x == column_y:
                i = i
                continue
            else:
                break

    cont_cont_brute_force["Predictors"] = col_combs
    cont_cont_brute_force["Difference of Mean Response"] = mse
    cont_cont_brute_force["Weighted Difference of Mean Response"] = w_mse

    print(cont_cont_brute_force)

    #####################################################################
    # CONT/CAT OUTPUT TABLE
    #####################################################################
    # defining the output table
    cont_cat_output_table = pd.DataFrame(
        columns=[
            "Predictors",
            "Correlation Ratio",
            "Absolute Value of Correlation",
            "Violin Plot",
            "Distribution Plot",
        ]
    )

    # correlation between cont/cat predictors
    cont_cat_corr = test_data.corr()
    print(cont_cat_corr)

    # this will hold out column pairs
    cols = []

    # loop through column names to get combinations
    for L in range(len(cont_cat_corr) + 1):
        for subset in itertools.combinations(cont_cat_corr, L):
            cols = list(subset)

    # getting combinations of columns
    col_combs = ["/".join(map(str, comb)) for comb in itertools.combinations(cols, 2)]

    # heatmap plot for cont/cat variables
    mask = np.triu(np.ones_like(cont_cat_corr, dtype=bool))
    rLT = cont_cat_corr.mask(mask)
    abs_rLT = abs(rLT)

    # getting them into a list for output table
    whole_list = []
    for row in rLT.values.tolist():
        partial_list = []
        for column in row:
            if pd.notna(column):
                partial_list.append(column)
        whole_list.append(partial_list)

    # flatten list for correlation values
    out_corr = [item for sublist in whole_list for item in sublist]

    # same but for abs values as above
    whole_list = []
    for row in abs_rLT.values.tolist():
        partial_list = []
        for column in row:
            if pd.notna(column):
                partial_list.append(column)
        whole_list.append(partial_list)

    # flatten list for correlation values
    abs_out_corr = [item for sublist in whole_list for item in sublist]

    # defining heatmap
    heat = go.Heatmap(
        z=rLT,
        x=rLT.columns.values,
        y=rLT.columns.values,
        zmin=-0.25,
        xgap=1,
        ygap=1,
    )

    # formatting heatmap
    title = "Cont/Cat Predictor Correlation Matrix"
    layout = go.Layout(
        title_text=title,
        title_x=0.5,
        width=600,
        height=600,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        yaxis_autorange="reversed",
    )

    # plotting figure
    fig = go.Figure(data=[heat], layout=layout)
    fig.write_html(file="cont_cat_corr_matrix.html")
    # fig.show()

    # filling columns on output table
    cont_cat_output_table["Predictors"] = col_combs
    cont_cat_output_table["Correlation Ratio"] = out_corr
    cont_cat_output_table["Absolute Value of Correlation"] = abs_out_corr

    print(cont_cat_output_table)


if __name__ == "__main__":
    sys.exit(main())
    # df, predictors, response = get_test_data_set(data_set_name=)
