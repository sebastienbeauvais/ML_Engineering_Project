import itertools
import random
import sys
from typing import List

import numpy as np
import pandas
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import scipy.stats as stats
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

    # get response out of test_data
    df = test_data.iloc[:, test_data.columns != response]

    # splitting dataset predictors into categorical and continuous
    df_continuous = df.select_dtypes(include="float")
    df_categorical = df.select_dtypes(exclude="float")
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
    fig.write_html(file="plots/corr/cont_cont_corr_matrix.html")
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
                fig.write_html(f"plots/lm/{column_x}_{column_y}_linear_model.html")
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

    #####################################################################
    # CONT/CONT BRUTE FORCE TABLE
    #####################################################################

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
    cont_cat_corr = df.corr()

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
    fig.write_html(file="plots/corr/cont_cat_corr_matrix.html")
    # fig.show()

    # filling columns on output table
    cont_cat_output_table["Predictors"] = col_combs
    cont_cat_output_table["Correlation Ratio"] = out_corr
    cont_cat_output_table["Absolute Value of Correlation"] = abs_out_corr

    # plots for cont/cat table
    i = 0
    for column_x in df:
        for column_y in df:
            if column_x != column_y and i <= len(cont_cat_output_table):
                # violin plot
                fig = px.violin(df, x=column_y, y=column_x, color=column_x)
                fig.update_layout(title_text=f"{column_x}/{column_y}: violin plot")
                fig.write_html(f"plots/violin/{column_x}_{column_y}_violin_plot.html")
                cont_cat_output_table["Violin Plot"][
                    i
                ] = f"{column_x}_{column_y}_violin_plot"
                cont_cat_output_table.style.format({"Violin Plot": make_clickable})

                # dist plot
                # switch x and y bc the plots made more sense upon display??
                fig_2 = px.histogram(
                    df, x=column_y, y=column_x, color=column_x, marginal="rug"
                )
                # fig_2.show()
                fig_2.update_layout(
                    title_text=f"{column_x}/{column_y}: dist plot",
                    xaxis_title_text=f"{column_y}",
                    yaxis_title_text=f"{column_x}",
                )
                fig_2.write_html(f"plots/hist/{column_x}_{column_y}_hist_plot.html")
                cont_cat_output_table["Distribution Plot"][
                    i
                ] = f"{column_x}_{column_y}_dist_plot"
                cont_cat_output_table.style.format({"Violin Plot": make_clickable})
                i += 1
            elif column_x == column_y:
                i = i
                continue
            else:
                break

    print(cont_cat_output_table)

    #####################################################################
    # CONT/CAT BRUTE FORCE TABLE
    #####################################################################
    cont_cat_brute_force = pd.DataFrame(
        columns=[
            "Predictors",
            "Difference of Mean Response",
            "Weighted Difference of Mean Response",
            "Bin Plot",
            "Residual Plot",
        ]
    )
    for variable in df:
        if df[variable].dtype != "float":
            df[variable] = df[variable].astype("category")
            df[variable] = df[variable].cat.codes

    print(df.head())
    # calculate mean response
    mse = []
    w_mse = []
    i = 0
    for column_x in df:
        for column_y in df:
            if column_x != column_y and i < len(cont_cat_output_table):
                a = np.array([list(df[column_x]), list(df[column_y])])
                # formula for standard error
                mse.append(np.std(a, ddof=1) / np.sqrt(np.size(a)))
                weight = sum(df[column_x] * df[column_y]) / sum(df[column_y])
                sample_size = np.count_nonzero(a)
                formula = np.sqrt(
                    np.sum(df[column_y] * (df[column_x] - weight) ** 2)
                    / np.sum(df[column_y] * (sample_size - 1) / sample_size)
                )
                w_mse.append(formula)
                # added binned plots later
                i += 1
            elif column_x == column_y:
                i = i
                continue
            else:
                break

    cont_cat_brute_force["Predictors"] = col_combs
    cont_cat_brute_force["Difference of Mean Response"] = mse
    cont_cat_brute_force["Weighted Difference of Mean Response"] = w_mse

    print(cont_cat_brute_force)

    #####################################################################
    # CAT/CAT OUTPUT TABLE
    #####################################################################
    # defining the output table
    cat_cat_output_table = pd.DataFrame(
        columns=[
            "Predictors",
            "Cramer's V",
            "Absolute Value of Correlation",
            "Heatmap",
        ]
    )

    # correlation between cont/cat predictors
    cat_cat_corr = df_categorical.corr()

    # this will hold our column pairs
    cols = []

    # loop through column names to get combinations
    for L in range(len(cat_cat_corr) + 1):
        for subset in itertools.combinations(cat_cat_corr, L):
            cols = list(subset)

    # getting combinations of columns
    col_combs = ["/".join(map(str, comb)) for comb in itertools.combinations(cols, 2)]

    # heatmap plot for all cat/cat variables
    mask = np.triu(np.ones_like(cat_cat_corr, dtype=bool))
    rLT = cat_cat_corr.mask(mask)
    abs_rLT = abs(rLT)

    # getting them into a list for output table
    whole_list = []
    for row in rLT.values.tolist():
        partial_list = []
        for column in row:
            if pd.notna(column):
                partial_list.append(column)
        whole_list.append(partial_list)

    # same but for abs values as above
    whole_list = []
    for row in abs_rLT.values.tolist():
        partial_list = []
        for column in row:
            if pd.notna(column):
                partial_list.append(column)
        whole_list.append(partial_list)

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
    title = "Cat/Cat Predictor Correlation Matrix"
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
    fig.write_html(file="plots/corr/cat_cat_corr_matrix.html")
    # fig.show()

    # filling columns on output table
    cat_cat_output_table["Predictors"] = col_combs

    # converting categoricals to numbers
    print(df_categorical.dtypes)
    for variable in df_categorical:
        if (
            df_categorical[variable].dtype == "object"
            or df_categorical[variable].dtype == "category"
        ):
            df_categorical[variable] = df_categorical[variable].astype("category")
            df_categorical[variable] = df_categorical[variable].cat.codes

    # plots for cat/cat table
    i = 0
    for column_x in df_categorical:
        for column_y in df_categorical:
            if column_x != column_y and i <= len(cat_cat_output_table):

                # cramers v calculation
                temp_dat = []
                dat1 = df_categorical[column_x].tolist()
                dat2 = df_categorical[column_y].tolist()
                temp_dat.append(dat1)
                temp_dat.append(dat2)
                cramer_data = np.array(temp_dat)
                X2 = stats.chi2_contingency(cramer_data, correction=False)[0]
                N = np.sum(cramer_data)
                minimum_dimension = min(cramer_data.shape) - 1
                result = np.sqrt((X2 / N) / minimum_dimension)
                cat_cat_output_table["Cramer's V"][i] = result
                cat_cat_output_table["Absolute Value of Correlation"][i] = result

                heat_data = df_categorical[[column_x, column_y]]
                heat_data = heat_data.corr()
                fig = go.Figure()
                fig.add_trace(
                    go.Heatmap(
                        x=heat_data.columns, y=heat_data.index, z=np.array(heat_data)
                    )
                )
                fig.write_html(
                    file=f"plots/corr/cat_cat_{column_x}_{column_y}_matrix.html"
                )

                # fig.show()

                cat_cat_output_table["Heatmap"][i] = f"{column_x}_{column_y}_heatmap"
                cat_cat_output_table.style.format({"Heatmap": make_clickable})

                i += 1
            elif column_x == column_y:
                i = i
                continue
            else:
                break

    print(cat_cat_output_table)

    #####################################################################
    # CAT/CAT BRUTEFORCE TABLE
    #####################################################################
    cat_cat_brute_force = pd.DataFrame(
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
    for column_x in df_categorical:
        for column_y in df_categorical:
            if column_x != column_y and i < len(cat_cat_output_table):
                a = np.array(
                    [list(df_categorical[column_x]), list(df_categorical[column_y])]
                )
                # formula for standard error
                mse.append(np.std(a, ddof=1) / np.sqrt(np.size(a)))
                weight = sum(df_categorical[column_x] * df_categorical[column_y]) / sum(
                    df_categorical[column_y]
                )
                sample_size = np.count_nonzero(a)
                formula = np.sqrt(
                    np.sum(
                        df_categorical[column_y]
                        * (df_categorical[column_x] - weight) ** 2
                    )
                    / np.sum(df_categorical[column_y] * (sample_size - 1) / sample_size)
                )
                w_mse.append(formula)
                # added binned plots later
                i += 1
            elif column_x == column_y:
                i = i
                continue
            else:
                break

    cat_cat_brute_force["Predictors"] = col_combs
    cat_cat_brute_force["Difference of Mean Response"] = mse
    cat_cat_brute_force["Weighted Difference of Mean Response"] = w_mse

    print(cat_cat_brute_force)

    #####################################################################
    # WRITING TO HTML TABLE OUTPUT
    #####################################################################
    with open("index.html", "w") as _file:
        _file.write(
            cont_cont_output_table.to_html()
            + "\n\n"
            + cont_cont_brute_force.to_html()
            + "\n\n"
            + cont_cat_output_table.to_html()
            + "\n\n"
            + cont_cat_brute_force.to_html()
            + "\n\n"
            + cat_cat_output_table.to_html()
            + "\n\n"
            + cat_cat_brute_force.to_html()
        )


if __name__ == "__main__":
    sys.exit(main())
    # df, predictors, response = get_test_data_set(data_set_name=)
