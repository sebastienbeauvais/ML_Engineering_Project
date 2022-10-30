import itertools
import random
import sys
from typing import List

import numpy as np
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
    name, url = val.split("#")
    return f'<a target="_blank" href="{url}">{name}</a>'


def make_clickable_names(val):
    name, url = val.split("#")
    return f'<a target="_blank" href="{url}">{name}</a>'


# dataset loader
def get_test_data_set(data_set_name: str = None) -> (pd.DataFrame, List[str], str):
    """Function to load a few test data sets

    :param:
    data_set_name : string, optional
        Data set to load

    :return:
    data_set : :class:`pd.DataFrame`
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
            data_set = pd.DataFrame(data.data, columns=data.feature_names)
            data_set["CHAS"] = data_set["CHAS"].astype(str)
        elif data_set_name == "diabetes":
            data = datasets.load_diabetes()
            data_set = pd.DataFrame(data.data, columns=data.feature_names)
            data_set["gender"] = ["1" if i > 0 else "0" for i in data_set["sex"]]
        elif data_set_name == "breast_cancer":
            data = datasets.load_breast_cancer()
            data_set = pd.DataFrame(data.data, columns=data.feature_names)

        data_set["target"] = data.target
        predictors = data.feature_names
        response = "target"

    print(f"Data set selected: {data_set_name}")
    return data_set, predictors, response


def main():
    # loading a test dataset
    # working: titanic, mpg, diabetes
    # broken: titanic_2, tips
    # very broken: breast_cancer, boston
    test_data, predictors, response = get_test_data_set(data_set_name="diabetes")

    # get response out of test_data
    all_preds = test_data.iloc[:, test_data.columns != response]

    # splitting dataset predictors into categorical and continuous
    df_continuous = all_preds.select_dtypes(include="float")
    df_categorical = all_preds.select_dtypes(exclude="float")
    # print(df_continuous.head(5))
    # print(df_categorical.head(5))

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
            "Absolute Value of Pearson",
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

    cont_cont_heat = go.Figure(data=[heat], layout=layout)

    # writing heatmap to html for linking in table
    cont_cont_heat.write_html(file="plots/corr/cont_cont_corr_matrix.html")
    # fig.show()

    # adding variables to cont/cont output table
    cont_cont_output_table["Predictors"] = col_combs
    cont_cont_output_table["Pearson's r"] = pearsons_r
    cont_cont_output_table["Absolute Value of Pearson"] = abs_pearson

    # Linear regression for each cont/cont predictors
    lm_l = []
    names = []
    urls = []
    links_df = pd.DataFrame(
        columns=[
            "name",
            "url",
        ]
    )
    if len(df_continuous.axes[1]) >= 2:
        for column_x in df_continuous:
            for column_y in df_continuous:
                if (
                    cont_cont_output_table["Predictors"]
                    .str.contains(f"{column_x}/{column_y}")
                    .any()
                ):
                    lm_l.append(f"{column_x}_{column_y}_linear_model")
                    lm = px.scatter(
                        df_continuous, x=column_x, y=column_y, trendline="ols"
                    )
                    results = px.get_trendline_results(lm)
                    # results = results.iloc[0]["px_fit_results"].summary()
                    t_val = results.iloc[0]["px_fit_results"].tvalues
                    t_val = round(t_val[1], 6)
                    p_val = results.iloc[0]["px_fit_results"].pvalues
                    p_val = p_val[1]
                    lm.update_layout(
                        title=f"{column_x}/{column_y}: (t-value={t_val} p-value={p_val})",
                    )
                    lm.write_html(
                        file=f"plots/lm/{column_x}_{column_y}_linear_model.html"
                    )
                    # add links
                    names.append(f"{column_x}_{column_y}_linear_model")
                    urls.append(f"plots/lm/{column_x}_{column_y}_linear_model.html")

        # print(cont_cont_output_table)
        links_df["name"] = names
        links_df["url"] = urls
        links_df["name_url"] = links_df["name"] + "#" + links_df["url"]

        cont_cont_output_table["Linear Regression Plot"] = links_df["name_url"]

    cont_cont_output_table = cont_cont_output_table.sort_values(
        by=["Absolute Value of Pearson"], ascending=False
    )

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
    cont_cont_brute_force["Predictors"] = col_combs

    bins_uw = []
    if len(df_continuous.axes[1]) >= 2:
        for col_x in df_continuous:
            for col_y in df_continuous:
                if (
                    cont_cont_brute_force["Predictors"]
                    .str.contains(f"{col_x}/{col_y}")
                    .any()
                ):
                    # added binned plots later
                    df = df_continuous[col_x], df_continuous[col_y]
                    uw = px.density_heatmap(
                        df,
                        x=df_continuous[col_x],
                        y=df_continuous[col_y],
                        histfunc="avg",
                        histnorm="probability",
                        text_auto=True,
                    )
                    uw.update_layout(
                        title_text=f"{col_x}_{col_y}_unweighted",
                        xaxis_title=f"{col_x}",
                        yaxis_title=f"{col_y}",
                    )
                    bins_uw.append(f"{col_x}_{col_y}_avg_response")
                    uw.write_html(
                        f"plots/binned_uw/{col_x}_{col_y}_probability_density_plot.html"
                    )
                    # uw.show()

    # calculate mean response
    mse = []
    w_mse = []
    i = 0
    if len(df_continuous.axes[1]) >= 2:
        for column_x in df_continuous:
            for column_y in df_continuous:
                if (
                    cont_cont_brute_force["Predictors"]
                    .str.contains(f"{column_x}/{column_y}")
                    .any()
                ):
                    a = np.array(
                        [list(df_continuous[column_x]), list(df_continuous[column_y])]
                    )
                    # difference of mean response
                    mse.append(np.std(a, ddof=1) / np.sqrt(np.size(a)))

                    # weighted difference
                    weight = sum(
                        df_continuous[column_x] * df_continuous[column_y]
                    ) / sum(df_continuous[column_y])
                    sample_size = np.count_nonzero(a)

                    formula = np.sqrt(
                        abs(
                            np.sum(
                                df_continuous[column_y]
                                * (df_continuous[column_x] - weight) ** 2
                            )
                            / np.sum(
                                df_continuous[column_y]
                                * (sample_size - 1)
                                / sample_size
                            )
                        )
                    )
                    w_mse.append(formula)
                    # added binned plots later
                    data = {
                        "x": df_continuous[column_x],
                        "y": df_continuous[column_y],
                    }
                    df = pd.DataFrame(data)
                    # weighted plot

        cont_cont_brute_force["Difference of Mean Response"] = mse
        cont_cont_brute_force["Weighted Difference of Mean Response"] = w_mse
        cont_cont_brute_force["Bin Plot"] = bins_uw
    # sorting values for output table
    cont_cont_brute_force = cont_cont_brute_force.sort_values(
        by=["Weighted Difference of Mean Response"], ascending=False
    )

    # print(cont_cont_brute_force)

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
    cont_cat_corr = all_preds.corr()

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
    cont_cat_heat = go.Figure(data=[heat], layout=layout)
    cont_cat_heat.write_html(file="plots/corr/cont_cat_corr_matrix.html")
    # fig.show()

    # filling columns on output table
    cont_cat_output_table["Predictors"] = col_combs
    cont_cat_output_table["Correlation Ratio"] = out_corr
    cont_cat_output_table["Absolute Value of Correlation"] = abs_out_corr

    # plots for cont/cat table
    v_l = []
    h_l = []
    names_1 = []
    urls_1 = []
    links_df_1 = pd.DataFrame(
        columns=[
            "name",
            "url",
        ]
    )
    names_2 = []
    urls_2 = []
    links_df_2 = pd.DataFrame(
        columns=[
            "name",
            "url",
        ]
    )
    for column_x in all_preds:
        if all_preds[column_x].dtype != "category":
            for column_y in all_preds:
                if (
                    cont_cat_output_table["Predictors"]
                    .str.contains(f"{column_x}/{column_y}")
                    .any()
                ):
                    # print({column_x},{column_y})
                    violin = px.violin(
                        all_preds, x=column_x, y=column_y, color=column_x
                    )
                    violin.update_layout(
                        title_text=f"{column_x}/{column_y}: violin plot"
                    )
                    v_l.append(f"{column_x}_{column_y}_violin_plot")
                    violin.write_html(
                        file=f"plots/violin/cont_cat_{column_x}_{column_y}_violin_plot.html"
                    )
                    names_2.append(f"{column_x}_{column_y}_violin_plot")
                    urls_2.append(
                        f"plots/violin/cont_cat_{column_x}_{column_y}_violin_plot.html"
                    )
                    # violin.show()
                    hist = px.histogram(
                        all_preds,
                        x=column_x,
                        y=column_y,
                        color=column_x,
                        marginal="rug",
                    )
                    # hist.show()
                    hist.update_layout(
                        title_text=f"{column_x}/{column_y}: dist plot",
                        xaxis_title_text=f"{column_x}",
                        # yaxis_title_text=f"{column_y}",
                    )
                    h_l.append(f"{column_x}_{column_y}_dist_plot")
                    hist.write_html(
                        file=f"plots/hist/cont_cat_{column_x}_{column_y}_hist_plot.html"
                    )
                    names_1.append(f"{column_x}_{column_y}_dist_plot")
                    urls_1.append(
                        f"plots/hist/cont_cat_{column_x}_{column_y}_hist_plot.html"
                    )

    # print(cont_cont_output_table)
    links_df_1["name"] = names_1
    links_df_1["url"] = urls_1
    links_df_1["name_url"] = links_df_1["name"] + "#" + links_df_1["url"]

    links_df_2["name"] = names_2
    links_df_2["url"] = urls_2
    links_df_2["name_url"] = links_df_2["name"] + "#" + links_df_2["url"]

    cont_cat_output_table["Violin Plot"] = links_df_2["name_url"]
    cont_cat_output_table["Distribution Plot"] = links_df_1["name_url"]

    # print(cont_cat_output_table)

    cont_cat_output_table = cont_cat_output_table.sort_values(
        by=["Absolute Value of Correlation"], ascending=False
    )

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

    calc_df = all_preds.select_dtypes(exclude=["category"])
    cont_cat_brute_force["Predictors"] = col_combs

    """# binned plot - probably wrong, very confused
    bins_uw = []
    for col_x in calc_df:
        for col_y in calc_df:
            if (
                cont_cat_brute_force["Predictors"]
                .str.contains(f"{col_x}/{col_y}")
                .any()
            ):
                df = calc_df[col_x], calc_df[col_y]
                uw = px.density_heatmap(
                    df,
                    x=calc_df[col_x],
                    y=calc_df[col_y],
                    histfunc="avg",
                    histnorm="probability",
                    text_auto=True,
                )
                uw.update_layout(
                    title_text=f"{col_x}_{col_y}_unweighted",
                    xaxis_title=f"{col_x}",
                    yaxis_title=f"{col_y}",
                )
                bins_uw.append(f"{col_x}_{col_y}_avg_response")
                uw.write_html(
                    f"plots/binned_uw/{col_x}_{col_y}_probability_density_plot.html"
                )
                uw.show()"""

    for variable in calc_df:
        if calc_df[variable].dtype != "float":
            calc_df[variable] = calc_df[variable].astype("category")
            calc_df[variable] = calc_df[variable].cat.codes

    # calculate mean response
    mse = []
    w_mse = []
    for column_x in calc_df:
        if all_preds[column_x].dtype != "category":
            for column_y in calc_df:
                if (
                    cont_cat_output_table["Predictors"]
                    .str.contains(f"{column_x}/{column_y}")
                    .any()
                ):
                    a = np.array([list(calc_df[column_x]), list(calc_df[column_y])])
                    # formula for standard error
                    mse.append(np.std(a, ddof=1) / np.sqrt(np.size(a)))
                    weight = sum(calc_df[column_x] * calc_df[column_y]) / sum(
                        calc_df[column_y]
                    )
                    sample_size = np.count_nonzero(a)
                    formula = np.sqrt(
                        abs(
                            np.sum(
                                calc_df[column_y] * (calc_df[column_x] - weight) ** 2
                            )
                            / np.sum(
                                calc_df[column_y] * (sample_size - 1) / sample_size
                            )
                        )
                    )
                    w_mse.append(formula)

    cont_cat_brute_force["Difference of Mean Response"] = mse
    cont_cat_brute_force["Weighted Difference of Mean Response"] = w_mse
    # cont_cat_brute_force["Bin Plot"] = bins_uw
    cont_cat_brute_force = cont_cat_brute_force.sort_values(
        by=["Weighted Difference of Mean Response"], ascending=False
    )

    # print(cont_cat_brute_force)

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
    cat_cat_heat = go.Figure(data=[heat], layout=layout)
    cat_cat_heat.write_html(file="plots/corr/cat_cat_corr_matrix.html")
    # fig.show()

    # filling columns on output table
    cat_cat_output_table["Predictors"] = col_combs

    # plots_df = df_categorical.select_dtypes(exclude="category")

    # converting categoricals to numbers
    for variable in df_categorical:
        if (
            df_categorical[variable].dtype == "object"
            or df_categorical[variable].dtype == "category"
        ):
            df_categorical[variable] = df_categorical[variable].astype("category")
            df_categorical[variable] = df_categorical[variable].cat.codes

    # plots for cat/cat table
    i = 0
    if len(df_categorical.axes[1]) >= 2:
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
                    i += 1
                elif column_x == column_y:
                    i = i
                    continue
                else:
                    break

    # getting heat plots for each pair
    names = []
    urls = []
    links_df = pd.DataFrame(
        columns=[
            "name",
            "url",
        ]
    )
    p_heat_l = []
    if len(df_categorical.axes[1]) >= 2:
        for column_x in df_categorical:
            for column_y in df_categorical:
                if (
                    cont_cat_output_table["Predictors"]
                    .str.contains(f"{column_x}/{column_y}")
                    .any()
                ):
                    heat_data = df_categorical[[column_x, column_y]]
                    heat_data = heat_data.corr()
                    p_heat = go.Figure()
                    p_heat.add_trace(
                        go.Heatmap(
                            x=heat_data.columns,
                            y=heat_data.index,
                            z=np.array(heat_data),
                        )
                    )
                    p_heat_l.append(f"{column_x}_{column_y}_heat_plot")
                    p_heat.write_html(
                        file=f"plots/corr/cat_cat_{column_x}_{column_y}_matrix.html"
                    )
                    # add links
                    names.append(f"{column_x}_{column_y}_heat_plot")
                    urls.append(f"plots/corr/cat_cat_{column_x}_{column_y}_matrix.html")

        # print(cont_cont_output_table)
        links_df["name"] = names
        links_df["url"] = urls
    if len(links_df) > 1:
        links_df["name_url"] = links_df["name"] + "#" + links_df["url"]
        cat_cat_output_table["Heatmap"] = links_df["name_url"]

    cat_cat_output_table = cat_cat_output_table.sort_values(
        by=["Absolute Value of Correlation"], ascending=False
    )

    # print(cat_cat_output_table)

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
    cat_cat_brute_force["Predictors"] = col_combs

    """# binned plot - probably wrong, very confused
    bins_uw = []
    for col_x in calc_df:
        for col_y in calc_df:
            if cat_cat_brute_force["Predictors"].str.contains(f"{col_x}/{col_y}").any():
                # added binned plots later
                df = calc_df[col_x], calc_df[col_y]
                uw = px.density_heatmap(
                    df,
                    x=calc_df[col_x],
                    y=calc_df[col_y],
                    histfunc="avg",
                    histnorm="probability",
                    text_auto=True,
                )
                uw.update_layout(
                    title_text=f"{col_x}_{col_y}_unweighted",
                    xaxis_title=f"{col_x}",
                    yaxis_title=f"{col_y}",
                )
                bins_uw.append(f"{col_x}_{col_y}_avg_response")
                uw.write_html(
                    f"plots/binned_uw/{column_x}_{column_y}_probability_density_plot.html"
                )
                # uw.show()"""

    # calculate mean response
    mse = []
    w_mse = []
    i = 0
    if len(df_categorical.axes[1]) >= 2:
        for column_x in df_categorical:
            for column_y in df_categorical:
                if column_x != column_y and i < len(cat_cat_output_table):
                    a = np.array(
                        [list(df_categorical[column_x]), list(df_categorical[column_y])]
                    )
                    # formula for standard error
                    mse.append(np.std(a, ddof=1) / np.sqrt(np.size(a)))
                    weight = sum(
                        df_categorical[column_x] * df_categorical[column_y]
                    ) / sum(df_categorical[column_y])
                    sample_size = np.count_nonzero(a)
                    formula = np.sqrt(
                        np.sum(
                            df_categorical[column_y]
                            * (df_categorical[column_x] - weight) ** 2
                        )
                        / np.sum(
                            df_categorical[column_y] * (sample_size - 1) / sample_size
                        )
                    )
                    w_mse.append(formula)
                    # added binned plots later
                    i += 1
                elif column_x == column_y:
                    i = i
                    continue
                else:
                    break

        cat_cat_brute_force["Difference of Mean Response"] = mse
        cat_cat_brute_force["Weighted Difference of Mean Response"] = w_mse
        # cat_cat_brute_force["Bin Plot"] = bins_uw
    cat_cat_brute_force = cat_cat_brute_force.sort_values(
        by=["Weighted Difference of Mean Response"], ascending=False
    )

    # print(cat_cat_brute_force)

    #####################################################################
    # WRITING TO HTML TABLE OUTPUT
    #####################################################################
    # using multiple files to still get output with broken datasets
    with open("cont_cont_out.html", "w") as _file:
        _file.write(
            cont_cont_output_table.style.format(
                {"Linear Regression Plot": make_clickable}
            ).to_html(escape=False)
            + "\n\n"
            + cont_cont_heat.to_html()
            + "\n\n"
            + cont_cont_brute_force.to_html(escape=False)
        )
    with open("cont_cat_out.html", "w") as _file:
        _file.write(
            cont_cat_output_table.style.format(
                {"Violin Plot": make_clickable, "Distribution Plot": make_clickable}
            ).to_html(escape=False)
            + "\n\n"
            + cont_cat_heat.to_html()
            + "\n\n"
            + cont_cat_brute_force.to_html(escape=False)
        )
    with open("cat_cat_out.html", "w") as _file:
        _file.write(
            cat_cat_output_table.style.format({"Heatmap": make_clickable}).to_html(
                escape=False
            )
            + "\n\n"
            + cat_cat_heat.to_html()
            + "\n\n"
            + cat_cat_brute_force.to_html(escape=False)
        )
    # also writing one big file with all tables if all tables run
    with open("midterm.html", "w") as _file:
        _file.write(
            cont_cont_output_table.style.format(
                {"Linear Regression Plot": make_clickable}
            ).to_html(escape=False)
            + "\n\n"
            + cont_cont_heat.to_html()
            + "\n\n"
            + cont_cont_brute_force.to_html(escape=False)
            + cont_cat_output_table.style.format(
                {"Violin Plot": make_clickable, "Distribution Plot": make_clickable}
            ).to_html(escape=False)
            + "\n\n"
            + cont_cat_heat.to_html()
            + "\n\n"
            + cont_cat_brute_force.to_html(escape=False)
            + cat_cat_output_table.style.format({"Heatmap": make_clickable}).to_html(
                escape=False
            )
            + "\n\n"
            + cat_cat_heat.to_html()
            + "\n\n"
            + cat_cat_brute_force.to_html(escape=False)
        )


if __name__ == "__main__":
    sys.exit(main())
    # df, predictors, response = get_test_data_set(data_set_name=)
