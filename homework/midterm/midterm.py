import itertools
import random
import sys
from typing import List

# import matplotlib.pyplot as plt
# import numpy as np
import pandas
import pandas as pd

# import plotly.express as px
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

    # continuous/continuous Predictor Pairs
    pearson_corr = df_continuous.corr()
    print(pearson_corr)
    col_1 = pearson_corr.loc[:, "age"][1:]
    print(col_1["fare"])
    abs_pearson = abs(pearson_corr)
    col_2 = abs_pearson.loc[:, "age"][1:]
    print(col_2["fare"])
    print(abs_pearson)

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

    # adding variables to cont/cont output table
    cont_cont_output_table["Predictors"] = cols
    cont_cont_output_table["Pearson's r"] = col_1["fare"]
    cont_cont_output_table["Abs Value of Pearson"] = col_2["fare"]
    print(cont_cont_output_table.head())

    # heatmap plot for cont/cont variables
    fig = go.Figure(
        data=go.Heatmap(z=pearson_corr, x=pearson_corr.columns, y=pearson_corr.columns)
    )

    # writing heatmap to html for linking in table
    fig.write_html(file="cont_cont_corr_matrix.html")
    # fig.show()


if __name__ == "__main__":
    sys.exit(main())
    # df, predictors, response = get_test_data_set(data_set_name=)
