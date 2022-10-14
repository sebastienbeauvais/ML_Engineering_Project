# libraries
import sys

import pandas as pd
import plotly.express as px
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
    cols = len(
        cont.columns
    )  # gets how ever many continous variables we have in out model
    titanic_features = list(cont.columns)
    predictor = sm.add_constant(cont.iloc[:, 0:cols])
    lm = sm.OLS(y, predictor)
    lm_fitted = lm.fit()
    print(f"Variable: {titanic_features}")
    print(lm_fitted.summary())

    # checking relationship between variables
    print(df_titanic.corr())
    heat_map = px.imshow(df_titanic)
    heat_map.show()


if __name__ == "__main__":
    sys.exit(main())
