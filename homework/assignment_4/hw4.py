# libraries
import sys

import pandas as pd

# import statsmodels.api as sm


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
    df_titanic = pd.read_csv("./datasets/titanic.csv")
    print(df_titanic.head())

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

    # get list of column headers to iterate over
    column_headers = list(X.columns)

    for index, col in enumerate(X.T):
        feature_name = column_headers[index]
        print(feature_name)
        # predictor = sm.add_constant(X.iloc[:, col])


if __name__ == "__main__":
    sys.exit(main())
