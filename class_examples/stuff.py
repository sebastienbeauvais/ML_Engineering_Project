import os
import sys

# NOTE: add pyproject.toml to repo
# from class_examples.csv_loader import CSVLoader
from class_examples.data_loader import DataLoader

# import numpy as np
# import pandas as pd


def main():

    thing = 50
    print("+" * 80)

    # normal string with in cast as str
    test = "What is the number? " + str(thing) + " Other stuff"
    print(test)

    # formatted string are way better and more reader friendly
    test_2 = f"What is the number? {thing} Other stuff"
    print(test_2)

    dir_path = os.path.dirname(
        os.path.abspath(__file__)
    )  # /Users/sbeauvais/Projects/src/mis_602/class_examples
    print(dir_path)

    # importing a class we made
    # if file is not in the same folder use /../
    filename = f"{dir_path}/titanic.csv"
    print(filename)
    data_loader = DataLoader(filename=filename)

    data_loader.print()

    # df = pd.read_csv('titanic.csv')
    # df = pd.read_csv(f"{dir_path}/../titanic.csv")

    # df.head(5) # first five elements in dataframe

    # csv loader
    # CSVLoader.print_any_dataframe(data_loader.data)

    # **this is a splat
    # we will need this in spark

    return


# keep this at the bottom
if __name__ == "__main__":
    sys.exit(main())
