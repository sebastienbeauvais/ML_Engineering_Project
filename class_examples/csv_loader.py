import pandas as pd

from class_examples.data_loader import DataLoader


class CSVLoader(DataLoader):
    # build a constructor
    def __init__(self, filename):
        print("Starting DataLoader")
        # self.data = pd.read_csv(filename)
        self._loader(filename=filename)
        return

    def _loader(self, filename):
        # if i make a class off this class we need to implement this loader
        # raise NotImplementedError
        self.data = pd.read_csv(filename)


"""
    @staticmethod
    def print_any_dataframe(df, n=n):
        print(df.head(n=5))
        return
"""
