import pandas as pd


class DataLoader:
    # build a constructor
    def __init__(self, filename):
        print("Starting DataLoader")
        self.data = pd.read_csv(filename)

    # _ denotes that it should only be used in this class
    def _loader(self, filename):
        # if i make a class off this class we need to implement this loader
        raise NotImplementedError

    def print(self, n=10):
        print(self.data.head(n=n))
        return

    def __del__(self):
        print("Ending DataLoader")
