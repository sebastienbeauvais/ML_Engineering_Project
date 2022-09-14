# main function
def main():
    # importing required libraries

    import pandas as pd
    import plotly.express as px
    from sklearn import datasets
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    # reading in UCI iris datasets into a pandas dataframe
    # naming columns to be more reader friendly
    csv_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    column_names = [
        "sepal_length",
        "sepal_width",
        "pedal_length",
        "pedal_width",
        "species",
    ]
    iris = pd.read_csv(csv_url, names=column_names)

    # checking data
    print("Checking the data ...")
    print(iris.head(5))
    print("\n")

    # some simple statistics of the data
    # using describe() for a quick output
    # print(iris.describe())

    # for loop to get stats on each column
    for col in iris[["sepal_length"]]:
        print("Mean of: \n{}\n".format(iris.iloc[:, :4].mean()))
        print("Max of: \n{}\n".format(iris.iloc[:, :4].max()))
        print("Min of: \n{}\n".format(iris.iloc[:, :4].min()))
        print(
            "Quantiles of: \n{}\n".format(iris.iloc[:, :4].quantile([0.25, 0.5, 0.75]))
        )

    # plotting different classes against each other
    # plot 1 - scatter plot
    df = px.data.iris()  # iris is a pandas DataFrame
    scatter = px.scatter(
        df,
        x="sepal_width",
        y="sepal_length",
        color="species",
        size="petal_length",
        hover_data=["petal_width"],
        title="Scatter plot of iris data set",
    )
    # scatter.write_html(file="scatter_plot.html", include_plotlyjs="cdn")
    scatter.show()

    # plot 2 - violin plot
    df = px.data.iris()
    violin = px.violin(
        df,
        y="sepal_width",
        color="species",
        violinmode="overlay",
        title="Violin plot of iris data set",  # draw violins on top of each other
        # default violinmode is 'group' as in example above
        hover_data=df.columns,
    )
    # violin.write_html(file="violin_plot.html", include_plotlyjs="cdn")
    violin.show()

    # plot 3 - scatter matrix
    df = px.data.iris()
    scatter_matrix = px.scatter_matrix(
        df,
        dimensions=["sepal_width", "sepal_length", "petal_width", "petal_length"],
        color="species",
        symbol="species",
        title="Scatter matrix of iris data set",
        labels={col: col.replace("_", " ") for col in df.columns},
    )  # remove underscore
    scatter_matrix.update_traces(diagonal_visible=False)
    # scatter_matrix.write_html(file="scatter_matrix.html", include_plotlyjs="cdn")
    scatter_matrix.show()

    # plot 4 - distribution plot
    df = px.data.iris()
    hist = px.histogram(
        df,
        x="sepal_width",
        y="sepal_length",
        color="species",
        marginal="violin",
        hover_data=df.columns,
        title="Distribution plot of iris data set",
    )
    # hist.write_html(file="hist_plot.html", include_plotlyjs="cdn")
    hist.show()

    # plot 5 - boxplot
    df = px.data.iris()
    box = px.box(
        df, y="sepal_length", color="species", title="Boxplot of iris data set"
    )
    # box.write_html(file="box_plot.html", include_plotlyjs="cdn")
    box.show()

    # Machine learning models
    # loading data (starting with fresh set in case of any alterations)
    iris = datasets.load_iris()

    # variable for feature data
    X = iris.data

    # variable for target data
    y = iris.target

    # splitting data into four new datasets (train, test for x and y)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    # Using a sklearn Pipeline on ML models
    # setting up training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    # chaining mulitple pipelines together
    pipeline_rf = Pipeline(
        [
            ("scaler1", StandardScaler()),
            ("pca1", PCA(n_components=2)),
            ("classification", RandomForestClassifier()),
        ]
    )
    pipeline_lr = Pipeline(
        [
            ("scaler2", StandardScaler()),
            ("pca2", PCA(n_components=2)),
            ("lr_classifier", LogisticRegression()),
        ]
    )
    pipeline_dt = Pipeline(
        [
            ("scaler3", StandardScaler()),
            ("pca3", PCA(n_components=2)),
            ("dt_classifier", DecisionTreeClassifier()),
        ]
    )

    # a list of pipelines
    pipelines = [pipeline_rf, pipeline_lr, pipeline_dt]

    # a dictionary of pipeline names for printing later
    pipe_dict = {0: "Random Forest", 1: "Logistic Regression", 2: "Decision Tree"}

    # fit each model
    for pipe in pipelines:
        pipe.fit(X_train, y_train)

    # print each models score
    for i, model in enumerate(pipelines):
        print("{} Test Accuracy:\n{}".format(pipe_dict[i], model.score(X_test, y_test)))


if __name__ == "__main__":
    main()
