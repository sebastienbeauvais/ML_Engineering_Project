#!/usr/bin/env python3
import itertools
import os
import sys
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import sqlalchemy
from plotly.subplots import make_subplots
from sklearn import metrics, tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

print("finished reading libraries")

print("defining clickable links")
# make links clickable


def make_clickable(val):
    name, url = val.split("#")
    return f'<a target="_blank" href="{url}">{name}</a>'


print("defining main function")


def main():
    print("starting py file")
    warnings.simplefilter(action="ignore", category=FutureWarning)

    # CONNECTING TO DB
    # print("creating connection to database")

    print("passing database connection")
    # pragma: allowlist nextline secret
    msqldb_uri = "mariadb+pymysql://root:password@mariadb:3306/baseball?charset=utf8mb4"
    engine = sqlalchemy.create_engine(msqldb_uri)

    print("getting query")
    query = """
        SELECT
            *
        FROM
            more_baseball_feats;
     """
    print("finished query")

    print("creating dataframe")
    df = pd.read_sql_query(query, engine)
    print("changing data types")
    # making year a category
    df = df.astype({"game_year": "category", "HomeTeamWins": "category"})
    print("dropping NaNs")
    # filling nan
    df = df.dropna()
    print("cleaning data")
    # removing degree from temp
    df["temp"] = df["temp"].str.replace(" degrees", "")
    df["temp"] = df["temp"].astype(int)

    print("creating train test split")
    # TRAIN TEST SPLIT
    train_df = df[df["game_year"] != 2011]

    # test will be last year available
    test_df = df[df["game_year"] == 2011]
    print("train test split successful")

    train_df = train_df.drop(columns=["game_year"])
    test_df = test_df.drop(columns=["game_year"])

    print("Separating wins into prediction")
    X_train = train_df.loc[:, train_df.columns != "HomeTeamWins"]
    y_train = train_df.loc[:, train_df.columns == "HomeTeamWins"]
    y_train = y_train.values.ravel()

    X_test = test_df.loc[:, test_df.columns != "HomeTeamWins"]
    y_test = test_df.loc[:, test_df.columns == "HomeTeamWins"]

    train_cont = X_train.select_dtypes(exclude="category")
    print("separated predictor successfully")

    # creating directory for graphs
    print("checking if graphs exist")
    dir = "graphs"
    # parent dir
    parent_dir = os.getcwd()
    # path
    path = os.path.join(parent_dir, dir)
    try:
        os.makedirs(path, exist_ok=True)
        print("Directory '%s' created successfully" % dir)
    except OSError as error:
        print("Directory '%s' cannot be created" % dir, error)

    #################################################
    # PREDICTORS TABLE
    #################################################

    print("predictor table is being made")
    # cont predictors
    cont_predictors_table = pd.DataFrame(columns=["Predictor", "Violin Plot"])
    predictor_l = []
    violin_l = []
    hist_l = []
    names = []
    urls = []
    links_df = pd.DataFrame(
        columns=[
            "name",
            "url",
        ]
    )
    names2 = []
    urls2 = []
    links_df2 = pd.DataFrame(
        columns=[
            "name",
            "url",
        ]
    )
    train_holder = train_df

    print("starting graphs for output table")
    for column in train_holder:
        predictor_l.append(column)
        violin = px.violin(
            train_holder,
            y=train_holder[column],
            x=train_holder["HomeTeamWins"],
            color="HomeTeamWins",
            box=True,
        )
        print("creating histogram")
        hist = px.histogram(
            train_holder, x=train_holder[column], marginal="rug", color="HomeTeamWins"
        )
        print("creating violin plot")
        violin_l.append(f"{column}_violin_plot")
        violin.write_html(f"graphs/{column}_violin_plot.html")

        hist_l.append(f"{column}_histogram")
        hist.write_html(f"graphs/{column}_histogram.html")

        # add links
        names.append(f"{column}_violin_plot")
        urls.append(f"graphs/{column}_violin_plot.html")

        names2.append(f"{column}_histogram")
        urls2.append(f"graphs/{column}_histogram.html")

    print("inserting predictors into table")
    cont_predictors_table["Predictor"] = predictor_l

    print("getting links for violin")
    links_df["name"] = names
    links_df["url"] = urls
    links_df["name_url"] = links_df["name"] + "#" + links_df["url"]
    cont_predictors_table["Violin Plot"] = links_df["name_url"]

    print("getting links for histogram")
    links_df2["name"] = names2
    links_df2["url"] = urls2
    links_df2["name_url"] = links_df2["name"] + "#" + links_df2["url"]
    cont_predictors_table["Histogram"] = links_df2["name_url"]

    cont_predictors_table = cont_predictors_table.drop(
        cont_predictors_table.tail(1).index
    )

    #################################################
    # OUTPUT TABLE
    #################################################
    print("output table created")
    # defining table
    output_table = pd.DataFrame(
        columns=[
            "Predictors",
            "Pearson's R",
        ]
    )

    print("getting corr for all variables")
    # correlation between variables
    pearson_corr = train_cont.corr()

    # defining variable to hold combos
    cols = []

    print("getting list of all unique combinations")
    # getting list of all combinations
    for L in range(len(pearson_corr) + 1):
        for subset in itertools.combinations(pearson_corr, L):
            cols = list(subset)

    print("formatting string of combinations")
    # concating combinations to place in table
    col_combos = ["/".join(map(str, comb)) for comb in itertools.combinations(cols, 2)]

    print("creating heatmap")
    # heatmap plot for cont/cont predictors
    heat = go.Heatmap(
        z=pearson_corr,
        x=pearson_corr.columns.values,
        y=pearson_corr.columns.values,
        zmin=-0.25,
        xgap=1,
        ygap=1,
        colorscale="haline",
    )

    # formatting heatmap
    title = "Cont/Cont Predictor Correlation Matrix"
    layout = go.Layout(
        title_text=title,
        title_x=0.5,
        width=1300,
        height=900,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        yaxis_autorange="reversed",
    )

    heat = go.Figure(data=[heat], layout=layout)

    # writing to html
    heat.write_html("graphs/continuous_predictors_heatmap.html")

    # extracting correlation values
    mask = np.tril(np.ones_like(pearson_corr, dtype=bool))
    rLT = pearson_corr.mask(mask)
    abs_rLT = abs(rLT)

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

    print("adding values to output table")
    # adding variables to cont/cont output table
    output_table["Predictors"] = col_combos
    output_table["Pearson's R"] = pearsons_r
    output_table["Absolute Value of Pearson"] = abs_pearson

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
    t_list = []
    p_list = []
    if len(train_cont.axes[1]) >= 2:
        for column_x in train_cont:
            for column_y in train_cont:
                if (
                    output_table["Predictors"]
                    .str.contains(f"{column_x}/{column_y}")
                    .any()
                ):
                    lm_l.append(f"{column_x}__{column_y}_lm")
                    lm = px.scatter(train_cont, x=column_x, y=column_y, trendline="ols")
                    results = px.get_trendline_results(lm)
                    # results = results.iloc[0]["px_fit_results"].summary()
                    t_val = results.iloc[0]["px_fit_results"].tvalues
                    t_val = round(t_val[1], 6)
                    p_val = results.iloc[0]["px_fit_results"].pvalues
                    p_val = p_val[1]
                    lm.update_layout(
                        title=f"{column_x}/{column_y}: (t-value={t_val} p-value={p_val})",
                    )
                    lm.write_html(f"graphs/{column_x}*{column_y}_lm.html")

                    # add links
                    names.append(f"{column_x}*{column_y}_linear_model")
                    urls.append(f"graphs/{column_x}*{column_y}_lm.html")

                    t_list.append(t_val)
                    p_list.append(p_val)

    links_df["name"] = names
    links_df["url"] = urls
    links_df["name_url"] = links_df["name"] + "#" + links_df["url"]

    output_table["t-value"] = t_list
    output_table["p-score"] = p_list
    output_table["Linear Regression Plot"] = links_df["name_url"]

    #################################################
    # CONT/CONT BRUTE FORCE TABLE
    #################################################
    print("starting brute force")
    brute_force = pd.DataFrame(
        columns=[
            "Predictor 1",
            "Predictor 2",
        ]
    )

    # getting list of predictor pairs
    brute_force["Predictor Pair"] = output_table["Predictors"]
    brute_force[["Predictor 1", "Predictor 2"]] = output_table["Predictors"].str.split(
        "/", expand=True
    )

    # mse df
    mse_df = pd.DataFrame()
    l1 = []  # need lists to store data for later
    l2 = []
    l3 = []

    graphs_df = pd.DataFrame()
    joiner = []
    names = []
    urls = []

    print("starting bins plots")
    # getting sample mean, pop_mean for each predictor
    for column in train_cont:
        temp_df = pd.DataFrame()  # initialize df to store calcs
        temp_df[column] = train_cont[column]  # take col of interest for calcs
        sample_mean = temp_df[column].mean()  # get mean of col
        temp_df["bin"] = pd.cut(
            train_cont[column], 7, right=True
        )  # separate into 7 bins
        temp_df["sample_mean"] = sample_mean  # set new col to mean
        temp_df["bin_mean"] = temp_df.groupby("bin")[column].transform(
            "mean"
        )  # get mean of each bin
        temp_df["bin_count"] = temp_df.groupby("bin")[column].transform(
            "count"
        )  # get count of each bin
        temp_df = temp_df.drop(columns=[column])  # dropping base col to condense df
        temp_df["diff_mean_resp"] = (
            (temp_df["bin_mean"] - temp_df["sample_mean"]) ** 2
        ) / 7  # calc mse
        temp_df["graph_mse"] = temp_df["bin_mean"] - temp_df["sample_mean"]
        temp_df = temp_df.drop_duplicates().sort_values(
            by="bin", ascending=True
        )  # dropping dup cols
        temp_df["population_proportion"] = temp_df["bin_count"].divide(
            temp_df["bin_count"].sum()
        )  # calc pop prop
        temp_df["weighted_diff"] = temp_df["diff_mean_resp"].multiply(
            temp_df["population_proportion"]
        )  # wmse

        # plots for each predictor
        temp_df["bin"] = temp_df["bin"].astype("str")  # need to convert to str to plot

        binplot = make_subplots(specs=[[{"secondary_y": True}]])

        binplot.add_trace(
            go.Bar(x=temp_df["bin"], y=temp_df["bin_count"], name="Population"),
            secondary_y=False,
        )
        binplot.add_trace(
            go.Scatter(
                x=temp_df["bin"], y=temp_df["sample_mean"], name="Upop", mode="lines"
            ),
            secondary_y=True,
        )
        binplot.add_trace(
            go.Scatter(
                x=temp_df["bin"],
                y=temp_df["graph_mse"],
                name="Ui-Upop",
                mode="lines",
            ),
            secondary_y=True,
        )
        binplot.update_xaxes(title_text="Bins")
        binplot.update_yaxes(title_text="Population", secondary_y=False)
        binplot.update_yaxes(title_text="Response", secondary_y=True)

        binplot.write_html(f"graphs/{column}_binplot.html")

        # getting links
        joiner.append(f"{column}")
        names.append(f"{column}_binplot")
        urls.append(f"graphs/{column}_binplot.html")

        pred_name = train_cont[column].name
        l1.append(pred_name)
        l2.append(temp_df["diff_mean_resp"].mean())
        l3.append(temp_df["weighted_diff"].mean())

    # predictors and mse, wmse
    mse_df["predictor"] = l1
    mse_df["mse"] = l2
    mse_df["weighted_mse"] = l3

    # making graphs df to merge with actual
    graphs_df["joiner"] = joiner
    graphs_df["name"] = names
    graphs_df["url"] = urls
    graphs_df["name_url"] = graphs_df["name"] + "#" + graphs_df["url"]

    # joining brute force and mse dataframes
    # merging on pred 1
    brute_force = brute_force.merge(mse_df, left_on="Predictor 1", right_on="predictor")
    brute_force = brute_force.drop(["predictor"], axis=1)
    brute_force = brute_force.rename(
        columns={"mse": "mse_1", "weighted_mse": "weighted_mse_1"}
    )

    # merging on pred 2
    brute_force = brute_force.merge(mse_df, left_on="Predictor 2", right_on="predictor")
    brute_force = brute_force.drop(["predictor"], axis=1)
    brute_force = brute_force.rename(
        columns={"mse": "mse_2", "weighted_mse": "weighted_mse_2"}
    )

    # calculating actual mse, wmse
    brute_force["Difference of Mean Response"] = (
        brute_force["mse_1"] * brute_force["mse_2"]
    )
    brute_force["Weighted Difference of Mean Response"] = (
        brute_force["weighted_mse_1"] * brute_force["weighted_mse_2"]
    )

    brute_force = brute_force.drop(
        ["mse_1", "mse_2", "weighted_mse_1", "weighted_mse_2"], axis=1
    )

    # correlation of predictors
    brute_force["Pearson"] = pearsons_r
    brute_force["Absolute Pearson"] = abs_pearson

    # merging graphs_df on actual to link graphs
    brute_force = brute_force.merge(graphs_df, left_on="Predictor 1", right_on="joiner")
    brute_force = brute_force.drop(columns={"joiner", "name", "url"})
    brute_force = brute_force.rename(columns={"name_url": "Predictor 1 Bin Plot"})
    brute_force = brute_force.merge(graphs_df, left_on="Predictor 2", right_on="joiner")
    brute_force = brute_force.drop(columns={"joiner", "name", "url"})
    brute_force = brute_force.rename(columns={"name_url": "Predictor 2 Bin Plot"})
    print("brute force calcs done")
    print("starting 2d hist")
    # creating 2D histograms for each pair
    names = []
    urls = []
    if len(train_cont.axes[1]) >= 2:
        for column1 in train_cont:
            for column2 in train_cont:
                if (
                    brute_force["Predictor Pair"]
                    .str.contains(f"{column1}/{column2}")
                    .any()
                ):
                    temp_df1 = pd.DataFrame()  # initialize df to store calcs
                    temp_df1[column1] = train_cont[
                        column1
                    ]  # take col of interest for calcs
                    sample_mean = temp_df1[column1].mean()  # get mean of col
                    temp_df1["bin"] = pd.cut(
                        train_cont[column1], 7, right=True
                    )  # separate into 10 bins
                    temp_df1["sample_mean"] = sample_mean  # set new col to mean
                    temp_df1["bin_mean"] = temp_df1.groupby("bin")[column1].transform(
                        "mean"
                    )  # get mean of each bin
                    temp_df1["bin_count"] = temp_df1.groupby("bin")[column1].transform(
                        "count"
                    )  # get count of each bin
                    temp_df1 = temp_df1.drop(
                        columns=[column1]
                    )  # dropping base col to condense df
                    temp_df1["diff_mean_resp"] = (
                        (temp_df1["bin_mean"] - temp_df1["sample_mean"]) ** 2
                    ) / 7  # calc mse
                    temp_df1[column2] = train_cont[
                        column2
                    ]  # take col of interest for calcs
                    sample_mean = temp_df1[column2].mean()  # get mean of col
                    temp_df1["bin2"] = pd.cut(
                        train_cont[column2], 7, right=True
                    )  # separate into 10 bins
                    temp_df1["sample_mean2"] = sample_mean  # set new col to mean
                    temp_df1["bin_mean2"] = temp_df1.groupby("bin2")[column2].transform(
                        "mean"
                    )  # get mean of each bin
                    temp_df1["bin_count2"] = temp_df1.groupby("bin2")[
                        column2
                    ].transform(
                        "count"
                    )  # get count of each bin
                    temp_df1 = temp_df1.drop(
                        columns=[column2]
                    )  # dropping base col to condense df
                    temp_df1["diff_mean_resp2"] = (
                        (temp_df1["bin_mean2"] - temp_df1["sample_mean2"]) ** 2
                    ) / 7  # calc mse
                    brute_force_pairs = px.density_heatmap(
                        temp_df1,
                        x="bin_mean",
                        y="bin_mean2",
                        title=f"{column1}/{column2}_bruteforce_plot",
                        labels={"bin_mean": f"{column1}", "bin_mean2": f"{column2}"},
                    )
                    brute_force_pairs.write_html(
                        f"graphs/{column1}*{column2}_brute_force_plot.html"
                    )

                    # add links
                    names.append(f"{column1}*{column2}_brute_force_plot")
                    urls.append(f"graphs/{column1}*{column2}_brute_force_plot.html")

    links_df["name"] = names
    links_df["url"] = urls
    links_df["name_url"] = links_df["name"] + "#" + links_df["url"]
    brute_force["Brute Force Plot"] = links_df["name_url"]

    # dropping extra column
    brute_force = brute_force.drop(columns=["Predictor Pair"], axis=1)

    # add random forest variable importance
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X_train, y_train)

    print("creating feature importance")
    # base RF
    feat_bar = px.bar(
        x=rf.feature_importances_,
        y=X_train.columns,
        labels=dict(x="Feature Importance", y="Feature Name"),
        orientation="h",
        title="Random Forest Feature Importance",
    )
    feat_bar.update_yaxes(categoryorder="total ascending")

    feat_bar.write_html("graphs/random_forest_classifier.html")

    # add t-score and p-val

    # add 2d hist on predictors pairs

    ##################################################
    # MODELS
    ##################################################

    print("starting model pipeline")
    # making a model pipeline
    seed = 55
    model_pipeline = []
    model_pipeline.append(LogisticRegression(solver="liblinear", random_state=seed))
    model_pipeline.append(SVC(random_state=seed))
    model_pipeline.append(KNeighborsClassifier())
    model_pipeline.append(tree.DecisionTreeClassifier(random_state=seed))
    model_pipeline.append(RandomForestClassifier(random_state=seed))
    model_pipeline.append(GaussianNB())

    model_list = [
        "Logistic Regression",
        "SVM",
        "KNN",
        "Decision Tree",
        "Random Forest",
        "Naive Bayes",
    ]
    acc_list = []
    auc_list = []
    cm_list = []

    for model in model_pipeline:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc_list.append(metrics.accuracy_score(y_test, y_pred))
        fpr, tpr, _thresholds = metrics.roc_curve(y_test, y_pred)
        auc_list.append(round(metrics.auc(fpr, tpr), 2))
        cm_list.append(confusion_matrix(y_test, y_pred))

    # comparing models
    results_df = pd.DataFrame(
        {
            "Model": model_list,
            "Accuracy": acc_list,
            "AUC": auc_list,
            # add ROC
        }
    )

    ##################################################
    # MODELING WITHOUT HIGHLY CORR PAIRS .75 CUTOFF
    ##################################################
    print("test 2")
    # making a model pipeline
    X_train2 = X_train.drop(
        columns=[
            # "home_20_game_hit2plate_avg",
            "home_20_game_hrh_avg",
            "home_20_game_soh_avg",
            "away_20_game_batting_avg",
            # "home_20_game_safe2plate_avg",
            "temp",
        ]
    )
    X_test2 = X_test.drop(
        columns=[
            # "home_20_game_hit2plate_avg",
            "home_20_game_hrh_avg",
            "home_20_game_soh_avg",
            "away_20_game_batting_avg",
            # "home_20_game_safe2plate_avg",
            "temp",
        ]
    )

    # making a model pipeline
    model_pipeline = []
    model_pipeline.append(LogisticRegression(solver="liblinear", random_state=seed))
    model_pipeline.append(SVC(random_state=seed))
    model_pipeline.append(KNeighborsClassifier())
    model_pipeline.append(tree.DecisionTreeClassifier(random_state=seed))
    model_pipeline.append(RandomForestClassifier(random_state=seed))
    model_pipeline.append(GaussianNB())

    model_list = [
        "Logistic Regression",
        "SVM",
        "KNN",
        "Decision Tree",
        "Random Forest",
        "Naive Bayes",
    ]
    acc_list = []
    auc_list = []
    cm_list = []

    for model in model_pipeline:
        model.fit(X_train2, y_train)
        y_pred = model.predict(X_test2)
        acc_list.append(metrics.accuracy_score(y_test, y_pred))
        fpr, tpr, _thresholds = metrics.roc_curve(y_test, y_pred)
        auc_list.append(round(metrics.auc(fpr, tpr), 2))
        cm_list.append(confusion_matrix(y_test, y_pred))

    # comparing models
    results_df2 = pd.DataFrame(
        {
            "Model": model_list,
            "Accuracy": acc_list,
            "AUC": auc_list,
            # add ROC
        }
    )
    ##################################################
    # MODELING WITH STRONGEST PREDICTORS
    ##################################################
    print("test 3")
    # making a model pipeline
    X_train3 = X_train.drop(
        columns=[
            # "away_20_game_safe2plate_avg",
            # "away_20_game_hit2plate_avg",
            "away_20_game_h2b_avg",
            "away_20_game_obp_avg",
            "away_20_game_k_bb_avg",
            "away_20_game_hso_avg",
            "away_20_game_bb_k_avg",
            "away_20_game_hrh_avg",
            "away_20_game_soh_avg",
            "away_20_game_batting_avg",
            "temp",
        ]
    )
    X_test3 = X_test.drop(
        columns=[
            # "away_20_game_safe2plate_avg",
            # "away_20_game_hit2plate_avg",
            "away_20_game_h2b_avg",
            "away_20_game_obp_avg",
            "away_20_game_k_bb_avg",
            "away_20_game_hso_avg",
            "away_20_game_bb_k_avg",
            "away_20_game_hrh_avg",
            "away_20_game_soh_avg",
            "away_20_game_batting_avg",
            "temp",
        ]
    )

    # making a model pipeline
    model_pipeline = []
    model_pipeline.append(LogisticRegression(solver="liblinear", random_state=seed))
    model_pipeline.append(SVC(random_state=seed))
    model_pipeline.append(KNeighborsClassifier())
    model_pipeline.append(tree.DecisionTreeClassifier(random_state=seed))
    model_pipeline.append(RandomForestClassifier(random_state=seed))
    model_pipeline.append(GaussianNB())

    model_list = [
        "Logistic Regression",
        "SVM",
        "KNN",
        "Decision Tree",
        "Random Forest",
        "Naive Bayes",
    ]
    acc_list = []
    auc_list = []
    cm_list = []

    for model in model_pipeline:
        model.fit(X_train3, y_train)
        y_pred = model.predict(X_test3)
        acc_list.append(metrics.accuracy_score(y_test, y_pred))
        fpr, tpr, _thresholds = metrics.roc_curve(y_test, y_pred)
        auc_list.append(round(metrics.auc(fpr, tpr), 2))
        cm_list.append(confusion_matrix(y_test, y_pred))

    # comparing models
    results_df3 = pd.DataFrame(
        {
            "Model": model_list,
            "Accuracy": acc_list,
            "AUC": auc_list,
            # add ROC
        }
    )
    print("starting html table")
    ##################################################
    # TABLE TO HTML
    ##################################################
    # formatting for html out
    html_string = """
    <html>
      <head><title>Final Output</title></head>
      <link rel="stylesheet" type="text/css" href="my_style.css"/>
      <body>
        {table}
      </body>
    </html>.
    """
    # sorting values
    output_table = output_table.reset_index(drop=True)
    output_table = output_table.sort_values(by=["p-score"], ascending=False)
    # this is breaking plots to preds line-up
    brute_force = brute_force.reset_index(drop=True)
    brute_force = brute_force.sort_values(
        by=["Weighted Difference of Mean Response"], ascending=False
    )

    results_df = results_df.sort_values(by=["Accuracy"], ascending=False)
    results_df2 = results_df2.sort_values(by=["Accuracy"], ascending=False)
    results_df3 = results_df3.sort_values(by=["Accuracy"], ascending=False)

    # creating clickable links
    cont_predictors_table = (
        cont_predictors_table.style.set_properties(**{"text=align": "center"})
        .format({"Violin Plot": make_clickable, "Histogram": make_clickable})
        .hide_index()
    )
    output_table = (
        output_table.style.set_properties(**{"text-align": "center"})
        .format({"Linear Regression Plot": make_clickable})
        .hide_index()
    )
    brute_force = (
        brute_force.style.set_properties(**{"text-align": "center"})
        .format(
            {
                "Predictor 1 Bin Plot": make_clickable,
                "Predictor 2 Bin Plot": make_clickable,
                "Brute Force Plot": make_clickable,
            }
        )
        .hide_index()
    )

    # OUTPUT HTML FILE
    with open("test_env.html", "w") as f:
        f.write(
            html_string.format(
                table=cont_predictors_table.to_html(justify="center", classes="mystyle")
            )
            + "\n\n"
            + html_string.format(
                table=output_table.to_html(
                    justify="center", col_space=10, classes="mystyle"
                )
            )
            + "\n\n"
            + heat.to_html()
            + "\n\n"
            + html_string.format(
                table=brute_force.to_html(justify="center", classes="mystyle")
            )
            + "\n\n"
            + feat_bar.to_html()
            + "\n\n"
            + html_string.format(
                table=results_df.to_html(justify="center", classes="mystyle")
            )
            + "\n\n"
            + html_string.format(
                table=results_df2.to_html(justify="center", classes="mystyle")
            )
            + "\n\n"
            + html_string.format(
                table=results_df3.to_html(justify="center", classes="mystyle")
            )
        )
    print("finished writing HTML table")
    # os.system("start test_env.html")


if __name__ == "__main__":
    sys.exit(main())
