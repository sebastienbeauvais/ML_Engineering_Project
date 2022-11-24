import itertools
import os
import sys
import warnings

import graphviz
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import sqlalchemy
from sklearn import metrics, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# make links clickable
def make_clickable(val):
    name, url = val.split("#")
    return f'<a target="_blank" href="{url}">{name}</a>'


def main():
    warnings.simplefilter(action="ignore", category=FutureWarning)

    # CONNECTING TO DB
    db_user = "root"
    # pragma: allowlist nextline secret
    db_pass = "password"
    db_host = "localhost"
    db_database = "baseball"
    # pragma: allowlist nextline secret
    connect_string = (
        f"mariadb+mariadbconnector://{db_user}:{db_pass}@{db_host}/{db_database}"
    )

    sql_engine = sqlalchemy.create_engine(connect_string)

    query = """
        SELECT
            *
        FROM
            baseball_feats;
    """
    df = pd.read_sql_query(query, sql_engine)

    # creating more features (if negative, away team is higher)
    df["better_batting"] = df["home_batting_avg"] - df["away_batting_avg"]
    df["better_soh"] = df["home_soh"] - df["away_soh"]
    df["better_hrh"] = df["home_hrh"] - df["away_hrh"]
    df["better_bb_k"] = df["home_bb_k"] - df["away_bb_k"]
    df["better_hso"] = df["home_hso"] - df["away_hso"]
    df["better_k_bb"] = df["home_k_bb"] - df["away_k_bb"]
    df["better_obp"] = df["home_obp"] - df["away_obp"]

    # dropping useless column(s)
    df = df.drop(["game_id", "home_team", "away_team"], axis=1)

    # making year a category
    df = df.astype(
        {
            "year": "category",
        }
    )
    # filling nan
    df["HomeTeamWins"] = df["HomeTeamWins"].fillna(0)

    # TRAIN TEST SPLIT
    train_df = df[df["year"] != 2012]

    # test will be last year available
    test_df = df[df["year"] == 2012]

    X_train = train_df.loc[:, train_df.columns != "HomeTeamWins"]
    y_train = train_df.loc[:, train_df.columns == "HomeTeamWins"]
    y_train = y_train.values.ravel()

    X_test = test_df.loc[:, test_df.columns != "HomeTeamWins"]
    y_test = test_df.loc[:, test_df.columns == "HomeTeamWins"]

    train_cont = X_train.select_dtypes(exclude="category")
    # train_cat = X_train.select_dtypes(include="category")

    # creating directory for graphs
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
    # OUTPUT TABLE
    #################################################

    # defining table
    output_table = pd.DataFrame(
        columns=[
            "Predictors",
            "Pearson's R",
        ]
    )

    # correlation between variables
    pearson_corr = train_cont.corr()

    # defining variable to hold combos
    cols = []

    # getting list of all combinations
    for L in range(len(pearson_corr) + 1):
        for subset in itertools.combinations(pearson_corr, L):
            cols = list(subset)

    # concating combinations to place in table
    col_combos = ["/".join(map(str, comb)) for comb in itertools.combinations(cols, 2)]

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

    links_df["name"] = names
    links_df["url"] = urls
    links_df["name_url"] = links_df["name"] + "#" + links_df["url"]

    output_table["Linear Regression Plot"] = links_df["name_url"]

    #################################################
    # CONT/CONT BRUTE FORCE TABLE
    #################################################
    brute_force = pd.DataFrame(
        columns=[
            "Predictor 1",
            "Predictor 2",
        ]
    )

    # getting list of predictor pairs
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

    # getting sample mean, pop_mean for each predictor
    for column in train_cont:
        temp_df = pd.DataFrame()  # initialize df to store calcs
        temp_df[column] = train_cont[column]  # take col of interest for calcs
        sample_mean = temp_df[column].mean()  # get mean of col
        temp_df["bin"] = pd.cut(
            train_cont[column], 10, right=True
        )  # separate into 10 bins
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
        ) / 10  # calc mse
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
        binplot = px.bar(temp_df, x="bin", y="bin_count", title=column)
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

    # add random forest variable importance

    # add t-score and p-val

    # add 2d hist on predictors pairs

    ##################################################
    # MODELS
    ##################################################

    # Decision tree
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    tree.plot_tree(clf)

    dot_data = tree.export_graphviz(
        clf,
        out_file=None,
        feature_names=X_train.columns,
        filled=True,
        rounded=True,
        special_characters=True,
    )
    graph = graphviz.Source(dot_data)
    graph.render("decision_tree")

    # making a model pipeline
    model_pipeline = []
    model_pipeline.append(LogisticRegression(solver="liblinear"))
    model_pipeline.append(SVC())
    model_pipeline.append(KNeighborsClassifier())
    model_pipeline.append(tree.DecisionTreeClassifier())
    model_pipeline.append(RandomForestClassifier())
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
        }
    )

    """
    Best model is Random Forest based on accuracy of prediction. Most likely over-fit due to the accuracy being 1.
    Second best is Decision Tree with .99 accuracy followed by KNN at .98. Both probably over-fit...
    Going off non over-fit models logistic regression actually performs the best at .78
    Curious as to why RF and DT could be so overfit? I did not think that I had included predictors that 'cheated'
    """

    ##################################################
    # TABLE TO HTML
    ##################################################
    # formatting for html out
    html_string = """
    <html>
      <head><title>Assignment 5 Output</title></head>
      <link rel="stylesheet" type="text/css" href="my_style.css"/>
      <body>
        {table}
      </body>
    </html>.
    """
    # sorting values
    brute_force = brute_force.sort_values(
        by=["Weighted Difference of Mean Response"], ascending=False
    )
    results_df = results_df.sort_values(by=["Accuracy"], ascending=False)

    # creating clickable links
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
            }
        )
        .hide_index()
    )

    # OUTPUT HTML FILE
    with open("assignment_5.html", "w") as f:
        f.write(
            html_string.format(
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
            + html_string.format(
                table=results_df.to_html(justify="center", classes="mystyle")
            )
        )


if __name__ == "__main__":
    sys.exit(main())
