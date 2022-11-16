import itertools
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import sqlalchemy

# from sklearn.model_selection import train_test_split_


def main():
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
            EXTRACT(YEAR FROM g.local_date) AS year,
            bc.team_id,
            t.league,
            t.division,
            bs.overcast,
            g.game_id,
            SUM(bc.Hit)/SUM(bc.atBat) AS batting_avg,
            SUM(bc.Strikeout)/SUM(bc.Hit) AS batting_soh,
            SUM(bc.Home_Run)/SUM(bc.Hit) AS batting_hrh,
            SUM(bc.atBat/bc.Home_Run) AS ab_hr,
            SUM(bc.Walk)/SUM(bc.Strikeout) AS bb_k,
            CASE
                WHEN SUM(pc.pitchesThrown) = 0 THEN 0
                ELSE SUM(pc.Hit)/SUM(pc.pitchesThrown)
            END AS htt,
            CASE
                WHEN SUM(pc.pitchesThrown) = 0 THEN 0
                ELSE SUM(pc.Strikeout)/SUM(pc.pitchesThrown)
            END AS stt,
            CASE
                WHEN SUM(pc.Strikeout) = 0 THEN 0
                ELSE SUM(pc.Hit)/SUM(pc.Strikeout)
            END AS hso,
            CASE
                WHEN SUM(pc.endingInning-pc.startingInning) = 0 THEN 0
                ELSE SUM(pc.Walk)+SUM(pc.Hit)/SUM(pc.endingInning-pc.startingInning)
            END AS whip,
            CASE
                WHEN SUM(pc.atBat+pc.Walk+pc.Hit_By_Pitch+pc.Sac_Fly) = 0 THEN 0
                ELSE SUM(pc.Hit+pc.Walk+pc.Hit_By_Pitch)/SUM(pc.atBat+pc.Walk+pc.Hit_By_Pitch+pc.Sac_Fly)
            END AS obp,
            SUM(pc.Strikeout)/SUM(pc.Walk) as k_bb,
            CASE
                WHEN tr.home_away = 'H' AND tr.win_lose = 'W' THEN 1
                ELSE 0
            END AS HomeTeamWins
        FROM
            batter_counts bc
        JOIN game g
        ON g.game_id = bc.game_id
        JOIN pitcher_counts pc
        ON pc.game_id = g.game_id
        JOIN team_results tr
        ON tr.game_id = g.game_id
        JOIN team t
        ON t.team_id = tr.team_id
        JOIN boxscore bs
        ON bs.game_id = g.game_id
        GROUP BY
            year, bc.team_id, g.game_id
        ORDER BY
            year, bc.team_id DESC
    """
    df = pd.read_sql_query(query, sql_engine)

    # dropping useless column(s), maybe drop team_id??
    df = df.drop(["game_id"], axis=1)

    # making year a category
    df = df.astype(
        {
            "year": "category",
            "team_id": "category",
            "league": "category",
            "division": "category",
            "overcast": "category",
        }
    )

    # TRAIN TEST SPLIT
    train_df = df[df["year"] != 2011]

    # test will be last year available
    # test_df = df[df['year'] == 2011]

    X_train = train_df.loc[:, train_df.columns != "HomeTeamWins"]
    # y_train = train_df.loc[:, train_df.columns == 'HomeTeamWins']

    # X_test = test_df.loc[:, test_df.columns != 'HomeTeamWins']
    # y_test = test_df.loc[:, test_df.columns == 'HomeTeamWins']
    # print(df.head())

    train_cont = X_train.select_dtypes(exclude="category")
    # train_cat = X_train.select_dtypes(include="category")

    # make table and graphs for train_cont and train_cat

    #################################################
    # CONT/CONT TABLE
    #################################################

    # defining table
    cont_cont_output_table = pd.DataFrame(
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
        width=900,
        height=600,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        yaxis_autorange="reversed",
    )

    cont_cont_heat = go.Figure(data=[heat], layout=layout)

    # display heatmap
    cont_cont_heat.show()

    # writing to html

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
    cont_cont_output_table["Predictors"] = col_combos
    cont_cont_output_table["Pearson's R"] = pearsons_r
    cont_cont_output_table["Absolute Value of Pearson"] = abs_pearson

    # Linear regression for each cont/cont predictors
    lm_l = []
    """names = []
    urls = []
    links_df = pd.DataFrame(
        columns=[
            "name",
            "url",
        ]
    )"""
    if len(train_cont.axes[1]) >= 2:
        for column_x in train_cont:
            for column_y in train_cont:
                if (
                    cont_cont_output_table["Predictors"]
                    .str.contains(f"{column_x}/{column_y}")
                    .any()
                ):
                    lm_l.append("linear model")
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

    cont_cont_output_table["Linear Regression Plot"] = lm_l

    #################################################
    # CONT/CONT BRUTE FORCE TABLE
    #################################################
    cont_cont_brute_force = pd.DataFrame(
        columns=[
            "Predictor 1",
            "Predictor 2",
        ]
    )
    # getting list of predictor pairs
    cont_cont_brute_force[["Predictor 1", "Predictor 2"]] = cont_cont_output_table[
        "Predictors"
    ].str.split("/", expand=True)

    # mse df
    mse_df = pd.DataFrame()
    l1 = []  # need lists to store data for later
    l2 = []
    l3 = []

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
        fig = px.bar(temp_df, x="bin", y="bin_count", title=column)
        fig.show()

        # making an MSE df to use in brute force

        pred_name = train_cont[column].name
        l1.append(pred_name)
        l2.append(temp_df["diff_mean_resp"].mean())
        l3.append(temp_df["weighted_diff"].mean())

    # predictors and mse, wmse
    mse_df["predictor"] = l1
    mse_df["mse"] = l2
    mse_df["weighted_mse"] = l3

    # joining brute force and mse dataframes
    # merging on pred 1
    cont_cont_brute_force = cont_cont_brute_force.merge(
        mse_df, left_on="Predictor 1", right_on="predictor"
    )
    cont_cont_brute_force = cont_cont_brute_force.drop(["predictor"], axis=1)
    cont_cont_brute_force = cont_cont_brute_force.rename(
        columns={"mse": "mse_1", "weighted_mse": "weighted_mse_1"}
    )

    # mergin on pred 2
    cont_cont_brute_force = cont_cont_brute_force.merge(
        mse_df, left_on="Predictor 2", right_on="predictor"
    )
    cont_cont_brute_force = cont_cont_brute_force.drop(["predictor"], axis=1)
    cont_cont_brute_force = cont_cont_brute_force.rename(
        columns={"mse": "mse_2", "weighted_mse": "weighted_mse_2"}
    )

    # calculating actual mse, wmse
    cont_cont_brute_force["mse"] = (
        cont_cont_brute_force["mse_1"] * cont_cont_brute_force["mse_2"]
    )
    cont_cont_brute_force["weighted_mse"] = (
        cont_cont_brute_force["weighted_mse_1"]
        * cont_cont_brute_force["weighted_mse_2"]
    )
    cont_cont_brute_force = cont_cont_brute_force.drop(
        ["mse_1", "mse_2", "weighted_mse_1", "weighted_mse_2"], axis=1
    )

    # correlation of predictors
    cont_cont_brute_force["pearson"] = pearsons_r
    cont_cont_brute_force["abs_pearson"] = abs_pearson


if __name__ == "__main__":
    sys.exit(main())
