import itertools
import sys

import numpy as np
import pandas as pd
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
        GROUP BY
            year, bc.team_id, g.game_id
        ORDER BY
            year, bc.team_id DESC
    """
    df = pd.read_sql_query(query, sql_engine)

    # dropping useless column(s), maybe drop team_id??
    df = df.drop(["game_id"], axis=1)

    # making year a category
    df = df.astype({"year": "category"})

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


if __name__ == "__main__":
    sys.exit(main())
