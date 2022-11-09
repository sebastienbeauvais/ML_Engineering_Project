import sys

import pandas
import sqlalchemy


def main():
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
    df = pandas.read_sql_query(query, sql_engine)
    print(df.head())


if __name__ == "__main__":
    sys.exit(main())
