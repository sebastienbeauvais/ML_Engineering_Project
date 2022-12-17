USE baseball;

####################################################################
# HOME TEAM WINS
####################################################################
DROP TABLE IF EXISTS home_team_wins;
CREATE OR REPLACE TABLE home_team_wins
SELECT
    EXTRACT(YEAR FROM tr.local_date) AS game_year
    , tr.game_id
    , tr.team_id
    , bs.temp
    , CASE
        WHEN tr.home_away = 'H' AND tr.win_lose = 'W' THEN 1
        ELSE 0
    END AS HomeTeamWins
FROM
    team_results tr
JOIN boxscore bs
ON bs.game_id = tr.game_id
WHERE tr.home_away = 'H'
;

#####################################################################
# GETTING HOME TEAM BATTING STATS
#####################################################################
DROP TABLE IF EXISTS home_batting_stats;
CREATE OR REPLACE TABLE home_batting_stats
SELECT
    EXTRACT(YEAR FROM g.local_date) AS game_year
    , htw.game_id
    , htw.temp
    , g.home_team_id AS home_team
    , CASE
        WHEN tbc.atBat = 0 THEN 0
        ELSE tbc.Hit/tbc.atBat
    END AS home_batting_avg
    , CASE
        WHEN tbc.Hit = 0 THEN 0
        ELSE tbc.Strikeout/tbc.Hit
    END AS home_soh
    , CASE
        WHEN tbc.Hit = 0 THEN 0
        ELSE tbc.Home_Run/tbc.Hit
    END AS home_hrh
    , CASE
        WHEN tbc.Strikeout = 0 THEN 0
        ELSE tbc.Walk/tbc.Strikeout
    END AS home_bb_k
    , CASE
        WHEN tbc.Hit = 0 THEN 0
        ELSE tbc.toBase/tbc.Hit
    END AS home_h2b_ratio
    , CASE
        WHEN tbc.plateApperance = 0 THEN 0
        ELSE tbc.Hit/tbc.plateApperance
    END AS home_hit2plate_ratio
    , CASE
        WHEN tbc.plateApperance = 0 THEN 0
        ELSE (tbc.Walk + tbc.Home_Run + tbc.Hit_By_Pitch + tbc.Hit + tbc.toBase)/tbc.plateApperance
    END AS home_safe2plate_ratio
FROM
    game g
JOIN home_team_wins htw
ON htw.game_id = g.game_id
JOIN team_batting_counts tbc
ON tbc.game_id = g.game_id
GROUP BY
    htw.game_id
ORDER BY
    htw.game_id, g.home_team_id DESC;

#####################################################################
# GETTING HOME TEAM PITCHING STATS
#####################################################################
DROP TABLE IF EXISTS home_pitching_stats;
CREATE OR REPLACE TABLE home_pitching_stats
SELECT
    EXTRACT(YEAR FROM g.local_date) AS game_year
    , htw.game_id
    , g.home_team_id AS home_team
    , CASE
        WHEN tpc.Strikeout = 0 THEN 0
        ELSE tpc.Hit/tpc.Strikeout
    END AS home_hso
    , CASE
        WHEN tpc.Walk = 0 THEN 0
        ELSE tpc.Strikeout/tpc.Walk
    END AS home_k_bb
    , CASE
        WHEN (tpc.atBat+tpc.Walk+tpc.Hit_By_Pitch+tpc.Sac_Fly) = 0 THEN 0
        ELSE (tpc.Hit+tpc.Walk+tpc.Hit_By_Pitch)/(tpc.atBat+tpc.Walk+tpc.Hit_By_Pitch+tpc.Sac_Fly)
    END AS home_obp
    , CASE
        WHEN (tpc.atBat - tpc.Home_Run - tpc.Strikeout + tpc.Sac_Fly) = 0 THEN 0
        ELSE ((tpc.Hit - tpc.Home_Run)/(tpc.atBat - tpc.Home_Run - tpc.Strikeout + tpc.Sac_Fly))
    END AS home_BABIP
FROM
    game g
JOIN home_team_wins htw
ON htw.game_id = g.game_id
JOIN team_pitching_counts tpc
ON tpc.team_id = g.home_team_id
GROUP BY
    htw.game_id
ORDER BY
    htw.game_id, g.home_team_id DESC;

#####################################################################
# GETTING AWAY TEAM STATS
#####################################################################
DROP TABLE IF EXISTS away_batting_stats;
CREATE OR REPLACE TABLE away_batting_stats
SELECT
    EXTRACT(YEAR FROM g.local_date) AS game_year
    , htw.game_id
    , g.away_team_id AS away_team
    , CASE
        WHEN tbc.atBat = 0 THEN 0
        ELSE tbc.Hit/tbc.atBat
    END AS away_batting_avg
    , CASE
        WHEN tbc.Hit = 0 THEN 0
        ELSE tbc.Strikeout/tbc.Hit
    END AS away_soh
    , CASE
        WHEN tbc.Hit = 0 THEN 0
        ELSE tbc.Home_Run/tbc.Hit
    END AS away_hrh
    , CASE
        WHEN tbc.Strikeout = 0 THEN 0
        ELSE tbc.Walk/tbc.Strikeout
    END AS away_bb_k
    , CASE
        WHEN tbc.Hit = 0 THEN 0
        ELSE tbc.toBase/tbc.Hit
    END AS away_h2b_ratio
    , CASE
        WHEN tbc.plateApperance = 0 THEN 0
        ELSE tbc.Hit/tbc.plateApperance
    END AS away_hit2plate_ratio
    , CASE
        WHEN tbc.plateApperance = 0 THEN 0
        ELSE (tbc.Walk + tbc.Home_Run + tbc.Hit_By_Pitch + tbc.Hit + tbc.toBase)/tbc.plateApperance
    END AS away_safe2plate_ratio
FROM
    game g
JOIN home_team_wins htw
ON htw.game_id = g.game_id
JOIN team_batting_counts tbc
ON tbc.team_id = g.away_team_id
GROUP BY
    htw.game_id
ORDER BY
    htw.game_id, g.away_team_id DESC;

#####################################################################
# GETTING AWAY TEAM PITCHING STATS
#####################################################################
DROP TABLE IF EXISTS away_pitching_stats;
CREATE OR REPLACE TABLE away_pitching_stats
SELECT
    EXTRACT(YEAR FROM g.local_date) AS game_year
    , htw.game_id
    , g.away_team_id AS away_team
    , CASE
        WHEN tpc.Strikeout = 0 THEN 0
        ELSE tpc.Hit/tpc.Strikeout
    END AS away_hso
    , CASE
        WHEN tpc.Walk = 0 THEN 0
        ELSE tpc.Strikeout/tpc.Walk
    END AS away_k_bb
    , CASE
        WHEN (tpc.atBat+tpc.Walk+tpc.Hit_By_Pitch+tpc.Sac_Fly) = 0 THEN 0
        ELSE (tpc.Hit+tpc.Walk+tpc.Hit_By_Pitch)/(tpc.atBat+tpc.Walk+tpc.Hit_By_Pitch+tpc.Sac_Fly)
    END AS away_obp
    , CASE
        WHEN (tpc.atBat - tpc.Home_Run - tpc.Strikeout + tpc.Sac_Fly) = 0 THEN 0
        ELSE ((tpc.Hit - tpc.Home_Run)/(tpc.atBat - tpc.Home_Run - tpc.Strikeout + tpc.Sac_Fly))
    END AS away_BABIP
FROM
    game g
JOIN home_team_wins htw
ON htw.game_id = g.game_id
JOIN team_pitching_counts tpc
ON tpc.team_id = g.away_team_id
GROUP BY
    htw.game_id
ORDER BY
    htw.game_id, g.away_team_id DESC;

#####################################################################
# HOME TEAM STATS
#####################################################################
DROP TABLE IF EXISTS home_team_stats;
CREATE OR REPLACE TABLE home_team_stats
SELECT
    hbs.*
    , hps.home_hso
    , hps.home_k_bb
    , hps.home_obp
    , hps.home_BABIP
FROM
    home_batting_stats hbs
LEFT JOIN home_pitching_stats hps
ON hbs.game_id = hps.game_id;

#####################################################################
# AWAY TEAM STATS
#####################################################################
DROP TABLE IF EXISTS away_team_stats;
CREATE OR REPLACE TABLE away_team_stats
SELECT
    abs.*
    , aps.away_hso
    , aps.away_k_bb
    , aps.away_obp
    , aps.away_BABIP
FROM
    away_batting_stats abs
LEFT JOIN away_pitching_stats aps
ON abs.game_id = aps.game_id;

#####################################################################
# FINAL QUERY
#####################################################################
DROP TABLE IF EXISTS baseball_feats;
CREATE OR REPLACE TABLE baseball_feats
SELECT
    hts.*
    , ats.away_batting_avg
    , ats.away_soh
    , ats.away_hrh
    , ats.away_bb_k
    , ats.away_hso
    , ats.away_k_bb
    , ats.away_obp
	, AVG(hts.home_batting_avg) OVER (PARTITION BY hts.home_team ORDER BY htw.game_id RANGE BETWEEN
	20 PRECEDING AND 1 PRECEDING) AS home_20_game_batting_avg
	, AVG(hts.home_soh) OVER (PARTITION BY hts.home_team ORDER BY htw.game_id RANGE BETWEEN
	20 PRECEDING AND 1 PRECEDING) AS home_20_game_soh_avg
	, AVG(hts.home_hrh) OVER (PARTITION BY hts.home_team ORDER BY htw.game_id RANGE BETWEEN
	20 PRECEDING AND 1 PRECEDING) AS home_20_game_hrh_avg
	, AVG(hts.home_bb_k) OVER (PARTITION BY hts.home_team ORDER BY htw.game_id RANGE BETWEEN
	20 PRECEDING AND 1 PRECEDING) AS home_20_game_bb_k_avg
	, AVG(hts.home_hso) OVER (PARTITION BY hts.home_team ORDER BY htw.game_id RANGE BETWEEN
	20 PRECEDING AND 1 PRECEDING) AS home_20_game_hso_avg
	, AVG(hts.home_k_bb) OVER (PARTITION BY hts.home_team ORDER BY htw.game_id RANGE BETWEEN
	20 PRECEDING AND 1 PRECEDING) AS home_20_game_k_bb_avg
	, AVG(hts.home_obp) OVER (PARTITION BY hts.home_team ORDER BY htw.game_id RANGE BETWEEN
	20 PRECEDING AND 1 PRECEDING) AS home_20_game_obp_avg
    , AVG(hts.home_BABIP) OVER (PARTITION BY hts.home_team ORDER BY htw.game_id RANGE BETWEEN
	20 PRECEDING AND 1 PRECEDING) AS home_20_game_BABIP_avg
	, AVG(ats.away_batting_avg) OVER (PARTITION BY ats.away_team ORDER BY htw.game_id RANGE BETWEEN
	20 PRECEDING AND 1 PRECEDING) AS away_20_game_batting_avg
	, AVG(ats.away_soh) OVER (PARTITION BY ats.away_team ORDER BY htw.game_id RANGE BETWEEN
	20 PRECEDING AND 1 PRECEDING) AS away_20_game_soh_avg
	, AVG(ats.away_hrh) OVER (PARTITION BY ats.away_team ORDER BY htw.game_id RANGE BETWEEN
	20 PRECEDING AND 1 PRECEDING) AS away_20_game_hrh_avg
	, AVG(ats.away_bb_k) OVER (PARTITION BY ats.away_team ORDER BY htw.game_id RANGE BETWEEN
	20 PRECEDING AND 1 PRECEDING) AS away_20_game_bb_k_avg
	, AVG(ats.away_hso) OVER (PARTITION BY ats.away_team ORDER BY htw.game_id RANGE BETWEEN
	20 PRECEDING AND 1 PRECEDING) AS away_20_game_hso_avg
	, AVG(ats.away_k_bb) OVER (PARTITION BY ats.away_team ORDER BY htw.game_id RANGE BETWEEN
	20 PRECEDING AND 1 PRECEDING) AS away_20_game_k_bb_avg
	, AVG(ats.away_obp) OVER (PARTITION BY ats.away_team ORDER BY htw.game_id RANGE BETWEEN
	20 PRECEDING AND 1 PRECEDING) AS away_20_game_obp_avg
    , AVG(ats.away_BABIP) OVER (PARTITION BY ats.away_team ORDER BY htw.game_id RANGE BETWEEN
	20 PRECEDING AND 1 PRECEDING) AS away_20_game_BABIP_avg
	, AVG(hts.home_h2b_ratio) OVER (PARTITION BY hts.home_team ORDER BY htw.game_id RANGE BETWEEN
	20 PRECEDING AND 1 PRECEDING) AS home_20_game_h2b_avg
	, AVG(ats.away_h2b_ratio) OVER (PARTITION BY ats.away_team ORDER BY htw.game_id RANGE BETWEEN
	20 PRECEDING AND 1 PRECEDING) AS away_20_game_h2b_avg
    , htw.HomeTeamWins
FROM
    home_team_stats hts
LEFT JOIN away_team_stats ats
ON hts.game_id = ats.game_id
LEFT JOIN home_team_wins htw
ON htw.game_id = hts.game_id
GROUP BY hts.game_id;

DROP TABLE IF EXISTS more_baseball_feats;
CREATE OR REPLACE TABLE more_baseball_feats
SELECT
    game_year
    , temp
    , home_20_game_batting_avg
    , away_20_game_batting_avg
    , home_20_game_soh_avg
    , away_20_game_soh_avg
    , home_20_game_hrh_avg
    , away_20_game_hrh_avg
    , home_20_game_bb_k_avg
    , away_20_game_bb_k_avg
    , home_20_game_hso_avg
    , away_20_game_hso_avg
    , home_20_game_k_bb_avg
    , away_20_game_k_bb_avg
    , home_20_game_obp_avg
    , away_20_game_obp_avg
    , home_20_game_h2b_avg
    , away_20_game_h2b_avg
    , (home_20_game_batting_avg - away_20_game_batting_avg) AS diff_batting_avg
    , (home_20_game_soh_avg - away_20_game_soh_avg) AS diff_soh_avg
    , (home_20_game_hrh_avg - away_20_game_hrh_avg) AS diff_hrh_avg
    , (home_20_game_bb_k_avg - away_20_game_bb_k_avg) AS diff_bb_k_avg
    , (home_20_game_hso_avg - away_20_game_hso_avg) AS diff_hso_avg
    , (home_20_game_k_bb_avg - away_20_game_k_bb_avg) AS diff_k_bb_avg
    , (home_20_game_obp_avg - away_20_game_obp_avg) AS diff_obp_avg
    , (home_20_game_BABIP_avg - away_20_game_BABIP_avg) AS diff_BABIP_avg
    , (home_20_game_h2b_avg - away_20_game_h2b_avg) AS diff_h2b_avg
    , HomeTeamWins
FROM
    baseball_feats;
