#####################################################################
# Select database
#####################################################################
USE baseball;
SHOW tables FROM baseball;
show columns from pitcher_stat;
show columns from pitcher_counts;
show columns from pitchersInGame;

#####################################################################
# Dropping all self made tables
#####################################################################
DROP TABLE IF EXISTS temp_stats;
DROP TABLE IF EXISTS hist_bat_avg;
DROP TABLE IF EXISTS temp_year;
DROP TABLE IF EXISTS yearly_bat_avg;
DROP TABLE IF EXISTS temp_rolling;
DROP TABLE IF EXISTS overall_rolling_avg;

#####################################################################
# creating indexes
#####################################################################
CREATE INDEX batter_stats_idx
ON batter_counts(batter, Hit, atBat, game_id);

CREATE INDEX games_idx
ON game(game_id);

#####################################################################
# Historical Batting Average
#####################################################################
CREATE TEMPORARY TABLE temp_stats
SELECT
	bc.batter, SUM(bc.Hit) AS total_hits,
	SUM(bc.atBat) AS total_atBat
FROM
	batter_counts bc
JOIN
	game g
ON
	g.game_id = bc.game_id
GROUP BY
	bc.batter;

CREATE TABLE hist_bat_avg (batter INT, total_hits INT, hist_batting_avg FLOAT(4,3)) ENGINE=MyISAM
SELECT *,
CASE
	WHEN ts.total_atBat = 0
	THEN 0
	ELSE (ts.total_hits/ts.total_atBat)
END AS hist_batting_avg
FROM
	temp_stats ts
GROUP BY
	ts.batter;

#####################################################################
# Yearly Batting Average
#####################################################################
CREATE TEMPORARY TABLE temp_year
SELECT DISTINCT bc.batter, EXTRACT(YEAR FROM g.local_date) AS year,
	SUM(bc.Hit) OVER (PARTITION BY bc.batter, year) AS total_hits,
	SUM(bc.atBat) OVER(PARTITION BY bc.batter, year) AS total_atBats
FROM
	game g
JOIN batter_counts bc
ON g.game_id = bc.game_id;

CREATE TABLE yearly_bat_avg (year YEAR, total_hits INT, total_atBats INT, yearly_batting_avg FLOAT(4,3)) ENGINE=MyISAM
SELECT *,
CASE
	WHEN total_atBats = 0
	THEN 0
	ELSE (total_hits/total_atBats)
END AS yearly_batting_avg
FROM temp_year;

#####################################################################
# Rolling average of game and last 100 days
#####################################################################
CREATE TEMPORARY TABLE temp_rolling
SELECT g.game_id,
	SUM(bc.Hit) AS total_hits, SUM(bc.atBat) AS total_atBats,
	(SUM(bc.Hit)/SUM(bc.atBat)) AS overall_batting_avg,
	DATE(g.local_date) AS ora_date
FROM
	batter_counts bc
JOIN
	game g
ON
	g.game_id = bc.game_id
GROUP BY
	g.game_id
ORDER BY
	g.game_id DESC
;

CREATE TABLE overall_rolling_avg
SELECT *,
	AVG(overall_batting_avg) OVER (ORDER BY game_id, ora_date ROWS BETWEEN
	100 PRECEDING AND 1 PRECEDING) AS rolling_avg
FROM temp_rolling
;

#####################################################################
# 1: K-BB% - strikeout per walks percentage
#####################################################################
SELECT
    player_id,
    team_id,
    career_so,
    career_bb,
    (career_so/career_bb) AS K_BB_PER
FROM
    pitcher_stat
WHERE
    career_so > 0
GROUP BY
    player_id, team_id
ORDER BY
    K_BB_PER ASC;

#####################################################################
# 2: BB/K% - Walk to strikeout ratio
#####################################################################
SELECT
    player_id,
    team_id,
    career_so,
    career_bb,
    (career_bb/career_so) AS BB_K_RATIO
FROM
    pitcher_stat
WHERE
    career_so > 0
GROUP BY
    player_id, team_id
ORDER BY
    BB_K_RATIO ASC;

#####################################################################
# 3: WHIP - walks+hits/innings pitched
#####################################################################
CREATE TEMPORARY TABLE temp_pitcher_whip
SELECT
    pitcher,
    team_id,
    game_id,
    Hit,
    Walk,
    (endingInning-startingInning) AS innings_played
FROM
    pitcher_counts
ORDER BY
    innings_played DESC, pitcher;

SELECT
    pitcher,
    SUM(Hit) AS total_hit,
    SUM(Walk) AS total_walk,
    SUM(innings_played) AS total_inning,
    (SUM(Walk)+SUM(Hit)/SUM(innings_played)) AS whip
FROM
    temp_pitcher_whip
GROUP BY
    pitcher
LIMIT
    0,20;

-- same query without using a temp table
SELECT
    pitcher,
    SUM(Hit) AS total_hit,
    SUM(Walk) AS total_walk,
    SUM(endingInning-startingInning) AS total_inning,
    (SUM(Walk)+SUM(Hit)/SUM(endingInning-startingInning)) AS whip
FROM
    pitcher_counts
GROUP BY
    pitcher
LIMIT
    0,20;

#####################################################################
# 4: Hits to strikeout ratio
#####################################################################
CREATE TEMPORARY TABLE temp_pitcher_hso
SELECT
    pitcher,
    team_id,
    game_id,
    Hit,
    Strikeout,
    pitchesThrown
FROM
    pitcher_counts
ORDER BY
    pitcher;

SELECT
    pitcher,
    SUM(Hit) AS total_hit,
    SUM(Strikeout) AS total_so,
    SUM(pitchesThrown) AS total_thrown,
    (SUM(Hit)/SUM(Strikeout)) AS hso
FROM
    temp_pitcher_hso
GROUP BY
    pitcher
LIMIT
    0,20;

#####################################################################
# 5,6: strikeouts and hits to total throws
#####################################################################
SELECT
    pitcher,
    SUM(Hit) AS total_hit,
    SUM(Strikeout) AS total_so,
    SUM(pitchesThrown) AS total_thrown,
    (SUM(Hit)/SUM(pitchesThrown)) AS htt,
    (SUM(Strikeout)/SUM(pitchesThrown)) AS stt
FROM
    temp_pitcher_hso
GROUP BY
    pitcher
LIMIT
    0,20;

#####################################################################
# 7: top 100 pitchers by WHIP based on over 100 innings played
#####################################################################
CREATE TEMPORARY TABLE temp_innings
SELECT
    pitcher,
    SUM(Hit) AS total_hit,
    SUM(Walk) AS total_walk,
    SUM(innings_played) AS total_inning,
    CASE
	    WHEN innings_played = 0
	THEN
	    0
	ELSE
	    (SUM(Walk)+SUM(Hit)/SUM(innings_played))
END AS whip
FROM
    temp_pitcher_whip
GROUP BY
    pitcher;

SELECT
    *
FROM
    temp_innings
WHERE
    total_inning >= 100
ORDER BY
    whip DESC
LIMIT
    0,100;

#####################################################################
# Test queries for each table
#####################################################################
SELECT *
FROM hist_bat_avg
LIMIT 0,20;

SELECT *
FROM yearly_bat_avg
LIMIT 0,20;

SELECT *
FROM overall_rolling_avg
LIMIT 0,20;

#####################################################################
# Dropping Indexes
#####################################################################
ALTER TABLE batter_counts
DROP INDEX batter_stats_idx;

ALTER TABLE game
DROP INDEX games_idx;



