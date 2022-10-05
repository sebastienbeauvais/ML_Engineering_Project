#####################################################################
# Select database
#####################################################################
USE baseball;

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



