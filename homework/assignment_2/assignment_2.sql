# Use baseball database
USE baseball;

#####################################################################
# Useful shortcuts
# how all tables in baseball db
# SHOW TABLES;

# show all columns in a table
# SHOW COLUMNS FROM table_name
#####################################################################


#####################################################################
# Temp table 
# gets total hits, atbats, and year
#####################################################################

#####################################################################
# Dropping all tables and indecies created tables
# if they exists for clean run
#####################################################################
DROP TABLE IF EXISTS temp_stats;
DROP TABLE IF EXISTS hist_bat_avg;
DROP TABLE IF EXISTS temp_year;
DROP TABLE IF EXISTS yearly_bat_avg;
DROP TABLE IF EXISTS temp_rolling;
DROP TABLE IF EXISTS temp_game_stats;
DROP TABLE IF EXISTS temp_game_stats_and_batting_avg;

ALTER TABLE batter_counts
DROP INDEX batter_stats_idx;

ALTER TABLE game
DROP INDEX games_idx;

ALTER TABLE battersInGame
DROP INDEX batter_in_game_idx;

# create an index on batter for quicker results
CREATE INDEX batter_stats_idx
ON batter_counts(batter, Hit, atBat, game_id);

CREATE INDEX games_idx
ON game(game_id);

CREATE INDEX batter_in_game_idx
ON battersInGame(batter, game_id);


CREATE TEMPORARY TABLE temp_stats
SELECT 
	bc.batter, SUM(bc.Hit) AS total_hits, 
	SUM(bc.atBat) AS total_atBat,
	EXTRACT(YEAR from g.local_date) AS year
FROM 
	batter_counts bc
JOIN 
	game g
ON
	g.game_id = bc.game_id
GROUP BY
	bc.batter;

#####################################################################
# Drop the table if it exists
#####################################################################

--DROP TABLE IF EXISTS hist_bat_avg;

#####################################################################
# Query for historical batting avg
# Case statement to handle division by 0
#####################################################################

CREATE TABLE hist_bat_avg
SELECT ts.batter, ts.total_hits, ts.total_atBat,
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
# Create temp table for yearly batting avg
# gets some stats but also uses a window function to break down
# the query into years for each player
#####################################################################

CREATE TEMPORARY TABLE temp_year
SELECT DISTINCT bc.batter, EXTRACT(YEAR FROM g.local_date) AS year,
	SUM(bc.Hit) OVER (PARTITION BY bc.batter, year) AS total_hits,
	SUM(bc.atBat) OVER(PARTITION BY bc.batter, year) AS total_atBats
FROM 
	game g
JOIN batter_counts bc
ON g.game_id = bc.game_id;

#####################################################################
# Dropping table if exists
#####################################################################

--DROP TABLE IF EXISTS yearly_bat_avg;

#####################################################################
# Query to get yearly batting avg
# Case statement to handle division by 0 
#####################################################################

CREATE TABLE yearly_bat_avg
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

--DROP TABLE IF EXISTS temp_rolling;

-- w index on battercurrently takes 34mins to run
-- 38 mins w/o index..
-- 2 seconds with 3 level index..
CREATE TEMPORARY TABLE temp_rolling
SELECT DISTINCT bc.batter, g.game_id, 
	bc.Hit, bc.atBat, 
CASE
	WHEN bc.atBat = 0
	THEN 0
	ELSE (bc.Hit/bc.atBat)
END AS batting_avg,
	DATE(g.local_date) AS date
FROM 
	batter_counts bc
JOIN 
	game g
ON 
	g.game_id = bc.game_id
JOIN 
	battersInGame big
ON 
	big.batter = bc.batter
ORDER BY 
	bc.batter, date
;



--select distinct batter from batter_counts limit 0,20;

-- using window function to get 10 day rolling for each player
SELECT *,
	AVG(batting_avg) OVER (PARTITION BY batter ORDER BY date ROWS BETWEEN 
	10 PRECEDING AND CURRENT ROW) AS rolling_average
FROM temp_rolling
WHERE batter = 110029
LIMIT 0,20;

SELECT *,
	AVG(batting_avg) OVER (PARTITION BY batter ORDER BY date ROWS BETWEEN 
	10 PRECEDING AND 1 PRECEDING) AS rolling_average
FROM temp_rolling
WHERE batter = 110029
LIMIT 0,20;

-- rolling avg over 100 days per player excluding day 1
SELECT *,
	AVG(batting_avg) OVER (PARTITION BY batter ORDER BY date ROWS BETWEEN
	100 PRECEDING AND 1 PRECEDING) AS rolling_average
FROM temp_rolling;
--WHERE batter = 407886
--LIMIT 0,20;

#####################################################################
# test queries for rolling
#####################################################################
--DROP TABLE IF EXISTS temp_game_stats;
--DROP TABLE IF EXISTS temp_game_stats_and_batting_avg;

CREATE TEMPORARY TABLE temp_game_stats
SELECT game_id, date,
	SUM(Hit) AS tot_hits, SUM(atBat) AS tot_bats
FROM temp_rolling
GROUP BY game_id;

CREATE TEMPORARY TABLE temp_game_stats_and_batting_avg
SELECT *, (tot_hits/tot_bats) AS game_bat_avg
FROM temp_game_stats;

SELECT *,
	AVG(game_bat_avg) OVER (ORDER BY date ROWS BETWEEN
	100 PRECEDING AND 1 PRECEDING) AS total_rolling_avg
FROM temp_game_stats_and_batting_avg
LIMIT 0,20;	




















