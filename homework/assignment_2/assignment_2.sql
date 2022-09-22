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

DROP hist_bat_avg IF EXISTS;

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

DROP yearly_bat_avg IF EXISTS;

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

-- lets get batters and games
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
drop table temp_rolling;

select distinct batter from batter_counts limit 0,20;

-- using window function to get 10 day rolling
SELECT *,
	AVG(batting_avg) OVER (ORDER BY date ROWS BETWEEN 
	100 PRECEDING AND CURRENT ROW) AS rolling_average
FROM temp_rolling;



















