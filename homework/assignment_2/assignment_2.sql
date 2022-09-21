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
SELECT DISTINCT bc.batter, EXTRACT(YEAR from g.local_date) AS year,
	SUM(bc.Hit) OVER (PARTITION BY bc.batter, year) AS total_hits,
	SUM(bc.atBat) OVER(PARTITION BY bc.batter, year) AS total_atBats
FROM 
	game g
JOIN batter_counts bc
ON g.game_id = bc.game_id;

-- dropping table if exists
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


-- getting rolling average of last 5 days for a specific player




















