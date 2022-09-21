-- Use baseball database
USE baseball;

-- show all tables in baseball db
SHOW TABLES;

-- show all columns in a table
SHOW COLUMNS FROM table_name

-- using temp table to store some data
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

-- using temp table to get historical batting avg
-- getting error of division by 0..
-- create a case statement where atBat is 0 then 0 else calc
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


-- using temp table to get year batting avg
-- query to get all year a batter played
CREATE TEMPORARY TABLE temp_year
SELECT DISTINCT bc.batter, EXTRACT(YEAR from g.local_date) AS year,
	SUM(bc.Hit) OVER (PARTITION BY bc.batter, year) AS total_hits,
	SUM(bc.atBat) OVER(PARTITION BY bc.batter, year) AS total_atBats
FROM 
	game g
JOIN batter_counts bc
ON g.game_id = bc.game_id;

-- create actual table with batting avg calc
CREATE TABLE yearly_bat_avg
SELECT *,
CASE
	WHEN total_atBats = 0
	THEN 0
	ELSE (total_hits/total_atBats)
END AS yearly_batting_avg
FROM temp_year;






















