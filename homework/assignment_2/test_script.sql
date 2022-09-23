USE baseball;

DROP TABLE IF EXISTS temp_stats;
DROP TABLE IF EXISTS hist_bat_avg;
DROP TABLE IF EXISTS temp_year;
DROP TABLE IF EXISTS yearly_bat_avg;
DROP TABLE IF EXISTS temp_rolling;
DROP TABLE IF EXISTS temp_game_stats;
DROP TABLE IF EXISTS temp_game_stats_and_batting_avg;
DROP TABLE IF EXISTS overall_rolling_avg;

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

CREATE TEMPORARY TABLE temp_year
SELECT DISTINCT bc.batter, EXTRACT(YEAR FROM g.local_date) AS year,
	SUM(bc.Hit) OVER (PARTITION BY bc.batter, year) AS total_hits,
	SUM(bc.atBat) OVER(PARTITION BY bc.batter, year) AS total_atBats
FROM 
	game g
JOIN batter_counts bc
ON g.game_id = bc.game_id;

CREATE TABLE yearly_bat_avg
SELECT *,
CASE
	WHEN total_atBats = 0
	THEN 0
	ELSE (total_hits/total_atBats)
END AS yearly_batting_avg
FROM temp_year;

ALTER TABLE batter_counts
DROP INDEX batter_stats_idx;

ALTER TABLE game
DROP INDEX games_idx;

ALTER TABLE battersInGame
DROP INDEX batter_in_game_idx;

CREATE INDEX batter_stats_idx
ON batter_counts(batter, Hit, atBat, game_id);

CREATE INDEX games_idx
ON game(game_id);

CREATE INDEX batter_in_game_idx
ON battersInGame(batter, game_id);

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
	bc.batter, date;

CREATE TEMPORARY TABLE temp_game_stats
SELECT game_id, date,
	SUM(Hit) AS tot_hits, SUM(atBat) AS tot_bats
FROM temp_rolling
GROUP BY game_id;

CREATE TEMPORARY TABLE temp_game_stats_and_batting_avg
SELECT *, (tot_hits/tot_bats) AS game_bat_avg
FROM temp_game_stats;

CREATE TABLE overall_rolling_avg
SELECT *,
	AVG(game_bat_avg) OVER (ORDER BY date ROWS BETWEEN
	100 PRECEDING AND 1 PRECEDING) AS total_rolling_avg
FROM temp_game_stats_and_batting_avg;

SELECT *
FROM overall_rolling_avg
LIMIT 0,20;