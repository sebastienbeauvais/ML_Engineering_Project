USE baseball;

CREATE TEMPORARY TABLE IF NOT EXISTS temp_rolling
SELECT g.game_id,
    bc.batter,
	SUM(bc.Hit) AS total_hits,
	SUM(bc.atBat) AS total_atBats,
	CASE
	    WHEN SUM(bc.atBat) = 0 THEN 0
	    ELSE (SUM(bc.Hit)/SUM(bc.atBat))
	END AS batting_avg,
	DATE(g.local_date) AS ora_date
FROM
	batter_counts bc
JOIN
	game g
ON
	g.game_id = bc.game_id
GROUP BY
	bc.batter, g.game_id
ORDER BY
	bc.batter, ora_date DESC
;

-- this takes
CREATE OR REPLACE TABLE _100_day_rolling_avg
SELECT *,
	AVG(batting_avg) OVER (PARTITION BY batter ORDER BY game_id RANGE BETWEEN
	100 PRECEDING AND CURRENT ROW) AS _100_day_rolling_avg
FROM temp_rolling
;

CREATE TEMPORARY TABLE IF NOT EXISTS temp_batter_avg
SELECT
    b.game_id AS game_id,
    DATE(local_date) AS game_date,
    batter,
    Hit,
    atBat
FROM
    batter_counts b
JOIN game g
ON b.game_id = g.game_id;

-- USING CODE FROM SLIDES LECTURE 12 (ASSIGNMENT 6, 2/4)
CREATE OR REPLACE TABLE batter_rolling_avg
SELECT
    game_id,
    game_date,
    batter,
    (SELECT SUM(Hit)/SUM(atBat)
        FROM temp_batter_avg brat3
        WHERE brat3.game_date > DATE_ADD(bra1.game_date, INTERVAL - 100 DAY) AND
            brat3.game_date < bra1.game_date AND bra1.batter = brat3.batter) AS last_100_days_rolling_avg
FROM temp_batter_avg bra1
WHERE game_id = 12560;

-- SELECT * FROM _100_day_rolling_avg INTO OUTFILE '100_day_rolling.txt';
-- SELECT * FROM batter_rolling_avg INTO OUTFILE '12560_output.txt';
