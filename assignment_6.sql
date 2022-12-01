USE baseball;

CREATE OR REPLACE TEMPORARY TABLE temp_rolling
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

CREATE OR REPLACE TABLE overall_rolling_avg
SELECT *,
	AVG(overall_batting_avg) OVER (ORDER BY game_id, ora_date ROWS BETWEEN
	100 PRECEDING AND 1 PRECEDING) AS rolling_avg
FROM temp_rolling
;
