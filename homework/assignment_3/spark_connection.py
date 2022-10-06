# libraries
import sys

from pyspark.sql import SparkSession


def main():

    appName = "PySpark Connection"
    master = "local"

    # create spark session
    spark = SparkSession.builder.appName(appName).master(master).getOrCreate()

    sql = """SELECT bc.game_id,
        SUM(bc.Hit) AS total_hits,
        SUM(bc.atBat) AS total_atBats,
        SUM(bc.Hit)/SUM(bc.atBat) AS batting_avg,
        DATE(g.local_date) AS ora_date,
        AVG(SUM(bc.Hit)/SUM(bc.atBat)) OVER (ORDER BY g.game_id, DATE(g.local_date)
            ROWS BETWEEN 100 PRECEDING AND 1 PRECEDING) AS rolling_avg
        FROM batter_counts bc
        JOIN game g
        ON g.game_id = bc.game_id
        GROUP BY g.game_id"""
    database = "baseball"
    user = "root"
    # pragma: allowlist nextline secret
    password = "password"  # your password
    server = "localhost"
    port = 3306
    jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
    jdbc_driver = "org.mariadb.jdbc.Driver"

    # create a dataframe by reading data via JDBC
    df = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("query", sql)
        .option("user", user)
        # pragma: allowlist nextline secret
        .option("password", password)
        .option("driver", jdbc_driver)
        .load()
    )

    df.show(15)


if __name__ == "__main__":
    sys.exit(main())
