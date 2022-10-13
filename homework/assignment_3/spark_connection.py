# libraries
import sys


def main():
    # libraries
    import shutil

    from pyspark import StorageLevel
    from pyspark.sql import SparkSession

    appName = "PySpark Connection"
    master = "local"

    # create spark session
    spark = SparkSession.builder.appName(appName).master(master).getOrCreate()

    # SQL query for 100 day rolling avg
    base_sql = """
        SELECT
            bc.game_id,
            SUM(bc.Hit) AS game_total_hits,
            SUM(bc.atBat) AS game_total_atBats,
            SUM(bc.Hit)/SUM(bc.atBat) AS game_batting_avg,
            DATE(g.local_date) AS game_date,
            AVG(SUM(bc.Hit)/SUM(bc.atBat)) OVER (ORDER BY g.game_id, DATE(g.local_date)
            ROWS BETWEEN 100 PRECEDING AND 1 PRECEDING) AS rolling_avg
        FROM
            batter_counts bc
        JOIN
            game g
        ON
            g.game_id = bc.game_id
        GROUP BY
            g.game_id"""
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
        .option("query", base_sql)
        .option("user", user)
        # pragma: allowlist nextline secret
        .option("password", password)
        .option("driver", jdbc_driver)
        .load()
    )
    # writing query as CSV to use in transformer
    df.write.options(header="true").csv("rolling_baseball.csv")

    baseball_df = spark.read.csv(
        "rolling_baseball.csv", inferSchema="true", header="true"
    )
    baseball_df.createOrReplaceTempView("rolling_baseball")
    baseball_df.persist(StorageLevel.MEMORY_ONLY)

    # Create a column that has all the words we want to encode for modeling
    baseball_df = spark.sql(
        """
        SELECT *,
            SPLIT(CONCAT(
                CASE WHEN game_id IS NULL THEN ""
                ELSE game_id END,
                " ",
                CASE WHEN game_total_hits IS NULL THEN ""
                ELSE game_total_hits END,
                " ",
                CASE WHEN game_total_atBats IS NULL THEN ""
                ELSE game_total_atBats END,
                " ",
                CASE WHEN game_batting_avg IS NULL THEN ""
                ELSE game_batting_avg END,
                " ",
                CASE WHEN game_date IS NULL THEN ""
                ELSE game_date END,
                " ",
                CASE WHEN rolling_avg IS NULL THEN ""
                ELSE rolling_avg END
            ), " ") AS categorical,
            AVG(game_total_hits/game_total_atBats) OVER (ORDER BY game_id, DATE(game_date)
            ROWS BETWEEN 100 PRECEDING AND 1 PRECEDING) AS avg_transform
        FROM rolling_baseball
        """
    )
    # checking fist 15 entries
    df.show(15)

    baseball_df.show(15)

    # checking metadata for dataframe after transformer
    for field in df.schema.fields:
        print(field.name + " , " + str(field.dataType))

    # removing csv created from dataframe to run w/o errors
    shutil.rmtree("rolling_baseball.csv")


if __name__ == "__main__":
    sys.exit(main())
