def main():
    from pyspark.sql import SparkSession

    appName = "PySpark Connection"
    master = "local"

    # create spark session
    spark = SparkSession.builder.appName(appName).master(master).getOrCreate()

    sql = "select * from batter_counts"
    database = "baseball"
    user = "root"
    password = ""  # your password
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
        .option("password", password)
        .option("driver", jdbc_driver)
        .load()
    )

    df.show()


if __name__ == "__main__":
    main()
