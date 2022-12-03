#!/usr/bin/env bash

sleep 10
# Check if db exists
echo "Checking if database exists..."
# shellcheck disable=SC2006
RESULT=`mysql -u root -ppassword -h mariadb -e "SHOW DATABASES" | grep "baseball"`
if [ "$RESULT" == "baseball" ]; then
    echo "Database exist"
else
    echo "Database does not exist"
    echo "Building database..."
    mysql -u root -ppassword -h mariadb -e "CREATE DATABASE baseball;"
    echo "Finished building database"
    echo "Dumping file..."
    mysql -u root -ppassword -h mariadb baseball < baseball.sql
    echo "Finished dumping"
fi

# making table
echo "Creating output assignment 6..."
mysql -u root -ppassword -h mariadb baseball < assignment_6.sql > ./output.txt
mysql -u root -ppassword -h mariadb baseball < 12560_rolling.sql > ./12560_output.txt
mysql -u root -ppassword -h mariadb baseball < 100_day_rolling.sql > ./100_day_rolling_output.txt
echo "Finished output"
