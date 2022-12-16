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

# Creating features
echo "Creating sql features..."
mysql -u root -ppassword -h mariadb baseball < more_baseball_features.sql
echo "Features created"

echo "Creating CSV"
mysql -u root -ppassword -h mariadb baseball < baseball_features.sql > ./baseball_features.txt

echo "Building model..."
python3 final.py
echo "Finished building model"
