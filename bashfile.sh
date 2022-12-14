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
echo "Checking if tables exist..."
#if [[ $(mysql -u root -ppassword -h mariadb -e\
#    "select count(*) from information_schema.tables where table_schema='baseball'\
#    and table_name='more_baseball_features';") ]]; then
#    echo "Features exist"
#else
#    echo "Features do not exist"
#    echo "Creating features. This may take a few moments..."
#    mysql -u root -ppassword -h mariadb baseball < more_baseball_features.sql
#    echo "Finished creating features"
#fi
echo "Creating sql features..."
mysql -u root -ppassword -h mariadb baseball < more_baseball_features.sql
echo "Features created"

echo "Running models..."
# python3 final.py
echo "Finished running models"
