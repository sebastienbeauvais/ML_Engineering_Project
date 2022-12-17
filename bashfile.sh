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
mysql -u root -ppassword -h mariadb baseball < test_env.sql
echo "Features created"

echo "Building model..."
python3 test_env.py
echo "Finished building model"

echo "Opening HTML file"
cat ./test_env.html
echo "Done running script"
