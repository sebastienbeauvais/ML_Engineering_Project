#!/usr/bin/env bash

sleep 10
# create baseball database
echo "building database"
mysql -u root -ppassword -h mariadb -e "CREATE DATABASE IF NOT EXISTS baseball;"

# dump into baseball
echo "dumping file"
mysql -u root -ppassword -h mariadb baseball < baseball.sql

# making table
echo "creating output assignment 6"
mysql -u root -ppassword -h mariadb baseball < assignment_6.sql > output.txt
