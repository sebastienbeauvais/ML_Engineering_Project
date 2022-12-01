FROM --platform=linux/amd64 python:3.10.6-buster

ENV APP_HOME /app
WORKDIR $APP_HOME
ENV PYTHONPATH /

# Get necessary system packages
RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
     build-essential \
     python3 \
     python3-pip \
     python3-dev \
     mariadb-client \
  && rm -rf /var/lib/apt/lists/*

# Get necessary python libraries
COPY requirements.txt .
RUN wget https://dlm.mariadb.com/2678574/Connectors/c/connector-c-3.3.3/mariadb-connector-c-3.3.3-debian-bullseye-amd64.tar.gz -O - | tar -zxf - --strip-components=1 -C /usr
RUN pip install --upgrade pip
RUN pip install mariadb
RUN pip3 install --compile --no-cache-dir -r requirements.txt

# Copy file to load database
COPY baseball.sql baseball.sql

# SQL - making baseball db
RUN wget teaching.mrsharky.com/data/baseball.sql.tar.gz
RUN tar -xvzf baseball.sql.tar.gz
CMD mariadb -u root -ppassword -e "CREATE DATABASE IF NOT EXIST baseball"
CMD mysql -u root -ppassword baseball < baseball.sql

#CMD mariadb -u root -ppassword -h maridb -e "SHOW DATABASES;" > "my_file"

# docker exec <> mysql -u root -ppassword -h mariadb -e "SHOW DATABASES;"

# docker exec -i mariadb mysql -uroot -ppassword baseball < baseball.sql;

