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
RUN pip3 install --upgrade pip
RUN pip3 install --compile --no-cache-dir -r requirements.txt

# Copy files
COPY more_baseball_features.sql more_baseball_features.sql
COPY bashfile.sh bashfile.sh
COPY final.py final.py
COPY baseball_features.sql baseball_features.sql

# SQL - making baseball db
CMD ./bashfile.sh

