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
#RUN pip install mariadb
RUN pip3 install --compile --no-cache-dir -r requirements.txt

# Copy files
COPY assignment_6.sql assignment_6.sql
COPY 12560_rolling.sql 12560_rolling.sql
COPY 100_day_rolling.sql 100_day_rolling.sql
COPY bashfile.sh bashfile.sh

# SQL - making baseball db
CMD ./bashfile.sh


