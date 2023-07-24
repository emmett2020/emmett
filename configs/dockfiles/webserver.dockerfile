FROM ubuntu
LABEL maintainer="xiaoming2020"
RUN apt-get update \ 
    && apt-get upgrade -y \
    && apt-get autoclean \
#    && rm -rf /var/lib/apt/lists/* \
    && apt-get install -y \
    mysql-client \
    mysql-server \
    libmysqlclient-dev \
    vim \
    git \
    make \
    g++ \
    net-tools \
    telnet \
    iputils-ping \
    && apt-get update \
    && apt-get upgrade -y \
    && apt-get autoclean \
  
  RUN git clone https://github.com/Runner-2019/WebServer

