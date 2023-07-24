FROM nginx:latest
LABEL maintainer="xiaoming2020"
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y vim git make g++ net-tools telnet iputils-ping


