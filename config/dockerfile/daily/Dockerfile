FROM ubuntu
LABEL maintainer="emmett"

ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8

# 1. Make the entire build process non-interactive
ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_BRANCH=master

# 2.Build a separate layer of basic tools.
#   Tools are arranged alphabetically for easy maintenance.
RUN echo "************* Core Frame *****************" \
    && apt-get update              \
    && apt-get install -y          \
               build-essential     \
               curl                \
               gdb                 \
               gnutls-bin          \
               git                 \
               git-lfs             \
               g++                 \
               htop                \
               iputils-ping        \
               libboost-all-dev    \
               libmysqlclient-dev  \
               libnghttp2-dev      \
               libssl-dev          \
               locales             \
               locales-all         \
               make                \
               net-tools           \
               nghttp2             \
               python3             \
               python3-pip         \
               sudo                \
               strace              \
               tcpdump             \
               telnet              \
               tree                \
               tzdata              \
               unzip               \
               vim                 \
               wget

RUN echo "************** Next Frame ******************"   \
# [git]
    && git config --global http.sslVerify  false          \
    && git config --global http.postBuffer 1048576000     \
    && git config --global https.postBuffer 1048576000    \
# [emmett git repo]
    && git clone -b ${GIT_BRANCH}                         \
           https://github.com/emmett2020/emmett.git       \
# [daily]
    && bash emmett/script/linux/ubuntu/install_daily.sh   \
# [other]
    sudo echo "127.0.0.1 emmett" >> /etc/hosts            \
# [clear]
    && rm -rf emmett                                      \
    && apt-get autoclean

CMD ["/bin/zsh"]
