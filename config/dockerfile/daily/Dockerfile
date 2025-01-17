FROM ubuntu
LABEL maintainer="emmett"

# 1. Make the entire build process non-interactive
ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_BRANCH=main

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
# [emmett git repo]
    && git clone https://github.com/emmett2020/emmett.git \
    && git checkout ${GIT_BRANCH}                         \
# [daily]
    && bash emmett/script/linux/ubuntu/install_daily.sh   \
# [clear]
    && rm -rf emmett                                      \
    && apt-get autoclean

CMD ["/bin/zsh"]
