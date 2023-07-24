FROM ubuntu
LABEL maintainer="xiaoming2020"

# 1. Make the entire build process non-interactive
ARG DEBIAN_FRONTEND=noninteractive

# 2.Build a separate layer of core tool images. 
#   Tools are arranged alphabetically for easy maintenance
RUN echo "**************Core Frame******************" \
    && apt-get update \
    && apt-get install -y \
    	build-essential \
    	cargo \ 
    	clang-format \ 
    	cmake \
    	cpplint \ 
    	curl \
    	gdb \
    	gnutls-bin \
    	git \
    	git-lfs \ 
    	g++ \
    	iputils-ping \
    	libboost-all-dev \
   	  libmysqlclient-dev \
    	libnghttp2-dev \
    	libssl-dev \
    	make \
    	net-tools \
    	npm \ 
    	nghttp2 \
	    python3 \
    	python3-pip \
    	pstack \
    	strace \
    	tcpdump \
    	telnet \
    	tree \
    	tzdata \
    	vim \
    	wget \
    	zsh

RUN echo "**************Next Frame******************" \
# [terminal]
# Add plugins/themes for zsh: download plugins/themes and modify plugin/themes in daily_zshrc.
# TODO: the follow linkage and associated operations should be adjusted.
    && chsh -s /bin/zsh \
    && sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" \
    && git clone https://github.com/zsh-users/zsh-syntax-highlighting ~/.zsh/zsh-syntax-highlighting \
    && git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions \
    && git clone --depth=1 https://gitee.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k \
    && git clone https://github.com/Runner-2019/resource.git \
    && rm -f ~/.zshrc && cp resource/config/daily_zshrc ~/.zshrc && rm -rf ./resource \
    && /bin/zsh -c "source ~/.zshrc" \
# [nvim].
    && curl https://docker-env-1305552539.cos.ap-shanghai.myqcloud.com/nvim.appimage.0.9.1 -o nvim.appimage \
    && chmod u+x nvim.appimage \
    && ./nvim.appimage --appimage-extract \
    && ln -s /squashfs-root/AppRun /usr/bin/nvim \
    && rm nvim.appimage \ 
    && nvim --version \
# [clangd]
    && apt-get install -y clangd-15 \
    && mv /usr/bin/clangd-15 /usr/bin/clangd \
    && clangd --version \
# [git]
    && git config --global http.sslVerify false \
    && git config --global http.postBuffer 1048576000 \
    && git config --global user.email "xiaomingZhang2020@outlook.com" \
    && git config --global user.name  "xiaomingZhang2020" \
# [clear]
    && apt-get autoclean

CMD ["/bin/zsh"]
