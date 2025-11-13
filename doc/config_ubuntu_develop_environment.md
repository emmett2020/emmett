# Catalog
1. [Docker](#Docker)
2. [Conda](#Conda)
3. [Manaual](#Manaual)

Personally, I recommend the following three methods to set up a local `Ubuntu` development environment for general use.

# 1. Docker
1. This method assumes you already have Docker installed. If not, you can refer to [this link](https://docs.docker.com/engine/install/) for installation instructions.
2. You can download a pre-built image and run the container to get an out-of-the-box C++ development environment, which includes tools like GCC, CMake, and Neovim.
To pull the stable image:

```bash
docker pull emmettzhang/daily:latest
```

Alternatively, you can find newer [nightly build images](https://hub.docker.com/repository/docker/emmettzhang/nightly/general).
Example command for a specific tag:
```bash
docker pull emmettzhang/nightly:tagname
```

3. Create and run the container.
First, retrieve the image ID:
```bash
docker images
```

Then run the container use above image ID:
```bash
# Add "--gpus all --cap-add=SYS_ADMIN" to follow commands to enable gpu support
docker run -it --name daily --network=host -v /root/.ssh:/root/.ssh/:r image-id
```

Additionally, you can build your own image using the Dockerfile available [here](https://github.com/emmett2020/emmett/blob/master/config/dockerfile/daily/Dockerfile).


# Conda
When root-level access is not granted, we recommend using Conda to install the required tools.

```bash
conda create -c conda-forge -n daily python=3.11 gcc=14.2.0 gxx=14.2.0
conda activate daily
```


# Manaual
For a manual installation that offers finer control over your environment, begin by installing these essential compilation tools:

| Tool      | Version  | Install method                                         |
| -------   | -------- | ----------------------------------------               |
| g++       | 14.2.0   |                                                        |
| cmake     | 14.2.0   |                                                        |


For an optimal experience, the following development tools are strongly recommended
- [neovim](./config_neovim.md)



# Pip
TODO


