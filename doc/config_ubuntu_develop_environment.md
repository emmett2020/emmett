Personally, I recommend the following three methods to set up a local development environment for general use.

# 1. Docker
----
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



# 2. apt
----


# 3. conda
----

