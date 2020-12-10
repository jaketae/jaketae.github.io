---
title: Docker Blitz
mathjax: false
toc: true
categories:
  - development
tags:
  - docker
---

Docker was one of these things that I always wanted to learn, but never got into. Part of the reason was that it seemed distant and even somewhat unnecessary to me. As someone who has only worked on relatively simple projects, I never felt the need to go beyond the notion of virtual environments. Indeed, when I first read about Docker in an attempt to learn more about what all the DevOps hype was about, I found myself wondering: is Docker really that much different from a Python virtual environment? 

Well, some time has passed since then, and I got lucky enough to have landed an internship at a small startup. Given that the team will be using some DevOps tools---Docker definitely being one of them---I thought I'd get my hands dirty to get a sense of what Docker is like and what it's primarily used for. Instead of the YouTube route, this time I decided to check out a book titled [Docker Deep Dive](https://www.amazon.com/Docker-Deep-Dive-Nigel-Poulton-ebook/dp/B01LXWQUFF) by Nigel Poulton. Throughout this post, I will be referring to examples from his book. For those who want to get a beginner-friendly introduction to Docker, I highly recommend this book. 

At the point of writing, I've read up to Chapter 8 of the book, "Containerizing an App," immediately before the next chapter on Docker compose. This post is not intended as a comprehensive, well-written introduction to Docker; instead, it is in fact a playground environment I used to test out some Docker commands as I was following along the book. With that out of the way, let's jump right in.

# Terminal in Jupyter

Before getting into any details about Docker, it's perhaps necessary for me to clarify the setup in which this post was written. In testing out Docker commands, I went back and forth between this Jupyter notebook and the terminal. I mainly tried to use Jupyter in order to record the commands I typed and their outputs in this post, but certain commands that require secondary input in interactive mode, such as `docker container run -it [...]` was tested in the terminal.

The `!` sign in front of every Docker command is necessary to run unix commands in Jupyter. An exception is `%cd`, which is a magic command in Jupyter that allows the use of `cd`; `! cd` does not work, because the way Jupyter interacts with the system is by attaching a shell subprocess. These details aside, the key takeaway is that the exclamation or percent symbols can be disregarded.

# Docker Basics

In this section, we will learn about some basic docker commands to get started. Here is the most basic one that allows us to check the version and configuration of Docker:


```python
! docker version
```

    Client: Docker Engine - Community
     Version:           19.03.8
     API version:       1.40
     Go version:        go1.12.17
     Git commit:        afacb8b
     Built:             Wed Mar 11 01:21:11 2020
     OS/Arch:           darwin/amd64
     Experimental:      false
    
    Server: Docker Engine - Community
     Engine:
      Version:          19.03.8
      API version:      1.40 (minimum version 1.12)
      Go version:       go1.12.17
      Git commit:       afacb8b
      Built:            Wed Mar 11 01:29:16 2020
      OS/Arch:          linux/amd64
      Experimental:     false
     containerd:
      Version:          v1.2.13
      GitCommit:        7ad184331fa3e55e52b890ea95e65ba581ae3429
     runc:
      Version:          1.0.0-rc10
      GitCommit:        dc9208a3303feef5b3839f4323d9beb36df0a9dd
     docker-init:
      Version:          0.18.0
      GitCommit:        fec3683


Notice that the Docker engine correctly identifies as `OS/Arch` as `darwin`, whereas that of the Server is noted as `linux`. In essence, this is saying that the server is running on a linux kernel. Running a linux kernel on a macOS host through Docker is made possible via Hypervisor and the LinuxKit. At this point, all there is to know about the details is that Docker originally used VirtualBox to run a linux VM, but now uses a more lightweight setup thanks to the aforementioned tools. 

## ls Commands

In unix, `ls` is a command that can be used to get a convenient list of files available in the current directory. Similarly, `docker [...] ls` can be used to look up what docker components are running or existent. For instance, to check which containers are running, we can type


```python
! docker container ls
```

    CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS               NAMES


If we want to check images instead of containers, we can simply replace the `container` with `image`.


```python
! docker image ls
```

    REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
    test                latest              3ad97d9a5a5a        13 minutes ago      82.7MB
    alpine              latest              a24bb4013296        6 weeks ago         5.57MB
    golang              1.11-alpine         e116d2efa2ab        10 months ago       312MB


We can also use some filtering along with the `ls` command to target or specify our search. For instance, to search for only those images whose tags are `latest`, we can run


```python
! docker image ls --filter=reference="*:latest"
```

    REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
    test                latest              3ad97d9a5a5a        2 hours ago         82.7MB
    ubuntu              latest              adafef2e596e        6 days ago          73.9MB
    alpine              latest              a24bb4013296        6 weeks ago         5.57MB


## Pulling an Image

To pull an image, we can use `docker pull [...]`, where the ellipses are the name of the repository and the tag. For example, let's try pulling the latest Ubuntu image from Docker hub. 


```python
! docker pull ubuntu:latest
```

    latest: Pulling from library/ubuntu
    
    [1B352adcf2: Pulling fs layer 
    [1B8a342707: Pulling fs layer 
    [1Bb8e766f4: Pulling fs layer 
    [1BDigest: sha256:55cd38b70425947db71112eb5dddfa3aa3e3ce307754a3df2269069d2278ce47[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[1A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[3A[2K[2A[2K[1A[2K
    Status: Downloaded newer image for ubuntu:latest
    docker.io/library/ubuntu:latest


If we now check what images we have, we see the Ubuntu image that was just pulled.


```python
! docker image ls
```

    REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
    test                latest              3ad97d9a5a5a        13 minutes ago      82.7MB
    ubuntu              latest              adafef2e596e        6 days ago          73.9MB
    alpine              latest              a24bb4013296        6 weeks ago         5.57MB
    golang              1.11-alpine         e116d2efa2ab        10 months ago       312MB


We can also pull from other sources as well. In Docker hub, there is this notion of namespaces. What this simply means is that some Docker accounts, most likely huge companies or other established developers, have a first class namespace status. This means that the name of their repository is absolute. A good example is `ubuntu`---`ubuntu:latest` is a valid name of an image. 

For third party or individual developers like us, however, the namespace becomes slightly different. For example, to pull from Poulton's repository on Docker hub, we need to reference his image as :`nigelpoulton/tu-demo:v2`. For me, it would be `jaketae/repo_title: tag`. Note that the name of the Docker repository is effectively the name of the image.

```
! docker image pull ubuntu:latest
! docker image pull redis:latest
! docker image pull mongo:3.3.11
! docker image pull nigelpoulton/tu-demo:v2
```

Another useful thing to know about pulling is that Docker intelligently knows when to pull new layers and when to use preexisting ones that are already on our system. For example, if I try pulling an image from Docker hub, here is the output message I get on the terminal:


```python
! docker image pull -a nigelpoulton/tu-demo
```

    latest: Pulling from nigelpoulton/tu-demo
    
    [1B3a933944: Pulling fs layer 
    [1B563217f5: Pulling fs layer 
    [1B7ec39263: Pulling fs layer 
    [1B26f0f7cc: Pulling fs layer 
    [1B2aee5115: Pulling fs layer 
    [1Be9939cc3: Pulling fs layer 
    [1B38d27074: Pulling fs layer 
    [1B8469a194: Pulling fs layer 
    [1BDigest: sha256:c9f8e1882275d9ccd82e9e067c965d1406e8e1307333020a07915d6cbb9a74cf[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[9A[2K[7A[2K[9A[2K[7A[2K[9A[2K[9A[2K[9A[2K[9A[2K[8A[2K[7A[2K[7A[2K[7A[2K[6A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[5A[2K[5A[2K[5A[2K[5A[2K[7A[2K[5A[2K[5A[2K[4A[2K[5A[2K[7A[2K[5A[2K[5A[2K[5A[2K[5A[2K[5A[2K[7A[2K[7A[2K[7A[2K[7A[2K[2A[2K[7A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[7A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[7A[2K[2A[2K[2A[2K[7A[2K[2A[2K[2A[2K[1A[2K[2A[2K[2A[2K[7A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[7A[2K[2A[2K[2A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[7A[2K[6A[2K[5A[2K[5A[2K[5A[2K[4A[2K[3A[2K[2A[2K[2A[2K[2A[2K[1A[2K
    v1: Pulling from nigelpoulton/tu-demo
    
    [1B3a933944: Already exists 
    [1B563217f5: Already exists 
    [1B7ec39263: Already exists 
    [1B26f0f7cc: Already exists 
    [1B2aee5115: Already exists 
    [1Be9939cc3: Already exists 
    [1B38d27074: Already exists 
    [1B8469a194: Already exists 
    [1BDigest: sha256:674cb034447ab34d442b8df03e0db6506a99390a1e282d126fb44af8598e4d2a
    v2: Pulling from nigelpoulton/tu-demo
    Digest: sha256:c9f8e1882275d9ccd82e9e067c965d1406e8e1307333020a07915d6cbb9a74cf
    Status: Downloaded newer image for nigelpoulton/tu-demo
    docker.io/nigelpoulton/tu-demo


Notice that layers that already exist are skipped. For example, consider a situation where the Docker image uses `alphine:latest` as a basis. Then, since we already have `alpine:latest` in our system, Docker simply assigns a pointer to reference that image instead of downloading duplicate contents again.

# Image and Containers

A useful concept to have in mind when dealing with Docker is the notion of images and containers. Simply put, a Docker image is a snapshot of this semi-virtual machine. One can think of it as some sort of frozen specimen from which we can only read, not write. Then how do use this image? This is where containers come in. Containers are based off of images and allow users to interact with the virtual environment. For example, we can run the Ubuntu image by spinning a container off of it through the following command:


```python
! docker container run -it ubuntu:latest /bin/bash
```

In Docker commands, `-it` means interactive mode, meaning that the current terminal will turn into a command line interface within the Docker container. Due to constraints in Jupyter, this process cannot be illustrated here, but you'll figure out what this means once you simply run the command. 

Since we have a container running, if we use the `ls` command again---but this time on containers---we get the running container. Notice that under the `IMAGE` tab, we see the original image from which this container was created: `ubuntu:latest`. 


```python
! docker container ls
```

    CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS               NAMES
    6258444a446a        ubuntu:latest       "/bin/bash"         35 seconds ago      Up 34 seconds                           compassionate_hofstadter


We can stop containers that are running simply by explicitly stopping it. We can use either names or container ids to target the container we want to stop.


```python
! docker container stop compassionate_hofstadter
```

    compassionate_hofstadter


But stopping a container doesn't mean that the container is gone. In fact, if we type `ls -a`, we see that `compassionate_hofstadter` is still on our system!


```python
! docker container ls -a
```

    CONTAINER ID        IMAGE               COMMAND             CREATED              STATUS                      PORTS               NAMES
    6258444a446a        ubuntu:latest       "/bin/bash"         About a minute ago   Exited (0) 45 seconds ago                       compassionate_hofstadter


A lot of times, we probably want to keep this docker container since we will probably be developing some application in this Docker container. However, if we want to erase the container completely, we can use the `rm` command.


```python
! docker container rm compassionate_hofstadter
```

    compassionate_hofstadter


And now we see that it is finally gone.


```python
! docker container ls -a
```

    CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS               NAMES


But of course, erasing a container doesn't mean that the image from which it was created is also deleted from the system. Indeed, if we run `ls` on Docker images, we still see `ubuntu:latest`. If we want to, we can always spin another container from this image.


```python
! docker image ls
```

    REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE
    test                latest              3ad97d9a5a5a        18 minutes ago      82.7MB
    ubuntu              latest              adafef2e596e        6 days ago          73.9MB
    alpine              latest              a24bb4013296        6 weeks ago         5.57MB
    golang              1.11-alpine         e116d2efa2ab        10 months ago       312MB


# Dockerfile

Next, let's talk about Dockerfiles. A Dockerfile is a file that tells Docker what sort of image we want to build. So far, we've only be dealing with default images available from Docker hub, such as the latest version of Ubuntu. But what if we want to build some customized image of our own for an application? After all, these images only contain absolutely necessary components. For instance, I've tried typing `whoami` in the Ubuntu image, but the command does not exist! 

So how do we build custom images?

Well, we basically stack images on top of each other. In this context, we call these images as layers. But the boundary between an image and a layer can get somewhat confusing, since an image composed of multiple images can be squashed into one layer, which would then produce one single-layered image. But the overall idea is that we can stack components on top of each other to build a customized image. 

Here is an example Dockerfile from Poulton's repository.


```python
! cat Dockerfile
```

    # Test web-app to use with Pluralsight courses and Docker Deep Dive book
    # Linux x64
    FROM alpine
    
    LABEL maintainer="nigelpoulton@hotmail.com"
    
    # Install Node and NPM
    RUN apk add --update nodejs nodejs-npm
    
    # Copy app to /src
    COPY . /src
    
    WORKDIR /src
    
    # Install dependencies
    RUN  npm install
    
    EXPOSE 8080
    
    ENTRYPOINT ["node", "./app.js"]


In summary, we might visualize this Docker image as follows:

```
layer 4: RUN  npm install
========================================
layer 3: COPY . /src
========================================
layer 2: RUN apk add --update nodejs nodejs-npm
========================================
layer 1: FROM alpine
```

While this file is certainly not written in vernacular prose, we can sort of see what it's doing. First, we start `FROM` some base image, which is `alpine` in this case. Then, we install some modules that will be necessary. We then copy the contents of the file to `/src`, a virtual directory in the Docker container. Then, we run some commands and expose the endpoint of the application. Exposing the endpoint simply means that there is a port or url through which we can access the web application living in Docker. 

As stated earlier, a Dockerfile is a method of building custom images. How do we actually build an image off of it? All we need is a simple `docker image build` command.


```python
! docker image build -t test:latest .
```

    Sending build context to Docker daemon  100.9kB
    Step 1/8 : FROM alpine
     ---> a24bb4013296
    Step 2/8 : LABEL maintainer="nigelpoulton@hotmail.com"
     ---> Using cache
     ---> 2ead764f71cf
    Step 3/8 : RUN apk add --update nodejs nodejs-npm
     ---> Using cache
     ---> 6a652e727789
    Step 4/8 : COPY . /src
     ---> Using cache
     ---> 33eed66ed95e
    Step 5/8 : WORKDIR /src
     ---> Using cache
     ---> e07f22f7a87b
    Step 6/8 : RUN  npm install
     ---> Using cache
     ---> 57fcc62715f2
    Step 7/8 : EXPOSE 8080
     ---> Using cache
     ---> 889b9b226806
    Step 8/8 : ENTRYPOINT ["node", "./app.js"]
     ---> Using cache
     ---> 3ad97d9a5a5a
    Successfully built 3ad97d9a5a5a
    Successfully tagged test:latest


The `.` in the command above simply tells Docker that the Dockerfile is available in the current directory. If it is in a subfolder, we will have to specify its location. 

Now let's run the app! Through `localhost` on port 8080, we can now access the web application running on the Docker container image.


```python
! docker container run -d --name web1 --publish 8080:8080 test:latest
```

    c6645ae79b55b87650c8468d1f605e34d3c22a948a2c99bf717f25753598f63a


If we check which Docker containers are up and running, we see the node application on the list right away. It also shows us the ports that are open.


```python
! docker container ls
```

    CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS                    NAMES
    c6645ae79b55        test:latest         "node ./app.js"     19 seconds ago      Up 18 seconds       0.0.0.0:8080->8080/tcp   web1


Let's gracefully stop the container. 


```python
! docker container stop c6645ae79b55
```

    c6645ae79b55


Note that we can chain the two command together to gracefully stop and remove the container in one chained command.


```python
! docker container stop 8b867dd4a284; docker container rm 8b867dd4a284
```

    8b867dd4a284
    8b867dd4a284


# Notes on Deletions

Earlier, we saw that the `rm` command could be used to delete Docker images or containers. While this is true, there are certain things that we need to be careful of when deleting an image or container. For example, if we try to delete `alpine:latest`, we run into the following message:


```python
! docker image rm alpine:latest
```

    Error response from daemon: conflict: unable to remove repository reference "alpine:latest" (must force) - container 6295af1857c5 is using its referenced image a24bb4013296


This simply means that the `alpine:latest` image is referenced by another container, namely `ubuntu:latest`. From this, we can deduce that the Dockerfile for `ubuntu:latest` probably starts off with `FROM alpine`, or at least uses `alpine` as one of its layers at one point of the building process. Like this, we need to make sure that one image is not a basis for another; only the children can be deleted, not its parent.

Sometimes, you might see `<none>:<none>` images when you run `ls` commands for Docker images. These might be dangling image layers, which can be checked for through the following command:


```python
! docker image ls --filter dangling=true
```

    REPOSITORY          TAG                 IMAGE ID            CREATED             SIZE


To remove dangling layers, we can prune Docker.


```python
! docker image prune
```

    WARNING! This will remove all dangling images.
    Are you sure you want to continue? [y/N] ^C


In some cases, however, pruning does not delete `<none>:<none>` images. This means that these images are not dangling; most commonly, I've realized that these seemingly dangling images are simply the intermediate layers of some custom created image. 

A final note on a convenient command with which we can remove all current containers. Although this is a one-liner, it is really just a nested command in which we first look for containers that are open, get their identifications, and forcibly remove them from the system with the `-f` flag. Note that enforcing `-f` does not constitute graceful shutdown and deletion, but it is a convenient command nonetheless.


```python
! docker container rm $(docker container ls -aq) -f
```

    da65774cecf9


# Searching Docker Hub

As mentioned earlier, Docker hub is sort of the GitHub for Docker images. Here, people can push and pull images that they themselves have created, or those that have been created by others. One convenient thing about Docker hub is that we can use the command line interface to perform some quick searches. In this example, we search for Poulton's images on Docker hub, then pipe that result onto `head` so that we don't end up getting too much search results. 


```python
! docker search nigelpoulton | head
```

    NAME                                 DESCRIPTION                                     STARS               OFFICIAL            AUTOMATED
    nigelpoulton/pluralsight-docker-ci   Simple web app used in my Pluralsight video …   23                                      [OK]
    nigelpoulton/tu-demo                 Voting web server used for various Pluralsig…   12                                      
    nigelpoulton/ctr-demo                Web server for simple Docker demos              3                                       
    nigelpoulton/k8sbook                 Simple web app used for demos in The Kuberne…   2                                       
    nigelpoulton/vote                    Fork of dockersamples Voting App for *Docker…   1                                       
    nigelpoulton/dockerbook              Repo for examples used in Docker Deep Dive b…   0                                       
    nigelpoulton/msb-hello                                                               0                                       
    nigelpoulton/web-fe1                 Web front end                                   0                                       
    nigelpoulton/workshop101             Kubernetes 101 Workshop.                        0                                       


We can also apply filters on our search, just like we saw earlier how we can use the `ls` command along with `filter`. For instance, let's try to search for an official Docker image whose name is `alpine`. Spoiler alert: turns out that there is only one, since `alpine` has first-class namespace status.


```python
! docker search alpine --filter "is-official=true"
```

    NAME                DESCRIPTION                                     STARS               OFFICIAL            AUTOMATED
    alpine              A minimal Docker image based on Alpine Linux…   6613                [OK]                


# Image Inspection

When we deal with custom created images, it's probably a good idea to run a quick inspection on the image, just to be sure that everything looks good and nothing is suspicious. The `inspect` command can be used in this context, and running it gives us this long JSON style output that tells us a lot about how the image was created and what layers there are within it.


```python
! docker image inspect ubuntu:latest
```

    [
        {
            "Id": "sha256:adafef2e596ef06ec2112bc5a9663c6a4f59a3dfd4243c9cabe06c8748e7f288",
            "RepoTags": [
                "ubuntu:latest"
            ],
            "RepoDigests": [
                "ubuntu@sha256:55cd38b70425947db71112eb5dddfa3aa3e3ce307754a3df2269069d2278ce47"
            ],
            "Parent": "",
            "Comment": "",
            "Created": "2020-07-06T21:56:31.471255509Z",
            "Container": "6255a9da773a5e0438e3c097b876a2de65d33f3fb57c4e515faed215d17b8b5d",
            "ContainerConfig": {
                "Hostname": "6255a9da773a",
                "Domainname": "",
                "User": "",
                "AttachStdin": false,
                "AttachStdout": false,
                "AttachStderr": false,
                "Tty": false,
                "OpenStdin": false,
                "StdinOnce": false,
                "Env": [
                    "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
                ],
                "Cmd": [
                    "/bin/sh",
                    "-c",
                    "#(nop) ",
                    "CMD [\"/bin/bash\"]"
                ],
                "ArgsEscaped": true,
                "Image": "sha256:8437baa15ca1576161e9e3f0981298a9c8f0c027e2f86b8d4336bb0d54c2896a",
                "Volumes": null,
                "WorkingDir": "",
                "Entrypoint": null,
                "OnBuild": null,
                "Labels": {}
            },
            "DockerVersion": "18.09.7",
            "Author": "",
            "Config": {
                "Hostname": "",
                "Domainname": "",
                "User": "",
                "AttachStdin": false,
                "AttachStdout": false,
                "AttachStderr": false,
                "Tty": false,
                "OpenStdin": false,
                "StdinOnce": false,
                "Env": [
                    "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
                ],
                "Cmd": [
                    "/bin/bash"
                ],
                "ArgsEscaped": true,
                "Image": "sha256:8437baa15ca1576161e9e3f0981298a9c8f0c027e2f86b8d4336bb0d54c2896a",
                "Volumes": null,
                "WorkingDir": "",
                "Entrypoint": null,
                "OnBuild": null,
                "Labels": null
            },
            "Architecture": "amd64",
            "Os": "linux",
            "Size": 73858282,
            "VirtualSize": 73858282,
            "GraphDriver": {
                "Data": {
                    "LowerDir": "/var/lib/docker/overlay2/a20140d993e4faac2bf8d1ab7aadc4aa5867fb7575a6f3a86a05e1b033df2ab8/diff:/var/lib/docker/overlay2/236b08d4cac34967fd2afe3effef4b8b5116a1ef7088cb1c6dbe216aabe920ca/diff:/var/lib/docker/overlay2/6aca6a67d2b1c73d377979b654af526637994474018c960915d1ac4a5503a353/diff",
                    "MergedDir": "/var/lib/docker/overlay2/5d99fbc21081542a4f8c520abc58119aebadc3b6de40adfe6e404ab74cd7bbb2/merged",
                    "UpperDir": "/var/lib/docker/overlay2/5d99fbc21081542a4f8c520abc58119aebadc3b6de40adfe6e404ab74cd7bbb2/diff",
                    "WorkDir": "/var/lib/docker/overlay2/5d99fbc21081542a4f8c520abc58119aebadc3b6de40adfe6e404ab74cd7bbb2/work"
                },
                "Name": "overlay2"
            },
            "RootFS": {
                "Type": "layers",
                "Layers": [
                    "sha256:d22cfd6a8b16689838c570b91794ed18acc752a08a10bce891cc64acc1533b3f",
                    "sha256:132bcd1e0eb5c706a017ff058b68d76c24f66f84120c51c7662de074a98cbe7a",
                    "sha256:cf0f3facc4a307e4c36e346ddb777a73e576393575043e89d2ea536b693c3ff5",
                    "sha256:544a70a875fc8e410b8a1389bf912e9536cf8167cbbfc1457bba355d5b7ce5c4"
                ]
            },
            "Metadata": {
                "LastTagTime": "0001-01-01T00:00:00Z"
            }
        }
    ]


Okay, this is perhaps too much data, but there there are parts that are interesting that require our attention. For example, notice that under `"RootFS"`, the image shows us how many layers there are. Granted, the layers are SHA256 encrypted, so we can't really see what these individual layers are right away. Nonetheless, we can still get an idea of who heavy the image is and how many layers it is composed of.

Potentially even more important that getting the number of layers from an inspection command is knowing what command the Docker is instructed to run. For a better example, let's take a look at another image. 


```python
! docker image inspect nigelpoulton/pluralsight-docker-ci
```

    [
        {
            "Id": "sha256:dd7a37fe7c1e6f3b9bcd1c51cad0a54fde3f393ac458af3b009b2032978f599d",
            "RepoTags": [
                "nigelpoulton/pluralsight-docker-ci:latest"
            ],
            "RepoDigests": [
                "nigelpoulton/pluralsight-docker-ci@sha256:61bc64850a5f2bfbc65967cc33feaae8a77c8b49379c55aaf05bb02dcee41451"
            ],
            "Parent": "",
            "Comment": "",
            "Created": "2020-01-18T15:29:24.3067368Z",
            "Container": "5e6c8e135f3504d8cdbb3b0f4f7658018f7eafa99011bcb0252c34bad246844f",
            "ContainerConfig": {
                "Hostname": "5e6c8e135f35",
                "Domainname": "",
                "User": "",
                "AttachStdin": false,
                "AttachStdout": false,
                "AttachStderr": false,
                "ExposedPorts": {
                    "8080/tcp": {}
                },
                "Tty": false,
                "OpenStdin": false,
                "StdinOnce": false,
                "Env": [
                    "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
                ],
                "Cmd": [
                    "/bin/sh",
                    "-c",
                    "#(nop) ",
                    "CMD [\"/bin/sh\" \"-c\" \"cd /src && node ./app.js\"]"
                ],
                "Image": "sha256:3eee35387b69036be84160c16d756c975ce6445f5460b19ada2c343d796a0a17",
                "Volumes": null,
                "WorkingDir": "",
                "Entrypoint": null,
                "OnBuild": null,
                "Labels": {
                    "MAINTAINER": "nigelpoulton@hotmail.com",
                    "org.label-schema.build-date": "20190927",
                    "org.label-schema.license": "GPLv2",
                    "org.label-schema.name": "CentOS Base Image",
                    "org.label-schema.schema-version": "1.0",
                    "org.label-schema.vendor": "CentOS"
                }
            },
            "DockerVersion": "19.03.4",
            "Author": "",
            "Config": {
                "Hostname": "",
                "Domainname": "",
                "User": "",
                "AttachStdin": false,
                "AttachStdout": false,
                "AttachStderr": false,
                "ExposedPorts": {
                    "8080/tcp": {}
                },
                "Tty": false,
                "OpenStdin": false,
                "StdinOnce": false,
                "Env": [
                    "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
                ],
                "Cmd": [
                    "/bin/sh",
                    "-c",
                    "cd /src && node ./app.js"
                ],
                "Image": "sha256:3eee35387b69036be84160c16d756c975ce6445f5460b19ada2c343d796a0a17",
                "Volumes": null,
                "WorkingDir": "",
                "Entrypoint": null,
                "OnBuild": null,
                "Labels": {
                    "MAINTAINER": "nigelpoulton@hotmail.com",
                    "org.label-schema.build-date": "20190927",
                    "org.label-schema.license": "GPLv2",
                    "org.label-schema.name": "CentOS Base Image",
                    "org.label-schema.schema-version": "1.0",
                    "org.label-schema.vendor": "CentOS"
                }
            },
            "Architecture": "amd64",
            "Os": "linux",
            "Size": 604213387,
            "VirtualSize": 604213387,
            "GraphDriver": {
                "Data": {
                    "LowerDir": "/var/lib/docker/overlay2/72128cfba240aa98c5b9d2b485463872b2e56f339ce69d5908afe9ca6e4fb31d/diff:/var/lib/docker/overlay2/45b41ad373af200b3751eeaaea1723f76af3340fe98518e31370cbb5c964a225/diff:/var/lib/docker/overlay2/3ac85bfcadfbf8445f596a31d93cf5e20cd4897986abea6a3d1f9d3f56026dee/diff:/var/lib/docker/overlay2/0c23839d4de34d644ad866812e2ba1a850d367fb903f1933f121acf74e677eff/diff:/var/lib/docker/overlay2/d98e0f531b232eeb37ddfa4f188c6518737322967bbbd02363e42808903b9d16/diff",
                    "MergedDir": "/var/lib/docker/overlay2/ea7d7135f789fe192fa518ac788d86331a50a512d5bb4c7f17ab4b898f1f3737/merged",
                    "UpperDir": "/var/lib/docker/overlay2/ea7d7135f789fe192fa518ac788d86331a50a512d5bb4c7f17ab4b898f1f3737/diff",
                    "WorkDir": "/var/lib/docker/overlay2/ea7d7135f789fe192fa518ac788d86331a50a512d5bb4c7f17ab4b898f1f3737/work"
                },
                "Name": "overlay2"
            },
            "RootFS": {
                "Type": "layers",
                "Layers": [
                    "sha256:9e607bb861a7d58bece26dd2c02874beedd6a097c1b6eca5255d5eb0d2236983",
                    "sha256:295c91644e82f1407550c700f1517e814dfa34512ee71ac82ccd4737ca44ea4d",
                    "sha256:07ef3e9a214efe1d68365952f4376a7f8699ce9a5f8b6dc5788347759f334e8c",
                    "sha256:ad1a639ad455b481e4723f3c546a623eac28c86ac961d8b173dab7507f62e122",
                    "sha256:13dba83733f937ac8633ce7b6ebec222ec51d6bbe3f247cf4e652d67fe22c913",
                    "sha256:35467005de8ad904fcc55d34fd5f6bcead2f8b9d97113aa4115130ee9dfa92d7"
                ]
            },
            "Metadata": {
                "LastTagTime": "0001-01-01T00:00:00Z"
            }
        }
    ]


If you look closely at the output, at one point you will see the `"Cmd"` section, which looks like this:

```
"Cmd": [
                "/bin/sh",
                "-c",
                "#(nop) ",
                "CMD [\"/bin/sh\" \"-c\" \"cd /src && node ./app.js\"]"
```

This section tells us exactly what command the Docker container is supposed to run. In this particular instance, we know that the command translates to 

```
/bin/sh -c "cd /src && node ./app.js"
```

The part that is in quotation marks is the actual command. From the looks of it, when the container is spun up, it will `cd` into the `/src` directly and run a node application. Nice!

# More on Running Containers

So far, the only thing we know about running a container is that `-it` is an interactive mode and that running can simply be achieved with `docker container run`. There are some other details that might be helpful to know, in particular relating to automatic restarts. For example, we can pass in some flags such as `always`, `unless-stopped`, and `on-failure` to specify what action the Docker container should take when something breaks down, causing a halt.

```
! docker container run --name neversaydie -it --restart always alpine sh
! docker container run --name neversaydie -it --restart unless-stopped alpine sh
! docker container run --name neversaydie -it --restart on-failure alpine sh
```

Also note that we specified the name of the container in the example commands above as `neversaydie`. We can also micro-configure the container by specifically mapping ports from one to another. For example, if we run 

```
! docker container run -d --name webserver -p 80:8080 nigelpoulton/pluralsight-docker-ci
```

Then we would be able to access the container on port 80. In other words, we would be browsing into `localhost:80`, which would effectively be equivalent to browsing into port `localhost:8080` within the container. These are useful techniques that might come in handy when building a web application. 

# Pushing Docker Image

So far, we've looked at pulling Docker images from Docker hub. We can also push our own images as well. As a simple example, let's take a look at how we might retag an image and perform a simple push. 


```python
! docker image ls
```

    REPOSITORY                           TAG                 IMAGE ID            CREATED             SIZE
    web                                  latest              34b07893e6cf        10 seconds ago      82.8MB
    ubuntu                               latest              adafef2e596e        6 days ago          73.9MB
    alpine                               latest              a24bb4013296        6 weeks ago         5.57MB
    nigelpoulton/pluralsight-docker-ci   latest              dd7a37fe7c1e        5 months ago        604MB
    golang                               1.11-alpine         e116d2efa2ab        10 months ago       312MB


The `docker image tag` command basically uses a preexisting image and re-tags it as specified. In this case, we've retagged `web:latest` into `jaketae/web:latest`. 


```python
! docker image tag web:latest jaketae/web:latest
```

If we look at the images that are on our system, we see the newly tagged image as well.


```python
! docker image ls
```

    REPOSITORY                           TAG                 IMAGE ID            CREATED             SIZE
    jaketae/web                          latest              34b07893e6cf        30 minutes ago      82.8MB
    web                                  latest              34b07893e6cf        30 minutes ago      82.8MB
    ubuntu                               latest              adafef2e596e        6 days ago          73.9MB
    alpine                               latest              a24bb4013296        6 weeks ago         5.57MB
    nigelpoulton/pluralsight-docker-ci   latest              dd7a37fe7c1e        5 months ago        604MB
    golang                               1.11-alpine         e116d2efa2ab        10 months ago       312MB


Now pushing is extremely easy: all we need to do is to use the command `docker image push [...]`, where the ellipses contain the repository and tag of the image that we want to push. Note that retagging was necessary for us to be able to use our own namespace---equivalently, the Docker id---on Docker hub.


```python
! docker image push jaketae/web:latest
```

    The push refers to repository [docker.io/jaketae/web]
    
    [1B8b6e0356: Preparing 
    [1B9a0747a8: Preparing 
    [1Ba1bd40b4: Preparing 
    [2Ba1bd40b4: Pushed   54.46MB/51MB5MBine [2K[1A[2K[4A[2K[2A[2K[4A[2K[2A[2K[4A[2K[2A[2K[4A[2K[2A[2K[2A[2K[4A[2K[3A[2K[4A[2K[4A[2K[4A[2K[2A[2K[4A[2K[4A[2K[4A[2K[2A[2K[4A[2K[2A[2K[4A[2K[2A[2K[4A[2K[4A[2K[2A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[4A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[4A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2K[2A[2Klatest: digest: sha256:ffac23f83cc6f8e6a888db08dc95eca411b13548db499be994f24c26826ac532 size: 1161


# Conclusion

In this post, we took a very quick blitz into the world of Docker, images, and containers. The more I self-study, the more I realize that I'm more of a person who learns through a hands-on approach. I think this is especially the case when learning a new technology which allows one to tinker with and interact with the tools being used. I felt this when learning things like Spark, and I feel it again in this post. 

On a special note, I will be working as a backend software development intern for a Yale SOM-based startup called [ReRent](https://www.rerent.co). I'm so excited about this opportunity, and I can't wait to apply my knowledge of Docker in real production environments as we develop and deploy apps into the cloud. At the same time, however, this also means that I will probably be unable to write as many posts as I used to prior to work. I hope to find a good balance between working and self-studying. I might also write posts about things I learn through the internship, such as Django, using AWS, and many more. 

Thanks for reading this post. See you in the next one!
