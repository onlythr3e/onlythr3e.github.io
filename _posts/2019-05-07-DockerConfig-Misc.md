---
layout:     post
title:      Deep Learning Docker环境配置
subtitle:   Ubuntu 16.04 + Docker-CE + Nvidia-docker + NGC Pytorch配置
date:       2019-05-07
author:     onlythr3e
header-img: img/post-bg-coffee.jpeg
catalog: true
tags:
    - Machine learning
    - Misc
    - Configuration
    - Docker
---

<!---
    - Title level: #
    - Inline math: $$ $$ or \( \)
    - Block math: \$$ $$ or \[ \]
    - Block Quote: > or >>
    - Bold: ** **
    - Bullet list: - or *
    - Number list: Number
    - Inline code block: ``
    - Image: ![image_name](image_url)
    - Line break: two spaces + \n
    - Link: [link_text](url).
    - Reference: [link_url][number]
    - 
-->
## 前言
之前在[这篇文章](https://onlythr3e.github.io/2019/04/24/DLConfig-Misc/)中介绍了深度学习本地环境的配置方案，但本地使用只适合开发，在确定要大规模开始训练时超算是少不了的。个人常用的是Pittsburgh Super Computing的GPU中心，默认支持singularity container的服务。但考虑到docker转singularity相当方便，且工业界主流云服务（AWS，Google Cloud等）均以docker为主，本文将就docker的使用作简要介绍，并在文末附上docker-to-singularity的解决方案。

## 准备工作
#### 安装最新版本的Nvidia驱动
这里注意CUDA并不是必须的，因为nvidia-docker提供的image中允许自由选择对应的CUDA+CUDNN版本。因此在本机配置最新的nvidia驱动可以获得最好的适配效果。Ubuntu的显卡驱动安装过程参见[前文](https://onlythr3e.github.io/2019/04/24/DLConfig-Misc/)。

#### 申请Nvidia GPU Cloud账号
前往[NGC](https://ngc.nvidia.com)注册一个账号即可，注册完毕后登陆，在configuration中可以生成一个API Key即可，注意保存这个API Key，之后在网站上是无法获取的。API Key的形式如下：
> `Username: $oauthtoken`  
`Password: <Your key>`

这里username必须严格用上面的形式，password则是生成的api key。

## 安装
#### 安装docker
参见官方的[docker-ce for ubuntu安装教程](https://docs.docker.com/install/linux/docker-ce/ubuntu/)，基本没有问题。

安装完成后执行：
>`sudo docker version`

如果显示正常，则安装成功。

#### 将当前用户添加到docker用户组中
这一步是为了不需要sudo也可以使用docker，会方便很多，执行以下命令即可。
> `sudo usermod -aG docker <your-username>`

#### 安装nvidia-docker
同样参照[官方教程](https://github.com/NVIDIA/nvidia-docker)即可。

#### 登陆ngc
这边利用前面已经获取的NGC API Key来登陆NGC，以开始访问Nvidia官方的镜像。
> `docker login nvcr.io`

弹出username和password时输入NGC的对应信息即可。

#### 获取NGC镜像
这里以最新的pytorch 19.04为例。注意，在实际使用中根据要跑的代码的pytorch版本，我们应该先到[pytorch-ngc](https://docs.nvidia.com/deeplearning/dgx/pytorch-release-notes/overview.html#overview)中查询应该下载哪个对应的image。执行下列命令pull对应的image到本地即可：

> `docker pull nvcr.io/nvidia/pytorch:19.04-py3`

等待安装完成后可以检查安装是否正确，这里可以先切换到包含了代码和数据的文件夹以便将代码mount到docker中，然后新建如下的交互式container：
> `docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -v $(pwd):/workspace --rm -it nvcr.io/nvidia/pytorch:19.04-py3`

这样本地的docker就配置完成了。

## 使用云计算和singularity
如果本地跑个小程序也使用docker的话，感觉还是没有一个原生的环境来的爽。因此，docker的优势更多的是体现在云计算和超算上，云计算使用docker和本地没什么区别，这里略过不谈。超算由于使用singularity进行container布置居多，本文以[PSC](https://www.psc.edu/)的使用为例进行讲解。

#### 基本配置
PSC要求通过[XSEDE](https://portal.xsede.org)进行资源申请，第一次进行资源申请顺利的话大概一周左右会得到结果。资源审批通过后XSEDE的账户会优先激活，此时PSC账户尚未激活，需要大概再过一周的时间PSC的账户才会可以使用，此时会收到PSC的邮件。这里我们的research申请了如下的资源：
> Bridges: 常规的CPU算力。  
> Bridges-Pylon: Pylon5的存储空间用于存放数据。  
> Bridges-GPU: GPU算力。

#### 转移数据
登陆到Bridges后确定自己的用户组以便获取数据的存放路径：

> `id -Gn`

目前pylon的路径为
> `/pylon5/groupname/username`

可以在里面创建文件夹并将代码和数据都转过去。

#### 直接使用云镜像配置singularity
超算一般使用的是singularity而不是docker，考虑到在超算的login节点（也就是不算钱的登陆界面）是无法访问到singularity的，而开着超算调singularity也未免太奢侈了一些，因此，我们可以选择在本地也配置一下singularity，确保运行正确后再到超算上去执行。

singularity安装参照[官方教程](https://www.sylabs.io/guides/3.2/user-guide/installation.html)即可，理论上2.6之后的版本都是可以直接build docker image的。singularity安装完毕后有两种方法可以转换一个docker image，如果前面没有下载nvidia的image的话，这里可以直接使用网上的image作为基础镜像。

为了让singularity可以使用nvidia-docker，参照[nvidia教程](https://docs.nvidia.com/ngc/ngc-user-guide/singularity.html#singularity)设置如下。首先设置singularity的NGC信息：
> `export SINGULARITY_DOCKER_USERNAME='$oauthtoken'`  
> `export SINGULARITY_DOCKER_PASSWORD=<NVIDIA NGC API key>`

为了方便可以将上述命令存到`~/.ngc`中，然后需要使用时`source ~/.ngc`即可，此时singularity可以正确访问到nvidia cloud的镜像。然后执行如下命令即可

> `singularity build [path_to_image]/[app_tag].simg docker://nvcr.io/nvidia/pytorch:19.04-py3`

镜像的名字可以自己设置，后面给出nvcr对应镜像的url，singularity会自动下载并转换该image。提示build complete之后即可使用，这种方法会创建一个可以传输到超算中的simg文件，比较适合斤斤计较的穷孩子。如果直接在云平台上使用singularity，那么直接:

> `singularity pull docker://nvcr.io/nvidia/pytorch:19.04-py3`

也可以获取该镜像到本地并完成转换操作。这样的方法适用于所有已经push到docker hub上的image。如果是private image，加入登陆信息即可：
> `singularity pull --docker-login docker://<url_to_image>`

#### 使用本地docker镜像配置singularity
有小伙伴说啊我本地还是喜欢用docker啊，只有上超算时才想着用singularity转一下，那也没啥问题。首先判断一下使用的本地registry是v1还是v2，这可以通过`docker info`来确定本地registry的url。如果是v1，可以执行以下命令：
> `curl -X GET http://myregistry:5000/v1/search?`

类似的，v2使用了新版的request如下：
> `curl -X GET http://myregistry:5000/v2/_catalog`

将前面的`<url_to_image>`换成registry中的地址即可。如果并没本地的registry，可以参照官方的教程如下：

```
# Start a docker registry
docker run -d -p 5000:5000 --restart=always --name registry registry:2
```

```
# Push local docker container to it
docker tag <complete_app_tag> localhost:5000/<namespace>/<app_tag> 
docker push localhost:5000/<namespace>/<app_tag> 
```

push完成后注意由于localhost是http而非https协议，我们需要相应地改变singularity的build命令如下：

```
# Build singularity container
$ SINGULARITY_NOHTTPS=1 singularity build <app_tag>.simg docker://localhost:5000/<namespace>/<app_tag>
```
完成后本地的image就转换完成啦！

#### 使用singularity执行程序
直接执行程序如下：
> `singularity exec --nv <path_to_app>/<app_tag>.simg <command_to_run>`

开启交互界面如下：
> `singularity exec --nv <path_to_app>/<app_tag>.simg /bin/bash`
或
> `singularity shell --nv <path_to_app>/<app_tag>.simg`

注意，用第一个命令打开interact模式下bash不会有任何改变，所以看起来好像完全没有变化！但实际上unix的哲学是没有报错就是成功，所以实际上你已经是在singularity的环境下运行程序了。这一点可以通过检查nvcc的版本和pytorch的版本得到确认。第二个命令则会在shell的光标处提示singularity，表示你已经处于singularity的环境中。

#### 定制化的container
很多时候跑一些开源的代码需要补充一些必要的package，此时就需要修改image，但是nvidia官方的image是不允许使用`--wriable`选项进行修改的，为此我们基于选择的镜像创建新的镜像。当然了熟悉singularity的小伙伴可以直接用`.def`文件修改，这和dockerfile类似，查阅官方文档即可。

不过我个人使用`.def`文件以NGC的镜像为基础系统还是遇到了一些问题，比如build完成后NGC镜像中的所有环境变量都丢失了，这一点就很糟糕。为此，我还是用了最蠢但是最保守的方法，也就是进入image中直接修改得到所需的环境，然后commit image。操作如下：

> `nvidia-docker run -ti nvcr.io/nvidia/<app_tag>`

打开交互界面后就像普通系统一样安装软件即可。完成后退出docker container，然后检查id如下：

> `docker ps -a`

找到刚刚修改完成的container，commit到一个新的位置：

> `docker commit <container_id> nvcr.io/nvidia_sas/<new_app_tag>`

这里前面的url和namespace都是可以修改的。这时`docker image ls`就可以看到修改后的镜像了。按照前面说的使用singularity build本地镜像的方法即可创建所需的singularity镜像。

## 后记

在漫长的等待之后PSC告诉我账号终于激活了，那可是P100上的1000 gpu-hours，够我玩好一阵的了。兴冲冲地deploy好我的container打算开始跑程序，结果发现PSC优先allocate了bridges和pylon，gpu资源仍然在审核中。绝望的心情堪比黑魂里对面一个后撤步背刺被你成功弹反，这时候打算一换掏出流放者大刀处决却换出了折断的直剑。无可奈何啊，phd不就是为了折腾么，只能继续望穿秋水了。




