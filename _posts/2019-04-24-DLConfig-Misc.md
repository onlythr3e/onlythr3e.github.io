---
layout:     post
title:      深度学习环境配置
subtitle:   Ubuntu 16.04 + Nvidia-418驱动 + CUDA10.1 & CUDNN 7.5/CUDA 9.0 & CUDNN 7.0 配置
date:       2019-04-24
author:     onlythr3e
header-img: img/post-bg-coffee.jpeg
catalog: true
tags:
    - Machine learning
    - Misc
    - Configuration
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
配置环境一直是很恶心的一件事情，不定期更新一下个人验证可用的方法吧。目前最稳定的依然是手动卸载，手动runfile的驱动和安装方法，然后根据个人喜好设置path配置多套cuda环境即可。高版本的驱动都是兼容低版本的驱动的，因此可以放心装高版的。

## 准备工作
#### 清理旧版本的驱动和cuda
不多说了，清就完事了。

> `sudo apt-get remove --purge nvidia*`
> `sudo /usr/local/cuda-*/bin/cuda-uninstaller`
> `sudo PATH_TO_OLD_DRIVER_INSTALLER.run --uninstall`

最后一步不是必须的，一般装新的驱动时会自动删除旧版本的。

#### 下载
从NVIDIA官网下载最新的驱动和计划配置的cuda+cudnn版本即可。

## 安装
#### 禁用nouveau驱动
> `sudo vi /etc/modprobe.d/blacklist.conf`

在最后加入下面三行并保存： 
> blacklist nouveau 
blacklist intel 
options nouveau modeset=0 

然后执行：
> `sudo update-initramfs -u`

重启后执行
>`lsmod | grep nouveau`

如果没有输出，表明禁用成功。

#### 关闭lightdm服务
这一步一般是连接了显示器才需要，如果是远程服务器实测没啥关系，保险起见可以都关上。
> `sudo service stop lightdm`

#### 安装驱动
授予运行权限并执行安装文件，以418版本为例：
> `sudo chmod +r ./NVIDIA-Linux-x86_64-418.56.run`
`sudo ./NVIDIA-Linux-x86_64-418.56.run`

开始后会自检，如有旧版本的驱动选择继续，script failure没有影响，继续即可，dkms建议选no，旧版本会存在opengl的问题，建议不安装，新版驱动r410以后官方已经不再支持opengl，同意协议后即可安装。安装完成后x-config视个人选择，我一般选yes也不会有啥问题。驱动安装完成后，输入`nvidia-smi`检查显卡信息正确即可。如果显示正确，可以重新启动显示服务：
>`sudo service start lightdm`

#### 安装cuda
授予运行权限并执行安装文件，这里先装cuda 9.0:
> `sudo chmod +r ./cuda_9.0.176_384.81_linux.run`
`sudo ./cuda_9.0.176_384.81_linux.run`

驱动安装必须选no，位置可以默认，symbolic link多个版本的话可以后面再自己弄，嫌麻烦可以把想要默认使用的版本选择创建link，这样安装时`/usr/local/cuda`会指向创建了symbolic link的版本。

然后一样的方法安装cuda 10.1如下：
> `sudo chmod +r ./cuda_10.1.105_418.39_linux.run`
`sudo ./cuda_10.1.105_418.39_linux.run`

同理选择不安装驱动，其他的自己酌情选。完成后安装cudnn，以cuda 10.1对应的7.5版本为例。
> `tar -xzvf cudnn-10.1-linux-x64-v7.5.0.56.tgz`
`sudo cp cuda/include/cudnn.h /usr/local/cuda-10.1/include`
`sudo cp cuda/lib64/libcudnn* /usr/local/cuda-10.1/lib64`
`sudo chmod a+r /usr/local/cuda-10.1/include/cudnn.h /usr/local/cuda-10.1/lib64/libcudnn*`

这里注意cudnn要拷到对应版本的cuda里而不要直接选择`/usr/local/cuda`。

安装完成后可以创建两个不同环境文件如下:
`sudo vim ~/.cuda-9.0-cudnn-7-0.env`
> `export PATH=$PATH:/usr/local/cuda-9.0/bin`
`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64`
`export CUDADIR=/usr/local/cuda-9.0`
`export NVIDIA_CUDNN=/usr/local/cuda-9.0`

`sudo vim ~/.cuda-10.1-cudnn-7-5.env`
> `export PATH=$PATH:/usr/local/cuda-10.1/bin`
`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.1/lib64`
`export CUDADIR=/usr/local/cuda-10.1`
`export NVIDIA_CUDNN=/usr/local/cuda-10.1`

其中后两行是为matlab配置的，当然了用matlab跑深度学习还需要Nvidia-tensor-rt。需要某个环境时打开新的terminal并`source ./cuda-10.1-cudnn-7-5-env`即可切换到对应的cuda环境了。

个人比较懒，通常会在`~/.bashrc`中设置alias如下：
>`alias cuda9='source ~/.cuda-9.0-cudnn-7-0.env'`
`alias cuda10='source ~/.cuda-10.1-cudnn-7-5.env'`

至此环境就配置完成了。


