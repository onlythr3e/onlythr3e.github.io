---
layout:     post
title:      Discrete Differential Geometry 离散微分几何笔记
subtitle:   Part 1. Basics
date:       2019-03-02
author:     onlythr3e
header-img: img/post-bg-coffee.jpeg
catalog: true
tags:
    - Computer Graphics
    - Computer Science
    - Discrete Differential Geometry
    - Notes
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

最近在上Keenan Crane的DDG课程，感觉过去在学CG时涉及到mesh处理很多不明就里的操作都有了理论上的支持，并深深地感受到了微分几何在CG领域的强大作用。离散微分几何（DDG）的这一研究主题的目的就是如何将微分几何中已经解决的问题转变为计算机可以处理的离散问题，并关心这样的转化会带来什么样的差异。

本系列笔记参考了[Keenan Crane DDG 2019](http://brickisland.net/DDGSpring2019/)的课程内容及他的一本小书[DDG: An Applied Introduction](http://www.cs.cmu.edu/~kmcrane/Projects/DDG/paper.pdf)。感兴趣的朋友可自行前往查阅英文原版教材及习题。

在学习离散微分几何的过程中，最大的感触就是对连续版本的微分几何不够熟悉，基础不够扎实，以至于在推导离散版本的公式的时候遇到了很多困难。但换一个方向说，这也恰恰是离散微分几何的魅力。在微分几何的理论世界中，很多概念无疑是很难理解的，或者说至少不是那么直观的。但是当这些问题被成功转化为离散的版本之后，我们就可以直观地体会到微分几何中所学习到的性质是多么的优美和简洁。

## 概览

在此先列出本系列笔记打算覆盖的内容：

* 基础的离散微分几何背景。
* Exterior Calculus基础及其与传统的向量空间的关系。
* 离散几何的基本形式，如单纯形、复合形、网格等。
* 离散几何的基本操作：基本运算、求导、积分、拉普拉斯算子（即求解Laplacian/Poisson Equation的过程）等。
* 离散几何的操作在CG中的应用，如平滑，参数化，优化网格等。

和Machine learning中一个核心的概念类似的是，DDG中也存在所谓的**No free lunch**原则，即**将某一个微分几何概念离散化的过程必然导致其某些性质的缺失**。换言之，对于任何一个微分几何中的概念，如斜率、曲率、微分算子、梯度算子等等，我们都可以找出许多种方法来对其离散化，而这些方法都只保留了原概念中一部分的性质。以下是Keenan总结的解决一个离散微分几何问题的通用思路：

* 对于要离散化的微分几何概念，先找出多个在Smooth setting中等价的定义。
* 找到对应于每一种定义的离散化表示。
* 研究这些离散化表示都保留了哪些性质，并针对不同的应用选择合适的离散化方式。

在原书中针对这个问题给出了一个很经典的关于curvature离散化的讨论，此处不做赘述。
  
## 工具

对离散微分几何的研究涉及到以下的知识点:

* Exterior Calculus：这是研究离散微分几何最有用的工具之一，必须灵活掌握。
* 传统微分几何。
* CG中基于Halfedge的网格表示。
* 线性方程组的求解及常见的优化问题的求解。


## Exterior Calculus
接下来我们来介绍基本的Exterior Calculus的内容。



