---
layout:     post
title:      Decision Theory Basics
subtitle:   决策论基础 Decision Theory Basics #1
date:       2019-02-12
author:     onlythr3e
header-img: img/post-bg-coffee.jpg
catalog: true
tags:
    - Machine Learning
    - Statistics
    - Decision Theory
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
# 前言

Decisoion Theory和Statistical Inference还不太一样，有些概念很容易混淆，尤其是不同流派之间基于不同的出发点推导的loss，risk和对应的action之间的关系经常忘记，最近看*Statistical Decision Theory and Bayesian Analysist* 一书发现里面介绍得挺详细的，所以写下这篇文章聊作备忘。

# 不确定性

Decision Theory的本意是用来解决以下的问题:

* 在面临的情景中存在不确定性的因素时应如何作出决定? 
* 如何刻画不同的面对不确定性的偏好？
* 在不同的偏好下如何最大化收益或者最小化损失？

基于以上的考虑，我们往往希望用数字去刻画这样的不确定性，在Decision theory中我们用以下的数学语言来描述一个决策问题:

* 问题的不确定性（State of nature）：$$\theta \in \Theta$$。
	* $$\theta$$ 即随机参数parameters。
	* $$\Theta$$ 即参数空间parameter space。
* 采样/调查 (Statistical investigation/Sampling)： 从$$\theta$$决定的一个分布中去采样得到随机样本 $$X_1, X_2, \dots, X_n$$的过程，可以用$$x$$来表示所有可能的样本取值。
* 行为（Action）：根据已有的信息作出的决定 $$a \in \mathcal{A}$$，对于不同的问题action有不同的表述。
	* 最简单的参数估计问题中，action就是给出一个参数$$\hat{\theta}$$对$$\theta$$进行估计。
	* 在经济学的问题中，action可以是要不要执行某个商业计划。
	* 在machine learning的问题中，action可以是对新的输入进行推测。
* 损失（Loss）：做出行为获得的收益和所需付出的代价，统一用损失刻画。
* 决策规则（Decision Rule）：根据不同的样本信息而选择不同的决策，本质上是这样的一个函数：$$\delta: \mathcal{X} \to \mathcal{A}$$。

我们应对不确定性的方法也有三种：

* 刻画不确定性的分布以及我们从这个分布采样得到的数据：$$X \sim P_{\theta \in \Theta}$$.
* 对$$\theta$$的先验知识：$$\pi(\theta)$$.
* 不同的参数和不同的行为所需要付出的代价：$$L(\theta, a)$$.

基于以上的设定，我们可以将decision theory的核心总结为这样的一个问题，在不确定性存在的情况下，如何通过决策者的偏好、先验知识、以及采样等方式实现Loss的最小化。
  
# Loss最小化的原则

在对Loss进行最小化的过程中，过去的研究者们总结出了不同的准则。这些不同的准则来自于对问题不同的理解，其核心区别主要在于以下几点：

* 参数是否会随着行为而改变，即$$\theta$$是否独立于$$a$$。在目前的研究中，我们通常认为$$\theta$$不受$$a$$的影响。
* 行为是否应该受采样/调查影响，即$$a$$是否取决于$$X$$。
	* 如果$$a$$独立于$$x$$，我们可以直接求最优的行为。
	* 如果$$a$$取决于$$x$$，那么此时$$a = \delta(x)$$。
* Loss应该对什么平均？这个问题乍看之下有点莫名其妙，但其实颇有几分哲学意味。举一个简单的例子，我们是否相信$$\theta$$是一个一成不变的值？这其实关乎到Frequentist和Bayesian的本源之争。

对于以上问题持不同的态度，会自然而然地得到最小化loss的不同原则，以下将经典的三个原则简要介绍如下：
#### Conditional Bayesian Principle 条件贝叶斯原则

Conditional Bayesian的核心假设是以下几点：

* $$\theta$$ 本身也是一个随机变量，可以用分布刻画。在没有观察数据时，人们可以用先验分布$$\pi(\theta)$$描述$$\theta$$，在存在观测数据时，我们可以用后验条件分布$$\pi^*(\theta | X)$$来描述$$\theta$$。
* 行为不影响参数。（影响的话也可以继续用Conditional Bayes，只是需要作调整）
* 行为不受观察数据影响，数据是用来更新先验的。
* 关心loss在后验分布上的表现，最优行为直接通过优化后验分布下的loss求解。换言之，我们认为问题的不确定性来自于$$\theta$$本身，因而对$$\theta$$求期望。

基于以上的假设，条件贝叶斯原则旨在优化下面的**Bayesian Expected Loss**:
\$$
	\rho(\pi^\*, a) = \mathbb{E}_{\pi^\*} [L(\theta, a)]
$$

其中$$\pi^*$$是后验分布。这也是为什么将这条准则叫做条件贝叶斯的原因。假定后验概率已知，则最优决策为：
\$$
	a^* = \arg\min_{a \in \mathcal{A}} \rho(\pi^\*, a)
$$ 
细心的朋友应该已经发现了这和statistics以及machine learning中Maximium a Posterior (MAP) Estimation 的关联。注意上式中的几个关键点：

* 对后验分布求期望：这本质上是loss对$\theta$的一个加权平均，希望我们选取的loss在$$\theta$$的后验分布的加权下平均最小。
* 没有考虑对$$x$$求期望：对$$x$$求期望反映了我们希望在考虑样本的随机性情况下作出决策，这里我们直接使用$$x$$更新了先验，目的是为了优化问题可以有一个简单的最优解。在后文对其他准则的讨论中我们还会提到这一点。

#### Frequentist Principle 频率学派准则

频率学派原则的核心假设是以下几点：

* $$\theta \in \Theta$$是一个固定而不可知的值。
* 行为受观察数据直接决定，即最优解是$$\delta^*(x)$$而不是$$a^*$$。
* 问题的随机性来自于采样的数据$$x$$，关心loss在可能的不同样本之间的平均表现。

为此，Frequentist Principle关心的是**Risk over Repetitive Experiments**:
\$$
	R(\theta, \delta) = \mathbb{E}_{x \sim P(X|\theta)}[L(\theta, \delta(x))] = \int_{x} L(\theta, \delta(x))f(x|\theta)dx
$$

