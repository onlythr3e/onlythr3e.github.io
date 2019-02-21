---
layout:     post
title:      Machine Learning - Decision Theory Basics I
subtitle:   Part 1. Settings and priniciples
date:       2019-02-12
author:     onlythr3e
header-img: img/post-bg-coffee.jpeg
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
## 前言

Decisoion Theory和Statistical Inference还不太一样，有些概念很容易混淆，尤其是不同流派之间基于不同的出发点推导的loss，risk和对应的action之间的关系经常忘记，最近看*Statistical Decision Theory and Bayesian Analysist* 一书发现里面介绍得挺详细的，所以写下这篇文章聊作备忘。

## 不确定性

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
  
## Loss最小化的原则

在对Loss进行最小化的过程中，过去的研究者们总结出了不同的准则。这些不同的准则来自于对问题不同的理解，其核心区别主要在于以下几点：

* 参数是否会随着行为而改变，即$$\theta$$是否独立于$$a$$。在目前的研究中，我们通常认为$$\theta$$不受$$a$$的影响。
* 行为是否应该受采样/调查影响，即$$a$$是否取决于$$X$$。
	* 如果$$a$$独立于$$x$$，我们可以直接求最优的行为。
	* 如果$$a$$取决于$$x$$，那么此时$$a = \delta(x)$$。
* Loss应该对什么平均？这个问题乍看之下有点莫名其妙，但其实颇有几分哲学意味。举一个简单的例子，我们是否相信$$\theta$$是一个一成不变的值？这其实关乎到Frequentist和Bayesian的本源之争。

对于以上问题持不同的态度，会自然而然地得到最小化loss的不同原则，以下将经典的三个原则简要介绍如下：
### Conditional Bayesian Principle 条件贝叶斯准则

Conditional Bayesian的核心假设是以下几点：

* 只关心发生了的事情（已经采样到的数据），而不关心可能发生却没发生的事件（没采样到但同样有可能存在的数据）。
* $$\theta$$ 本身也是一个随机变量，可以用分布刻画。在没有观察数据时，人们可以用先验分布$$\pi(\theta)$$描述$$\theta$$，在存在观测数据时，我们可以用后验条件分布$$\pi^*(\theta \mid X)$$来描述$$\theta$$。
* 行为不影响参数。（影响的话也可以继续用Conditional Bayes，只是需要作调整）
* 行为不受观察数据影响，数据是用来更新先验的。
* 关心loss在后验分布上的表现，最优行为直接通过优化后验分布下的loss求解。换言之，我们认为问题的不确定性来自于$$\theta$$本身，因而对$$\theta$$求期望。

基于以上的假设，条件贝叶斯原则旨在优化下面的**Bayesian Expected Loss**:
\$$
	\rho(\pi^\ast, a) = \mathbb{E}_{\pi^\ast} [L(\theta, a)]
$$

其中$$\pi^*$$是后验分布。这也是为什么将这条准则叫做条件贝叶斯的原因。假定后验概率已知，则最优决策为：
\$$
	a^* = \arg\min_{a \in \mathcal{A}} \rho(\pi^\*, a)
$$ 
注意上式中的几个关键点：

* 对后验分布求期望：这本质上是loss对$\theta$的一个加权平均，希望我们选取的loss在$$\theta$$的后验分布的加权下平均最小。
* 没有考虑对$$x$$求期望：对$$x$$求期望反映了我们希望在考虑样本的随机性情况下作出决策，这里我们直接使用$$x$$更新了先验，目的是为了优化问题可以有一个简单的最优解。在后文对其他准则的讨论中我们还会提到这一点。

可以看到，Conditional Bayes的核心就在于将观察到的数据作为条件。问题在于，我们应该如何利用观察数据进行条件分析。以下是常见的基于观察数据的分析方法，其中最有名的当然就是Maximum Likelihood Estimation和Maximum A Posterior了，之后有机会我们再对这些方法展开详细的讨论。

* Likelihood methods：直接使用模型来表示一个条件分布，其目的是使得该分布的likilihood $$P(X \mid \theta)$$最大。Likilihood包含了**样本数据**（而非全部数据）的一切与参数有关的信息。
* Bayesian methods：利用Bayes rules将posterior变为prior * likilihood，即$$P(\theta \mid X) \propto P(X \mid \theta)P(\theta)$$。
* Structural inference：略
* Pivital inference：略

### Frequentist Principle 频率学派准则

频率学派原则的核心假设是以下几点：

* 不仅关心当前观察到的数据，也关心未观察到的数据。
* $$\theta \in \Theta$$是一个固定而不可知的值。
* 行为受观察数据直接决定，即最优解是$$\delta^*(x)$$而不是$$a^*$$。
* 问题的随机性来自于采样的数据$$x$$，关心loss在可能的不同样本之间的平均表现。

为此，Frequentist Principle关心的是**Risk over Repetitive Experiments**:
\$$
	R(\theta, \delta) = \mathbb{E}_{x \sim P(X \mid \theta)}[L(\theta, \delta(x))] = \int L(\theta, \delta(x))f(x \mid \theta)dx
$$

问题在于，尽管$$\theta$$是一个确定的值，但却是不可知的，因此上式无法求最优解。给定不同的$$\delta$$，当$$\theta$$在变化时，我们其实得到了两个函数，$$R(\theta, \delta_1)$$和$$R(\theta, \delta_2)$$。那么问题来了，我们要如何比较这两个函数以选出更好的决策规则$$\delta^*$$呢？为此我们定义了以下的简单比较risk函数的方式：

* R-better：$$\delta_1$$ is R-better than $$\delta_2$$ if $$R(\theta, \delta_1) \leq R(\theta, \delta_2)$$ for all $$\theta \in \Theta$$, with strict inequality for some $$\theta$$.
	* 在最简单的情况下，假如其中一个risk函数所有可能的取值都比另一个risk函数低，那该决策规则显然是一个更好的选择。
	* 只要$$\delta_1$$的risk在$$\theta$$取任意值的情况下都不差于$$\delta_2$$，并且在某些$$\theta$$取值下更优，那么我们认为$$\delta_1$$比$$\delta_2$$更好。
* R-equivalent: $$\delta_1$$ is R-equivalent to $$\delta_2$$ if $$R(\theta, \delta_1) = R(\theta, \delta_2)$$ for all $$\theta \in \Theta$$.
* Admissibility: A decision rule $$\delta$$ is admissible if there exists no R-better decision rule. 
	* 没有任何规则能够在**所有情况下**都比当前规则更好，此时的$$\delta$$为一个**可行的**选项。
	* 反之则为**不可行**的选项，即inadmissible。

我们可以发现，同样是admissible的规则也许不仅一条，换言之，在某些$$\theta$$下，$$\delta_1$$的risk更小，而在某些$$\theta$$下，$$\delta_2$$的risk更小。在Frequentist Principle的大背景下，我们同样面临着对risk的偏好问题，这些形色的偏好形成了如下的不同的子原则：

#### 1. Minimax

Minimax原则可以理解为保守主义，即尽量不让最坏的事情发生。即便执行某个行为的收益很高，但对于某些情况可能损失也很高的话，那么minimax倾向于拒绝这样的规则。一个最直观的例子是在金融投资中做出的决策可以少赚，但一定不能大赔。

这种规避最大风险的偏好可以用数学语言描述如下：
\$$
\delta_{\text{Minimax}} = \arg\inf_{\delta \in \mathcal{D}} \sup_{\theta \in \Theta}R(\theta, \delta)
$$

上式值得注意的有以下几点：

* 我们只关心最坏的情况，即$$\theta$$变化过程中最大的loss。
* 在最常见的参数估计问题中，Minimax往往不太好求最优解。由于涉及到两次优化问题，通常只能对一些较为简单的参数估计问题给出Minimax估计量。

#### 2. Bayes

Bayes原则理解为平均主义，即考虑如何让risk在加权平均的意义上更低。换言之，我们允许某些$$\theta$$下loss很高，但我们希望$$\theta$$处于不利情形下的概率较低。这样，我们就把先验知识反映在了$$\delta$$的选择中，从而得到Bayes偏好如下：
\$$
\delta_{\text{Bayes}} = \arg\inf_{\delta \in \mathcal{D}} \mathbb{E}_{\theta \sim \pi} R(\theta, \delta)
$$

我们将$$r(\pi, \delta) = \mathbb{E}_{\theta \sim \pi} R(\theta, \delta)$$称作Bayes Risk。

#### 3. Invariance
Invariance原则暂略。

## 总结

本文主要讨论了基本的Decision theory问题设定以及两种主流的思考问题的范式：Conditional及Frequentist。下表列出了二者最主要的区别。

| | Conditional | Frequentist |
| ---|--- | --- |
| 如何看待数据？ | 只关心观察的数据，因此数据是不变的。 | 关心观察到的数据和可能的数据，因此数据是可变的。 |
| 如何看待参数？ | 参数服从某种分布，因此参数是可变的。 | 参数固定但不可知，因此参数是不可变的。 |
| 关心的risk？| $$\rho(\pi^*, a) = \mathbb{E}_{\pi^*} [L(\theta, a)]$$ | $$R(\theta, \delta) = \mathbb{E}_{x \sim P(X \mid \theta)}[L(\theta, \delta(x))] = \int L(\theta, \delta(x))f(x \mid \theta)dx$$ |
| 选取最优行动的方法 | 最小化Conditional Bayesian Expected Loss | 依据不同的准则选择Minimax或Bayes |

然而我们很快就会看到，二者表面上看似出发点和解决问题的方式大相径庭，但其内核却有着异曲同工之妙，这也是回顾Statistics历史时有趣的一点吧。

## Reference
1. James O.Berger, *Statistical Decision Theory and Bayesian Analysis*, 1980.
