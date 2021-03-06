---
layout:     post
title:      Machine Learning - Decision Theory Basics II
subtitle:   Part 2. Bayesian and Minimax
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
    - Inline math: $$ $$
    - Block math: \$$ $$
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

[上文](https://onlythr3e.github.io/2019/02/12/DecisionTheoryBasics1-ML/)非常简略地介绍了Decision Theory的基础，本文会略微补充一些相关的细节，并就Bayes和Minimax这两种方法展开更深入的讨论。

以下是本文将要探讨的问题：

* 随机决策（Randomized decision policy）的引入及意义。
* 充分统计量（Sufficient Statistics）的意义以及Rao-Blackwell定理究竟说了个啥？
* Conditional Bayes / Frequentist Bayes方法的具体介绍与他们之间的联系。
* Minimax方法的具体介绍。

## 背景补充

#### 随机决策

决策可以是随机的，举个例子，MIT和伯克利实在不知道读哪个，这时掏出一个硬币，正面去波士顿，反面去旧金山（多希望我也可以有这样的烦恼）。一般来说，随机化的决策往往源自以下两种动机：

* 对抗性的决策：在博弈的过程中，总是按照当前最优的方式行动可能会被对手预测到你下一步的行动，为此，可以给行动加入一定的随机性，以防止自己的行动被预测到。
* 另一个更直接的原因是，在统计学的参数估计问题中，我们经常会遇到使用统计量(Statistics)来进行参数估计的情况。注意，统计量是样本数据的一个函数，由于样本数据是随机的，统计量也是一个随机变量。此时使用统计量作为某个参数的估计就相当于做出了一个随机化的决策。

为此，我们给出随机化的决策定义如下：
\$$
    \delta(x, A) = Pr(A \subseteq \mathcal{A} \space \text{is picked})
$$

其中$$x$$是我们观察到的样本数据，$$A$$是action空间的一个子集，上式表示：在观察到的样本为$$x$$的情况下，采取行动$$A$$的概率为$$P$$。

有时我们需要从随机化的decision policy中获取deterministic的decision，这时derandomization就很有用。举个例子，在高斯分布的参数估计中我们决定使用样本均值来估计参数$$\mu$$，此时样本均值本身也是一个随机变量，可以视作一个随机化的decision policy。我们很自然的一个想法就是求这样一个随机变量的期望并以此作为我们对$$\mu$$的估计。当然了，期望的本质是对分布进行加权平均，我们并不总能保证期望一定落在decision space中，为此需要对loss函数和决策空间都作出一定的限制。定义去随机化的过程如下：

**去随机化 Derandomization**: 给定一个随机化的决策规则$$\delta^*(x, A)$$， 如$$\mathcal{A} \in \mathbb{R}^d$$为凸集，损失函数$$L(\theta, a)$$为凸函数，那么我们可以得到deterministic的决策如下：
\$$
    \delta = \mathbb{E}_{a \sim \delta^*(x, A)}[a] < \infty
$$

且有：
\$$
L(\theta, \delta(X)) \leq L(\theta, \delta^*(x, A)) \quad \text{by Jensen's inequality}
$$

这表明在非特殊情况下，去随机化可以保证loss不会高于随机化之前的方案。

#### 充分统计量 Sufficient Statistics
关于Sufficient Statistics有很多的东西可以讲，以后有机会再补充，这里只简单谈两点：

* Sufficient Statistics是T(X)中的一个特例，它包含了参数估计所需的全部信息。
* 从降维的角度看，Sufficient Statistics可以视作一种数据降维的方法，将样本中与参数无关的信息去掉，只保留与参数估计有关的信息。

这两点性质令充分统计量变得很有用，试想一下，在实际使用中我们不再关心冗余的数据，而只提取出其中与参数有关的部分，这很大程度上可以使计算得到简化。一个原本将样本空间$$\mathcal{X}$$映射到决策空间$$\mathcal{D}$$的规则$$\delta$$可以如下转变为一个将统计量空间$$\mathcal{T}$$映射到决策空间$$\mathcal{D}$$的规则：
\$$
\delta^\ast(t, A) = \mathbb{E}_{x \mid t}\delta^\ast(x, A)
$$

上式的期望指的是对$$x \mid t$$这样一个条件随机量求期望，即我们关注的是$$x$$在给定充分统计量$$T = t$$之后的条件分布。

#### Rao-Blackwell Theorem

在背景补充的最后一部分，我们简要地介绍一下Rao-Blackwell Theorem。有了前面提到的充分统计量，很自然的一个想法是能不能利用充分统计量来帮助找到最优的决策。答案是肯定的，换言之，Rao-Blackwell定理指出，任何基于数据本身构造的决策，在加入Sufficient Statistics的信息后，改善后的决策都不会比原来的更差。用数学语言来描述如下：

给定任意为凸的决策空间和凸的loss函数，我们有：

\$$
\begin{align*}
R(\theta, \delta_{\text{RB}}(t)) &= \mathbb{E}_{t}[L(\theta, \delta_{\text{RB}}(t))]\\
& = \mathbb{E}_{t}[L(\theta, \mathbb{E}_{x \mid t}[\delta(x)])]\\
& \leq \mathbb{E}_{t}\left[\mathbb{E}_{x \mid t}[L(\theta, \delta(x))]\right]\\
&= R(\theta, \delta)
\end{align*}
$$

上面的结果给了我们几个重要的启发：

* 构造decision policy时应尽量选那些以Sufficient Statistics作为自变量的函数，这样可以保证不会存在一个Rao-Blackwellization后的policy比原方案更好。
* 当我们可以很轻松地找到一个合理的policy的时候，总可以用Sufficient Statistics来对其进行优化。如Plug-in方法中所使用的Empirical Statistics是非常暴力且有效的一类参数估计量，假定我们可以对原问题找出一个Sufficienct Statistics，那么就可以对Empirical方法的估计量进行改善。
* R-B方法改善过后的结果无法再次通过R-B方法改善。

以上对随机决策、充分统计量、R-B定理等话题的讨论内容为后文深入研究Bayes和Minimax方法提供了基础。
  
## Bayes Method
#### Bayes Rule
著名的贝叶斯法则此处就不赘述了，简单给出定义如下：
\$$
\pi(\theta \mid X) = \frac{\pi(\theta)f(X \mid \theta)}{f(X)} = \frac{\pi(\theta)f(X \mid \theta)}{\int \pi(X)f(X \mid \theta) d\theta} \propto \pi(\theta)f(X \mid \theta) 
$$
由于贝叶斯法则在statistical inference中非常重要，我们也套用其中的定义来称呼公式中的每一项：

* 先验Prior: $$\pi(\theta)$$，即我们对于natural state的先验知识，如用一个随机变量来刻画水的温度的话，那么一般情况下液态水的温度在0到100之间。
* 似然Likelihood：$$f(X \mid \theta)$$，即如果真实参数为$$\theta$$，则观察到样本为$$X$$的可能性。
* 后验Posterior：$$f(\theta \mid X)$$，即观察到数据$$X$$之后对参数$$\theta$$的认识发生的改变。 

值得注意的是，分布$$\mathcal{P}$$被称作**共轭于$$\mathcal{F}$$(Conjugate to $$\mathcal{F}$$)**，如果其先验为$$\mathcal{F}$$而其后验为$$\mathcal{P}$$。共轭分布可以显著地简化参数推断的计算，因此人们常常选择那些有良好共轭的分布作为先验。

#### Bayesian Decision Rule

如前文所述，Conditional Bayesian关注的是Bayesian Expected Loss：
\$$
    \rho(\pi^\ast, a) = \mathbb{E}_{\pi^\ast} [L(\theta, a)]
$$

该loss假定数据是给定且不变的，变化的是观察到样本$$X$$之后$$\theta$$的分布，利用Bayes rule，我们很容易可以对先验分布进行更新并得到后验分布，将后验分布代入得$$\pi^\ast = \pi(\theta \mid X)$$并求$$\rho$$的最小值可得Bayes Decision Rule (Bayes estimator)：
\$$
\delta_{\pi}(x) = \inf_{a \in \mathcal{A}} L(\pi(\theta \mid x), a)
$$

这时细心的读者可能会感觉到不对劲，Bayes estimator不是通常用来指Bayes Risk最小的estimator吗？那不应该是在Frequentist视角下对样本数据$$X$$求了期望的结果吗？为什么这里也叫Bayes estimator呢？事实上，我们马上可以看到，**在Conditional Bayesian视角下最小化Bayesian Expected Loss得到的结果与在Frequentist视角下最小化Bayes Risk得到的结果是一样的**。


给定Bayes Risk的定义：


\$$
    r(\pi, \delta) = \mathbb{E}_{\theta \sim \pi} R(\theta, \delta) = \mathbb{E}_{\theta \sim \pi}\left[\mathbb{E}_{x \sim f(X \mid \theta)}[L(\theta, \delta(x))]\right]
$$

我们有**Minimization Equivalence Theorem**，即最小化Bayes Expected Loss和最小化Bayes Risk等价。证明如下：

\$$
\begin{align*}
    r(\pi, \delta) &= \int_{\theta \in \Theta}\left[\int_{x \in \mathcal{X}} L(\theta, \delta(x))f(x \mid \theta)dx \right] \pi(\theta) d\theta\\
    &= \int_{\theta \in \Theta}\left[\int_{x \in \mathcal{X}} L(\theta, \delta(x))\pi(\theta \mid x)m(x)dx \right] d\theta && (f(x \mid \theta)\pi(\theta) = \pi(\theta \mid x)m(x))\\
    &= \int_{x \in \mathcal{X}}\left[\int_{\theta \in \Theta} L(\theta, \delta(x))\pi(\theta \mid x)d\theta \right] m(x)dx \\
    &= \int_{x \in \mathcal{X}}\rho(\pi(\theta \mid x), \delta(x)) m(x) dx
\end{align*}
$$

此时由于可变的只有$$\delta(x)$$，当选择Bayes estimator $$\delta_{\pi}(x)$$最小化$$\rho$$的时候，$$r$$也取到最小值，二者等价。两个流派从不同的出发点做出的最优选择，在Bayes rule下其实是统一的。

#### 实际计算

实际计算Bayes Estimator时有两种方法，一是直接代入原定义，将loss，likelihood和prior代入求解期望，并最小化Bayes risk，二是如果后验是一个很好求的分布的话，可以直接写出对后验和loss积分的形式，求解期望并最小化Bayes risk。通常我们会选择后者，且针对常见的两种loss后者有非常简洁的解，即：

* Square loss: $$L(\theta, a) = (\theta - a)^2$$，Bayes统计量即后验的mean。
* Absolute loss： $$L(\theta, a) = \lvert \theta - a \rvert$$，Bayes统计量即后验的median。
* Zero-one loss: $$L(\theta, a) = \mathbf{1}\{ \theta \neq a \}$$，Bayes统计量即后验的mode。

其他的loss也可以相应地求期望进行计算，计算时注意将期望中的项凑成一个分布的形式并使得积分为1，而不要暴力地去求积分（算力惊人可以忽略这句话）。总的来说，求解Bayes统计量在计算上是可行的。

## Minimax Method
#### 对抗视角

我们可以将Minimax的思想看做是统计学家与上帝的一场游戏，游戏的规则是统计学家要选择一个行为使得loss最小，而上帝要做的是改变自然参数$$\theta$$使得loss尽可能地大。显而易见的是先做出决策的一方有优势，为此我们可以针对双方谁先行分别找到游戏的最优解如下：

\$$
\begin{align*}
    \overline{V} &= \inf_{a \in \mathcal{A}} \sup_{\theta \in \Theta}R(\theta, a) && \text{上帝先选}\theta\\
    \underline{V} &= \sup_{\theta \in \Theta}  \inf_{a \in \mathcal{A}} R(\theta, a) && \text{统计学家先选}a\\
    \overline{V} &\geq \underline{V} && \text{上帝先选时有优势，因为目的是最大化loss}
\end{align*}
$$

当然了Bayesian们会说怎么能认为$$\theta$$就是一个固定的值呢，它也可能是一个分布嘛。我们可以将上帝一方的行动扩展为$$\sup_{\pi \in \Pi} R(\pi, a)$$。对应地我们也可以将统计学家的行为扩展为选择一个deterministic或者是random的行动规则，只要将$$a$$换成$$\delta$$即可，这些都不影响下面的讨论。根据这样的设定，我们会相应得到Minimax和maximum的定义如下：

如果对任意$$\theta \in \Theta$$和任意$$a \in \mathcal{A}$$都有$$R(\theta, \delta_0) \leq R(\pi_0, a)$$，那么$$\delta_0$$是Minimax（即统计学家后手的选择），$$\pi_0$$是Maximum（上帝先手的选择）。这会给我们计算Minimax提供帮助。

#### Minimax Rule

从上面我们可以更清楚地了解到Minimax是一种保守主义的哲学，上帝总是先行并且总会找到最不利于统计学家的参数。为了做出应对，我们必须考虑最坏情况下的loss，但这在实际中往往是不可行的，因为$$\theta$$的真实取值和分布是未知的。这就要求我们必须用其他的方法来辅助求解Minimax decision。一个常见的方法是利用Bayes的equalizer rule：

**Equalizer Rule**： 如果$$\delta_{0}$$是基于先验$$\pi$$的Bayes decision，且$$\pi$$是对于$$\delta_0$$而言的最不利先验(Least favorable prior)，即$$R(\delta_0, \theta) \leq R(\delta_0, \pi)$$，那么$$\delta_0$$也是Minimax。

>证明： 如果$$\delta_0$$不是Minimax，则另有Minimax $$\delta_{M}$$ 满足 $$\sup_{\theta}R(\delta_M, \theta) < \sup_{\theta}R(\delta_0, \theta)$$，我们可以构造反例如下
\$$
    r(\pi, \delta_M) \leq \sup_{\theta} R(\delta_M, \theta) \leq \sup_{\theta}R(\delta_0, \theta) \leq r(\pi, \delta_0)
$$

 第一个不等式来自于平均loss必定小于或等于最大loss，第二个不等式来自于$$\delta_M$$才是Minimax的假设，第三个不等式来自于$$\pi$$是最不利先验。由此可见存在$$\delta_M$$使得其Bayes risk比$$\delta_0$$更小，违背了$$\delta_0$$是Bayes decision的假定。因此，$$\delta_0$$必定是Minimax。

上面的证明中有一个有趣的点，我们左边用了risk对某个分布的期望应该小于等于其最大值的结论，然而右边的最不利先验却显示risk的最大值也小于最不利先验下的Bayes risk。由此我们可以引出一个更清晰的表述：

如果Bayes risk取到了$$\delta_0$$下的maximum risk，即
\$$
    r(\pi, \delta_0) = \sup_{\theta}R(\delta_0, \theta)
$$
那么：

* $$\pi$$是最不利先验。
* $$\delta_0$$是Minimax。
* $$\delta_0$$是唯一的Bayes的话，那么它也是唯一的Minimax。
* 如果对于一个Bayes decision $$\delta$$有$$R(\delta_0, \theta)$$是常数，即risk与$$\theta$$无关，那么此时Bayes与Minimax等价。

这也是计算Minimax一个较为有效的方法，但对比起Bayes，Minimax总体而言更难找到最优解。

#### 其他与Minimax有联系的方法
除了Bayes和Minimax，我们之前还提到过一些其他的方法，这里简要介绍两个和minimax有关联的方法。

* MLE: Maximum Likelihood Method可以看作是近似版本的Minimax，或者说是渐进意义上的Minimax。此处不作过多的讨论，仅以square loss 下 $$ R = bias^2 + var$$为例。当$$n$$变大而$$\theta$$固定时，可以认为loss近似等于var，此时对于MLE估计量有$$var(\delta_{MLE}) \approx 1 / nI(\theta)$$，可以看出随着$$n$$变大$$\delta_{MLE}$$的risk可以小于任何值，由此推断它是渐进的minimax统计量。
* Minimax Regret：同样是minimax的思路，只是将关心的量由risk换成了regret，即当前方案与最优方案的risk差值。


## 总结
至此我们已经对Decision Theory的背景有了一定的了解，后面我们会看到这里的setting是如何应用到machine learning当中的。Machine Learning的理论基础就在于**可学习性**，即一个算法在观察到足够的数据之后是否能够将我们关心的risk降低到某个特定的界之内。从这个角度来看，以下的几个问题都是值得深入思考的：

* 我们应该关心什么risk的界？或者说，什么样的界对我们设计一个算法是有指导意义的。
* 这个界是确切的吗？还是说我们只能保证它渐进意义上有一定概率可以满足。
* 什么条件下这样的界存在？对模型有要求吗？对样本有要求吗？

在之后的文章中，我们会深入探讨上面的这些问题。在正式进入可学习性和Empirical Risk Minimization的话题之前，我们还将补充几个部分的内容，包括常用的tail bounds、常用的分布及其关系、以及常用的证明技巧。

## Reference
1. James O.Berger, *Statistical Decision Theory and Bayesian Analysis*, 1980.
