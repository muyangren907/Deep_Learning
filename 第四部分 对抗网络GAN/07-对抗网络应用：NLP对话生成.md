# 对抗网络应用：NLP对话生成

在自然语言处理NLP中，对抗网络也有它的应用空间。我们从NLP最经典的应用：
Dialog Generation 对话生成开始。

传统的对话生成使用的是Maxlikelihood的思路，就是使对话生成的每一个词的概率的乘积最大。 但是效果不是很好（词库量、训练量都太大）

我们先看看用Reinforcement learning的方法

## 1.Reinforcement learning（human feedback)

在Reinforcement learning 中人类给予每组对话一个反馈（Reward）

![7-1-1](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/7-1-1.png)

我们的目标就是让`Reward`最大

![7-1-2](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/7-1-2.jpg)


我们讲过`Policy Gradient`，使用PG，达成我们的目标
$$
\theta ^{new} \leftarrow \theta ^{old} + \eta \triangledown \overline{R}_{\theta ^{old}}\\
\triangledown \overline{R}_\theta \approx \frac{1}{N} \sum_{i=1}^{N} R(h^i,x^i) \triangledown \log P_\theta (x^i\mid h^i)
$$

整体的算法是：

![7-1-3](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/7-1-3.jpg)

RL与Maxlikelihood的区别是：

![7-1-4](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/7-1-4.jpg)

从上面可以看出，虽然考虑问题的角度不同，归纳到数学公式上差距比想象中小很多。

但是RL 处理NLP对话生成问题的时候，存在一个很大的问题： Reward是人给出的，在训练中人参与的比重很大，而且所有的Reward（上千万次迭代），均由人给出，不现实。

## 2.GAN（像AlphaGo一样训练）

我们需要再建立一个talk agent ，两个agent互相对话，相互成长。

![7-2-1](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/7-2-1.jpg)

2017年 NLP顶会EMNLP上 Li et.al 提出, 使用Condition GAN的模型生成对话

![7-2-2](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/7-2-2.jpg)

具体的算法是：

![7-2-3](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/7-2-3.jpg)

是不是觉得顶会的算法，只要是掌握了基础算法原理，也是能看得懂了哦。

但是这个算法是不是就这样结束了呢。远远没有！

和CV图片处理不同，NLP是离散的数据，离散的数据抽样的过程中无法微分。所以我们有如下技巧。

## 3.GAN在NLP中的训练技巧

### (1) Gumbel-softmax

来自Matt J. Kusner, et al, arXiv, 2016，

### (2) 鉴别器的输入连续化 Continuous Input for Discriminator

我们将输Discriminator的离散输入用连续概率替代，这样，可以使输入连续化，结果变得可微分。

![7-3-1](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/7-3-1.jpg)

但是，这样会存在什么问题？

![7-3-2](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/7-3-2.jpg)

WGAN可以帮助消除生成的问题。

### (3) 强化学习RL在NLP中的进阶

在强化学习中，我们一般使用人在给Reward，我们能不能让机器来给Reward呢。

我们使用强化学习+对抗网络的思路（RL+GAN）, 在G步骤更新参数，生成对话，在D步骤，训练能够判断真假（输出Reward的 鉴别器D）

![7-3-3](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/7-3-3.jpg)

但是，NLP的特点是生成序列化的离散数据，一句话的Reward低，并不代表里面的单词的Reward都低

举例来讲： 

> 问题： How old are you？
回答1： I dont know Reward -5
回答2： I am 16. Reward +10

两个回答中都有单词 I，所以我们需要更加深入的去考虑Reward的设计

![7-3-4](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/7-3-4.jpg)

有两个方法，供大家参考：

> Method 1. Monte Carlo (MC) Search [Yu, et al., AAAI, 2017]
Method 2. Discriminator For Partially Decoded Sequences [Li, et al., EMNLP, 2017]

本专栏图片、公式很多来自台湾大学李宏毅老师的深度学习课程,在这里，感谢这些经典课程，向李老师致敬！