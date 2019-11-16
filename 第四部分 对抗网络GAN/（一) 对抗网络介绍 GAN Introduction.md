# 对抗网络介绍 GAN Introduction

## 1. 背景介绍

对抗网络GAN的全称，Generative Adversarial Network (GAN) [1]是由机器学习大神，深度学习(花书)的作者lan J. Goodfellow在2014年提出。

机器学习泰斗，CNN之父，在他的twitter中如此评价GAN：
> 这是十年中，机器学习领域最伟大的算法之一。堪比于他自己在1990年提出的卷积神经网络。
（评论内容来自：https://www.quora.com/What-are-some-recent-and-potentially-upcoming-breakthroughs-in-deep-learning）
![1-1-1](https://raw.githubusercontent.com/muyangren907/Deep_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/1-1-1.png)

- GAN 的种类
    
    目前已经有100多种GAN的变形。
    ![1-1-2](https://raw.githubusercontent.com/muyangren907/Deep_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/1-1-2.png)
    （参考https://github.com/hindupuravinash/the-gan-zoo）
- GAN 的火爆程度

    GAN自从2104年提出后，就成了学术界和工业界的明星。我们从ICASSP 会议提交的论文就可以一看端倪。 GAN和强化学习，越来越受到大家的关注。
    ![1-1-3](https://raw.githubusercontent.com/muyangren907/Deep_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/1-1-3.png)

## 2. GAN的基本想法

我们从基本的输入(向量)，通过由神经网络组成的生成器，生成结构化的高维数据。例如:图片, 句子
![1-2-1](https://raw.githubusercontent.com/muyangren907/Deep_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/1-2-1.png)

但是生成是目的，生成的过程非常曲折，就像人生一样，成功的过程一定要经历挫折。

GAN生成的过程中遇到的”挫折“就是 `评价器”Discriminator“`，Discriminator也是由神经网络实现的。

![1-2-2](https://raw.githubusercontent.com/muyangren907/Deep_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/1-2-2.png)

生成器在生成的过程中被评价器矫正，评价器也是在不断进化，两者之间构成了相辅相成的对抗网络。

### 2.1 基本思路

我们举一个例子，枯叶蝶蝶和老鹰的故事。

枯叶蝶本是一只普通的蝴蝶。
老鹰本是一只小鸟。

刚开始的时候，小鸟很容易发现蝴蝶的藏身之处，蝴蝶家族伤亡惨重。 蝴蝶家族紧急召开会议，讨论对策。他们在讨论中发现，是由于自己的服饰太过鲜艳，导致总是被小鸟发现。 于是乎，他们把服饰颜色低调处理了。发现蝴蝶的伤亡开始减少。

小鸟家族慢慢发现，蝴蝶变少了，食物开始变得稀缺。小鸟家族召开紧急会议，会议经过讨论，发现蝴蝶是伪装了，小鸟需要升级自己发现蝴蝶的能力，于是他们升级到了大鸟。

故事又开始了循环，蝴蝶家族伤亡开始增加。他们开会决定将自己的衣服颜色变成树叶，这样，蝴蝶的伤亡就开始减少了。

小鸟家族认识到问题的严重性，他们一直决定，进化成老鹰，不然无法存活。
…
自然界的进化就是这样周而复始。

![1-2-3](https://raw.githubusercontent.com/muyangren907/Deep_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/1-2-3.png)

## 3 GAN的算法过程

### 算法简单描述

1. 首先初始化Generator和Discrimintor

    ![1-3-1](https://raw.githubusercontent.com/muyangren907/Deep_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/1-3-1.jpg)

2. 固定住D, 先update G 目的是骗过D
3. 然后固定住G，update D 目的是识别骗局
4. 返回2

本专栏图片、公式很多来自台湾大学李宏毅老师的深度学习课程，在这里，感谢这些经典课程，向李老师致敬！

## 参考资料 References

1. Ian Goodfellow et al. (2014). Generative Adversarial Networks.