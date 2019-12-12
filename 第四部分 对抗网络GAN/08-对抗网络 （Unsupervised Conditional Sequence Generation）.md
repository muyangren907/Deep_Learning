# 对抗网络 （Unsupervised Conditional Sequence Generation）

现实世界中，非监督数据占据绝大多数，GAN可以利用非监督数据进行结构化数据生成。

## 1. 文本转换 Text Transfer

我们经常使用GAN做图片生成，其实除了图片这样的结构化数据外，文本转换也是GAN可以做到的。

![8-1-1](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/8-1-1.png)

我们可以使用Cycle GAN的技术进行文本的“情绪”转换

![8-1-2](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/8-1-2.png)

由于文本是离散的数据，不可微分和使用BP算法，我们将文本one-hot 内容转换成wordembedding 连续的数据。

Shen, et al., NIPS, 2017 使用映射到空间信息的方法。

![8-1-3](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/8-1-3.png)

## 2. 内容摘要 Summarization

在自然语言处理应用中内容摘要 Summarization一直是常见应用之一。
我们先看一下常规的Seq2Seq在监督学习下的摘要生成模式

![8-2-1](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/8-2-1.png)

GAN可以进行非监督学习完成内容摘要是用的是我们讲到过的Unsupervised Conditional Generation。

![8-2-2](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/8-2-2.png)

整个GAN生成摘要的过程是这样的：

![8-2-3](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/8-2-3.png)

## 3. 非监督翻译 Unsupervised Machine Translation

众所周知，翻译是NLP中最重要的一环。 基于监督学习的机器翻译已经达到了很好的效果。

但是非监督学习的机器翻译呢？

![8-3-1](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/8-3-1.jpg)

我们还是是Condition Domain的方式用GAN来解决问题。可以参考论文：

[Alexis Conneau, et al., ICLR, 2018 和 Guillaume Lample, et al., ICLR, 2018]

本专栏图片、公式很多来自台湾大学李宏毅老师的深度学习课程,在这里，感谢这些经典课程，向李老师致敬！

