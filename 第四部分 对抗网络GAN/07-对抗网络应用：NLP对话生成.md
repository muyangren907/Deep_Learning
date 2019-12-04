# 对抗网络应用：NLP对话生成

在自然语言处理NLP中，对抗网络也有它的应用空间。我们从NLP最经典的应用：
Dialog Generation 对话生成开始。

传统的对话生成使用的是Maxlikelihood的思路，就是使对话生成的每一个词的概率的乘积最大。 但是效果不是很好（词库量、训练量都太大）

我们先看看用Reinforcement learning的方法