# 对抗网络 （RankGAN + GAN家族总结）

## １.RankGAN

这个模型不一样的地方在于,将原来的Discriminator从二分类模型变为一个排序模型,也就是一个Leaning to Rank的问题.所以模型的两个神经网络为:一个generator和一个ranker.

![9-1-1](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/9-1-1.jpg)

其中G为生成的样本,H为抽样的真实的样本,U为抽样的真实的样本.(没错，这里是两个都是真实样本)

基本思想就是分别计算G和U,H和U的相似度,

正常来说,H和U应该更相似.因为都是来自真实样本的分布.

这里H和U的采样方式论文中没有提,感觉构造的时候混入一些和G更相似的U和H可以有更好的训练效果.

计算rank得分的时候,是通过计算多组样本得到期望得分,具体为:

$$
\alpha (s\mid u)= \cos (y_s,y_u)=\frac{y_s \cdot y_u}{\left \| y_s\right \|\left \| y_u\right \|}\\
P(s\mid u,C)=\frac{\exp (\gamma \alpha (s\mid u))}{\sum _{{s}'\in {C}'}\exp (\gamma \alpha (s'\mid u))}\\
\log R_\phi (s\mid U,C)= \underset{u\in U}{E} \log [P(s\mid u,C)]
$$

## 2.GAN家族

![9-2-1](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/9-2-1.jpg)

GAN可以说是近年来，在人工智能领域最重要的算法贡献！

其实GAN也是符合自然规律的，从进化论甚至中国的古代阴阳八卦中都能找到GAN算法的影子。

大道至简、大道相同，自然之道，相生相克！

很荣幸能够让大家看到我的文章，希望大家一起学习，一起进步！

本专栏图片、公式很多来自台湾大学李宏毅老师的深度学习课程,在这里，感谢这些经典课程，向李老师致敬！
