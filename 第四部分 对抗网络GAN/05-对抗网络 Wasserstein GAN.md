# 对抗网络 Wasserstein GAN

## 1. 传统Traditional GAN的问题

### 1. JS 距离衡量存在问题

在大多数例子中$P_{G}$和$P_{data}$中间是不重叠的

![5-1-1](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/5-1-1.jpg)

生成器的目的就是要让$P_{G}$接近$P_{data}$,那么GAN中的JS divergence 无法有一个过渡的过程，在让$P_{G}$接近$P_{data}$接近之前，他们的距离适中是一个固定的值

![5-1-2](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/5-1-2.jpg)

我们需要一个将$P_{G}$和$P_{data}$距离衡量的一种方式

## 2. “挖掘机距离” Earth Mover’s Distance

这里的“挖掘机”可不是蓝翔哦，不过在概率密度分布上，和真实的挖掘机是一样的，“铲土”！

![5-2-1](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/5-2-1.jpg)

如何铲土呢：

我们看P和Q两个概率分布：（图片来自 https://vincentherrmann.github.io/blog/wasserstein）
P通过铲土的形式（方式1），将概率密度函数的图形，向Q靠拢。

![5-2-2](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/5-2-2.jpg)

但是铲土的方式由很多，这种放方式也行( 方式2）

![5-2-3](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/5-2-3.jpg)

它们之间的区别是，方式1的运土的距离比方式2的短
我们的目标就是要找到，运土过程中的最短距离：

![5-2-4](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/5-2-4.jpg)

- 横轴一行加起来就是P对应的概率分布（土的多少）
- 纵轴一列加起来就是Q对应的概率分布（土的多少）

我们有:

$$moving\ plan\ \gamma$$

我们需要找到，一个最佳plan

$$
Earth\ Mover's\ Distance:\\
W(P,Q)=\underset{\color{blue} {The\ best\ plan}}{\underset{\gamma \in \Pi}{\min }B(\gamma)}
$$

![5-2-5](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/5-2-5.jpg)

找到最佳的迁移方案往往需要遍历所有的方案，这种办法显然是效率最低的。

## 3. WGAN

WGAN 就是用 Wasserstein distance 来衡量$P_{G}$和$P_{data}$
衡量的方式是：

![5-3-1](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/5-3-1.png)

D 属于 1-Lipschitz 函数，其实可以理解为D是平滑的。

![5-3-2](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/5-3-2.png)

由于为了让V(G,D)最大，V的公式让real最大，让generated最小，但是需要限制。如果generated的结果接近无权小，V的虽然最大，但是没有达到GAN的最初目的。

![5-3-3](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/5-3-3.png)

所以有`Weight Clipping`的技巧（Martin Arjovsky, et al., arXiv, 2017）：

强制性让所有的参数$w\in [-c,c]$
if w > c, w = c;
if w < -c, w = -c

## 4. Improved WGAN（WGAN-GP）

一种和Lipschitz 函数等价的计算方式：在x所有区域内，D(x)的导数都小于或等于1

$$D \in 1-Lipschitz \Leftrightarrow \left \| \triangledown _x D(x) \right \| \leqslant 1\ for\ all\ x$$

但是x的范围比较大，在整个区域内进行限制有点困难，那么我们就找到一个区域叫做penalty，在penalty中进行限制。

![5-4-1](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/5-4-1.png)

- penalty 是怎么得到的呢？
$P_{G}$和$P_{data}$之间的连线的中点

![5-4-2](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/5-4-2.png)

## 5. WGAN 整体算法

![5-5-1](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/5-5-1.png)
