# 对抗网络 GAN背后的理论

对抗网络GAN是由生成器Generator最终生成图片、文本等结构化数据。

生成器能生成结构化数据的原理是什么呢？

简而言之：就是让生成器Generator找到目标图片、文本的信息的概率密度函数。通过概率密度函数 $P_{data}(x)$ 生成数据。

![2-0-1](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/2-0-1.png)

## 1. 最大似然估计近似

任何复杂的问题都可以拆解为简单的问题。 在机器学习中最大似然估计就是基本问题。

我们再对抗网络中使用最大似然估计：

1. 我们首先获得目标数据的概率密度函数 $P_{data}(x)$

2. 我们设定Generator的概率密度函数为 $P_{G}(x; \theta )$

- 找到 $\theta$能够让 $P_{G}(x;\theta)$越来越接近 $P_{data}(x)$

- 举例： 假设$P\_{G}(x; \theta )$属于高斯分布，$\theta$ 就代表高斯分布的参数均值`mean`和方差`variance`。

具体的做法是：

- 抽取sample $x_{1},x_{2},...,x_m\ from\ P_{data}(x)$

- 计算最大似然函数 $$L = \prod_{i=1}^{m}P_{G}(x^{i};\theta )$$

目的是让似然函数的结果最大，我们就找到了 $\theta$

计算最大似然值可以推导如下：
$$
\begin{equation}\nonumber\\
\begin{split}\\
\theta ^{*}&=\arg\underset{\theta }{\max}\prod_{i=1}^{m}P_{G}(x^{i};\theta )\\
&=\arg\underset{\theta }{\max}\log\prod_{i=1}^{m}P_{G}(x^{i};\theta )\\
&=\arg\underset{\theta }{\max}\sum_{i=1}^{m}\log P_{G}(x^{i};\theta )\ \ \ \ \color{red} {\{x^{1},x^{2},...,x^{3}\}\ from\ P_{data}(x)}\\
&\approx\arg\underset{\theta }{\max}E_{x\sim P_{data}}[\log P_{G}(x;\theta )]\\
&=\arg\underset{\theta }{\max}\int_{x}^{ }P_{data}(x)\log P_{G}(x;\theta )\text{d}x-\int_{x}^{ }P_{data}(x)\log P_{data}(x)\text{d}x
\end{split}
\end{equation}
$$

注：上式中，有半部分是一个固定的值，求最小值的时候，减去固定的值，对参数$\theta$ 的结果不影响。（目的是凑出KL距离公式）

实际上划归为计算$P_{G}(x;\theta )$和$P_{data}(x)$KL距离最小值的问题：
$$\arg\underset{\theta }{\min}KL(P_{data}\parallel P_{G})$$

## 2. 生成器Generator

Generator G就是一个神经网络，它定义了生成器的$P_{G}(x;\theta )$
G的目标是：找到$P_{G}(x;\theta )$和$P_{data}(x)$之间的最小差距
![2-2-1](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/2-2-1.png)

$$G^{*} = \arg \underset{G}{\min} \color{red}  { Div(P_{G},P_{data})}$$

## 3. 鉴别器 Discriminator

鉴别器Discriminator D 就是需要鉴别那些数据是来自Generator G的$P_{G}(x;\theta )$，那些数据是来自真实数据$P_{data}(x)$

D的目标是：更可能的能区分真实数据和生成数据，做好一个质量检查员，而且还需要在工作中不断学习。

举例：
我们有数据
![2-3-1](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/2-3-1.png)

鉴别器对于数据的鉴别难度，取决于数据的概率分布的差距：

![2-3-2](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/2-3-2.png)

用公式表示鉴别器的目标（G是固定的）：

$$D^{*}=\arg \color{red} {\underset{D}{\max} V(D,G)}$$

其中V的值（G是固定的）推导如下：
$$
\begin{equation}\nonumber\\
\begin{split}\\
V&=E_{x\sim P_{data}}[\log D(x)]+E_{x\sim P_{G}}[\log (1-D(x))]\\
&=\int_{x}^{}P_{data}(x)\log D(x)\text{d}x+\int_{x}^{}P_{G}(x)\log (1-D(x))\text{d}x\\
&=\int_{x}^{}[P_{data}(x)\log D(x)+P_{G}(x)\log (1-D(x))]\text{d}x\\
&\color{blue} {Assume\ that\ D(x)\ can\ be\ any\ function}\\
\end{split}
\end{equation}
$$

我们需要找到一个最好的鉴别器D
就是最大化：
$$P_{data}(x)\log D(x)+P_{G}(x)\log (1-D(x))$$

## 4. 算法的详细过程

### 4.1 数学推导

算法的核心是：

$$G^{*} = \arg \underset{G}{\min} \color{red} {\underset{D}{\max}V(G,D)}$$

这个公式看上去一头雾水，我们慢慢拆解它。
首先，我们去看：

$$D^{*}=\arg \color{red} {\underset{D}{\max} V(D,G)}$$

我们需要挑出最好的鉴别器：让V最大。
那么对于V我们知道：

$$V=E_{x\sim P_{data}}[\log D(x)]+E_{x\sim P_{G}}[\log (1-D(x))]$$

我们把它转换为求最大值的普通数学问题（大一或者高三知识就可以求解）

$$\underset {\color{blue} a}{P_{data}(x)}\underset {\color{blue} D}{\log D(x)}+\underset {\color{blue} b}{P_{G}(x)}\underset {\color{blue} D}{\log (1-D(x))}$$

其中a,b 都是固定值，求最大值D，我们推导一下：

![2-4-1](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/2-4-1.png)

求出最优的$D^{*}$我们把它代回得到：

![2-4-2](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/2-4-2.png)

注明：其中最后的结果中有JS距离。和KL一样，JS距离也是衡量两种概率分布的工具。

$$
JSD(P\parallel Q)=\frac{1}{2}D(P\parallel M)+\frac{1}{2}D(Q\parallel M)\\
M=\frac{1}{2}(P+Q)
$$

求解完D后我们再看下，最小化$\underset{D}{\max} V(G,D)$,是什么意思。我们假设存在三个G：G1,G2,G3, 每一个G都有一个$\underset{D}{\max} V(G,D)$

![2-4-3](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/2-4-3.png)

很显然，算法最终的结果是选择G3。

### 4.2 算法过程

![2-4-4](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/2-4-4.png)

算法过程看起来比较简单，但是实际操作中遇到很多很问题。GAN是比较难以“驯服”的。

实际操作：

- 给定G，计算$\underset{D}{\max} V(G,D)$
抽取sample $x_{1},x_{2},...,x_{m}\ from\ P_{data}(x)$，抽取sample $x_{1}^{'},x_{2}^{'},...,x_{m}^{'}\ from\ P_{G}(x)$，计算最大值。

![2-4-5](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/2-4-5.png)

D实际上是我们学过的最简单的二元分类器。

![2-4-6](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/2-4-6.png)

我们需要找到一个最好的D。

- 给定D，找到能让$P_{data}(x)$和$P_{G}(x)$分布距离最小的G。

整体算法过程：

![2-4-7](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/2-4-7.png)

注明：GAN的object 函数很难训练，刚开始的变化比较小。

$$V=E_{x\sim P_{data}}[\log D(x)]+E_{x\sim P_{G}}[\log (1-D(x))]$$

其中给定D的情况下，V的左半部分是固定值，我们可以不用考虑。

![2-4-8](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/2-4-8.png)

实操中：V可以写作

$$V=E_{x\sim P_{G}}[-\log (D(x))]$$

这样，函数图像变为：

![2-4-9](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/2-4-9.png)

这样的函数，就相对好train许多。

本专栏图片、公式很多来自台湾大学李宏毅老师的深度学习课程,在这里，感谢这些经典课程，向李老师致敬！