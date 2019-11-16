# 对抗网络 GAN背后的理论

对抗网络GAN是由生成器Generator最终生成图片、文本等结构化数据。

生成器能生成结构化数据的原理是什么呢？

简而言之：就是让生成器Generator找到目标图片、文本的信息的概率密度函数。通过概率密度函数 $$P_{data}(x)$$ 生成数据。

![2-0-1](https://raw.githubusercontent.com/muyangren907/Deep_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/2-0-1.png)

## 1. 最大似然估计近似

任何复杂的问题都可以拆解为简单的问题。 在机器学习中最大似然估计就是基本问题。

我们再对抗网络中使用最大似然估计：

1. 我们首先获得目标数据的概率密度函数 $$P_{data}(x)$$

2. 我们设定Generator的概率密度函数为 $$P_{G}(x; \theta )$$

找到 $$\theta$$能够让 $$P\_{G}(x;\theta)$$ 越来越接近 $$P_{data}(x)$$

举例： 假设$$P\_{G}(x; \theta )$$属于高斯分布，$$\theta$$ 就代表高斯分布的参数均值`mean`和方差`Variance`。
