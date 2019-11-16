# 对抗网络 GAN背后的理论

对抗网络GAN是由生成器Generator最终生成图片、文本等结构化数据。

生成器能生成结构化数据的原理是什么呢？

简而言之：就是让生成器Generator找到目标图片、文本的信息的概率密度函数。通过概率密度函数 $$P_{data}(x)$$ 生成数据。

![2-0-1](https://raw.githubusercontent.com/muyangren907/Deep_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/2-0-1.png)
