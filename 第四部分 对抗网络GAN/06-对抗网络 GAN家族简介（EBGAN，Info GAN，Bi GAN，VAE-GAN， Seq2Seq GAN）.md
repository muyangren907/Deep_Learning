# 对抗网络 GAN家族简介（EBGAN，Info GAN，Bi GAN，VAE-GAN， Seq2Seq GAN）

## 1. Energy-based GAN

EBGAN 其实就是 鉴别器Discriminator，提前用AutoEncoder训练过的GAN

![6-1-1](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/6-1-1.png)

EBGAN的优势：

1. GAN的训练比较麻烦，而AutoEncoder相对简单，EBGAN可以减少训练时间
2. EBGAN的 discriminator 只会给有限的空间的值大的估计

![6-1-2](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/6-1-2.png)

## 2. Info GAN

我们在生成图片或者内容的时候，肯定有这样的想法，生成的内容可以轻微调整吗。比如，数字的颜色，数字的大小，数字的形状，或者人的头发的颜色，皮肤的颜色。

简而言之，我们想“驾驭”GAN。

但是事与愿违，我们在参数集z的轻微调整，可能会有意想不到的影响。

![6-2-1](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/6-2-1.jpg)

Info GAN 就是帮助我们找到GAN的生成器的生成规律

![6-2-2](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/6-2-2.jpg)

Info GAN的架构刚开始可能让人感到一头雾水，我们慢慢拆开看：

1. 我们把Z 拆成 C和Z’
2. C的内容我们用一个AutoEncoder（图片黄色部分）进行Decoder能够反向生成 X
3. X的内容和常规的GAN一样，有个鉴别器Discriminator 进行鉴别

区分出C的意义就是C能够代表图片的特征。

论文 https://arxiv.org/abs/1606.03657 中，我们得到，修改C中的内容，可以得到不同的MNIST效果图：

![6-2-3](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/6-2-3.jpg)

## 3. Bi GAN

一般的，我们将Encoder和Decoder 是相连的，即Encoder的信息输入到Decoder中，但是BiGAN就将他们隔离开来，分别输入鉴别器Discriminator 来判断输入的image 和z 的pair 是来自encoder还是decoder。

![6-3-1](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/6-3-1.jpg)

具体的算法如下：

- Initialize encoder En, decoder De,discriminator Dis
- In each iteration:

    - Sample M images $x^1,x^2,\cdots ,x^M$ from database
    - Generate M codes $\widetilde{z}^1,\widetilde{z}^2,\cdots ,\widetilde{z}^M$ from encoder
    
        - $\widetilde{z}^i=En(x^i)$
        
    - Sample M codes $z^1,z^2,\cdots ,z^M$ from prior P(z)
    - Generate M codes $\widetilde{x}^1,\widetilde{x}^2,\cdots ,\widetilde{x}^M$ from decoder
    
        - $\widetilde{x}^i=De(z^i)$
    
    - Update Dis to increase $Dis(x^i,\widetilde{z}^i)$,decrease $Dis(\widetilde{x}^i,z^i)$
    - Update En and De to decrease $Dis(x^i,\widetilde{z}^i)$,increase $Dis(\widetilde{x}^i,z^i)$
    
## 4. 对抗领域训练 Domain adversarial training

我们直觉上认为，GAN在训练完黑白Minist后，生成器的特征feature和彩色Minist的生成器的特征应该分布接近，但是实际情况并非如此。

![6-4-1](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/6-4-1.jpg)

所以我们需要加入领域分类模块

![6-4-2](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/6-4-2.jpg)
        
## 5. VAE-GAN

我们之前介绍过VAE，通过VAE可以画图，但是加上GAN后，“双剑合璧”，威力就会更大。

![6-5-1](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/6-5-1.jpg)

## 6. Seq2Seq-GAN

在传统的语音识别中：

![6-6-1](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/6-6-1.jpg)

我们可以加入语言encoder和说话者encoder

![6-6-2](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/6-6-2.jpg)

使用GAN来区分说话者

![6-6-3](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/6-6-3.jpg)

本专栏图片、公式很多来自台湾大学李宏毅老师的深度学习课程,在这里，感谢这些经典课程，向李老师致敬！
