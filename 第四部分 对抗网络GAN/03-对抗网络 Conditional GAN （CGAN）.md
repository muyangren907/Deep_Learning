# 对抗网络 Conditional GAN （CGAN）

首先，我们先举一个例子，文本转换成图片:Text to Image.
有过机器学习经验的同学肯定会想到，用监督学习就可以做到：

![3-0-1](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/3-0-1.png)

不断地用监督的数据进行训练。

但是，真实世界中，非监督数据远远大于带标记的监督数据。 那我们考虑，如何利用GAN，来生成图片呢。

Scott Reed, et al 在2016年ICML 提出Conditional GAN 的概念。

## 1. Generator 生成器

与传统的GAN的相比，生成器中除了输入数据分布z带，增加了要生成数据的条件。目的是要告诉生成器，我要生成什么。

![3-1-1](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/3-1-1.png)

大道至简。 这个思路其实相对来讲，比较符合信息论的基本原理，一个系统给以的信息越多，它的结果会更加准确。 相对于传统的GAN， Condition GAN就是有的放矢的去生成结构化数据，而不是仅仅很盲目的开始尝试。

## 2. Discriminator 鉴别器

传统的鉴别器是鉴别生成数据和真实数据之间的区别，而 Condition GAN
的鉴别器不同，它的数据有Condition。

![3-2-1](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/3-2-1.png)

还可以结合传统的鉴别器的模式：

![3-2-2](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/3-2-2.png)

比如我想生成动漫头像，头发和眼睛的颜色我可以规定哦

![3-2-3](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/3-2-3.png)

## 3. Condition GAN 的应用举例

### 1. Stack GAN

我们以文字生成图片为例，输入的是一段文字，输出的是一张文字所描述的图片。

![3-3-1](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/3-3-1.jpg)

看着这个图，大家可能比较晕，但是复杂模型都是简单模型的叠加。我们一步一步看：

 1. 首先输入一段文字，经过整合，输入生成器$\color {blue} {G_{0}}$
 2. G1生成的图片是64*64的小图片，然后用鉴别器 $\color {blue} {D_{0}}$ 进行鉴别
 3. G1生成的图片和文字内容一起输入到生成器 $\color {purple} {G}$
 4. G2生成的图片是256*256的图片，然后用鉴别器 $\color {purple} {D}$ 进行鉴别

从简单到复杂，不断生成，不断鉴别。

### 2. Image to Image

我们以草图生成实际图片为例，看一看图片转换成图片的方法。

使用传统监督学习方法的思路：

![3-3-2](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/3-3-2.jpg)

但是效果不佳：

![3-3-3](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/3-3-3.jpg)


于是，我们用GAN来解决问题，

![3-3-4](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/3-3-4.jpg)

实验结果：

![3-3-5](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/3-3-5.jpg)

实验结果比监督学习好，但是图片中会有很多奇怪的东西：图片左上角有烟囱和窗户的“综合体”

![3-3-6](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/3-3-6.jpg)

于是我们使用Autoencoder的思路，加入图片限制，结果入下（结果更加真实）：

![3-3-7](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/3-3-7.jpg)

### 3. Patch GAN

由于一张图片太大，直接GAN的速度很慢，我们可以并行选取patch进行GAN

![3-3-8](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/3-3-8.jpg)

### 4. 语音处理Speech Enhancement

GAN技术不仅仅使用在图片中，在语音处理中，也可以大展身手。

我们在过滤声音中的噪声的过程中：

![3-3-9](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/3-3-9.jpg)

传统的深度学习方法：

![3-3-10](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/3-3-10.jpg)

使用Condition GAN之后，结果提升很大：

![3-3-11](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/3-3-11.jpg)

### 5. 视频生成 Video Generation

GAN不仅适用于静态图片，而且适用于视频生成。

G通过以前的图片帧来生成后面的图片。

D来判断生成的视频帧是否是原生视频

![3-3-12](https://raw.githubusercontent.com/muyangren907/Machine_Learning/master/%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86%20%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9CGAN/images/3-3-12.jpg)

本专栏图片、公式很多来自台湾大学李宏毅老师的深度学习课程,在这里，感谢这些经典课程，向李老师致敬！