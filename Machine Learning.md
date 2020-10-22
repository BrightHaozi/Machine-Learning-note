# 回归（regression）

## 1.函数空间（function space）

一个函数越复杂， 它的function space就越大。我们回归任务实际上是在函数空间中找到效果最好的function。

![image-20200910190952931](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200910190952931.png)



找到合适的函数空间非常重要。

若函数空间过小，最有函数不在所选的函数空间内，则不论怎么优化都无法找到最优函数。

e.g.如图，model是选择的function space，红色圆心是target function。则在当前model里无法得到target function。

![image-20200910202701835](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200910202701835.png)

如果函数空间过小，则可能会产生过拟合。



## 2.正则化（regularization）

在Loss函数中增加一个惩罚项
$$
L=\sum_{n}\left(\hat{y}^{n}-\left(b+\sum w_{i} x_{i}\right)\right)^{2}+\lambda \sum\left(w_{i}\right)^{2}
$$
增加正则项，可以使最后优化得到的函数更加**平滑**。

平滑是指当$x$发生变化时，$y$发生的变化较小。

因为我们的目标是优化$L$使其最小，优化过程是求取梯度，优化$w_i$。那么我们通过对$w_i$增加惩罚项，在优化的过程中就在$w_i$上产生更大的梯度，则对应$w_i$会被优化得更小。

我们使用$\lambda$来控制对$w_i$惩罚的程度。

**通常我们认为拥有较小$w_i$的函数是更好的**

原因：如果有一些噪声出现在数据中，那么平滑的函数收到的影响更小。

但并不是越平滑越好：

![image-20200910195042023](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200910195042023.png)

直观理解：如果过于平滑，函数变成一条直线（很多参数被惩罚地接近于0（不起作用）），则事实上函数空间变小，不能拟合到最优的function。

## 3.偏差（bias）与方差（variance）

### 3.1直观理解bias和variance

![image-20200910200244395](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200910200244395.png)

### 3.2Bias和Variance的关系

![image-20200910203345226](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200910203345226.png)

### 3.3如何判断是高Bias还是高Variance

- 如果模型不能很好地fit训练数据，那么具有高bias    **Underfitting**
- 如果可以很好地fit训练数据，但在测试数据上有很大错误，则据由高variance    **Overfitting**

### 3.4如何处理高Bias

重新设计模型：

- 在输入数据集中增加特征
- 选用更复杂的模型

### 3.5如何处理高Variance

- 采用更多数据
- 正则化

### 3.6模型选择

- 我们通常会遇到对于bias和variance的trade-off
- 我们需要选择一个平衡两种误差的模型，使总误差最小

#### 3.6.1交叉验证

讲训练集划分为训练集和验证集。用验证集去选择模型。

![image-20200910205720398](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200910205720398.png)

#### 3.6.2N-fold交叉验证

在不同验证集上取平均来选择模型。

![image-20200910210214254](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200910210214254.png)

## 4.学习率

学习率选择十分重要，如果选择地过小，会导致模型收敛较慢。如果选择地过大，可能会导致模型不收敛。

### 4.1自适应学习率（adaptive learning rate）

学习率在训练的过程中是逐步变化的，同时每个参数都拥有自己的学习率。

#### 4.1.1Adagrad

![image-20200913093126050](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200913093126050.png)

如图所示，$\sigma ^t$是之前参数$w$导数的均方根

e.g.

![image-20200913093624165](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200913093624165.png)

**表达式**
$$
w^{t+1}\leftarrow w^t-\frac{\eta^t}{\sigma^t}g^t\\
\eta^t=\frac{\eta}{\sqrt{t+1}}\\
\sigma^t=\sqrt{\frac{1}{t+1}\sum^t_{i=0}(g^i)^2}
$$
**化简得**
$$
w^{t+1}\leftarrow w^t-\frac{\eta}{\sum^t_{i=0}(g^i)^2}g^t
$$

## 5.随机梯度下降（Stochastic Gradient Descent）

### 5.1与梯度下降的区别

**梯度下降**是计算所有样本的Loss后，在所有样本的Loss上做梯度下降
$$
L=\sum_n(\hat{y}^n-(b+\sum w_ix^n_i))^2\\
\theta^{i}=\theta^{i-1}-\eta \nabla L^{n}\left(\theta^{i-1}\right)
$$
注意上式开始的求和符号

**随机梯度下降**一次只看一个样本，计算完一个样本的Loss后就进行一次梯度下降
$$
L=(\hat{y}^n-(b+\sum w_ix^n_i))^2\\
\theta^{i}=\theta^{i-1}-\eta \nabla L^{n}\left(\theta^{i-1}\right)
$$

### 5.2直观理解

![image-20200913113727483](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200913113727483.png)

### 5.3优点

梯度下降要对所有样本求平均，一轮只能做一次梯度下降，速度较慢。

随机梯度下降一个样本即可做一次梯度下降，收敛速度快。

## 6.特征缩放（Feature Scaling）（归一化）

特征缩放是用来解决不同参数取值范围相差较大的问题

### 6.1为什么要特征缩放

参考下述例子：

![image-20200913153024660](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200913153024660.png)

在**左边图**中，由于$x_2$的的取值范围明显大于$x_1$，这就导致$w_2$发生微小变化会使$y$发生较大变化。
在优化时，$w_1,w_2$需要选择不同的学习率，一般需要使用如Adagrade等方法。

在右边图中，通过归一化，$w_1,w_2$可选择相同的学习率，方便进行参数优化。

### 6.2如何特征缩放

特征缩放的方法有很多种

e.g.

![image-20200913154236736](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200913154236736.png)

# 分类（classification）

分类的任务是给定一个样本，根据这个样本的特征输出这个样本属于哪一个类别。训练过程是使模型在训练集和测试集上达到最高的分类准确率。

**分类模型的构建分为两种：Discriminative和Generative**

Generative的方法是首先假设数据服从某种分布，而后基于这种假设去找到数据集服从分布的均值$\mu$和方差$\sigma$，而后根据此求得模型参数$w,b$，再根据贝叶斯公式求得样本$x_i$属于类别$C_i$的概率。

Discriminative的方法则是像线性回归那样，直接构建表达式，通过优化的方式得到参数$w,b$。一个典型的方法是逻辑回归（logistic regression）。

## 1.Generative Model举例

此处以李宏毅教授上课所举宝可梦分类为例

1.首先假设水系精灵服从高斯分布：
$$
P(x|水系精灵)=f_{\mu^{1}, \Sigma^{1}}(x)=\frac{1}{(2 \pi)^{D / 2}} \frac{1}{\left|\Sigma^{1}\right|^{1 / 2}} \exp \left\{-\frac{1}{2}\left(x-\mu^{1}\right)^{T}\left(\Sigma^{1}\right)^{-1}\left(x-\mu^{1}\right)\right\}
$$
![image-20200914144633679](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200914144633679.png)



2.而后根据样本点，写出所有样本都发生的概率表达式：

![image-20200914144859095](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200914144859095.png)

即图中的
$$
L(\mu,\Sigma)=f_{\mu,\Sigma}(x^1)f_{\mu,\Sigma}(x^2)f_{\mu,\Sigma}(x^3)......f_{\mu,\Sigma}(x^{79})
$$
3.我们想使这79个样本出现的概率最大，即我们想找到一组参数$\mu,\Sigma$使$L(\mu,\Sigma)$最大。则我们使用**极大似然估计**求出对应的$\mu,\Sigma$

4.有了参数$\mu,\Sigma$，我们即知道了步骤1中的水系精灵分布的先验概率。同理我们可以求出普通系精灵的先验概率。

5.有了先验概率，我们就可以通过**贝叶斯公式**，求得后验概率，即一个精灵是水系/普通系的概率。

![image-20200914145719812](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200914145719812.png)

## 2.Discriminative Model

### 2.1由Generative过度到逻辑回归

在这里仍然借用李宏毅教授的课程PPT推导

从后验概率出发

![image-20200914150842083](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200914150842083.png)

此时原后验概率表达式简化为
$$
P(C_1|x)=\sigma(z) \\
z=ln\frac{P(x|C_1)P(C_1)}{P(x|C_2)P(C_2)}
$$
对$z$进行变形：

![image-20200914151126087](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200914151126087.png)

![image-20200914151413314](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200914151413314.png)

![image-20200914151536006](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200914151536006.png)

我们假设$\Sigma_1=\Sigma_2$

则

![image-20200914151621207](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200914151621207.png)

我们可以看到，最终后验概率$P(C_1|x)$可表示成$P(C_1|x)=\sigma(w\cdot x+b)$

在generative model里，我们是进行假设， 得到分布的参数，而后得到模型参数，那么我们能不能根据化简后的表达式直接通过优化的方法得到模型参数呢？这就引入了逻辑回归。

### 2.2逻辑回归（logistic regression）

#### 2.2.1交叉熵

$$
H(p,q)=-\sum_xp(x)ln(q(x))
$$

用于衡量两个分布之间的相似程度。

在分类问题中，我们可以把$P(x)$作为训练集的概率分布，即$P(training)$，把$q(x)$作为学到的模型的分布$P(model)$。

我们模型的训练目标是使$P(model)$尽可能等于$P(real)$，但我们无法得到真实分布$P(real)$，于是我们就假定测试集是从真实数据中独立同分布采样而来，希望$P(training)\simeq P(real)$，而后我们通过训练模型，使$P(model)\simeq P(training)$来达到$P(model) \simeq P(real)$

这样，我们通过交叉熵，就可以衡量$P(training)$和$P(model)$的相似程度，通过优化这个相似程度，就可以使模型变得更好。

#### 2.2.2逻辑回归过程

第一步，构建模型的function set：

![image-20200914203631925](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200914203631925.png)

第二步，使用交叉熵损失构建损失函数：

![image-20200914203711903](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200914203711903.png)

第三步：使用梯度下降优化：

![image-20200914203759951](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200914203759951.png)

#### 2.2.3能否使用均方差优化

![image-20200914205329559](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200914205329559.png)

如图我们看到，如果使用Square Error来进行优化，在step3中会出现梯度消失为0的问题，不能正常进行优化。因此不能使用Square Error作为优化函数。

![image-20200914205451539](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200914205451539.png)

## 3.Generative V.S. Discriminative

这两个模拥有相同的function set，但是训练出的function可能是不同的。

原因：我们在使用Generative model时，第一步是要对数据的分布进行假设，后面的优化过程是基于此假设的。而Discriminative的模型没有这一前提假设。因此我们通常认为Discriminative model是更好的。

Generative model的好处：

1. 根据概率分布的假设
   需要的数据更少
   模型对于噪声有更强的鲁棒性
2. 先验和类依赖概率可以从不同的来源进行估计。

## 4.多分类问题

![image-20200914210603870](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200914210603870.png)

## 5.逻辑回归不能解决的问题及深度学习的引入

逻辑回归是线性分类，因此其不能直接解决异或（xor）问题：

![image-20200914210754672](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200914210754672.png)

但是，我们可以通过特征转换(feature transformation)，对数据集进行相应变换，协助逻辑回归完成分类：

![image-20200914210928637](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200914210928637.png)

但是手动进行feature transformation并不总是容易的，因此我们可以引入深度学习，令神经网络进行feature transformation：

![image-20200914211054630](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200914211054630.png)

# Deep Learning

## 1.深度学习简介

### 1.1全连接神经网络示例

![image-20200915201646463](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200915201646463.png)

### 1.2使用深度学习的好处

我们知道，在机器学习中，特征工程（Feature engineering）是非常重要的，我们需要利用特征工程，为模型找到合适的参数，使模型表现得更好。这对于不复杂的模型，或者我们熟知工作方式的模型是合适的。但对于特征较多，或者我们不太了解工作机制（比如人怎么进行图像的识别）的模型时，特征工程是困难的。此时我们可以引入深度学习，令深度神经网络帮助我们进行特征提取：

![image-20200915202040766](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200915202040766.png)

如上图所示，我们可以将深度神经网络看成三个部分：数据输入(Input Layer)、特征提取(Hidden Layer)、分类输出(Output Layer)

### 1.3深度学习的训练过程

![image-20200915202902013](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200915202902013.png)

由上图可以看到，深度学习训练的三个Step与机器学习相同，都是构建模型(function set)->判定function好坏->优化得到最好的function

区别在于深度学习构建function set的方式与机器学习不同：

![image-20200915203050879](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200915203050879.png)

以识别手写数据集为例，在深度学习中，通过构建不同结构的神经网络，我们可以得到不同的function set。

对于优化的过程，分类问题我们同样可以使用与逻辑回归相同的交叉熵损失，使用梯度下降或者反向传播的方式进行优化。

### 1.4为什么Deep

已经证明，仅有一个隐藏层的神经网络就可以表示出所有的函数。那么我们为什么还要设计层数很多的深度网络。

**事实上，深度学习网络是一个模块化(Modularization)的网络**

以做长短头发男生女生分类为例：

![image-20200915204923812](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200915204923812.png)

我们可以看到，上述例子中，我们有很少的长发男生的数据，那么在进行训练时，由于缺少数据，我们模型对长发男生的分类效果就会很差

**模组化：**

我们不直接对原问题进行分类，我们先将原问题切成小的问题（划分成小的模组），比如先区分是男生还是女生以及是长头发还是短头发，而后再根据这些小的模组的分类结果，完成原问题的分类

![image-20200915205340917](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200915205340917.png)

这样，在训练小的模组时，每个模组都有足够的数据进行训练。这就使最终的分类结果得到很大的改进。

![image-20200915205640467](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200915205640467.png)

直观理解：如果只有一层，同层的神经元之间不能进行信息共享。而在多层中，将原来大的分类问题划分成许多小的分类问题，每一层解决一个简单的分类。后面层可以共用前面层的结果，使数据的利用率更高，模型准确性更好。

### 1.5反向传播(Backpropagation)

深度学习网络可以抽象表示为下图:

![image-20200915211515097](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200915211515097.png)

定义损失函数：
$$
L(\theta)=\sum^N_{n=1}l^n(\theta)
$$
![image-20200915211737439](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200915211737439.png)

以上图为例演示优化过程中，$L$对$w_1$求梯度
$$
\frac{\partial l}{\partial w_1}=\frac{\partial z}{\partial w_1}\cdot \frac{\partial l}{\partial z}=x_1\cdot \frac{\partial l}{\partial z}=x_1\cdot\frac{\partial l}{\partial a}\cdot\frac{\partial a}{\partial z}=x_1\cdot \sigma'(z)\cdot\frac{\partial l}{\partial a}
$$
而
$$
\frac{\partial l}{\partial a}=\frac{\partial l}{\partial z'}\cdot\frac{\partial z'}{\partial a}+\frac{\partial l}{\partial z''}\cdot\frac{\partial z''}{\partial a}
$$
则
$$
\frac{\partial l}{\partial z}=\sigma'(z)\cdot[w_3\cdot\frac{\partial l}{\partial z'}+w_4\cdot\frac{\partial l}{\partial z''}]
$$
此时，只有$w_3\cdot\frac{\partial l}{\partial z'}$与$w_4\cdot\frac{\partial l}{\partial z''}$未知。但我们可以用于计算$\frac{\partial l}{\partial z}$相同的方式，使用后续的链式法则进行计算。

由上述公式，我们可以抽象出来一个反向计算的神经网络：

![image-20200915213243647](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200915213243647.png)

对于正向传播的最后一层(output layer)，方向传播的第一层，计算方法是：

![image-20200915213415655](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200915213415655.png)

当这一层计算完毕，即可从反向的第一层向前计算，最终获得所有$L$对参数$w_i$的梯度$\frac{\partial l}{\partial w_i}$

## 2.卷积神经网络（CNN）

### 2.1为什么要使用卷积神经网络

1. 在图像识别领域，图片是由许多像素构成的，每个像素又可能由三个数据表示（RGB）。通常我们把每个像素当作一个特征，这时如果还使用**全连接神经网络**，我们就需要训练大量的参数，导致训练时间大大延长。
2. 神经网络中，每一个神经元观察一些特征，处理一部分数据。而在图像中，特征具有一定的局部性特征。举例来说，如果我们要训练一个识别鸟的神经网络。对于某个神经元，它的任务是观察鸟喙。而鸟喙只在图像的一部分位置出现，这时就没有必要使此神经元观察所有图像特征，只需观察鸟喙部分特征即可。因此对多数神经元来说，进行全连接是没有必要的。
3. 基于2，同样的特征可能会在一个图片中多次出现。比如图中可能有两只鸟，那么就会有两个鸟喙的特征。
4. 从人的直观角度，一张图片进行一定程度的缩小并不会影响人类对图片的识别。而缩小图片对于神经网络来说能够减少特征，从而降低模型的复杂度，提高训练效率。

基于但不限于以上四个问题，我们使用卷积神经网络，通过卷积和池化等操作使网络的工作方式更符合图像识别特点。

### 2.2卷积神经网络的基本结构

![image-20200917215813306](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200917215813306.png)

如上图所示，一个卷积神经网络主要由若干个卷积层(Convolution)，池化层(Pooling，这里的例子是Max Pooling)以及一个全连接神经网络构成。

### 2.3卷积（Convolution）

#### 2.3.1卷积操作

![image-20200917220816188](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200917220816188.png)

在CNN中，我们通常设置**许多**filter来进行训练。**每个**filter所做的工作是每次识别一个小的pattern，查找是否有想要的特征。

以识别鸟的照片为例，某个filter任务就是在整张图片中每次查看一个区域（pattern）中是否有鸟喙。当对应pattern中有鸟喙时，就会使该filter的输出最大。（比如上图中的Filter1，如果原图片中某个区域有斜对角线是1，1，1，就会使该pattern对该Filter的输出最大）

如2.1所说，一张图片中可能有多个区域有我们想要的特征，因此filter是在图片中以一定步长（stride）移动的。

filter中的内容，就是我们要训练的参数。

#### 2.3.2使用卷积的优势

1.通过卷积，我们可以训练大量的具有识别不同特征的filter，最终共同完成整个图片的识别工作。

2.![image-20200917222241905](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200917222241905.png)

如图所示，在卷积过程中，对于单个神经元，其只需连接与filter中参数数量相同的input，而不需要全连接，减少了参数数量。

对于多个神经元，他们对同一个filter中的参数是进行共享的，这就更加减少了参数的数目。

### 2.4池化（Pooling）

池化的作用是对卷积层提取出来的神经元进行压缩。直观理解是对原图片进行适当的压缩不会影响我们对图像的识别。因此我们就在一个小的pattern中找到最具代表性的元素来代替整个pattern。

以Max Pooling为例

![image-20200917222940256](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200917222940256.png)

图中下方的每个4*4矩阵是输入层通过filter卷积运算后得到的输出。我们取四个输出为一个pattern，使用Max Pooling，找出每个pattern中最大的元素代表这个区域。

直观理解：当输入的特征最符合Filter要探查的内容时，会使该Filter的输出最大。我们的目的是找到是否存在一个pattern存在对应filter的特征。因此，我们在池化层可以在通过对若干个输出（对应更大的pattern）取最大值（在更大的pattern中是否存在filter要探查的特征），一方面减小少了对下一层的输入数据量，一方面表示出了上层输出的信息。

![image-20200917224025156](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200917224025156.png)

### 2.5Flatten和全连接神经网络

因为对于多层卷积运算会使数据具有多个深度（channel），因此Flatten的作用就是将不同深度的数据展开放在同一层，将其作为输入放入全连接神经网络。

对于全连接神经网络，我是这样理解的：

深度神经网络通过模块化（modulation）将大问题分解成一个个小问题。前面的卷积网络是将一个大的图片识别任务划分成了许多小的识别任务，比如通过训练许多filter分别探测鸟的喙，翅膀等。而在全连接神经网络中，通过将前面的这些modulation进行整合，最终完成对原图片的识别任务。

### 2.6两个例子

#### 2.6.1语音识别

![image-20200918085444407](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200918085444407.png)

语音识别通常是识别一定时间内的语音频率。因此主要特征是一段时间宽度内不同频段的信号。由于不同频率上可能会出现相似的波形，因此这时我们通常设置filter的宽度与识别图片宽度相同，而后逐次向下移动，探查不同频段之间有没有相似的特征。

#### 2.6.2文本识别

![image-20200918090511105](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200918090511105.png)

而对于文本识别，由于相似的特征更可能在横向上出现（比如一个句子中有两个相同的字），通常我们是使filter的高度与图片相同，每次横向移动一定单位。

## 3.深度学习技巧

### 3.1深度学习判断改进模型的方法

![image-20200918091809233](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200918091809233.png)

欠拟合：在训练集上表现不好

方法：

1. 寻找新的激活函数
2. 采用动态学习率

过拟合：在训练集上表现很好，在测试集上表现很差

方法：

1. 提前结束训练
2. 正则化
3. Dropout

另外，不能单纯根据测试集的结果判断是否过拟合。比如一个20-layer的和56-layer的神经网络。如果testing error上56-layer的大，还需要去看看training error，如果也是56-layer的大，那就说明不是过拟合。这也说明并不一定更深的网络训练效果更好。训练过程中有很多因素会导致训练效果变差，比如local minimum。

### 3.2激活函数

#### 3.2.1梯度消失

![image-20200918095940920](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200918095940920.png)

在使用sigmoid function作为激活函数时，我们经常会发现，靠近输出层的神经元有较大的梯度，很快就收敛了。但与此同时，靠近输出层的神经元由于梯度很小，更新慢，还几乎处于随机参数状态。这就导致模型训练效果很差。

导致此问题的原因：

![image-20200918100139112](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200918100139112.png)

由于sigmoid function将$[-\infty,\infty]$的数据压缩到$[0,1]$，因此靠近输入层的参数变化$\Delta w$只能对输出层产生较小的影响。这就使靠近输入层的参数训练较慢。

#### 3.2.2ReLU激活函数

![image-20200918101141607](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200918101141607.png)

ReLU激活函数更加简单，训练迅速。

使用ReLU不会出现sigmoid导致的梯度消失问题。

一些变种：

![image-20200918102933505](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200918102933505.png)

#### 3.2.3Maxout激活函数

![image-20200918103014673](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200918103014673.png)

Maxout的功能更加强大，其能完成ReLU完成的功能（ReLU是Maxout的一个特例）：

![image-20200918103114376](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200918103114376.png)

Maxout可以拟合出任意分段线性凸函数：

![image-20200918103243340](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200918103243340.png)

### 3.3可调整学习率

#### 3.3.1Adagrad

在回归中总结过，每个参数有自己的学习率，学习率的衰减受前面训练轮次累加梯度影响。

![image-20200918103750297](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200918103750297.png)

#### 3.3.2RMSProp

Adagrad解决的是不同参数的梯度变化速率不同的问题。但对于一个参数，是假定其梯度变化是均匀的。但对于下述例子，该假设不成立：

![image-20200918104153437](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200918104153437.png)

此时我们引入RMSProp解决上述问题。

![image-20200918104456507](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200918104456507.png)

#### 3.3.3Momentun

Momentum不是一种具体的学习率算法，它是将物理中惯性的思想引入了参数更新。在每次更新参数时，不仅需要考虑当前的梯度，还要考虑由之前梯度更新积累的惯性。最终参数的更新方向是当前梯度与momentum的矢量和。

![image-20200918104808041](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200918104808041.png)

$v^i$的计算方式：

![image-20200918105334336](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200918105334336.png)

对这种方法带来的优势的直观理解：

![image-20200918105441017](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200918105441017.png)

在上图中，对于鞍点和local minimum，如果使用常规的参数更新，由于此处的梯度为0，则梯度不再惊醒更新。而由于我们有momentum的存在，就**有可能**冲出这些梯度为0的点，找到global minimum。但这只是给了我们冲出鞍点和local minimum的希望，并不保证一定能找到global minimum。

#### 3.3.4Adam

Adam=RMSProp+Momentum

![image-20200918110034078](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200918110034078.png)

### 3.4正则化

[L1，L2正则化详细解释]: https://blog.csdn.net/jinping_shi/article/details/52433975

以下内容为直观理解，详细内容查看上述博客。

#### 3.4.1 $L2$正则化

![image-20200918160126643](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200918160126643.png)

由更新公式可得，每次参数更新前先乘一个接近于1的数，使参数越来越接近0.同时，参数越大，每次相乘后参数向0移动的方向就越多。最终前面这一项会和后面这一项保持平衡，使训练出的模型参数偏小。**最终使整个模型更加平滑。**



#### 3.4.2 $L1$正则化

![image-20200918161323863](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200918161323863.png)

观察$L1$正则化后的参数更新公式，每次参数都向0多走$\eta \lambda sgn(w^t)$。与$L2$相比，其每次变化是固定的。而$L2$则是大参数变化多，小参数变化少。**因此$L1$正则化训练出的模型是稀疏(sparse)的。**

### 3.5Dropout

Dropout的思想是在每轮训练中，每个神经元以一定机率$p$被丢弃，不参加本轮训练。

![image-20200918150334407](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200918150334407.png)

在dropout后，相当于本轮训练的网络结构发生了改变。

需要注意的是，网络训练好后，在**测试集**上测试时，所有网络参数都要乘$1-p$

![image-20200918151341999](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200918151341999.png)

原因：

![image-20200918151428107](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200918151428107.png)

Dropout是一种Ensemble的思想：

![image-20200918152549748](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200918152549748.png)

![image-20200918152559308](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200918152559308.png)

Ensemble是将原数据集划分成若干集合，分别训练一个神经网络。在进行测试时，将数据分别输入不同神经网络，对输出取平均。

Dropout也是相似的思想，也可以看作每次训练是使用一个mini-batch数据训练一个神经网络。不同的是一些参数在不同轮次中对应的网络是共享的。

![image-20200918154058432](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20200918154058432.png)

# Explainable Machine Learning

以识别猫的分类器为例，**可解释的机器学习**主要分为两种：

1. Local Explanation(局部可解释性)
   目的：Why do you think this image is a cat?
2. Global Explanation(全局可解释性)
   目的：What do you think a "cat" looks like?

## 1. 不同模型的解释性

一些模型具有较强的解释性。

1. Linear model线性模型)
   线性模型我们可以通过权重的大小，直接判断哪个特征对模型的影响更大。
   但线性模型的拟合能力较弱。
2. Decision Tree(决策树)
   我们可以直接通过决策树的分支节点判断特征对模型的影响。
   决策树兼具有强的解释性和拟合能力。

通常，我们使用可解释性强的模型来解释可解释差的模型。比如，深度神经网络具有很强的拟合能力，但其可解释性很差，会给使用人员带来一些困扰。此时我们可以使用一些可解释性强的模型进行解释。

## 2. Local Explanation

e.g.Why do you think this image is ...

### 2.1 Basic Idea

![image-20201009214415520](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201009214415520.png)

一个值观的想法，对于我们要识别的一个图片$x$，将其划分成多个components，每次对components中的一个分量进行微调，调整后对输出结果影响大的，就是决定该图片分类结果的重要因素。

基于机上方法，我们将一个输入的图片看作$\{x_1,...,x_n,...,x_N \}$，每次对一个分量$x_i$进行微调$x_n+\Delta x$，则输出由$y_k$变为$y_k+\Delta y$。我们可以通过判断$|\frac{\Delta y}{\Delta x}|$的大小来判断分量$x_i$对该图像识别的影响。

在实际应用中我们通过$|\frac{\partial y_k}{\partial x_n}|$来判断$x$的对图片输出为某一类别的影响。但这不总是对的。

e.g.

![image-20201009215903761](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201009215903761.png)

大象的鼻子长度是判断一张图片中有大象的关键因素，但当大象鼻子非常长时，该特征对应的梯度可能会很小。

## 3. Global Explanation: Explain the whole model

e.g. What do you think a ... looks like?

### 3.1 **Activation Minimization** 

在卷积神经网络中，以图片分类问题为例。对于一个训练好的网络，我们想要知道什么情况下会使输出某类别$y_i$的概率最大。此时，我们可以调整输入$x$，找到使$y_i$最大的$x^*$，即：
$$
x^{*}=\arg \max _{x} y_{i}
$$
得到会使网络认为是$y_i$的图片。

![image-20201009221436082](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201009221436082.png)

但如上图所示，通常结果是人类所不能理解的，我们可以加一些约束项使输入结果尽可能便于辨认：

![image-20201009221524213](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201009221524213.png)

### 3.2 Constraint from Generator 

![image-20201009222152965](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201009222152965.png)

对于本方法，我们首先拥有一个训练好的generator和classifier，而后固定generator和classifier。通过调整generator的输入$z$，找到满足
$$
z^{*}=\arg \max _{z} y_{i}
$$
的$z^*$。再将其放入generator生成$x^*$，即
$$
x^*=G(z^*)
$$
即可得到模型在模型看来，输入最大$y_i$的最大概率时对应的$x^*$。即回答了从模型的角度，“What do you think a ... looks like?”的问题。

## 4. Using a Model to Explain Another

### 4.1 使用线性模型解释神经网络

![image-20201009223348596](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201009223348596.png)

使用线性模型存在的问题是，线性模型的函数空间太小，拟合能力弱。无法正确模仿要解释的模型。

但可以使用线性模型去模仿一个局部的区域

#### 4.1.1 Local Interpretable Model-Agnostic Explanations (LIME)

![image-20201009223804943](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201009223804943.png)

在我们想要模仿数据点周围进行取样，利用取样样本训练一个线性模型并用其解释该区域内的复杂模型特点。

在图像识别中使用LIME：

1. Given a data point you want to explain
2. Samble at the nearby
   ![image-20201010085919958](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201010085919958.png)

3. Fit with linear (or interpretable) model
   ![image-20201010090012480](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201010090012480.png)

4. Interpret the model you learned
   ![image-20201010090151152](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201010090151152.png)

### 4.2 Decision Tree

我们知道，决策树在有很好的解释性的同时，也有很强的拟合能力。因此我们可以用决策树来模仿待解释模型。

![image-20201010090427277](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201010090427277.png)

但同时，并不是所有的决策树都有很好的解释性：

![image-20201010090501024](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201010090501024.png)

如上图所示，当我们的决策树深度较深，或者使用森林时，决策树模型的解释性同样很差。

因此，我们引入$O(T_\theta)$来表示决策树的复杂度(e.g. average depth of $T_\theta$)

在我们训练决策树模型时，加入一项$O(T_\theta)$，来尽可能使训练出的决策树模型复杂度较小。

![image-20201010090831445](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201010090831445.png)

# Attack ML Models

研究机器学习模型攻击与防御的动机：

1. 我们不仅仅需要在实验室中部署机器学习模型，也需要在现实世界中部署模型。
2. 分类器在工作"大部分时间"具有好的鲁棒性是不够的。
3. 我们希望分类器对于恶意欺骗分类器的输入有较好的鲁棒性。（应对人类的欺骗）

## 1. Attack

### 1.1 攻击的目的

![image-20201011100148375](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201011100148375.png)

以上图为例，我们想要对原图片加入一些噪声，使分类器不能正确识别出原图像。

### 1.2 Loss Function for Attack

![image-20201011100435645](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201011100435645.png)

#### 1.2.1 Training

$$
L_{train}(\theta)=C(y^0,y^{true})
$$

在我们训练神经网络时，通常以上述损失函数进行训练。

#### 1.2.2 Non_targeted Attack

无目的性的攻击：
$$
L(x')=-C(y',y^{true})
$$
目的是让我们加噪声后的输入$x'$，经过神经网络后输出的结果距离加噪声前的输出结果越远越好。

#### 1.2.3 Targeted Attack

有目的性的攻击：
$$
L(x')=-C(y',y^{true})+C(y',y^{false})
$$
与1.2.2相比，除了尽可能与加噪声前的输出结果越远越好之外，我们又增加了第二项，希望其与我们目标让其变成的label$y^{false}$越近越好。

#### 1.2.4 Constraint

$$
d(x^0,x') \le \epsilon
$$

通过增加一项约束，使加噪声后的输入与原输入距离不要太远。防止因差距太大被发现。

常用的限制函数：

1. L2-norm
   $d(x^0,x')=||x^0-x'||_2=(\Delta x_1)^2+(\Delta x_2)^2+(\Delta x_1)^3...$
2. L-infinity
   $d(x^0,x')=||x^0-x'||_{\infty}=\max\{\Delta x_1,\Delta x_2,\Delta x_3,...\}$

![image-20201011103900522](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201011103900522.png)

我们更常用L-infinity来进行衡量。

### 1.3 How to Attack

就像训练神经网络一样，只是我们现在神经网络的参数$\theta$是固定的，转而训练输入$x'$。
$$
x^*=arg \min_{d(x^0,x')\le \epsilon}L(x')
$$
找到能使损失在$d$约束下能使损失函数$L$最小的输入$x^*$

![image-20201011104748862](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201011104748862.png)

![image-20201011104839984](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201011104839984.png)

上图中，蓝色的$x^t$是执行梯度下降后$x$落到的位置，如果离开了限制函数$d$的区间，则根据采用的$d$找到限制范围内离$x^t$最近的点。

**一个问题：**

有时我们会发现，训练出来的图片，人看起来几乎没有区别，但机器却会将其识别错误。而人手动加噪声，即使人一眼就能看出来加了噪声，但对机器识别结果却没有很大影响。这是为什么呢？

![image-20201011105849191](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201011105849191.png)

如图所示，通常图片分类识别都是在很高维度上进行训练。可能在某些维度上，其鲁棒性较好，进行一定的波动对识别效果影响不大。而在一些维度上，轻微的扰动就会对结果产生很大的影响。

### 1.4 White Box v.s. Black Box

#### 1.4.1 White Box Attack

在上述的工作中，我们都是固定了神经网络的参数$\theta$去优化$x'$。为了能够攻击，我们需要事先知道神经网络的参数$\theta$。因此，以上工作都称为白盒攻击。

我们可能会想，在实际应用中，我们通常是向用户提供API，用户无法知道模型的参数。那么我们的模型是不是安全的呢？

答案是否定的，因为我们可以进行黑盒攻击。

#### 1.4.2 Black Box Attack

![image-20201011112404785](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201011112404785.png)

如果我们有目标网络的训练数据，那么我们可以自己训练一个神经网络进行攻击。

如果没有，那我们可以制造大量的数据输入目标网络并得到网络的输出结果。我们有了输入及其对应的输出，就可以将其作为训练集，再自己训练一个网络。

## 2. Defense

权值正则化，dropout，模型集成是无法防御对抗性攻击的。

两种类型的防御：

1. Passive defense：不修改模型，找到被附加噪音的图片。
2. Proactive defense：训练一个对抗攻击的健壮模型

### 2.1 Passive Defense

![image-20201011163933183](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201011163933183.png)

在图片输入后，首先将图片经过一个过滤层(Filter)，对图片进行平滑处理，减少恶意添加的噪声对神经网络识别结果的影响。

### 2.2 Proactive Defense

![image-20201011170418598](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201011170418598.png)

主动防御的精神就是找出漏洞并补起来。

在模型训练的过程中，我们就使用相关的攻击算法，将其加入到模型训练。这样训练好的模型就可以抵御这些算法的攻击。

问题：如果在实际应用中遇到了使用不同算法进行的攻击，模型仍然会做出错误判断。

# Network Compression

## 1. Network Pruning

深度神经网络通常是参数过多了，通常我们可以修剪掉一部分的参数或者神经元。

修剪流程：

![image-20201011211233528](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201011211233528.png)

如上图所示：

1. 将神经网络的权重(衡量标准可以是L1，L2)排序
2. 删减掉重要性低的权重
3. 在训练数据上再进行训练
4. 如果修建后的模型已经满足要求，则输出。否则，返回第一步。

**注意：**不要一次修剪过多的参数，否则可能对模型伤害过大无法恢复。

### 1.1 Weight pruning

![image-20201011213650296](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201011213650296.png)

修剪掉一些参数：但这样会使模型难以实现，同时也很难加速运算。

### 1.2 Neuron pruning

![image-20201011213834684](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201011213834684.png)

修剪掉一些神经元是比较好的方法。方便实现，同时也方便进行加速运算。

## 2. Knowledge Distillation

知识蒸馏的原理是，在已经有的完备模型模型的基础上，构建一个简单模型。通过完备模型的输出来教简单模型。

![image-20201011214527379](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201011214527379.png)

以上图为例，完备模型不仅教简单模型当前看到的图片是1(因为识别是1的概率最大)，还教会简单模型，7和9与1形状相似(看到1时7和9也有一定概率输出)

为了能够让完备模型教会简单模型更多东西，我们引入“**Temperature**”的概念：

![image-20201011215226716](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201011215226716.png)

通常在分类问题中，模型最后的输出要经过softmax转化为概率的分布。但当出现上图左边的情况时，复杂模型只能教会简单模型当前输出时$y_1$类。而我们引入一个温度参数$T$，在输出前首先除以温度参数，使不同类别更为接近，就能教给简单模型更多的知识(当前输入是$y_1$类，同时$y_1$类与$y_2,y_3$类较为相似)。

## 3.Parameter Quantization

![image-20201011220300129](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201011220300129.png)

**Binary Weights:**

![image-20201011221546889](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201011221546889.png)

# Anomaly Detection

异常检测：目的是让机器知道自己不知道。

## 1. Problem Formulation

我们给一个训练数据集 $\{x^1,x^2,...,x^N \}$，我们想要找到一个函数去检测输入的$x$是否与训练数据是相似的。

![image-20201015164404474](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201015164404474.png)

**直观想法：**

获得正常数据normal data$\{x^1,x^2,...,x^N \}$

获得异常数据anomal data $\left\{\tilde{x}^{1}, \tilde{x}^{2}, \cdots, \tilde{x}^{N}\right\}$

然后训练一个二分类器

**问题：**

1. 异常数据太多，无法穷举。

e.g.训练一个识别宝可梦的分类器，那么所有不是宝可梦的图片都是异常的

2. 异常数据有时难以获得。

## 2. How to use the Classifier

![image-20201015184916786](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201015184916786.png)

我们在原来训练的分类器的基础上，除了输出输入$x$对应的标签$y$，，还对这个$y$输出一个信心分数$c$。同时，我们定义一个阈值(threshold)$\lambda$，如果信心分数大于$\lambda$，则认为是正常的，否则是异常的。

**Framework：**

![image-20201015193125500](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201015193125500.png)

# Unsupervised Learning

在无监督学习中，有三种主要问题：

1. Dimension Reduction（降维） （化繁为简）
   ![image-20201020215547419](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201020215547419.png)
2. Generation（无中生有）
   ![image-20201020215619167](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201020215619167.png)

3. Clustering（聚类）

## Clustering

### 1. K-means

![image-20201020220259948](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201020220259948.png)

K-means算法比较简单，就是初始化若干个中心点，每轮训练将每个样本都与其距离最近的中心点归为一类，而后更新中心点，直至算法收敛。

**问题：**需要事先选定要聚为多少类。即，需要预先选定类别数$K$

### 2. Hierarchical Agglomerative Clustering (HAC)

![image-20201020220639979](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201020220639979.png)

算法思路：初始状态将每个样本看成一类，而后每轮循环找到距离最近的两类将其归为一类，保留聚类过程。训练结束后所有样本都会被聚成一类。聚类过程如上图所示是一棵树。
完成聚类后，可以通过选择$threshold$的方式，选择将原样本划分为多少类。
e.g. 如果将距离根最近的那条线作为阈值进行划分，则原数据集被划分为两类。如果是第二近的那条线，则原数据集被划分为3类。

## Dimension Reduction

目标：找到一个函数，将原来的高维输入降到低维度

![image-20201020221339249](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201020221339249.png)

举例直观说明这种想法：

![image-20201020221413017](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201020221413017.png)

在MNIST识别手写数字数据集中，所有的图片是$28\times 28$维的，但并不是左右的的维度都用来表示数字。对于上图的五张图，可能使用一个用来表示角度的特征就可以进行区分。因此，可以对原来高维数据进行适当降维来降低运算的复杂度。

### Principle component analysis (PCA) (主成分分析)

#### 1.以一个二维数据为例说明PCA的目标

![image-20201020222605930](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201020222605930.png)

如上图所示，我们要在二维空间中找到一个维度（一个vector），将原数据集上的数据映射到这个vector上进行降维。如果没有施加限制，那么我们有无穷多种映射方法。
但是，我们知道，为了使数据集含有更多的信息，我们应该尽可能将降维后的数据区分开。以上图为例，如果选择Small variance的那条向量，很多数据点映射后挤在一起，那么我们就会损失许多有用信息。因此，PCA就是要找出一条vector，使数据降维后，有最大的方差。转化成数学公式就是：
$$
Var(z_1)=\frac{1}{N}\sum_{z_z}(z_1-\overline{z_1})\\
s.t. ||w^1||_2=1
$$

#### 2. 高维情况

![image-20201020223614461](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201020223614461.png)

扩展1.中的情况，如果我们的原数据是一个很高维度的数据，我们要对其降维。设我们降维后的向量组成的矩阵为$W$。对于第一维$w_1$，我们只需要求数据映射到本维度后方差最大即可。而对于后续维度，以$w_2$为例。除了要求数据映射到本维度后方差最大，还需限制本维度($w_2$)与之前的维度($w_!$)是正交(orthogonal)的。

**解释：**如果不施加限制，显然我们找到的第二个维度与第一个维度一定是相同的。同时，限制$w_2$与$w_1$正交可以使降维后的各个维度之间是**不相关**的，**能够降低后续模型过拟合的风险。**

后续维度$w_i$与上述分析相同。

那么，最后我们得到的降维后的维度矩阵$W$，一定是一个**正交阵**(Orthogonal matrix)。

**数学推导：**

设映射到的第一个维度为$w_1$，数据$x$映射到$w_1$后的数据集为$z_1$

则$z_1=\frac{1}{N}\sum z_1 = \frac{1}{N}\sum w^1 \cdot x=w^1 \cdot \frac{1}{N} \sum{x} =w^1 \cdot \overline{x}$

我们的目标是使$Var(z_1)$最大
$$
\begin{aligned}
Var(z_1)&=\frac{1}{N} \sum_{z_1}(z_1-\overline{z_1})^2\\
&=\frac{1}{N}\sum_{x} (w^1 \cdot x - w^1 \cdot \overline{z_1})^2\\
&=\frac{1}{N}\sum (w^1 \cdot (x- \overline{x}))^2 ①\\
&=\frac{1}{N}\sum(w^1)^T(x-\overline{x})(x-\overline{x})^Tw^1\\
&=(w^1)^T\frac{1}{N}\sum(x-\overline{x})(x-\overline{x})^Tw^1\\
&=(w^1)^TCov(x)w^1
\end{aligned}
$$
①：$(a\cdot b)=(a^Tb)^2=a^Tba^Tb=a^Tb(a^Tb)^T=a^Tbb^Ta$

令$S=Cov(x)$（协方差矩阵） 显然$S$是一个实对称矩阵

那么我们的算法目标变为：
$$
w_1=\max_{w_1} (w^1)^TSw^1\\
s.t. (w^1)^Tw^1=1
$$
只是一个优化问题，我们使用拉格朗日乘子法进行优化：

![image-20201021094248946](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201021094248946.png)

在优化的过程中我们可以发现，$w^1$是$S$对应的一个特征向量，$\alpha$是$w^1$对应的特征值。

我们的优化目标变为最大化$\alpha$，那么显而易见，我们应该选择此时最大的特征值$\lambda_1$，其对应的特征向量就是我们要找的目标向量$w_1$

同理我们求$w_2$，**同时对$w_2$施加与$w_1$正交的约束。**

![image-20201021101206213](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201021101206213.png)

而$(w^1)^TSw^2=((w^1)^TSw^2)^T=(w^2)^TS^Tw^1=(w^2)^TSw^1=\lambda_1(w^2)^Tw^1=0$

同时$(w^2)^Tw^1=0$

因此：

![image-20201021101909019](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201021101909019.png)

即，$\beta=0$

则：

![image-20201021102052625](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201021102052625.png)

$w_3,w_4...w_k$推导同理。

![image-20201021102739341](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201021102739341.png)

由上图推导可得，PCA最终降维得到的$z$的协方差矩阵是一个对角阵，说明经过PCA后得到 的特征之间是没有相关性的。

#### 3. 另一种观点的PCA

![image-20201021103224425](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201021103224425.png)

以MNIST识别手写数字为例，由上图都，每一个数字都可以看作是多个小的component($u^i$)组成的。那么，我们就可以用一个权值向量$c=[c_1 \ c_2 \ ... \ c_K]$来表示原图。

此时，相当于我们将原来图片的高维向量进行降维，使用$u=[u_1 \ u_2 \ ... \ u_K]$来表示当前的特征。

那么我们的优化目标就变为：

![image-20201021104215318](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201021104215318.png)

数学上可以证明，最后优化出的$u=[u_1 \ u_2 \ ... \ u_K]$与PCA得到的$w=[w_1 \ w_2 \ ... \ w_K]$是同一个向量。

过程如下：

![image-20201021105256819](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201021105256819.png)

我们的目标是使右边相乘的矩阵与左边的矩阵$X$越接近越好。

![image-20201021105552312](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201021105552312.png)

而我们知道，对矩阵$X$使用奇异值分解(SVD)后得到的相乘矩阵是与A最接近的矩阵(相似程度取决于$K$的选取)。如上图所示，我们把$\Sigma V$看作上面的矩阵$C$，而后进行优化，最后即可得到矩阵$U$。

#### 4. 使用神经网络

PCA看起来像一个只有一个隐藏层(线性激活函数)的神经网络。

![image-20201021110130583](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201021110130583.png)

不过需要注意的是，该神经网络解出的结果与PCA的结果是不同的。**原因在于**PCA要求$w_i$之间是正交的，而该神经网络没有这个要求。

#### 5. 应用例子

##### 5.1 神奇宝贝

![image-20201021110944575](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201021110944575.png)

我们有800个宝可梦的样本，每个样本有6个feature。对其做特征值分解后，我们可以发现，前四个特征值占的比例较大。则我们可以认为前四个特征值对应的维度已经足够用来区分不同的宝可梦。

那么我们分别将不同样本投影到着四个维度：

![image-20201021111134693](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201021111134693.png)

投影到这四个维度后，原来不同的特征会有不同的投影值，其中数值较大的说明本维度的影响较大。

##### 5.2 MNIST

![image-20201021111301972](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201021111301972.png)

我们同样进行特征值分解后，取出影响最大的30个components。对其可视化后得到上图。

我们发现，PCA得到的component不是我们想象中的一个个小的片段。每一个component都有完整的轮廓。

这是因为，$w^i$的系数$a_i$可以取负值。举例来说，我们要组成一个数字9，那么我们可以先找到一个8，然后把多余的部分再删除掉。

#### 6. PCA的缺点

1. PCA算法是线性的，只能进行线性变换导致其对数据的变化能力较弱，可能使降维后的数据无法区分：

![image-20201021111728982](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201021111728982.png)

2. PCA是unpervised的，其只考虑了数据之间分布的方差。在某些问题上可能效果不是很好：

   ![image-20201021111816541](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201021111816541.png)

   对于上图中下面部分的图，可以使用有监督的LDA进行划分。

## Word Embedding

机器可以在**没有监督**的情况下通过阅读大量文件来学习单词的意思。

Word Embedding通常运用在自然语言处理领域，目的是用词向量表示词，同时相比于传统方法如1-of-N Encoding，可以实现降维。

1-of-N Encoding，为了表示所有的单词，需要使用维度很高的词向量进行表示。且不同单词间的词向量是相互独立的，不能表示词义之间的联系。

Word Class的方法将具有同样性质的word分簇成一个class。用word所属的class来表示word。但这种方法也缺少了一些信息。比如下图中就不能体现class1和class3在能否运动这个角度的联系。

Word Embedding将单词project到一个高维空间（通常50-100维）中，在这个空间中能够很好地体现不同单词之间的联系，类似语义的词汇在word embedding的投影空间比较接近。同时其相比于1-of-N Encoding（通常上万维）维度也更低。

![image-20201022090516004](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201022090516004.png)

### Basic idea

word embedding 的基本思想就是，每一个词汇的含义都可以根据它的上下文得到。

### How to exploit the context

#### 1.Count based

我们用$V(w_i)$来表示单词$w_i$的词向量。`Count based`方法的是，在一个document中如果单词$w_i$和$w_j$总是同时出现，那么$V(w_i)$和$V(w_j)$将会很接近。

举一个`Glove Vector`的例子：

令$N_{i,j}$表示$w_i$和$w_j$在同一document中出现的次数，那么我们令$V(w_i)$和$V(w_j)$的内积$V(w_i)\cdot V(w_j)$与$N_{i,j}$越接近越好。

#### 2. Prediction based

##### 2.1 算法思想

![image-20201022151841598](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201022151841598.png)

给定若干单词$w_i，w_{i-1},...$，使用神经网络去预测$w_{i+1}$，使神经网络的输出结果与文中的真实输出越接近越好。

##### 2.2 为什么可以这样做

![image-20201022152113406](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201022152113406.png)

如上图所示，我们使用$w_{i-1}$去预测$w_i$。输入的是$w_{i-1}$的1-of-N编码。

假设两篇文章中分别有"张三去吃饭"，“李四去吃饭”。那么“张三”和“李四”就代表$w_{i-1}$，“去吃饭”代表$w_{i}$。我们希望神经网络输入两个不同的$w_{i-1}$后输出“去吃饭”的概率都是最高的。

那么，为了使这两个不同的输入最终得到的输出相同，神经网络中就需要对输入做一些变换，将两个不同的vector投影到**位置相近的低维空间**上。这样在后续的预测工作中才能使其有相同输出的概率最大。

那么，对于上图的例子，神经网络中第一层hidden layer所作的工作就是对输入的1-of-N编码进行降维，我们把其转换后的结果的$[z_1 \ z_2 \ ...]^T$拿出来，得到的就是单词1-of-N编码降维后的词向量。

这样，我们可以在prediction based方法的model中控制第一层hidden layer的大小，从而控制目标降维空间的维度。

##### 2.3 共享参数

![image-20201022153755661](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201022153755661.png)

由于现实世界中不同单词搭配情况过多，仅使用一个单词去预测下一个单词的效果往往是很差的。我们可以使用多个单词去预测下一个单词。

这里我们以两个单词为例。如果是一般是神经网络，我们直接把$w_{i-2}$和$w_{i-1}$这两个vector拼接成一个更长的vector作为input即可。但这样会产生一个问题：把同一个word放在$w_{i-2}$的位置和放在$w_{i-1}$的位置，得到的Embedding结果是会不一样的。那么，如果把两组weight设置成相同（即共享参数），可以使$w_{i-2}$与$w_{i-1}$的相对位置不会对结果产生影响，同时可以有效减少参数量。

由上图所示，$x_{i-2}$和$x_{i-1}$都是1-of-N编码的单词，$W_i$是用来进行降维的矩阵，$z$是降维后的结果。

在没有进行参数共享时，降维的结果为
$$
z=W_1x_{i-2}+W_2x_{i-1}
$$
而我们强制令$W1=W2=W$

则降维后的结果为
$$
z=W(x_{i-2}+x_{i-1})
$$
那么，当我们得到了这组参数$W$，就可以与1-of-N编码$x$相乘得到降维后的结果$z$。

**如何实际用于训练：**

- 首先在训练的时候就要给它们一样的初始值
- 然后分别计算loss function $C$对$w_i$和$w_j$的偏微分，并对其进行更新 $$ w_i=w_i-\eta \frac{\partial C}{\partial w_i}\ w_j=w_j-\eta \frac{\partial C}{\partial w_j} $$ 这个时候你就会发现，$C$对$w_i$和$w_j$的偏微分是不一样的，这意味着即使给了$w_i$和$w_j$相同的初始值，更新过一次之后它们的值也会变得不一样，因此我们必须保证两者的更新过程是一致的，即： $$ w_i=w_i-\eta \frac{\partial C}{\partial w_i}-\eta \frac{\partial C}{\partial w_j}\ w_j=w_j-\eta \frac{\partial C}{\partial w_j}-\eta \frac{\partial C}{\partial w_i} $$
- 这个时候，我们就保证了$w_i$和$w_j$始终相等：
  - $w_i$和$w_j$的初始值相同
  - $w_i$和$w_j$的更新过程相同

并且，由于这个算法是unsupervised的，因此我们只需要存取大量的文章数据输入给神经网络即可。

##### 2.4 多种变形

使用不同位置的词汇去预测下一个词汇可以产生word embedding的多种变形：

![image-20201022160159023](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201022160159023.png)

## Neighbor Embedding

PCA和Word Embedding都是线性降维思想，而Neighbor Embedding介绍的是非线性的降维。

### Manifold Learning

![image-20201022161640062](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201022161640062.png)

样本点的分布可能是在高维空间里的一个流行(Manifold)，也就是说，样本点其实是分布在低维空间里面，只是被扭曲地塞到了一个高维空间里。

地球的表面就是一个流行(Manifold)，它是一个二维的平面，但是被塞到了一个三维空间里。

在Manifold中，只有距离很近的点欧氏距离(Euclidean Distance)才会成立，而在下图的S型曲面中，欧氏距离是无法判断两个样本点的相似程度的。

而Manifold Learning要做的就是把这个S型曲面降维展开，把塞在高维空间里的低维空间摊平，此时使用欧氏距离就可以描述样本点之间的相似程度。

### Locally Linear Embedding(LLE)(局部线性嵌入)

![image-20201022162818124](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201022162818124.png)

算法思想：假设每一个$x_i$可以用它周围的点($x_j$)做linear combination得到。

那么我们这个问题就转换成了，找一组使所有样本点与周围样本点线性组合的差距能够最小的参数$w_{i,j}$

![image-20201022163604412](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201022163604412.png)

LLE的具体做法如下：

![image-20201022170421952](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201022170421952.png)

- 在原先的高维空间中找到$x^i$和$x^j$之间的关系$w_{ij}$以后就把它固定住

- 使$x^i$和$x^j$降维到新的低维空间上的$z^i$和$z^j$

- $z^i$和$z^j$需要minimize下面的式子： $$ \sum\limits_i||z^i-\sum\limits_j w_{ij}z^j ||_2 $$

  **即在原本的空间里，$x^i$可以由周围点通过参数$w_{ij}$进行线性组合得到，则要求在降维后的空间里，$z^i$也可以用同样的线性组合得到**

实际上，LLE并没有给出明确的降维函数，它没有明确地告诉我们怎么从$x^i$降维到$z^i$，只是给出了降维前后的约束条件。

在实际应用LLE的时候，对$x^i$来说，**需要选择合适的邻居点数目K**才会得到好的结果。

### T-distributed Stochastic Neighbor Embedding (t-SNE) (t分布随机邻居嵌入)

LLE只假设了相邻的点要接近，但没有假设不相近的点要分开。这就导致，虽然说同一个class的点会聚集在一起，但没法避免不同class的点重叠在一起。

#### 1. 算法思想：

![image-20201022171211558](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201022171211558.png)

1. 在原数据$x$的分布空间上，计算**所有**$(x^i,x^j)$对的相似度$S(x^i,x^j)$

2. 对其归一化：$P(x^j|x^i)=\frac{S(x^i,x^j)}{\sum_{k\ne i}S(x^i,x^k)}$

3. 将$x$降维到$z$，同样计算相似度$s'(z^i,z^j)$，并作归一化：$Q(z^j|z^i)=\frac{S'(z^i,z^j)}{\sum_{k\ne i}S'(z^i,z^k)}$

4. 利用`KL散度`来衡量分布$P,Q$的相似程度，目标是使其分布越接近越好。
   $$
   L=\sum\limits_i KL(P(|x^i)||Q(|z^i))\ =\sum\limits_i \sum\limits_jP(x^j|x^i)log \frac{P(x^j|x^i)}{Q(z^j|z^i)}
   $$

**注意：**归一化是有必要的，因为我们无法判断在$x$和$z$所在的空间里，$S(x^i,x^j)$与$S'(z^i,z^j)$的范围是否是一致的，需要将其映射到一个统一的概率区间。

#### 2. t-SNE算法的使用

因为t-SNE要为每一个pair都计算相似度，因此在原数据维度很高时，运算量非常巨大，通常不会直接使用。常用的方式是先使用PCA将输入集降到50维左右，而后再使用t-SNE降维到更低的目标维度。

另外，在使用t-SNE时，如果给一个新的$x$，则需要重新再跑一遍算法，因此通常不直接将t-SNE用于训练。而是训练前先使用t-SNE可视化一下降维后数据的分布。

#### 3. 相似函数的选择

如果根据欧氏距离计算降维前的相似度，往往采用**RBF function** $S(x^i,x^j)=e^{-||x^i-x^j||_2}$，这个表达式的好处是，只要两个样本点的欧氏距离稍微大一些，相似度就会下降得很快

还有一种叫做SNE的方法，它在降维后的新空间采用与上述相同的相似度算法$S'(z^i,z^j)=e^{-||z^i-z^j||_2}$

对t-SNE来说，它在降维后的新空间所采取的相似度算法是与之前不同的，它选取了**t-distribution**中的一种，即$S'(z^i,z^j)=\frac{1}{1+||z^i-z^j||_2}$

以下图为例，假设横轴代表了在原先$x$空间上的欧氏距离或者做降维之后在$z$空间上的欧氏距离，红线代表RBF function，是降维前的分布；蓝线代表了t-distribution，是降维后的分布

你会发现，降维前后相似度从RBF function到t-distribution：

- 如果原先两个点距离($\Delta x$)比较近，则降维转换之后，它们的相似度($\Delta y$)依旧是比较接近的
- 如果原先两个点距离($\Delta x$)比较远，则降维转换之后，它们的相似度($\Delta y$)会被拉得更远

![image-20201022172451292](C:\Users\dell\AppData\Roaming\Typora\typora-user-images\image-20201022172451292.png)

也就是说t-SNE可以聚集相似的样本点，同时还会放大不同类别之间的距离，从而使得不同类别之间的分界线非常明显。