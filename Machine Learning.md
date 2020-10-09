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