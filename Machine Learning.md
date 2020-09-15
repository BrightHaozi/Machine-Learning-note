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