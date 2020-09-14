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