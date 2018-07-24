 深度学习技术入门与实战

MLP--任海霞部分

---
多层感知器（MLP）

---

多层感知器（Multilayer Perceptron,缩写MLP）是一种前向结构的人工神经网络，映射一组输入向量到一组输出向量。MLP可以被看作是一个有向图，由多个的节点层所组成，每一层都全连接到下一层。除了输入节点，每个节点都是一个带有非线性激活函数的神经元（或称处理单元）。一种被称为反向传播算法的监督学习方法常被用来训练MLP。本章将从MLP的组成：感知器、激活函数、多层感知器和任何训练、以及Python代码实例来讲述多层感知器。

## 感知器（类神经元）
感知器最初是Frank Rosenblatt在1957年就职于康奈尔航空实验室（Cornell Aeronautical Laboratory）时所发明的一种人工神经网络。它可以被视为一种最简单形式的前馈神经网络，是一种二元线性分类器。

Frank Rosenblatt给出了相应的感知机学习算法，常用的有感知机学习、最小二乘法和梯度下降法。譬如，感知机利用梯度下降法对损失函数进行极小化，求出可将训练数据进行线性划分的分离超平面，从而求得感知机模型。

感知机是生物神经细胞的简单抽象。神经细胞结构大致可分为：树突、突触、细胞体及轴突。单个神经细胞可被视为一种只有两种状态的机器——激动时为‘是’，而未激动时为‘否’。神经细胞的状态取决于从其它的神经细胞收到的输入信号量，及突触的强度（抑制或加强）。当信号量总和超过了某个阈值时，细胞体就会激动，产生电脉冲。电脉冲沿着轴突并通过突触传递到其它神经元。为了模拟神经细胞行为，与之对应的感知机基础概念被提出，如权量（突触）、偏置（阈值）及激活函数（细胞体）。

在人工神经网络领域中，感知机也被指为单层的人工神经网络，以区别于较复杂的多层感知机（Multilayer Perceptron）。作为一种线性分类器，（单层）感知机可说是最简单的前向人工神经网络形式。尽管结构简单，感知机能够学习并解决相当复杂的问题。感知机主要的本质缺陷是它不能处理线性不可分问题。

感知器本质上是给予不同的输入不同的权重，然后得出最终输出，给出决策。举一个简单的例子，要不要买iPhone手机价钱这件事。它可能的关联因素有很多，比如市场上其他品牌的手机、iPhone手机自身的特点、该类型手机出厂的时间。但是作为一个买家，ta对不同的因素的看重点不同，有些人很喜欢iPhone以至于其他品牌的手机对他的影响很小，那这条输入的权重就很小，而另一些人只是想买手机，所以其他品牌的手机对他的影响很大，那他的这条输入的权重就很大。

感知器可以表示为 f:RN→{−1,1} 的映射函数。其中 f 的形式如下：

![]( https://upload.wikimedia.org/wikipedia/commons/9/97/Ncell.png)

其中，w 和 b 都是 N 维向量，是感知器的模型参数。感知器的训练过程其实就是求解w 和 b 的过程。正确的 w 和 b 所构成的超平面 w.x+b=0 恰好将两类数据点分割在这个平面的两侧。


## 激活函数（sigmoid,Relu等)

激活函数的主要作用是提供网络的非线性建模能力。如果没有激活函数，那么该网络仅能够表达线性映射，此时即便有再多的隐藏层，其整个网络跟单层神经网络也是等价的。因此也可以认为，只有加入了激活函数之后，深度神经网络才具备了分层的非线性映射学习能力。

可微性： 当优化方法是基于梯度的时候，这个性质是必须的。

单调性： 当激活函数是单调的时候，单层网络能够保证是凸函数。

输出值的范围： 当激活函数输出值是 有限 的时候，基于梯度的优化方法会更加 稳定，因为特征的表示受有限权值的影响更显著;当激活函数的输出是 无限 的时候，模型的训练会更加高效，不过在这种情况小，一般需要更小的learning rate

### 激活函数如何选择

从定义来看，几乎所有的连续可导函数都可以用作激活函数。但目前常见的多是分段线性和具有指数形状的非线性函数。

选择的时候，就是根据各个函数的优缺点来配置，例如：

如果使用 ReLU，要小心设置 learning rate，注意不要让网络出现很多 “dead” 神经元，如果不好解决，可以试试 Leaky ReLU、PReLU 或者 Maxout。

最好不要用 sigmoid，你可以试试 tanh，不过可以预期它的效果会比不上 ReLU 和 Maxout。

一般来说，在分类问题上建议首先尝试 ReLU，其次ELU，这是两类不引入额外参数的激活函数。然后可考虑使用具备学习能力的PReLU和MPELU，并使用正则化技术，例如应该考虑在网络中增加Batch Normalization层。

通常来说，很少会把各种激活函数串起来在一个网络中使用的。

### 常见的激活函数

1.linear activation function

![](https://img-blog.csdn.net/20180409094955629?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zODM2ODk0MQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

2.logistic function

![](https://img-blog.csdn.net/20180409094103904?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zODM2ODk0MQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

![](https://img-blog.csdn.net/20180409094120461?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zODM2ODk0MQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

优点：

在特征相差比较复杂或是相差不是特别大时效果比较好。

缺点：

1 激活函数计算量大，反向传播求误差梯度时，求导涉及除法；

2 反向传播时，很容易就会出现梯度消失的情况，从而无法完成深层网络的训练。Sigmoids saturate and kill gradients. （sigmoid饱和和消失梯度）sigmoid 有一个非常致命的缺点，当输入非常大或者非常小的时候（saturation），这些神经元的梯度是接近于0的（因为反向传播时候δj = σ'(aj)∑δkwkj，即梯度计算要用到前面的输入），从图中可以看出梯度的趋势（越是趋于两极，z的变化对于f(z)就变得非常小，再经过反向传导的链式法则运算，最后得到的梯度就会变得很小，出现所谓的梯度消失的情况）。具体来说，由于在后向传递过程中，sigmoid向下传导的梯度包含了一个f'(x)因子（sigmoid关于输入的导数），因此一旦输入落入饱和区，f'(x)就会变得接近于0，导致了向底层传递的梯度也变得非常小。所以，你需要尤其注意参数的初始值来尽量避免saturation的情况。如果你的初始值很大的话，大部分神经元可能都会处在saturation的状态而把gradient kill掉，这会导致网络变的很难学习。

一般来说， sigmoid 网络在 5 层之内就会产生梯度消失现象。

梯度消失问题至今仍然存在，但被新的优化方法有效缓解了，例如DBN中的分层预训练，Batch Normalization的逐层归一化，Xavier和MSRA权重初始化等代表性技术。这个问题也可以通过选择激活函数进行改善，比如PReLU；在LSTM中你可以选择关闭“遗忘闸门”来避免改变内容，即便打开了“遗忘闸门”，模型也会保持一个新旧值的加权平均。

Sigmoid 的饱和性虽然会导致梯度消失，但也有其有利的一面。例如它在物理意义上最为接近生物神经元。 (0, 1) 的输出还可以被表示作概率，或用于输入的归一化，代表性的如Sigmoid交叉熵损失函数。

3 Sigmoid 的 output 不是0均值。这是不可取的，因为这会导致后一层的神经元将得到上一层输出的非0均值的信号作为输入。产生的一个结果就是：如果数据进入神经元的时候是正的(e.g. x>0elementwise in f=wTx+b)，那么 w 计算出的梯度也会始终都是正的或者始终都是负的 then the gradient on the weights w" role="presentation">w will during backpropagation become either all be positive, or all negative (depending on the gradient of the whole expression f" role="presentation">f)。 当然了，如果你是按batch去训练，那么那个batch可能得到不同的信号，所以这个问题还是可以缓解一下的。因此，非0均值这个问题虽然会产生一些不好的影响，不过跟上面提到的 kill gradients 问题相比还是要好很多的。

3.softmax function
![](https://img-blog.csdn.net/20180409092201199?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zODM2ODk0MQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

softmax 会给出每个分类的类别概率，且其概率之后为1。

可用于计算与标准样本之间的差距、且优化参数时方便计算。

4.hyperbolic tangent（双曲正切）
![](https://img-blog.csdn.net/20180409092502456?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zODM2ODk0MQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
![](https://img-blog.csdn.net/20180409092808856)

双曲正切优于逻辑函数的优点是它具有更宽的输出频谱和开放间隔（-1,1）范围，可以改善反向传播算法的收敛性。

优点：

tanh在特征相差明显时的效果会很好，在循环过程中会不断扩大特征效果。

与 sigmoid 的区别是，tanh 是 0 均值的，因此实际应用中 tanh 会比 sigmoid 更好。文献 [LeCun, Y., et al., Backpropagation applied to handwritten zip code recognition. Neural computation, 1989. 1(4): p. 541-551.] 中提到tanh 网络的收敛速度要比sigmoid快，因为 tanh 的输出均值比 sigmoid 更接近 0，SGD会更接近 natural gradient[4]（一种二次优化技术），从而降低所需的迭代次数。

缺点：

也具有软饱和性。

5.rectified linear unit activation （ReLU）
![](https://img-blog.csdn.net/20180409093328852?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zODM2ODk0MQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

ReLU解决（渐变消失问题）随着Z变大，学习参数的速度变得越来越慢，甚至趋向于0的问题。

ReLU 在x<0 时硬饱和。由于 x>0时导数为 1，所以，ReLU 能够在x>0时保持梯度不衰减，从而缓解梯度消失问题。但随着训练的推进，部分输入会落入硬饱和区，导致对应权重无法更新。这种现象被称为“神经元死亡”。

优点：

虽然2006年Hinton教授提出通过分层无监督预训练解决深层网络训练困难的问题，但是深度网络的直接监督式训练的最终突破，最主要的原因是采用了新型激活函数ReLU。与传统的sigmoid激活函数相比，ReLU能够有效缓解梯度消失问题，从而直接以监督的方式训练深度神经网络，无需依赖无监督的逐层预训练。

Krizhevsky et al. 发现使用 ReLU 得到的SGD的收敛速度会比 sigmoid/tanh 快很多(看右图)。有人说这是因为它是linear，而且非饱和的 non-saturating。

相比于 sigmoid/tanh，ReLU 只需要一个阈值就可以得到激活值，而不用去算一大堆复杂的运算。

ReLU另外一个性质是提供神经网络的稀疏表达能力。PReLU[10]、ELU[7]等激活函数不具备这种稀疏性，但都能够提升网络性能。

缺点：

当然 ReLU 也有缺点，就是训练的时候很”脆弱”，很容易就”die”了. 什么意思呢？举个例子：一个非常大的梯度流过一个 ReLU 神经元，更新过参数之后（lz：参数变为一个很大的负值，之后正向传播时很大概率会使激活为0，就死了δj = σ'(aj)∑δkwkj中aj为0之后δj也一直为0），这个神经元可能再也不会对任何数据有激活现象了。如果这个情况发生了，那么这个神经元的梯度就永远都会是0。实际操作中，如果你的learning rate 很大，那么很有可能你网络中的40%的神经元都”dead”了。 当然，如果你设置了一个合适的较小的learning rate（大的梯度流过一ReLU 神经元会变成小梯度，参数不会变得特别小，正向传播时激活就不一定为0了），这个问题发生的情况其实也不会太频繁。

### 激活函数对比

![](https://img-blog.csdn.net/20180409093656794)

## 多个感知器级联（图示结构、权重）
多层感知器（Multi Layer Perceptron，即 MLP）包括至少一个隐藏层（除了一个输入层和一个输出层以外）。单层感知器只能学习线性函数，而多层感知器也可以学习非线性函数。
![](http://obmpvqs90.bkt.clouddn.com/muti-layer-perceptron.png)
![](http://obmpvqs90.bkt.clouddn.com/cost_func_quadratic.png)

如图所示的感知器由输入层、隐藏层和输出层。其中输入层有四个节点， X取外部输入（皆为根据输入数据集取的数字值）。在输入层不进行任何计算，所以输入层节点的输出是四个值被传入隐藏层。
隐藏层有三个节点，隐藏层每个节点的输出取决于输入层的输出以及连接所附的权重。输出层有两个节点，从隐藏层接收输入，并执行类似高亮出的隐藏层的计算。输出值分别是预测的分类和聚类的结果。
给出一系列特征 X = (x1, x2, ...) 和目标 Y，一个多层感知器可以以分类或者回归为目的，学习到特征和目标之间的关系。

## 如何训练（反向传播）
神经网络是一个模型，我们需要让模型自己学习它的参数也就是神经网络中的权重。当然一个神经网络的连接方式、网络的层数、每层的节点数这些参数，则不是学习出来的，而是人为事先设置的。对于这些人为设置的参数，我们称之为超参数(Hyper-Parameters)。
接下来，我们将要介绍神经网络的训练算法：反向传播算法。
其主要思想是：

（1）将训练集数据输入到神经网络输入层，经过隐藏层，最后达到输出层并输出结果，这是ANN的前向传播过程；

（2）由于神经网络的输出结果与实际结果有误差，则计算估计值与实际值之间的误差，并将该误差从输出层向隐藏层反向传播，直至传播到输入层；

（3）在反向传播的过程中，根据误差调整各种参数的值；不断迭代上述过程，直至收敛。

它的数学推导方法是：

1.变量定义

![](https://img-blog.csdn.net/20160401202509000)
 上图是一个三层人工神经网络，layer1至layer3分别是输入层、隐藏层和输出层。如图，先定义一些变量：
![](https://img-blog.csdn.net/20160401202738142)表示第L-1层的第k个神经元连接到第L层的第j个神经元的权重
![](https://img-blog.csdn.net/20160401202925814)表示第L层的第j个神经元的偏置
![](https://img-blog.csdn.net/20160401202949221)表示第L层的第j个神经元的输入，也即：
![](https://img-blog.csdn.net/20160401203046815)
![](https://img-blog.csdn.net/20160401203055737)表示第L层的第j个神经元的输出，也即
![](https://img-blog.csdn.net/20160401203135690)
其中![](https://img-blog.csdn.net/20160401203156534)表示激活函数

2.代价函数

以训练集![](https://www.zhihu.com/equation?tex=%5Cleft%5C%7B+%28x%5E%7B%281%29%7D%2Cy%5E%7B%281%29%7D%29%2C+%28x%5E%7B%282%29%7D%2Cy%5E%7B%282%29%7D%29%2C+...+%2C+%28x%5E%7B%28m%29%7D%2Cy%5E%7B%28m%29%7D%29+%5Cright%5C%7D)
为例，其中有 m 个训练样本，每个包含一组输入 x^{(i)} 和一组输出 y^{(i)} 。一般来说，我们使用的神经网络的代价函数是逻辑回归里代价函数的一般形式。
在逻辑回归中，我们的代价函数通常为： ![](https://www.zhihu.com/equation?tex=J%28%5CTheta%29%3D-%5Cfrac%7B1%7D%7Bm%7D%5B%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%5Csum_%7Bk%3D1%7D%5E%7BK%7D%7By_%7Bk%7D%5E%7B%28i%29%7Dlog%28h_%7B%5CTheta%7D%28x%5E%7B%28i%29%7D%29%29_%7Bk%7D%2B%281-y_%7Bk%7D%5E%7B%28i%29%7D%29log%281-%28h_%7B%5CTheta%7D%28x%5E%7B%28i%29%7D%29%29_%7Bk%7D%29%7D%5D%2B%5Cfrac%7B%5Clambda%7D%7B2m%7D%5Csum_%7Bl%3D1%7D%5E%7BL-1%7D%5Csum_%7Bi%3D1%7D%5E%7Bs_%7Bl%7D%7D%5Csum_%7Bj%3D1%7D%5E%7Bs_%7Bl%2B1%7D%7D%28%5CTheta_%7Bji%7D%5E%7B%28l%29%7D%29%5E%7B2%7D)其中![](https://www.zhihu.com/equation?tex=h_%7B%5CTheta%7D%28x%29%5Cin+R%5E%7BK%7D)（K 维向量），![](https://www.zhihu.com/equation?tex=%28h_%7B%5CTheta%7D%28x%29%29_%7Bi%7D)表示第 i 个输出。
即每一个的逻辑回归算法的代价函数，按输出的顺序 1 ～ K，依次相加。

3.公式及推导

这里我们先定义第l层的第j个神经元产生的误差为：![](https://img-blog.csdn.net/20160401203433129)当输入的样本数量为1时，代价函数为：![](https://img-blog.csdn.net/20160401203458879)
首先我们来计算最后一层神经网络产生的错误为：![](https://img-blog.csdn.net/20160401203552879)
其中![](https://img-blog.csdn.net/20160401203529707)表示Hadamard乘积，用于矩阵或向量之间点对点的乘法运算。
由此出发，我们可以计算每层神经网络的误差：![](https://img-blog.csdn.net/20161230105754424)
其中权重的梯度是：![](https://img-blog.csdn.net/20160401203634161)
偏置的梯度是：![](https://img-blog.csdn.net/20160401203656833)
综上所诉：反向传播的实现过程如下：
对于训练集中的每个样本x，设置输入层（Input layer）对应的激活值：
前向传播：![](https://img-blog.csdn.net/20160401203739849)
计算输出层产生的错误：![](https://img-blog.csdn.net/20160401203820380)
反向传播错误：![](https://img-blog.csdn.net/20160401203833583)
使用梯度下降（gradient descent），训练参数：![](https://img-blog.csdn.net/20160401203848037)
## 损失函数

损失函数（loss function）是用来估量你模型的预测值f(x)与真实值Y的不一致程度，它是一个非负实值函数,通常使用L(Y, f(x))来表示，损失函数越小，模型的鲁棒性就越好。损失函数是经验风险函数的核心部分，也是结构风险函数重要组成部分。模型的结构风险函数包括了经验风险项和正则项，通常可以表示成如下表达式：
![](https://img-blog.csdn.net/20180712203110396?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zODM2ODk0MQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
其中，前面的均值函数表示的是经验风险函数，L代表的是损失函数，后面的Φ是正则化项（regularizer）或者叫惩罚项（penalty term），它可以是L1，也可以是L2，或者其他的正则函数。整个式子表示的意思是找到使目标函数最小时的θ值。下面主要列出几种常见的损失函数。

一、log对数损失函数（逻辑回归）

有些人可能觉得逻辑回归的损失函数就是平方损失，其实并不是。平方损失函数可以通过线性回归在假设样本是高斯分布的条件下推导得到，而逻辑回归得到的并不是平方损失。在逻辑回归的推导中，它假设样本服从伯努利分布（0-1分布），然后求得满足该分布的似然函数，接着取对数求极值等等。而逻辑回归并没有求似然函数的极值，而是把极大化当做是一种思想，进而推导出它的经验风险函数为：最小化负的似然函数（即max F(y, f(x)) —> min -F(y, f(x)))。从损失函数的视角来看，它就成了log损失函数了。

log损失函数的标准形式：
![](https://img-blog.csdn.net/20180713140524863?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zODM2ODk0MQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70) 刚刚说到，取对数是为了方便计算极大似然估计，因为在MLE中，直接求导比较困难，所以通常都是先取对数再求导找极值点。损失函数L(Y, P(Y|X))表达的是样本X在分类Y的情况下，使概率P(Y|X)达到最大值（换言之，就是利用已知的样本分布，找到最有可能（即最大概率）导致这种分布的参数值；或者说什么样的参数才能使我们观测到目前这组数据的概率最大）。因为log函数是单调递增的，所以logP(Y|X)也会达到最大值，因此在前面加上负号之后，最大化P(Y|X)就等价于最小化L了。

逻辑回归的P(Y=y|x)表达式如下（为了将类别标签y统一为1和0，下面将表达式分开表示）：![](https://zhihu.com/equation?tex=P%28Y%3Dy%7Cx%29+%3D+%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%0Ah_%5Ctheta%28x%29+%3D+g%28f%28x%29%29+%3D+%5Cfrac%7B1%7D%7B1+%2B+exp%5C%7B-f%28x%29%5C%7D+%7D%26+%2Cy%3D1%5C%5C+%0A1+-+h_%5Ctheta%28x%29+%3D+1+-+g%28f%28x%29%29+%3D+%5Cfrac%7B1%7D%7B1+%2B+exp%5C%7Bf%28x%29%5C%7D+%7D+%26+%2Cy%3D0%0A%5Cend%7Bmatrix%7D%5Cright.)
将它带入到上式，通过推导可以得到logistic的损失函数表达式，如下：
![](https://zhihu.com/equation?tex=L%28y%2CP%28Y%3Dy%7Cx%29%29+%3D+%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%0A+%5Clog+%281%2Bexp%5C%7B-f%28x%29%5C%7D%29+%26+%2Cy%3D1%5C%5C+%0A+%5Clog+%281%2Bexp%5C%7B+f%28x%29%5C%7D%29++%26+%2Cy%3D0%5C%5C+%0A%5Cend%7Bmatrix%7D%5Cright.)
逻辑回归最后得到的目标式子如下：
![](https://img-blog.csdn.net/20180713140554331?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zODM2ODk0MQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)上面是针对二分类而言的。这里需要解释一下：之所以有人认为逻辑回归是平方损失，是因为在使用梯度下降来求最优解的时候，它的迭代式子与平方损失求导后的式子非常相似，从而给人一种直观上的错觉。

这里有个PDF可以参考一下：Lecture 6: logistic regression.pdf.

二、平方损失函数（最小二乘法, Ordinary Least Squares ）

最小二乘法是线性回归的一种，OLS将问题转化成了一个凸优化问题。在线性回归中，它假设样本和噪声都服从高斯分布（为什么假设成高斯分布呢？其实这里隐藏了一个小知识点，就是中心极限定理，可以参考【central limit theorem】），最后通过极大似然估计（MLE）可以推导出最小二乘式子。最小二乘的基本原则是：最优拟合直线应该是使各点到回归直线的距离和最小的直线，即平方和最小。换言之，OLS是基于距离的，而这个距离就是我们用的最多的欧几里得距离。为什么它会选择使用欧式距离作为误差度量呢（即Mean squared error， MSE），主要有以下几个原因：
简单，计算方便；
欧氏距离是一种很好的相似性度量标准；
在不同的表示域变换后特征性质不变。
平方损失（Square loss）的标准形式如下：
![image](https://img-blog.csdn.net/20180713141310776?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zODM2ODk0MQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
当样本个数为n时，此时的损失函数变为：

![image](http://latex.codecogs.com/gif.latex?L%28Y%2C%20f%28X%29%29%20%3D%20%5Csum%20_%7Bi%3D1%7D%5E%7Bn%7D%28Y%20-%20f%28X%29%29%5E2)
Y-f(X)表示的是残差，整个式子表示的是残差的平方和，而我们的目的就是最小化这个目标函数值（注：该式子未加入正则项），也就是最小化残差的平方和（residual sum of squares，RSS）。

而在实际应用中，通常会使用均方差（MSE）作为一项衡量指标，公式如下：
![image](https://img-blog.csdn.net/20180713140713998?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zODM2ODk0MQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

三、指数损失函数

学过Adaboost算法的人都知道，它是前向分步加法算法的特例，是一个加和模型，损失函数就是指数函数。在Adaboost中，经过m此迭代之后，可以得到fm(x):![image](http://latex.codecogs.com/gif.latex?%24%24f_m%20%28x%29%20%3D%20f_%7Bm-1%7D%28x%29%20+%20%5Calpha_m%20G_m%28x%29%24%24)
每次迭代的目的都是为了找到最小化的参数α 和G：也即

![image](http://latex.codecogs.com/gif.latex?%24%24%5Carg%20%5Cmin_%7B%5Calpha%2C%20G%7D%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20exp%5B-y_%7Bi%7D%20%28f_%7Bm-1%7D%28x_i%29%20+%20%5Calpha%20G%28x_%7Bi%7D%29%29%5D%24%24)

指数损失函数(exp-loss）的标准形式如下
![image](http://latex.codecogs.com/gif.latex?L%28y%2C%20f%28x%29%29%20%3D%20%5Cexp%5B-yf%28x%29%5D)
可以看出，Adaboost的目标式子就是指数损失，在给定n个样本的情况下，Adaboost的损失函数为：

![image](http://latex.codecogs.com/gif.latex?L%28y%2C%20f%28x%29%29%20%3D%20%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%5Cexp%5B-y_if%28x_i%29%5D)

四、Hinge损失函数（SVM）

在机器学习算法中，hinge损失函数和SVM是息息相关的。在线性支持向量机中，最优化问题可以等价于下列式子：

![image](http://latex.codecogs.com/gif.latex?%24%24%5Cmin_%7Bw%2Cb%7D%20%5C%20%5Csum_%7Bi%7D%5E%7BN%7D%20%5B1%20-%20y_i%28w%5Ccdot%20x_i%20+%20b%29%5D_%7B+%7D%20+%20%5Clambda%7C%7Cw%7C%7C%5E2%20%24%24)

如若取λ=12C且对原式进行变行，即可得到

![image](http://latex.codecogs.com/gif.latex?%24%24%5Cmin_%7Bw%2Cb%7D%20%5Cfrac%7B1%7D%7BC%7D%5Cleft%20%28%20%5Cfrac%7B1%7D%7B2%7D%5C%20%7C%7Cw%7C%7C%5E2%20%24%24%20+%20C%20%5Csum_%7Bi%7D%5E%7BN%7D%20%5Cxi_i%5Cright%20%29%24%24)
## python代码实例（API调用和手写代码实现）
本节我将搭建一个MLP来帮助识别MNIST库中的数字。实现的过程如下：
首先我们下载MNIST数据，然后对得到的数据做一个预处理，这里是将图片中的像素标准化为-1到1。
```
import os
import struct
import numpy as np
def load_mnist(path, kind='train'):
labels_path = os.path.join(path,
'%s-labels-idx1-ubyte' % kind)
images_path = os.path.join(path,'%s-images-idx3-ubyte' % kind)
with open(labels_path, 'rb') as lbpath:
 
#其中>表示big-endian，I表示无符号整数
 
magic, n = struct.unpack('>II',lbpath.read(8))
labels = np.fromfile(lbpath,
dtype=np.uint8)
with open(images_path, 'rb') as imgpath:
magic, num, rows, cols = struct.unpack(">IIII",
imgpath.read(16))
images = np.fromfile(imgpath,
dtype=np.uint8).reshape(
len(labels), 784)

#标准化像素值到-1到1
 
images = ((images / 255.) - .5) * 2
return images, labels
```
处理完成后，我们可以看一下现有数据是如何划分训练集和测试集的，这里调用shape函数实现，得到的运行结果告诉我们，我们下载的7000组数据中，60000组作为训练集，10000组作为测试集合

```
X_train, y_train = load_mnist('', kind='train')
   print('Rows: %d, columns: %d'% (X_train.shape[0], X_train.shape[1]))
X_test, y_test = load_mnist('', kind='t10k')
print('Rows: %d, columns: %d'% (X_test.shape[0], X_test.shape[1]))
```

接着我们使用tensorflow搭建神经网络，这里我们搭建一个3层的神经网络，h1和h2使用双曲正切函数作为激活函数，h3使用softmax来得到输入的最大可能性值。
```
import tensorflow as tf
n_features = X_train_centered.shape[1]
n_classes = 10
random_seed = 123
np.random.seed(random_seed)
g = tf.Graph()
with g.as_default():
tf.set_random_seed(random_seed)
tf_x = tf.placeholder(dtype=tf.float32,
shape=(None, n_features),
name='tf_x')
tf_y = tf.placeholder(dtype=tf.int32,
shape=None, name='tf_y')
y_onehot = tf.one_hot(indices=tf_y, depth=n_classes)
h1 = tf.layers.dense(inputs=tf_x, units=50,
activation=tf.tanh,
name='layer1')
h2 = tf.layers.dense(inputs=h1, units=50,
activation=tf.tanh,
name='layer2')
logits = tf.layers.dense(inputs=h2,
units=10,
activation=None,
name='layer3')
predictions = {
'classes' : tf.argmax(logits, axis=1,
name='predicted_classes'),
'probabilities' : tf.nn.softmax(logits,
name='softmax_tensor')
}
```

接着我们定义损失函数和优化器，损失函数这里我们使用softmax_cross_entropy来计算softmax交叉熵损失，同时我们使用梯度下降算法来优化网络。
```
with g.as_default():
cost = tf.losses.softmax_cross_entropy(
onehot_labels=y_onehot, logits=logits)
optimizer = tf.train.GradientDescentOptimizer(
learning_rate=0.001)
train_op = optimizer.minimize(
loss=cost)
 
init_op = tf.global_variables_initializer()
 
#return a generator batches of data
 
def create_batch_generator(X, y, batch_size=128, shuffle=False):
 
X_copy = np.array(X)
 
y_copy = np.array(y)
 
if shuffle:
data = np.column_stack((X_copy, y_copy))
np.random.shuffle(data)
X_copy = data[:, :-1]
y_copy = data[:, -1].astype(int)
for i in range(0, X.shape[0], batch_size):
yield (X_copy[i:i+batch_size, :], y_copy[i:i+batch_size])>>> ## create a session to launch the graph
>>> sess = tf.Session(graph=g)
>>> ## run the variable initialization operator
>>> sess.run(init_op)
>>>
>>> ## 50 epochs of training:
>>> for epoch in range(50):
... training_costs = []
... batch_generator = create_batch_generator(
... X_train_centered, y_train,
... batch_size=64, shuffle=True)
... for batch_X, batch_y in batch_generator:
... ## prepare a dict to feed data to our network:
... feed = {tf_x:batch_X, tf_y:batch_y}
... _, batch_cost = sess.run([train_op, cost],
feed_dict=feed)
... training_costs.append(batch_cost)
... print(' -- Epoch %2d '
... 'Avg. Training Loss: %.4f' % (
... epoch+1, np.mean(training_costs)
... ))
```
在得到参数最优的网络后，我们使用测试集来验证网络的性能

```
>>> ## do prediction on the test set:
>>> feed = {tf_x : X_test_centered}
>>> y_pred = sess.run(predictions['classes'],
... feed_dict=feed)
>>>
>>> print('Test Accuracy: %.2f%%' % (
 
... 100*np.sum(y_pred == y_test)/y_test.shape[0]))
```
![image](https://img-blog.csdn.net/20180404152702228?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zODM2ODk0MQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

###     前沿进展
近些年随着深度学习的高性能，被广泛使用来解决一些复杂问题，于是许多研究人员对其进行了深入研究，使得其发展很快。本节我们分别从初始化方法、网络层数和激活函数的选
择、模型结构、学习算法这三个方面对近几年MLP研究的新进展进行介绍。
### 初始化方法、网络层数和激活函数的选择
首先是初始化方法部分，当我们组建一个MLP网络时，一个困扰我们的问题是初始值的设定与学习结果之间的关系。Erhan等人在轨迹可视化研究中指出即使从相近的值开始训练深度结构神经网络，不同的初始值也会学习到不同的局部极值，同时发现用无监督预训练初始化模型的参数学习得到的极值与随机初始化学习得到的极值差异比较大，用无监督预训练初始化模型的参数学习得到的模型具有更好的泛化
误差。Bengio与Krueger等人指出用特定的方法设定训练样例的初始分布和排列顺序可以产生更好的训练结果，用特定的方法初始化参数，使其与均匀采样得到的参数不同，会对梯度下降算法训练的结果产生很大的影响。Giorot等人指出通过设定一组初始权值使得每一层深度结构神经网络的 Jacobian矩阵的奇异值接近1,在很大程度上减小了监督深度结构神经网络和有预训练过程设定初值的深度结构神经网络之间的学习结果差异。另外，用于深度学习的学习算法通常包含许多超参数，LeCUNY推荐一些常用的超参数，尤其适用于基于反向传播的学习算法和基于梯度的优化算法中；并讨论了如何解决有许多可调超参数的问题，描述了实际用于有效训练常用的大型深度结构神经网络的超参数的影响因素，指出深度学习训练中存在的困难。选择不同的网络隐层数和不同的非线性激活函数会对学
习结果产生不同的影响。Glorot等人研究了隐层非线性
映射关系的选择和网络的深度相互影响的问题，讨论了随机初
始化的标准梯度下降算法用于深度结构神经网络学习得到不
好的学习性能的原因。Glorot等人观察不同非线性激活函数
对学习结果的影响，得到逻辑斯蒂 Ｓ型激活单元的均值会驱使
顶层和隐层进入饱和，因而逻辑斯蒂 Ｓ型激活单元不适合用随
机初始化梯度算法学习深度结构神经网络；并据此提出了标准
梯度下降算法的一种新的初始化方案来得到更快的收敛速度，
为理解深度结构神经网络使用和不使用无监督预训练的性能
差异作出了新的贡献。Bengio等人从理论上说明深度学
习结构的表示能力随着神经网络深度的增加以指数的形式增
加，但是这种增加的额外表示能力会引起相应局部极值数量的
增加，使得在其中寻找最优值变得困难。
### 模型结构
１）DBN的结构及其变种

采用二值可见单元和隐单元RBM作为结构单元的DBN，在 MINIST等数据集上表现出很好
的性能。近几年，具有连续值单元的RBM，如mcRBM、mPot模 型和Spike-and-slabRBM  等 已 经 成 功 应 用。
Spike-and-slabRBM中Spike表示以 ０为中心的离散概率分布，
slab表示在连续域上的稠密均匀分布，可以用吉布斯采样对
Spike-and-slabRBM进行有效推断，得到优越的学习性能。

２）和—积网络 

深度学习最主要的困难是配分函数的学习，如何选择深度结构神经网络的结构使得配分函数更容易计算困扰我们很多。Poon等人
提出一种新的深度模型结构———和—积网络
（sumproductnetwork，SPN)，引入多层隐单元表示配分函数，
使得配分函数更容易计算。SPN是有根节点的有向无环图，图
中的叶节点为变量，中间节点执行和运算与积运算，连接节点
的边带有权值，它们在caltech-101和Olivetti 两个数据集上进
行实验证明了SPN的性能优于DBN和最近邻方法。

３）基于rectified单元的学习 

Glorot与Mesnil等人用降噪自编码模型来处理高维输入数据。与通常的Ｓ型和正切非线性隐单元相比，该自编码模型使用rectified 单元，使隐单元产生更加稀疏的表示。在此之前，文献已经对随机
rectified单元进行了介绍；对于高维稀疏数据，dauphin等人
采用抽样重构算法，训练过程只需要计算随机选择的很小的样
本子集的重构和重构误差，在很大程度上提高了学习速度，实
验结果显示提速了20倍。Glorot等人提出在深度结构神经
网络中，在图像分类和情感分类问题中用 rectified非线性神经
元代替双曲正切或 Ｓ型神经元，指出 rectified神经元网络在零
点产生与双曲正切神经元网络相当或者有更好的性能，能够产
生有真正零点的稀疏表示，非常适合本质稀疏数据的建模，在
理解训练纯粹深度监督神经网络的困难，搞清使用或不使用无监督预训练学习的神经网络造成的性能差异方面，可以看做新
的里程碑；Glorot等人还提出用增加L1正则化项来促进模型
稀疏性，使用无穷大的激活函数防止算法运行过程中可能引起
的数值问题。在此之前，Nair等人提出在RBM环境中rectified神经元产生的效果比逻辑斯蒂Ｓ型激活单元好，他们用
无限数量的权值相同但是负偏差变大的一组单元替换二值单
元，生成用于 ＲＢＭ的更好的一类隐单元，将RBM泛化，可以用
噪声 rectified线性单元（rectifiedlinearunits）有效近似这些 Ｓ型单元。用这些单元组成的RBM在NORB数据集上进行目标
识别以及在数据集上进行已标记人脸实际验证，得到比二值单
元更好的性能，并且可以更好地解决大规模像素强度值变化很
大的问题。

４）卷积神经网络 。

LEEH研究了用生成式子抽样单元
组成的卷积神经网络，在MNIST数字识别任务和Caltech-101 
目标分类基准任务上进行实验，显示出非常好的学习性能。
Huang等人提出一 种 新 的 卷 积 学 习 模 型———局部卷积RBM，利用对象类中的总体结构学习特征，不假定图像具有平稳特征，在实际人脸数据集上进行实验，得到性能很好的实验结果。
###  学习算法

１）深度费希尔映射方法 

Wong等人提出一种新的特
征提取方法———正则化深度费希尔映射（regularizeddeepFishermapping,RDFM）方法，学习从样本空间到特征空间的显式
映射，根据Fisher准则用深度结构神经网络提高特征的区分
度。深度结构神经网络具有深度非局部学习结构，从更少的样
本中学习变化很大的数据集中的特征，显示出比核方法更强的
特征识别能力，同时RDFM方法的学习过程由于引入正则化
因子，解决了学习能力过强带来的过拟合问题。在各种类型的
数据集上进行实验，得到的结果说明了在深度学习微调阶段运
用无监督正则化的必要性。

２）非线性变换方法 

Raiko等人提出了一种非线性变
换方法，该变换方法使得多层感知器（multi-layerperceptron,MLP
）网络的每个隐神经元的输出具有零输出和平均值上的零
斜率，使学习MLP变得更容易。将学习整个输入输出映射函
数的线性部分和非线性部分尽可能分开，用shortcut权值
（shortcutweight）建立线性映射模型，令 Fisher信息阵接近对角
阵，使得标准梯度接近自然梯度。通过实验证明非线性变换方
法的有效性，该变换使得基本随机梯度学习与当前的学习算法
在速度上不相上下，并有助于找到泛化性能更好的分类器。用
这种非线性变换方法实现的深度无监督自编码模型进行图像
分类和学习图像的低维表示的实验，说明这些变换有助于学习
深度至少达到五个隐层的深度结构神经网络，证明了变换的有
效性，提高了基本随机梯度学习算法的速度，有助于找到泛化
性更好的分类器。

３）稀疏编码对称机算法 

Ranzato等人提出一种新的
有效的无监督学习算法———稀疏编码对称机（sparseencoding symmetricmachine，SESM），能够在无须归一化的情况下有效产
生稀疏表示。SESM的损失函数是重构误差和稀疏罚函数的
加权总和，基于该损失函数比较和选择不同的无监督学习机，
提出一种与文献算法相关的迭代在线学习算法，并在理
论和实验上将SESM与RBM和PCA进行比较，在手写体数字识别 MNIST数据集和实际图像数据集上进行实验，表明该方
法的优越性。

４）迁移学习算法 

在许多常见学习场景中训练和测试数据集中的类标签不同，必须保证训练和测试数据集中的相似性进行迁移学习。Mesnil等人研究了用于无监督迁移学习场景中学习表示的不同种类模型结构，将多个不同结构的层堆栈使用无监督学习算法用于五个学习任务，并研究了用于少量已标记训练样本的简单线性分类器堆栈深度结构学习算法。Bengio研究了无监督迁移学习问题，讨论了无监督预训练有用的
原因，如何在迁移学习场景中利用无监督预训练，以及在什么情
况下需要注意从不同数据分布得到的样例上的预测问题。

５）自然语言解析算法 

Collobert基于深度递归卷积图
变换网络（graphtransformernetwor,GTN）提出一种快速可扩
展的判别算法用于自然语言解析，将文法解析树分解到堆栈层
中，只用极少的基本文本特征，得到的性能与现有的判别解析
器和标准解析器的性能相似，而在速度上有了很大提升。

６）学习率自适应方法

学习率自适应方法可用于提高深度结构神经网络训练的收敛性并且去除超参数中的学习率参数，其中包括全局学习率、层次学习率、神经元学习率和参数学习率等。最近研究人员提出了一些新的学习率自适应方法，如Duchi等人提出的自适应梯度方法和Schaul等人提出的学习率自适应方法；Hinton提出了收缩学习率方法使得平均权值更新在权值大小的1/1000数量级上；LeRoux等人提出自然梯度的对角低秩在线近似方法，并说明该算法在一些学习场景中能加速训练过程。
