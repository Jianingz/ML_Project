# 深度学习技术入门与实战

标签（空格分隔）： MLP--任海霞部分

---
多层感知器（MLP）

---

多层感知器（Multilayer Perceptron,缩写MLP）是一种前向结构的人工神经网络，映射一组输入向量到一组输出向量。MLP可以被看作是一个有向图，由多个的节点层所组成，每一层都全连接到下一层。除了输入节点，每个节点都是一个带有非线性激活函数的神经元（或称处理单元）。一种被称为反向传播算法的监督学习方法常被用来训练MLP。本章将从感知器、激活函数、多层感知器、任何训练已经python代码实例来讲述多层感知器。

##感知器（类神经元）
感知器最初是Frank Rosenblatt在1957年就职于康奈尔航空实验室（Cornell Aeronautical Laboratory）时所发明的一种人工神经网络。它可以被视为一种最简单形式的前馈神经网络，是一种二元线性分类器。

Frank Rosenblatt给出了相应的感知机学习算法，常用的有感知机学习、最小二乘法和梯度下降法。譬如，感知机利用梯度下降法对损失函数进行极小化，求出可将训练数据进行线性划分的分离超平面，从而求得感知机模型。

感知机是生物神经细胞的简单抽象。神经细胞结构大致可分为：树突、突触、细胞体及轴突。单个神经细胞可被视为一种只有两种状态的机器——激动时为‘是’，而未激动时为‘否’。神经细胞的状态取决于从其它的神经细胞收到的输入信号量，及突触的强度（抑制或加强）。当信号量总和超过了某个阈值时，细胞体就会激动，产生电脉冲。电脉冲沿着轴突并通过突触传递到其它神经元。为了模拟神经细胞行为，与之对应的感知机基础概念被提出，如权量（突触）、偏置（阈值）及激活函数（细胞体）。

在人工神经网络领域中，感知机也被指为单层的人工神经网络，以区别于较复杂的多层感知机（Multilayer Perceptron）。作为一种线性分类器，（单层）感知机可说是最简单的前向人工神经网络形式。尽管结构简单，感知机能够学习并解决相当复杂的问题。感知机主要的本质缺陷是它不能处理线性不可分问题。

感知器本质上是给予不同的输入不同的权重，然后得出最终输出，给出决策。举一个简单的例子，要不要买iPhone手机价钱这件事。它可能的关联因素有很多，比如市场上其他品牌的手机、iPhone手机自身的特点、该类型手机出厂的时间。但是作为一个买家，ta对不同的因素的看重点不同，有些人很喜欢iPhone以至于其他品牌的手机对他的影响很小，那这条输入的权重就很小，而另一些人只是想买手机，所以其他品牌的手机对他的影响很大，那他的这条输入的权重就很大。

感知器可以表示为 f:RN→{−1,1} 的映射函数。其中 f 的形式如下：
![]( https://upload.wikimedia.org/wikipedia/commons/9/97/Ncell.png)

其中，w 和 b 都是 N 维向量，是感知器的模型参数。感知器的训练过程其实就是求解w 和 b 的过程。正确的 w 和 b 所构成的超平面 w.x+b=0 恰好将两类数据点分割在这个平面的两侧。


##激活函数（sigmoid,Relu等)

激活函数的主要作用是提供网络的非线性建模能力。如果没有激活函数，那么该网络仅能够表达线性映射，此时即便有再多的隐藏层，其整个网络跟单层神经网络也是等价的。因此也可以认为，只有加入了激活函数之后，深度神经网络才具备了分层的非线性映射学习能力。

可微性： 当优化方法是基于梯度的时候，这个性质是必须的。

单调性： 当激活函数是单调的时候，单层网络能够保证是凸函数。

输出值的范围： 当激活函数输出值是 有限 的时候，基于梯度的优化方法会更加 稳定，因为特征的表示受有限权值的影响更显著;当激活函数的输出是 无限 的时候，模型的训练会更加高效，不过在这种情况小，一般需要更小的learning rate

###激活函数如何选择

从定义来看，几乎所有的连续可导函数都可以用作激活函数。但目前常见的多是分段线性和具有指数形状的非线性函数。

选择的时候，就是根据各个函数的优缺点来配置，例如：

如果使用 ReLU，要小心设置 learning rate，注意不要让网络出现很多 “dead” 神经元，如果不好解决，可以试试 Leaky ReLU、PReLU 或者 Maxout。

最好不要用 sigmoid，你可以试试 tanh，不过可以预期它的效果会比不上 ReLU 和 Maxout。

一般来说，在分类问题上建议首先尝试 ReLU，其次ELU，这是两类不引入额外参数的激活函数。然后可考虑使用具备学习能力的PReLU和MPELU，并使用正则化技术，例如应该考虑在网络中增加Batch Normalization层。

通常来说，很少会把各种激活函数串起来在一个网络中使用的。

###常见的激活函数

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

###激活函数对比

![](https://img-blog.csdn.net/20180409093656794)

##多个感知器级联（图示结构、权重）
多层感知器（Multi Layer Perceptron，即 MLP）包括至少一个隐藏层（除了一个输入层和一个输出层以外）。单层感知器只能学习线性函数，而多层感知器也可以学习非线性函数。
![](http://obmpvqs90.bkt.clouddn.com/muti-layer-perceptron.png)
![](http://obmpvqs90.bkt.clouddn.com/cost_func_quadratic.png)
如图所示的感知器由输入层、隐藏层和输出层。其中输入层有四个节点， X取外部输入（皆为根据输入数据集取的数字值）。在输入层不进行任何计算，所以输入层节点的输出是四个值被传入隐藏层。
隐藏层有三个节点，隐藏层每个节点的输出取决于输入层的输出以及连接所附的权重。输出层有两个节点，从隐藏层接收输入，并执行类似高亮出的隐藏层的计算。输出值分别是预测的分类和聚类的结果。
给出一系列特征 X = (x1, x2, ...) 和目标 Y，一个多层感知器可以以分类或者回归为目的，学习到特征和目标之间的关系。
##如何训练（反向传播）

##损失函数

##python代码实例（API调用和手写代码实现）

