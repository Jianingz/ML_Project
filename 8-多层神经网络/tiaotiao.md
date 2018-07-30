深度学习技术入门与实战

MLP--任海霞部分

---
## 8. 多层感知器（MLP）

---
多层感知器（Multilayer Perceptron,缩写MLP）是一种由一组输入向量映射到一组输出向量前向结构的人工神经网络。MLP可以被看作是一个有向图，由多个的节点层所组成，每一层都全连接到下一层。除了输入节点，每个节点都是一个带有非线性激活函数的神经元（或称处理单元）。反向传播算法的监督学习方法常被用来训练MLP。MLP的出现有效地解决了感知器无法处理线性不可分的问题，本章将从MLP的组成：感知器、激活函数、多层感知器和任何训练、Python代码实例以及MLP的前沿进展来讲述多层感知器。

## 8.1感知器（类神经元）
感知器最初是在1957年由康奈尔航空实验室（Cornell Aeronautical Laboratory）的Frank Rosenblatt发明的一种人工神经网络。它可以被视为一种最简单形式的二元线性分类器。Frank Rosenblatt给出了相应的感知机学习算法，常用的有感知机学习、最小二乘法和梯度下降法。

感知机抽象自生物神经细胞。神经细胞由树突、突触、细胞体及轴突组成。工作的时候是电信号从一端到另一端，沿着轴突从树突到树突。当电信号输入的信息量超过细胞的阈值，细胞就会“激动”，产生一个在数学上理解的“1”这样的信号，传递给其他的神经元。与之对应，在细胞安静的时候对应的是“0”。Frank使用了权量（突触）、偏置（阈值）及激活函数（细胞体）来模拟神经细胞行为。作为一种线性分类器，（单层）感知机可说是最简单的前向人工神经网络形式。

感知器本质上是给予不同的输入不同的权重，加上权重为1 的偏置经过激活函数得出最终输出，给出决策。关于权重的设置，这里举一个简单的例子，要不要买iPhone手机价钱这件事。它可能的关联因素有很多，比如市场上其他品牌的手机、iPhone手机自身的特点、该类型手机出厂的时间。但是作为一个买家，ta对不同的因素的看重点不同，有些人很喜欢iPhone以至于其他品牌的手机对他的影响很小，那这条输入的权重就很小，而另一些人只是想买手机，所以其他品牌的手机对他的影响很大，那他的这条输入的权重就很大。

感知器可以表示为n维输入向量到输出t的映射函数。 感知器的结构如下：

![]( https://upload.wikimedia.org/wikipedia/commons/9/97/Ncell.png)

其中，w1到wn是分别表示输入a1到an到感知器的权重和 b表示偏置（神经元中的阈值）。接下来的我们用w表示w1到wn，因为他们都表示输入的权重。感知器的训练过程本质上是求解参数w和 b 的过程。经过训练得到的正确的权重值和b所构成的w*a+b=0超平面 可以作为二元分类器使得不同类的数据刚好均匀分布在超平面的两侧。

## 8.2 多个感知器级联（图示结构、权重）

从上面可以看出，单个感知器可以用来处理与、或运算，但是遇到XOR这样的负责逻辑结构时，就只能束手无策，但是使用多个感知器级联就可以分割复杂的神经网络，于是多层感知器应运而生。多层感知器（Multi Layer Perceptron，即 MLP）包括至少一个隐藏层（除了一个输入层和一个输出层以外）。

![](http://obmpvqs90.bkt.clouddn.com/muti-layer-perceptron.png)

如图所示的感知器由输入层、隐藏层和输出层。其中输入层有四个节点， X表示外部输入，在输入层不进行任何计算，所以输入层节点的输出是四个值被传入隐藏层。
隐藏层有三个节点，隐藏层每个节点的输出取决于输入层的输出以及连接所附的权重。输出层有两个节点，从隐藏层接收输入，并执行计算。输出值分别是预测的分类和聚类的结果。
给出一系列特征 X = (x1, x2, ...) 和目标 Y，一个多层感知器可以以分类或者回归为目的，学习到特征和目标之间的关系。

这里举一个简单的例子，多层神经网络实现XOR运算，如果算上输入层我们的网络共有三层，如下图所示，其中第1层和第2层中的1分别是这两层的偏置单元。连线上是连接前后层的权重。

![image](https://images2015.cnblogs.com/blog/1035701/201704/1035701-20170414211000814-334243844.png)

输入：我们一共有四个训练样本，每个样本有两个特征，分别是(0, 0), (1, 0), (0, 1), (1, 1);
理想输出：真值表，样本中两个特征相同时为0，相异为1

![image.png](https://upload-images.jianshu.io/upload_images/13064452-3838e6149cd27955.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

参数：随机初始化，范围为(-1, 1)
代码实现可以参考：
```
# encoding: utf-8
#!/usr/bin/env python
import tensorflow as tf
import math
import numpy as np

INPUT_COUNT = 2
OUTPUT_COUNT = 2
HIDDEN_COUNT = 2
LEARNING_RATE = 0.1
MAX_STEPS = 5000

# 每个训练循环，我们将提供相同的输入和期望的输出数据。
INPUT_TRAIN = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
OUTPUT_TRAIN = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

# 我们使用占位符来创建节点，之后我们会输入占位符的值然后进行运算。 
# 这里我们创建二维浮点数x
inputs_placeholder = tf.placeholder("float",
shape=[None, INPUT_COUNT])
labels_placeholder = tf.placeholder("float",
shape=[None, OUTPUT_COUNT])

# 我们需要创建一个带有占位符作为关键字的Python字典对象，并将张量作为值输入。
feed_dict = {
inputs_placeholder: INPUT_TRAIN,
labels_placeholder: OUTPUT_TRAIN,
}

# 定义从输入层到隐藏层的权重和偏置
WEIGHT_HIDDEN = tf.Variable(tf.truncated_normal([INPUT_COUNT, HIDDEN_COUNT]))
BIAS_HIDDEN = tf.Variable(tf.zeros([HIDDEN_COUNT]))

#这里我们使用使用sigmoid函数作为隐藏层的激活函数
AF_HIDDEN = tf.nn.sigmoid(tf.matmul(inputs_placeholder, WEIGHT_HIDDEN) + BIAS_HIDDEN)

#  这里我们为隐藏层到输出层定义权重和偏置，这里的偏置使用tf.zero初始化，确保他们从0开始
WEIGHT_OUTPUT = tf.Variable(tf.truncated_normal([HIDDEN_COUNT, OUTPUT_COUNT]))
BIAS_OUTPUT = tf.Variable(tf.zeros([OUTPUT_COUNT]))

logits = tf.matmul(AF_HIDDEN, WEIGHT_OUTPUT) + BIAS_OUTPUT
# 这里我们用softmax计算每种输出可能性的最大值
y = tf.nn.softmax(logits)

# 使用 tf.nn.softmax_cross_entropy_with_logits op 来比较模型输出和准确的输出
#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, y)
cross_entropy = -tf.reduce_sum(labels_placeholder * tf.log(y))
# 然后使用tf.reduce_mean来计算交叉熵值作为总损失。
loss = tf.reduce_mean(cross_entropy)

# 接下来，我们实例化一个tf.train.GradientDescentOptimizer，然后使用梯度下降法来优化参数。
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

# 接下来我们利用tf.Session () 来运行表格
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
# 接下来我们用元祖来返回训练的训练进行到的程度和损失
	for step in range(MAX_STEPS):
		loss_val = sess.run([train_step, loss], feed_dict)
		if step % 100 == 0:
			print ("Step:", step, "loss: ", loss_val)
			for input_value in INPUT_TRAIN:
				print (input_value, sess.run(y, 
				feed_dict={inputs_placeholder: [input_value]}))
```

## 8.3 激活函数（sigmoid,Relu等)

加拿大蒙特利尔大学的Bengio教授在 ICML 2016的文章中给出了激活函数的定义：激活函数是映射 h:R→R，且几乎处处可导。激活函数的主要作用是提供网络的非线性建模能力。这句话是没有激活函数的神经网络本质上只能简单的线性运算，那么为了改善神经网络性能加入隐藏层的操作，将没有任何意义。故引入激活函数，神经网络才能处理非线性的问题。所以激活函数一般都是非线性的。

综上所述，激活函数具有如下性质：

1.可微性： 只有当激活函数是可微的，我们采用梯度下降的方法来优化参数。

2.单调性： 当激活函数是单调的时候，单层网络能够保证是凸函数。

3.输出值的范围选择上需要注意，当激活函数输出值是有限的时候，这时输入收到有限的权重影响，此时基于梯度的优化方法会更加稳定;当激活函数的输出是无限的时候，模型的训练会更加高效，不过在这种情况小，一般需要更小的学习率（learning rate）

### 一、常见的激活函数

1.Sigmoid 函数

sigmoid是使用频率最高的激活函数，它的表达式可以表示为：f(x)=1/（1+e^−x） ，函数值映射到（0，1）之间。

![image](https://upload-images.jianshu.io/upload_images/1667471-d2c5493f3380d6f2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/600)
sigmoid函数的优点有：

I.单调连续，且输出范围有限，因此优化稳定，可以用作输出层。

II.求导容易。

当然它的缺点也很明显。

I.函数饱和使梯度消失，使得优化难以进行

从sigmoid函数的图像可以看出，sigmoid 函数值在接近 0 或 1 的时候接近饱和，这些区域，函数的导数几乎为0，因此当使用反向传播算法优化参数时，sigmoid函数局部梯度为0的部分与整个损失函数关于该单元输出的梯度相乘，结果也会接近为 0，此时对于调整参数、改善模型性能的作用微乎其微。为了防止这样的情况出现，通常需要小心地设置权重的初始值。
  
II.sigmoid 函数不是0均值的

选择的激活函数不是0均值会导致后一层的神经元将得到上一层输出的非0均值的信号作为输入。也就是说如果数据进入神经元的时候是正的那么 w 计算出的梯度也会始终都是正的。 
这个缺点可以使用批（batch）训练来避免，每批训练的结果呈现差异的正负性，但是综合一下，仍然可以得到比较好的结果。

2.归一化指数函数（softmax function）

归一化指数函数是将一个含任意实数的K维向量 z“压缩”到另一个K维实向量中，使得每一个元素的范围都在(0,1)之间，并且所有元素的和为1。表达式如下：

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/e348290cf48ddbb6e9a6ef4e39363568b67c09d3)

softmax经常被使用做多分类网络输出。举一个简单的例子，当输入分别为3，1和-3的时候，经过softmax函数得到的输出值分别为0.88、0.12和0，可以明显的看到当输入较大时，经过softmax函数，能放大这种差异，在完成多元分类时，非常有效。

![image](https://upload-images.jianshu.io/upload_images/1667471-5bf75eefed2154f7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/643)


3.双曲正切（hyperbolic tangent）

双曲正切函数的表达式为：Y = tanh(X)，查看下面曲线，我们可以清楚地看到双曲正切函数取值（-1，1）且输出以0为中心。
![](https://ww2.mathworks.cn/help/matlab/ref/graphhyperbolictangentfunctionexample_02_zh_CN.png)


优点：

I.相比于sigmoid函数（0，1）的取值范围，tanh将值放大到了（-1，1），因此有了较sigmoid更快的收敛速度。

II.相比Sigmoid函数，其输出以0为中心，这样也避免了反向传播过程中全负或者全正的情况。

缺点：

双曲正切也具有软饱和性，也需要谨慎地选择初始化权重。

4.线性整流函数（rectified linear unit activation ReLU）

线性整流函数表达式如下，本质当输入值小于0的情况下，输出值均为0，当输入值大于等于0的时候，输出值等于输入值。

![](https://upload-images.jianshu.io/upload_images/1667471-776c7cf54955c1fd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/153)



优点：

I、与上述sigmoid、tanh函数、softmax函数不同，线性整流函数的取值范围是无穷，它的导数不会随着输入的变化而变化，以此也避免了软饱和带来的无法继续优化参数的困扰

II、Krizhevsky et al. 发现使用 ReLU 得到的参数收敛速度会比 sigmoid/tanh 快很多。

III、相比于 sigmoid/tanh，ReLU只需要一个阈值就可以得到激活值，而不用去算一大堆复杂的运算。


缺点：

当然 ReLU 也有缺点，像所有取值范围为无穷的激活函数一样，ReLU需要选择较小的学习率，才能让神经元中权重为0的比例足够小。出现权重为0的神经元的情况是：一个非常大的梯度流过一个 ReLU神经元，之后更新过参数，参数变为一个很大的负值，正向传播时很大概率会使激活为0，这个神经元再也不会对任何数据有激活现象了。当然这个问题在使用中不会频繁发生。


### 二、激活函数如何选择

从定义来看，几乎所有的连续可导函数都可以用作激活函数。但目前常见的多是分段线性和具有指数形状的非线性函数。

选择的时候，需要充分考虑各个函数的优缺点，例如：如果使用 线性整流函数（ReLU），由于它的输出值范围是无限的，因此要小心设置学习率（ learning rate），注意不要让网络出现很多权重值为0的神经元，如果没有办法避免，建议使用Leaky ReLU、PReLU 或者 Maxout代替。

一般来说，在分类问题上建议首先尝试不引入额外参数的激活函数 线性整流函数（ReLU），其次指数线性函数（ELU），然后可考虑使用具备学习能力的PReLU和MPELU，并使用正则化技术，例如应该考虑在网络中增加Batch Normalization层。

通常来说，很少会把各种激活函数串起来在一个网络中使用的。

## 8.4 损失函数（loss function）

损失函数是用来估量模型的预测值f(x)与真实值Y的不一致程度，通常使用L(Y, f(x))来表示，一个模型的损失函数越小，意味着模型越能很好的适用于各种的输入值。接下来介绍常见的损失函数。
一、0-1损失函数

0-1损失函数表达式如下图，由表达式可以看出，0-1损失函数是直接将模型输出值与准确答案做对比，这样的损失函数适用于验证当前模型的输出是否正确，但是不适合用来优化参数。

![image.png](https://upload-images.jianshu.io/upload_images/13064452-3f07e9db5189cd5c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

二、合页（Hinge）损失函数

Hinge损失函数的表达式如下图，从表达式可以看出，当数据点被正确分类，损失为0，如果没有被正确分类，损失为z。Hinge可以解间距最大化的问题，通常与SVM结合使用。

![image.png](https://upload-images.jianshu.io/upload_images/13064452-858290e7ea2e7085.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


三、平方损失函数

配方损失函数的表达式如下，由表达式可以看出是取预测值与真实值差距大的平方。

![image](https://upload-images.jianshu.io/upload_images/4155986-dfb3e453e1976caf.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/494)

当样本个数为n时，此时的损失函数变为

![image](https://upload-images.jianshu.io/upload_images/4155986-c25862df8df6f742.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/528)

在实际应用中通常使用均方差作为损失函数，其表达式如下：

![image](https://upload-images.jianshu.io/upload_images/4155986-c1dfdd8d5a14699b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/492)

四、指数损失函数

指数损失函数表达式如下：

![image](https://upload-images.jianshu.io/upload_images/4155986-b59dfa2ea516f570.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/480)

指数损失函数通常与Adaboost一起使用，Adaboost是前向分步加法算法的特例，是一个加和模型。Adaboost的表达式是

![image](https://upload-images.jianshu.io/upload_images/4155986-34c64df9d39409d5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/544)

当给定样本数量为n时，此时Adaboost的损失函数如下：

![image](https://upload-images.jianshu.io/upload_images/4155986-6bb4e5d3bebdd5cf.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/628)

使用这样的损失函数，就可以迭代求解找到最小的α和G，从而得到最优的网络模型。

五、绝对值损失函数

绝对值损失函数的表达式如下图，由表达式可以看出，绝对值损失函数较平方损失函数没有放大误差。

![image](https://upload-images.jianshu.io/upload_images/4155986-87221830ceb67bcd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/484)

六、对数损失函数

对数损失函数表达式如下，由表达式可以看出，对数损失表示在当前模型的基础上，对于样本X，Y是预测正确的概率。由于概率之间的同时满足需要使用乘法，使用对数函数将乘法转化为加法简化运算。但是由于损失函数与预测准确性有着相反的增减性。即当准确性增加时损失函数减少，故加一个负号。
![image](https://upload-images.jianshu.io/upload_images/4155986-a5fdf3873dfb2bb3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/558)

## 8.5 如何训练（反向传播）

神经网络是一个模型，它包含两类参数，一类是需要让模型自己学习它的参数：神经网络中的权重和偏置。另一类称为超参数，不需要学习：一个神经网络的连接方式、网络的层数、每层的节点数这些参数。

在正式开始介绍BP算法前，先来介绍一下BP算法的由来。1969年,人工神经网络创始人明斯基(Marrin M insky)和佩珀特(Seymour Papert)合作出版了《感知器》一书,论证了简单的线性感知器功能有限,例如不能解决如“异或”(XOR )这样的基本问题,而且对多层网络也持悲观态度。这些论让学界对神经网络研究陷入低谷期。虽然在1974年哈佛大学的Paul Werbos发明BP算法，但是当时神经网络的研究处于低潮期，并未受到改有的重视。终于在1983年，加州理工学院的物理学家John Hopfield利用神经网络，在旅行商这个NP完全问题的求解上获得当时最好成绩，引起了轰动，但是Hopfield仍然没能打消人们对多层网络的顾虑。直到David Rumelhart等学者出版的《平行分布处理:认知的微观结构探索》一书。书中完整地提出了BP算法,系统地解决了多层网络中隐单元连接权的学习问题,并在数学上给出了完整的推导。BP算法才得以出现在人们的视线中，用于解决多层神经网络中参数优化的问题。

BP算法具有实现映射能力、记忆机制和容错性的三种特性。实现映射能力是指，记忆机制是指具有足够多隐单元的三层神经同络可以记忆任给的样本集。同时孙德保、高超对三层BP同络的容错性和抗干扰性进行了研究，指出三层BP网络的容错能力取决于输入层到隐含层的连接权值矩阵与隐含层到辖出层连接权值矩阵的乘积的结果。

接下来，我们介绍反向传播算法的主要思想：

（1）首先是前向传播过程：这个过程中需要将训练集数据输入到神经网络输入层，经过隐藏层，最后达到输出层并输出结果。

（2）反向传播过程：使用损失函数计算神经网络的实际输出结果与真实结果的误差，将该误差利用梯度下降的方法从输出层向隐藏层反向传播，直至传播到输入层。

（3）优化参数：在反向传播的过程中，根据误差调整各种参数的值。

（4）不断迭代上述过程，直至收敛。

它的数学推导方法是：

首先我们对进行变量定义，如下：

![image](https://upload-images.jianshu.io/upload_images/1241397-6a97cc9d982c4fcb.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/700)

以给定数据集![image](https://upload-images.jianshu.io/upload_images/13064452-93a43d724bc5ad77.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)为例，接下来进行参数的说明：

d：表示输入层神经元个数

l：输出层神经元的个数

q: 隐藏层神经元的个数

![image](https://upload-images.jianshu.io/upload_images/13064452-377fc678f23c301b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)：输出层第j个神经元的阈值

![image](https://upload-images.jianshu.io/upload_images/13064452-9662b094a1424da6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)：隐藏层第h个神经元的阈值

![image](https://upload-images.jianshu.io/upload_images/13064452-048c9cd3e43b31ce.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)：表示输入层i个神经元与隐藏层第h个神经元的连接权重

![image](https://upload-images.jianshu.io/upload_images/13064452-a97b9c694482e4f7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)：表示隐藏层h个神经元与输出层第j个神经元的连接权重

![image](https://upload-images.jianshu.io/upload_images/13064452-b9f0108f951f91b8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)：表示隐藏层第h个神经元的输出

![image.png](https://upload-images.jianshu.io/upload_images/13064452-5b6967fb317e9697.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)：表示第h个神经元的输入

![image.png](https://upload-images.jianshu.io/upload_images/13064452-d0c88fc8c4bfb938.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)：表示输出层第j个神经元的输入

接着我们需要明确损失函数，我们以均方误差函数为例：

对于给定的样本![image.png](https://upload-images.jianshu.io/upload_images/13064452-4fb3f15938fb6c80.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)假设神经网络的输出为![image.png](https://upload-images.jianshu.io/upload_images/13064452-4fb3f15938fb6c80.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)，则此时样本的均方误差为：![image](https://upload-images.jianshu.io/upload_images/13064452-6c03fb830e103a2f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

接下来我们使用梯度下降的方式调整参数，调整参数的公式如下：

![image](https://upload-images.jianshu.io/upload_images/13064452-f6e2c3070e3e764c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

在开始优化参数前，我们需要计算神经网络各层的梯度，首先是输出层

使用链式法则求导输出层阈值关于损失函数的梯度，公式如下：

![image.png](https://upload-images.jianshu.io/upload_images/13064452-f7593e9b7799be6b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

若我们选择sigmoid函数作为激活函数并带入损失函数的表达式，则可得到阈值关于损失函数的梯度如下：

![image.png](https://upload-images.jianshu.io/upload_images/13064452-25e37af1acb40e9f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

可将其标记为：
![image.png](https://upload-images.jianshu.io/upload_images/13064452-b2306a378e0301c2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

同理我们计算输出层权重关于损失函数的梯度，计算公式如下：

![image.png](https://upload-images.jianshu.io/upload_images/13064452-7dc629ce84f3efa5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

同理我们选择sigmoid函数为例，此时可得到

![image.png](https://upload-images.jianshu.io/upload_images/13064452-8d4c7500d0190e38.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

接下来是隐藏层阈值关于损失函数的梯度，仍然可以由链式法则推导得到

![image.png](https://upload-images.jianshu.io/upload_images/13064452-155fcd3091197101.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image.png](https://upload-images.jianshu.io/upload_images/13064452-d896060f47b4f169.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

观察上述表达式可以得出隐藏层梯度取决于隐藏层的输出、输出层阈值的梯度和隐藏层与输出层的权重。这也揭示了神经网络的精髓：在阈值调整过程中，当前层的参数的梯度取决于后一层参数的梯度。
接下来我将举一些简单的数字来推导一下整个过程：
假设现在有这样一个网络层：第一层是输入层，包含两个输入i1、i2和一个偏置b1，第二层是隐藏层，包含两个神经元，分别是h1，h2和偏置b2，第三层是输出层，两个神经元分别是o1和o2。

![image.png](https://upload-images.jianshu.io/upload_images/13064452-33e3ccf0142ebb7f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

接下来我们随机给这个简单的网络设置初值

![image.png](https://upload-images.jianshu.io/upload_images/13064452-642ceca7f4920015.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
其中，输入数据  i1=0.05，i2=0.10;

输出数据 o1=0.01,o2=0.99;

初始权重  w1=0.15,w2=0.20,w3=0.25,w4=0.30;

w5=0.40,w6=0.45,w7=0.50,w8=0.55

 目标：给出输入数据i1,i2(0.05和0.10)，使输出尽可能与原始输出o1,o2(0.01和0.99)接近。
接下来我们来计算模拟反向传播过程。
首先是正向传播：
从输入层到隐藏层：
首先计算神经元h1和h2的输入加权和：
![image.png](https://upload-images.jianshu.io/upload_images/13064452-bec4da3053c484e8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

选择sigmoid函数作为激活函数时，可得到h1的输出为：

![](https://upload-images.jianshu.io/upload_images/13064452-0d77f0ae9ee9f880.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

同理可得到，h2的输出为：
![image.png](https://upload-images.jianshu.io/upload_images/13064452-de6920ad19b34595.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

从隐藏层到输出层：
同样以sigmoid函数作为激活函数，可以得到o1和o2的值：
![image.png](https://upload-images.jianshu.io/upload_images/13064452-c9b5612b056c82f2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image.png](https://upload-images.jianshu.io/upload_images/13064452-34cdff0946a9f4ec.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

查看结果，我们发现神经网络的输出与我们预期的值存在差异，接下来我们定义损失函数，使用BP算法来优化参数。

这里我们使用平方误差来作为损失函数进行计算。
![image.png](https://upload-images.jianshu.io/upload_images/13064452-e598b85be93e9a7d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


因为有两个输出，所以分别计算o1和o2的误差，总误差为两者之和：

![image](http://upload-images.jianshu.io/upload_images/13064452-bca6bbb778b1cf98.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240) 

同理可以得到：

![image](http://upload-images.jianshu.io/upload_images/13064452-c015326c38de6bb7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

二者相加得到总共的损失如下：

![image](http://upload-images.jianshu.io/upload_images/13064452-8e982a5b02230e37.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

2.隐含层到输出层的权值更新：

以更新权重参数w5为例，我们可以用链式法则求的总误差对权重参数w5的偏导数：

![image](http://upload-images.jianshu.io/upload_images/13064452-469bb94a3d272565.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

下面的图可以更直观的看清楚误差是怎样反向传播的：

![image](http://upload-images.jianshu.io/upload_images/13064452-b55300854ec2737e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

现在我们来分别计算每个式子的值：

首先是![image](http://upload-images.jianshu.io/upload_images/13064452-da7ecc5df7ddc215.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

：

![image](http://upload-images.jianshu.io/upload_images/13064452-ba61f1f136c89ee4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

接着计算![image.png](https://upload-images.jianshu.io/upload_images/13064452-d540c15d802305ba.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


：

![image](http://upload-images.jianshu.io/upload_images/13064452-e9c61f1e36cf393f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
附上sigmoid函数求导数的推导过程

![image.png](https://upload-images.jianshu.io/upload_images/13064452-242e4e83bf593237.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

最后计算![image](http://upload-images.jianshu.io/upload_images/13064452-3abd08a5007441d2.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

：

![image](http://upload-images.jianshu.io/upload_images/13064452-ef6fe35137541615.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

最后三者相乘：

![image](http://upload-images.jianshu.io/upload_images/13064452-f0ad1986f4b45de8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这样我们就计算出整体误差E(total)对w5的偏导值。

回过头来再看看上面的公式，我们发现：

![image](http://upload-images.jianshu.io/upload_images/13064452-01d27d3a16697bd4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

为了表达方便，用![image](http://upload-images.jianshu.io/upload_images/13064452-8b01c0abf39ea992.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

来表示输出层的误差：

![image](http://upload-images.jianshu.io/upload_images/13064452-a1d0c4dfca9d8939.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

因此，整体误差E(total)对w5的偏导公式可以写成：

![image](http://upload-images.jianshu.io/upload_images/13064452-17fceba53995e36e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

如果输出层误差计为负的话，也可以写成：

![image](http://upload-images.jianshu.io/upload_images/13064452-11d66bad21195e90.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

最后我们来更新w5的值：

![image](http://upload-images.jianshu.io/upload_images/13064452-faa202ecd1a7ff8f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

（其中，![image](http://upload-images.jianshu.io/upload_images/13064452-5c8d6407e17639f4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

是学习速率，这里我们取0.5）

同理，可更新w6,w7,w8:

![image.png](https://upload-images.jianshu.io/upload_images/13064452-d23ee7390cbbb726.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

3.输入层隐含层的权值更新：

　方法和上面类似，但是需要注意w1的值与o1和o2都有关系，求导数的过程中都要包含在内：

![image](http://upload-images.jianshu.io/upload_images/13064452-8ed6bc27658d466f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

计算![image](http://upload-images.jianshu.io/upload_images/13064452-4ea6df4ffbcf9942.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

：

![image](http://upload-images.jianshu.io/upload_images/13064452-4ac1a4cc39086a69.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

先计算![image](http://upload-images.jianshu.io/upload_images/13064452-dbbdb2fce03179ed.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

：

![image](http://upload-images.jianshu.io/upload_images/13064452-f544518d9b717c47.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image](http://upload-images.jianshu.io/upload_images/13064452-45bb74d7b87fa411.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image](http://upload-images.jianshu.io/upload_images/13064452-836656f62a0dcd43.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![image.png](https://upload-images.jianshu.io/upload_images/13064452-cc11ac3d37995626.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

同理，计算出：

　　　　　　　　　　![image](http://upload-images.jianshu.io/upload_images/13064452-b2a25b88a56b7fa7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

两者相加得到总值：

![image.png](https://upload-images.jianshu.io/upload_images/13064452-b05dcfff8ec9a2f0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


再计算![image](http://upload-images.jianshu.io/upload_images/13064452-ffbc425ddf548a40.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

：

![image](http://upload-images.jianshu.io/upload_images/13064452-006867b3670e905b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

再计算![image](http://upload-images.jianshu.io/upload_images/13064452-4b10610f72487bce.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

：

![image](http://upload-images.jianshu.io/upload_images/13064452-c2a97ca3037e07d6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

最后，三者相乘：

![image](http://upload-images.jianshu.io/upload_images/13064452-c506277d1283fe78.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

 为了简化公式，用sigma(h1)表示隐含层单元h1的误差：

![image](http://upload-images.jianshu.io/upload_images/13064452-d8b38e174cdbeffd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

最后，更新w1的权值：

![image](http://upload-images.jianshu.io/upload_images/13064452-9a3b2504c54a3eec.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

同理，可更新w2,w3,w4的权值：

![image](http://upload-images.jianshu.io/upload_images/13064452-b4743a76bf8a116c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

这样误差反向传播法就完成了，最后我们再把更新的权值重新计算，不停地迭代，在这个例子中第一次迭代之后，误差由0.298371109下降至0.291027924。迭代多次后误差会有显著的减少。

## 8.5 python代码实例（API调用和手写代码实现）

首先我们先手动搭建一个MLP网络，代码如下：
```
##读取图片并将图片中的像素点数据标准化
import os
import struct
import numpy as np
def load_mnist(path, kind='train'):
"""Load MNIST data from `path`"""
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
##展示数据的维数
X_train, y_train = load_mnist('', kind='train')
   print('Rows: %d, columns: %d'% (X_train.shape[0], X_train.shape[1]))
X_test, y_test = load_mnist('', kind='t10k')
print('Rows: %d, columns: %d'% (X_test.shape[0], X_test.shape[1]))
```
此处可以得到我们下载的MNIST数据集包含6万组训练数据和1万组测试数据。
![image.png](https://upload-images.jianshu.io/upload_images/13064452-1ebd5970430cf689.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

展示示例特征矩阵中的784像素向量重新整形后的数字0-9原始的28×28图像，不同的手写字体的不同，代码如下：
```
fig, ax = plt.subplots(nrows=5,
... ncols=5,
... sharex=True,
... sharey=True)
ax = ax.flatten()
for i in range(25):
... img = X_train[y_train == 7][i].reshape(28, 28)
... ax[i].imshow(img, cmap='Greys')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
```
![12_06.png](https://upload-images.jianshu.io/upload_images/13064452-604e2edcc9a87b92.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
```
#train the MLP using 55,000 samples from the already shuffled MNIST training dataset 
#and use the remaining 5,000 samples for validation during training
import numpy as np
import sys
class NeuralNetMLP(object):
    """ Feedforward neural network / Multi-layer perceptron classifier.
    Parameters
    ------------
    n_hidden : int (default: 30)
        Number of hidden units.
    l2 : float (default: 0.)
        Lambda value for L2-regularization.
        No regularization if l2=0. (default)
    epochs : int (default: 100)
        Number of passes over the training set.
    eta : float (default: 0.001)
        Learning rate.
    shuffle : bool (default: True)
        Shuffles training data every epoch if True to prevent circles.
    minibatch_size : int (default: 1)
        Number of training samples per minibatch.
    seed : int (default: None)
        Random seed for initalizing weights and shuffling.
    Attributes
    -----------
    eval_ : dict
      Dictionary collecting the cost, training accuracy,
      and validation accuracy for each epoch during training.
    """
    def __init__(self, n_hidden=30,
                 l2=0., epochs=100, eta=0.001,
                 shuffle=True, minibatch_size=1, seed=None):
 
        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.l2 = l2
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size
 
    def _onehot(self, y, n_classes):
        """Encode labels into one-hot representation
        Parameters
        ------------
        y : array, shape = [n_samples]
            Target values.
        Returns
        -----------
        onehot : array, shape = (n_samples, n_labels)
        """
        onehot = np.zeros((n_classes, y.shape[0]))
        for idx, val in enumerate(y):
            onehot[val, idx] = 1.
        return onehot.T
 
    def _sigmoid(self, z):
        """Compute logistic function (sigmoid)"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
 
    def _forward(self, X):
        """Compute forward propagation step"""
 
        # step 1: net input of hidden layer
        # [n_samples, n_features] dot [n_features, n_hidden]
        # -> [n_samples, n_hidden]
        z_h = np.dot(X, self.w_h) + self.b_h
 
        # step 2: activation of hidden layer
        a_h = self._sigmoid(z_h)
 
        # step 3: net input of output layer
        # [n_samples, n_hidden] dot [n_hidden, n_classlabels]
        # -> [n_samples, n_classlabels]
 
        z_out = np.dot(a_h, self.w_out) + self.b_out
 
        # step 4: activation output layer
        a_out = self._sigmoid(z_out)
 
        return z_h, a_h, z_out, a_out
 
    def _compute_cost(self, y_enc, output):
        """Compute cost function.
        Parameters
        ----------
        y_enc : array, shape = (n_samples, n_labels)
            one-hot encoded class labels.
        output : array, shape = [n_samples, n_output_units]
            Activation of the output layer (forward propagation)
        Returns
        ---------
        cost : float
            Regularized cost
        """
        L2_term = (self.l2 *
                   (np.sum(self.w_h ** 2.) +
                    np.sum(self.w_out ** 2.)))
 
        term1 = -y_enc * (np.log(output))
        term2 = (1. - y_enc) * np.log(1. - output)
        cost = np.sum(term1 - term2) + L2_term
 
        # If you are applying this cost function to other
        # datasets where activation
        # values maybe become more extreme (closer to zero or 1)
        # you may encounter "ZeroDivisionError"s due to numerical
        # instabilities in Python & NumPy for the current implementation.
        # I.e., the code tries to evaluate log(0), which is undefined.
        # To address this issue, you could add a small constant to the
        # activation values that are passed to the log function.
        #
        # For example:
        #
        # term1 = -y_enc * (np.log(output + 1e-5))
        # term2 = (1. - y_enc) * np.log(1. - output + 1e-5)
 
        return cost
 
    def predict(self, X):
        """Predict class labels
        Parameters
        -----------
        X : array, shape = [n_samples, n_features]
            Input layer with original features.
        Returns:
        ----------
        y_pred : array, shape = [n_samples]
            Predicted class labels.
        """
        z_h, a_h, z_out, a_out = self._forward(X)
        y_pred = np.argmax(z_out, axis=1)
        return y_pred
 
    def fit(self, X_train, y_train, X_valid, y_valid):
        """ Learn weights from training data.
        Parameters
        -----------
        X_train : array, shape = [n_samples, n_features]
            Input layer with original features.
        y_train : array, shape = [n_samples]
            Target class labels.
        X_valid : array, shape = [n_samples, n_features]
            Sample features for validation during training
        y_valid : array, shape = [n_samples]
            Sample labels for validation during training
        Returns:
        ----------
        self
        """
        n_output = np.unique(y_train).shape[0]  # number of class labels
        n_features = X_train.shape[1]
 
        ########################
        # Weight initialization
        ########################
 
        # weights for input -> hidden
        self.b_h = np.zeros(self.n_hidden)
        self.w_h = self.random.normal(loc=0.0, scale=0.1,
                                      size=(n_features, self.n_hidden))
 
        # weights for hidden -> output
        self.b_out = np.zeros(n_output)
        self.w_out = self.random.normal(loc=0.0, scale=0.1,
                                        size=(self.n_hidden, n_output))
 
        epoch_strlen = len(str(self.epochs))  # for progress formatting
        self.eval_ = {'cost': [], 'train_acc': [], 'valid_acc': []}
 
        y_train_enc = self._onehot(y_train, n_output)
 
        # iterate over training epochs
        for i in range(self.epochs):
 
            # iterate over minibatches
            indices = np.arange(X_train.shape[0])
 
            if self.shuffle:
                self.random.shuffle(indices)
 
            for start_idx in range(0, indices.shape[0] - self.minibatch_size +
                                   1, self.minibatch_size):
                batch_idx = indices[start_idx:start_idx + self.minibatch_size]
 
                # forward propagation
                z_h, a_h, z_out, a_out = self._forward(X_train[batch_idx])
 
                ##################
                # Backpropagation
                ##################
 
                # [n_samples, n_classlabels]
                sigma_out = a_out - y_train_enc[batch_idx]
 
                # [n_samples, n_hidden]
                sigmoid_derivative_h = a_h * (1. - a_h)
 
                # [n_samples, n_classlabels] dot [n_classlabels, n_hidden]
                # -> [n_samples, n_hidden]
                sigma_h = (np.dot(sigma_out, self.w_out.T) *
                           sigmoid_derivative_h)
 
                # [n_features, n_samples] dot [n_samples, n_hidden]
                # -> [n_features, n_hidden]
                grad_w_h = np.dot(X_train[batch_idx].T, sigma_h)
                grad_b_h = np.sum(sigma_h, axis=0)
 
                # [n_hidden, n_samples] dot [n_samples, n_classlabels]
                # -> [n_hidden, n_classlabels]
                grad_w_out = np.dot(a_h.T, sigma_out)
                grad_b_out = np.sum(sigma_out, axis=0)
 
                # Regularization and weight updates
                delta_w_h = (grad_w_h + self.l2*self.w_h)
                delta_b_h = grad_b_h  # bias is not regularized
                self.w_h -= self.eta * delta_w_h
                self.b_h -= self.eta * delta_b_h
 
                delta_w_out = (grad_w_out + self.l2*self.w_out)
                delta_b_out = grad_b_out  # bias is not regularized
                self.w_out -= self.eta * delta_w_out
                self.b_out -= self.eta * delta_b_out
 
            #############
            # Evaluation
            #############
 
            # Evaluation after each epoch during training
            z_h, a_h, z_out, a_out = self._forward(X_train)
            cost = self._compute_cost(y_enc=y_train_enc,
                                      output=a_out)
 
            y_train_pred = self.predict(X_train)
            y_valid_pred = self.predict(X_valid)
 
            train_acc = ((np.sum(y_train == y_train_pred)).astype(np.float) /
                         X_train.shape[0])
            valid_acc = ((np.sum(y_valid == y_valid_pred)).astype(np.float) /
                         X_valid.shape[0])
 
            sys.stderr.write('\r%0*d/%d | Cost: %.2f '
                             '| Train/Valid Acc.: %.2f%%/%.2f%% ' %
                             (epoch_strlen, i+1, self.epochs, cost,
                              train_acc*100, valid_acc*100))
            sys.stderr.flush()
 
            self.eval_['cost'].append(cost)
            self.eval_['train_acc'].append(train_acc)
            self.eval_['valid_acc'].append(valid_acc)
 
        return self
nn.fit(X_train=X_train[:55000],y_train=y_train[:55000],
 
X_valid=X_train[55000:],y_valid=y_train[55000:])
```

```
import matplotlib.pyplot as plt
plt.plot(range(nn.epochs), nn.eval_['cost'])
 plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.show()
```
![12_07.png](https://upload-images.jianshu.io/upload_images/13064452-c11055c432f4ef4f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
对图像进行分析可知，当ecochs=100时，cost逐渐开始收敛，坡度减缓，到145,200时仍然逐渐减小。
```
<code class="language-python">plt.plot(range(nn.epochs), nn.eval_['train_acc'],label='training')  
plt.plot(range(nn.epochs), nn.eval_['valid_acc'],label='validation', linestyle='--')  
plt.ylabel('Accuracy')  
plt.xlabel('Epochs')  
plt.legend()  
plt.show()</code>  
```
![12_08.png](https://upload-images.jianshu.io/upload_images/13064452-12ec7174e3fd8417.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
该图显示训练和验证准确度之间的差距增加我们训练网络的Epochs越来越多。 
大约在第50个Epochs，训练和验证准确度值相等，然后网络开始过拟和训练数据。
```
y_test_pred = nn.predict(X_test)
acc = (np.sum(y_test == y_test_pred).astype(np.float) / X_test.shape[0])
print('Training accuracy: %.2f%%' % (acc * 100))
```
![image.png](https://upload-images.jianshu.io/upload_images/13064452-1ec8dfb6e3df0fab.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
```
<code class="language-python">#查看25个被错误分类的例子  
miscl_img = X_test[y_test != y_test_pred][:25]  
correct_lab = y_test[y_test != y_test_pred][:25]  
miscl_lab= y_test_pred[y_test != y_test_pred][:25]  
fig, ax = plt.subplots(nrows=5,  
ncols=5,sharex=True, sharey=True,)  
ax = ax.flatten()  
for i in range(25):  
... img = miscl_img[i].reshape(28, 28)  
... ax[i].imshow(img,  
... cmap='Greys',  
... interpolation='nearest')  
... ax[i].set_title('%d) t: %d p: %d'  
... % (i+1, correct_lab[i], miscl_lab[i]))  
ax[0].set_xticks([])  
ax[0].set_yticks([])  
plt.tight_layout()  
plt.show()</code>  
```
![12_09.png](https://upload-images.jianshu.io/upload_images/13064452-0c098277ba51ee20.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

接下来我将基于tensorflow搭建一个简单的神经网络，在开始介绍代码前先介绍一下tensorflow,tensorflow是最初由Google大脑小组的研究员和工程师们开发出来，用于机器学习和深度神经网络方面的研究，但这个系统同样试用于其他的计算系统。TensorFlow 是一个采用数据流图（data flow graphs），用于数值计算的开源软件库。节点（Nodes）在图中表示数学操作，图中的线（edges）则表示在节点间相互联系的多维数据数组，即张量（tensor）。它灵活的架构让你可以在多种平台上展开计算，例如台式计算机中的一个或多个CPU（或GPU），服务器，移动设备等等。接下来我们介绍一下MNIST数据集，
MNIST数据集来自美国国家标准与技术研究所，National Institute of Standards and Technology (NIST). 训练集 (training set) 由来自 250 个不同人手写的数字构成, 其中 50% 是高中学生, 50% 来自人口普查局 (the Census Bureau) 的工作人员. 测试集(test set) 也是同样比例的手写数字数据。可在 http://yann.lecun.com/exdb/mnist/ 下载的MNIST数据集，数据集中包含6万个训练集合和1万个测试集，以及他们对应的标签。

话不多说，接下来我们训练手写数字识别模型。
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
![image.png](https://upload-images.jianshu.io/upload_images/13064452-37af58d395326c96.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

###     8.6 MLP的发展

深度学习在语言、图像、自然语言处理方面展现出了超高的性能，受到了包括政府和产业界的广泛关注。接下来我将从语音、图像和自然语言处理三个方面展开叙述。

一、深度学习在语音识别领域的应用

  传统的语音识别算法首先会对声音进行分帧，接着做波形变换，也称为声学特征提取，提取到的特征识别成状态，接着将状态组合成音素，最后将音素组合成单词。在神经网络出现前，语音识别系统大多是采用高斯混合模型（GMM）来描述每个建模单元的概率模型。这种模型的优点是有较好的区分度，且训练简单，故被广泛使用。但是相比于深度学习算法，GMM建模能利用的特征值有限，而且特征之间的相关性也不能很好表示。
  
  从2009年开始，微软亚洲研究院和深度学习领军人物Hinton合作。在2011年推出基于深度神经网络的语音识别系统，这一系统使得样本数据特征间相关性信息得以充分表示，将连续的特征信息结合构成高维特征，通过高维特征样本对深度神经网络模型进行训练，性能得到了很大的提升。

二、深度学习在图像识别领域的应用

  图像识别的原理是：将图像信号传送给专用的图像处理系统，然后系统根据像素分布和亮度、颜色等信息，转变成数字化信号；图像系统对这些信号进行各种运算来抽取目标的特征，进而根据判别的出图像的信息。图像的处理是深度学习算法最早尝试应用的领域。
 早在1989年，加拿大多伦多大学教授Yann LeCun就和他的同事，受到动物视觉皮层细胞的启发（感光细胞和折光细胞组成）提出了卷积神经网络（Convolutional Neural Networks, CNN），它是一种包含卷积层的深度神经网络模型。通常一个卷机神经网络架构包含两个可以通过训练产生的非线性卷积层，两个固定的子采样层和一个全连接层，隐藏层的数量一般至少在5个以上。起初卷积神经网络在小规模的问题上取得了当时世界最好成果。但是在很长一段时间里一直没有取得重大突破。直到2012年Hinton构建深度神经网络才产生了惊人的成果。主要是因为改变是在网络的训练中引入了权重衰减的概念，有效的减小权重幅度，防止网络过拟合。更关键的是计算机计算能力的提升，GPU加速技术的发展，使得在训练过程中可以产生更多的训练数据，使网络能够更好的拟合训练数据。这部分将在9章节具体介绍。

三、深度学习在自然语言处理领域的应用

  自然语言处理问题是深度学习在除了语音和图像处理之外的另一个重要的应用领域。在神经网络被广泛使用前，业界通常试用有些简单的模型来处理比如：邮件过滤、词性标注等常见问题。但是这些简单的模型存在不能准确结合语境来理解文字表达的情绪。
  在这个过程中，首先出现的Word2Vec改变了现状，它可在保留单词语意结构的前提下，生成一个有效的向量表示。但是它仍然存在无法获得文本短期顺序依赖关系的问题。这一点可以通过循环神经网络（Recurrent Neural Networks）RNN解决。RNN利用数据的时间性质，使用存储在隐含状态中的先前单词信息，有序地将每个单词传输到训练网络中。在之后的10章节会对RNN进行详细的说明。


