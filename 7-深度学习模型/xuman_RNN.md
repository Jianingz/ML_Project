[deep learning book ](http://www.deeplearningbook.org/contents/rnn.html)
# chapter 10 
# Sequence Modeling: Recurrent and Recursive Nets

循环神经网络，RNN，是用来处理序列数据的网络之一。卷积网络CNN是用来处理网格一样的数据，循环网络专门用来处理序列数据比如<a href="https://www.codecogs.com/eqnedit.php?latex=x^{1}..x^{T}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x^{1}..x^{T}" title="x^{1}..x^{T}" /></a>。就像CNN可以很方便的缩放长宽很大的图像，也可以处理长宽变化的图像，RNN可以缩放长度很长的**序列数据**，大多数RNN也可以处理可变长度的序列数据。

 从多层网络到循环网络，我们可以利用机器学习和统计模型中观点：在网络的不同部分**共享参数**。参数共享才能将模型应用到不同长度的数据序列中然后进行泛化。如果我们对于不同的时间索引有不同的参数，那么我们就不能在训练过程中对未知长度的序列进行泛化，也不能在不同的时间位置和不同的序列长度之间共享统计强度。当一个特殊的信息可以出现在同一序列的不同位置时，参数共享就更加重要了。比如，“I went to Nepal in 2009” 和 “In 2009,I went to Nepal.‘如果想用机器学习的方法读取序列并提取出时间，通常是先将2009看作一个信息片段，不管他的位置在哪。假设我们训练了一个前向网络可以处理固定长度的句子。传统的全连接网络对于每个输入特征都有独立的参数，所以在句子的每个位置都要学习不同的语言规则。但是在RNN中，在不同的时间片段中共享共同的参数。

近几年有把卷积神经网络应用与一维时间序列上，这种卷积的方法是时间延迟神经网络的基础。这种卷积的方法也能实现浅显的参数共享，当每个输出单元是少量输入数据临近单元的函数时，输出单元是序列数据。这种参数共享在每个时间段都应用同一个卷积核时有明显的效果。
循环神经网络的参数共享是另一种方式。RNN中，每个输出单元是先前输出单元的函数，每个输出单元都使用与先前输出单元相同的更新规则产生。这种循环的结构使得RNN中的参数共享通过一个非常深的计算图谱实现。

简便叙述下RNN。将RNN看作是由向量<a href="https://www.codecogs.com/eqnedit.php?latex=x^{t}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x^{t}" title="x^{t}" /></a>组成的序列，其中t为从1到<a href="https://www.codecogs.com/eqnedit.php?latex=\tau" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tau" title="\tau" /></a>时间索引.RNN通常是作用于这样一个序列的一小部分，每一部分的序列长度也不同。

本章引入计算图谱的概念来说明循环的概念。这种循环阐述了当前变量在下一个时间点对它自身的影响。本章接下来会通过不同的方式来描述RNN网络的构建、训练和使用。

# 10.1 展开的计算图谱

![fig1](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN_images/fig10.1.png)

本节将递归或者循环计算展开为计算图谱进行讲解。该图谱是一个链式规则的典型结构。
![equ1](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN_images/equ10.1.png)
$$s^{t}$$ 为系统的当前状态。这就是一个循环，因为系统在t时刻的状态与t-1时刻有关。
![equ2](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN_images/equ10.3:3.png)
![fig10.2](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN_images/fig10.2.png)
上图为一个没有输出的循环结构。
![equ4](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN_images/equ10.4.png)
公式10.4中状态s包含了之前所有状态的信息

循环神经网络有很多实现方式，一般都可以看作是前向神经网络。RNN中一般用公式10.5表示隐藏层。然后RNN会增加一个**输出层**来读取隐藏层的输出信息，进行预测。
![equ5](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN_images/equ10.5.png)
如果用RNN来做预测，一般隐藏层都是有损的，因为隐藏层要将不定长的输入序列转换为定长的序列，肯定会产生损失。隐藏层只需要存储能够作出正确预测的信息即可，不需要保存输入的所有信息。

将公式10.5写成一下形式；
![equ6](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN_images/equ10.6:7.png)
这样有两个优点：
1. 不必考虑序列的长度问题。因为模型关注的是从一个状态到另一个状态的转变，而不是历史状态的变化长度。
2. 在每一个时刻，可以使用具有同样参数的转换函数f。

这样就可以将简单模型f应用到不同时刻以及不同长度的序列上。
展开的计算图谱更加简明，也很清楚的说明了每一步是如何进行的。也说明了信息流随着时间向前传递时计算输出和损失，向后传递时计算梯度的过程。

# 10.2 循环神经网络
在了解了参数共享和展开的计算图谱之后，可以继续学习不同种类的RNN了。
几种不同类型的循环神经网络如下：
* 在每一个时刻都产生输出，且所有隐藏层之间循环相连。如图10.3所示。
* 在每一个时刻都产生输出，但是只在该时刻的输出层和下一时刻的隐藏层之间相连。如图10.4所示。
* 所有隐藏层之间循环相连，读取所有的输入序列，只产生一个输出。如图10.5所示。

1. 在每一个时刻都产生输出，且所有隐藏层之间循环相连

![](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN_images/fig10.3.png)
这种RNN是本书中应用到的最多的一种，任何一种可以由图灵机计算的函数都可以在有限长度内由这种RNN结构实现。
当把这种RNN当作图灵机使用时，输入为二进制序列数据，输出一定是离散的二进制输出。
我们用前向传播的方式实现图10.3中展示的RNN。假设图10.3中使用的激活函数是tangent，假设输出为离散数据，一个最常用的展示离散数据的方式就是计算离散数据中每个离散值的log概率，然后用离散输出<a href="https://www.codecogs.com/eqnedit.php?latex=y^{t}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y^{t}" title="y^{t}" /></a>的softmax得到输出向量<a href="https://www.codecogs.com/eqnedit.php?latex=y^{t}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y^{t}" title="y^{t}" /></a>。前向传播的初始状态是<a href="https://www.codecogs.com/eqnedit.php?latex=h^{0}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h^{0}" title="h^{0}" /></a>,之后从时间t=1到t=<a href="https://www.codecogs.com/eqnedit.php?latex=\tau" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tau" title="\tau" /></a>应用以下公式进行更新。

![](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN_images/equ10.8:11.png)
其中，b,c是偏置值，U输入层-隐藏层连接矩阵，V隐藏层-输出层连接矩阵,W隐藏层-隐藏层连接矩阵。
**损失函数：**就是所有时刻给定输入x与输出y之间损失的加和。一般用输出对数似然值表示

![](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN_images/equ10.12-14.png)
计算该损失函数对于每个参数的梯度值花销会很大，因为它包括了图10.3种从左向右的前向传播，紧接着是从右向左的反向传播。。时间和空间上复杂度都是<a href="https://www.codecogs.com/eqnedit.php?latex=O(\tau)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?O(\tau)" title="O(\tau)" /></a>。而且时间复杂度不能降低，因为输入数据是序列，只能计算完前一个才能计算下一个。空间复杂度同样原因，存储完前一个才能存储下一个。其中反向传播时间称为BPTT（back-propagation through time）。这种RNN功能强大但是时间和空间复杂度很高。





















