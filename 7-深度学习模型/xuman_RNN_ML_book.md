## 目录
1. RNN基本数学原理，设计思想   
[参考：deep learning book notes](https://github.com/xuman-Amy/deeplearningbook-chinese)     
[动手搭建最简单的神经网络](http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/)

2. cs224d 《deep learning for NLP》课堂笔记和课后习题   
[参考：cs224d](http://cs224d.stanford.edu/syllabus.html)

3. RNN常见的算法BPTT等以及梯度消失问题等.   
[参考：CSDN](https://blog.csdn.net/heyongluoyao8/article/details/48636251).   
[BPTT](http://colah.github.io/posts/2015-08-Backprop/)

4. 详解LSTM模型,代码举例详解。  
[参考知乎——超智能体](https://zhuanlan.zhihu.com/p/25518711)  
[大牛讲解lstm] (http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

5. RNN项目实践  
[参考：自动补全唐诗](https://github.com/jinfagang/tensorflow_poems)
 
6. CNN和RNN的对比   
[参考：文本分类](https://github.com/xuman-Amy/text-classification-cnn-rnn)


## 杂记
**知乎-超智能体：**

* 深度学习中‘层’的概念  
每一层就是 $ \vec{y}= a( W \cdot \vec{x} + b)$ ,a:激活函数，y:输出向量，x：输入向量，W：权重矩阵，b：偏移向量。用线性变换$( W \cdot \vec{x} + b)$加上激活函数的非线性变换，将输入空间映射到新的输出空间。

* 训练模型  
通过计算目标函数和预测函数之间的差值——损失函数（loss function）。用梯度下降法(Gradient Descent)减小损失函数,每次减小多少由学习速率（learning rate）决定。

**神经网络三大模块**

1. 结构模块 ——前向 | 反向 
2. 训练模块——初始化 | 训练集 | 验证集
3. 应用模块——预测 | 分析

**2017.07.02 前向神经网络的实现——基于tensorflow，学会了大体的框架，深度网络实现的流程，相关参数的含义。**
[代码示例](http://211.159.179.239:6007/notebooks/xuman/ML/forward_NN.ipynb)

tf.redude_mean()求平均值

tf.scalar_summary()  

RNN原理
应用场景
变体



# 1、常规循环神经网络简介
## 1.1 提出背景
**1. 处理序列信息**

循环神经网络（Recurrent Neural Network）的提出主要是为了更好的利用序列信息（sequential Information）。在传统的神经网络中，一般假设所有的输入数据和所有的输出数据之间都是相互独立的，即前一个输入和后一个输入之间没有关联。传统神经网络忽略了数据之间的时序性，从输入层到隐藏层再到输出层，层与层之间是全连接的，而在同一层内的各节点之间是没有连接的，这种架构限制了传统神经网络的应用场景，如情感分析，文本预测等需要考虑前后数据关联性的任务，传统神经网络不能很好的解决此类问题。

**窗处理机制** 

随着神经网络的不断发展，传统神经网络增加了窗处理机制来处理时序数据，即将每个时刻视为一个窗口，通过将前后窗口向量并接成一个更大的向量，以此利用前后时刻的信息预测当前信息。

![RNN_5_from_zhihu](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN/RNN_5.png)


如图所示，要想预测$t_{1}$时刻的信息，传统的神经网络是通过将$t_{0}、t_{1}和t_{2}$时刻的向量并接起来，然后进行预测。
    
* 举例说明：  
任务：“小明做错了事情，老师批评了___。”很明显我们想要的结果是“小明”。这个问题就要考虑前文词语的输入，而且各词语的输入顺序很重要。 

基于此背景，循环神经网络应运而生。循环神经网络的提出最早能够追溯到1990年由EFFREY L. ELMAN发表的文章《Finding Structure in Time》中，这篇文章中论述了如何在时间序列信息中找到一种特定的模式（pattern）或者结构（structure），提出了现在RNN的基本框架。

**2. 具有记忆能力**

循环神经网络是具有记忆能力的架构，它对于之前计算过的输入序列信息都保持一定的记忆。这是因为在RNN中，隐藏层之间采用了循环连接的方式，即隐藏层的输入除了当前t时刻的输入信息的输出$x_{t}$外，还包括前一时刻的隐藏层$ h_{t-1}$的输出信息。也正是因为RNN的隐藏层之间是循环连接的，所以RNN的输入序列长度也是不限制的。

如下图的左边为典型的RNN逻辑图，右边为RNN在时间序列上前向计算的展开图。


![RNN_1](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN/RNN_1.jpg)

图1 RNN逻辑图

## 1.2   数学原理
###1.2.1 常规循环神经网络的逻辑图

RNN的数学本质与深度学习中传统网络的数学本质相同，都是通过权重矩阵与输入信息的线性变换加上激活函数的非线性变换，将输入空间映射到新的输出空间中去。


![RNN_2_from_cs224d](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN/RNN_2.png)


![RNN_6_from_zhihu](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN/RNN_6.gif)

### 1.2.2 常规神经网络的数学公式：

#### $$ 隐藏层： h_{t} = \sigma( W_{(hh)} h_{t-1} + W_{(xh)} x_{t} + b ) \qquad (1)$$
#### $$ 输出层： \hat{y_{t}} = \sigma (W_{(ho)} h_{t})  \qquad (2) $$

**参数讲解：**

* $ x_{1}...x_{t-1},x_{t},x_{t+1}...x_{T}$ 为网络中每一时刻t的输入信息。

* $ h_{t} = \sigma( W_{(hh)} h_{t-1}) + W_{(hx)} x_{t}$：计算t时刻隐藏层的输出特征  $h_{t}$
	- $ x_{t}$ : t 时刻的输入，假设维度维input_dim
	
	- $ W_{(hx)}$: 输入层与隐藏层之间的权重矩阵，形状为$ hidden\_dim(隐藏层维度) \times input		\_dim(输入维度)$
	
	- $ W_{(hh)}$: 隐藏层之间的权重矩阵，形状为$hidden\_dim(隐藏层维度) \times hidden\_dim(隐藏层维度)$
	
	- $ h_{t-1}$ 前一时刻隐藏层的输出。
	
	- $ \sigma$ :非线性函数，比如tanh，sigmoid，relu。

* $\hat{y_{t}} = \sigma (W_{ho} h_{t})$ ：计算输出层的输出信息。权重矩阵为隐藏层与输出层之间的权重矩阵，激活函数为softmax。



**非线性函数 （non-linear function）**
	- 可以拓展下sigmoid/tanh/relu的选择
###1.2.3 损失函数（loss function）
1. 交叉熵
目前为止，所使用的循环神经网络的损失$L^{t}$都是用训练目标$y^{t}$和输出$o^{t}$之间的交叉熵（cross entropy）计算。
###$$ C\_entropy = - p(y^{t}) log (o^{t})$$
可以看出交叉熵与似然函数表示的损失函数是一样的。
2. 梯度求解
BPTT

## 1.3 RNN的特点

**1. 参数共享性**

>从多层网络出发到循环网络，我们需要利用上世纪80年代机器学习和统计模型早期思想的优点:在模型的不同部分共享参数。参数共享使得模型能够扩展到不 同形式的样本(这里指不同长度的样本)并进行泛化。如果我们在每个时间点都有 一个单独的参数，我们不但不能泛化到训练时没有见过序列长度，也不能在时间上 共享不同序列长度和不同位置的统计强度。当信息的特定部分会在序列内多个位置 出现时，这样的共享尤为重要。$^{[1]}$

卷积神经网络的参数共享体现在：在每个时间步内的卷积核是共享的。
 
循环神经网络的参数共享性体现在：在时间结构上，所有时刻的权重矩阵$W_{hx}、W_{hh}和W_{ho}$是共享的。这是循环神经网络作为突出的优势之一。循环神经网络的循环体现在隐藏层中t时刻的输出正是t+1时刻的输入，且每一层对于输入的信息都采用同样的权重矩阵和计算规则更新，这种循环也使得RNN的参数能够在很深的计算图中进行共享。
 
**2. 长短可变性**

循环神经网络在时间结构上的参数共享特性赋予了RNN能够处理任意长度时间序列的能力。从公式(1)可以看出，只要知道当前时刻的输入信息$x_{t}$和前一时刻隐藏层的输出$h_{t-1}$，又因为循环神经网络的权重矩阵在时间结构上的共享特性，所以RNN可以处理任意长度的时间序列信息，得出当前时刻隐藏层的输出$h_{t}$。

**3. 信息依赖性**

循环神经网络的主要思想就是利用序列信息时间上的依赖特性，从数学公式（1）可以看出RNN对于网络过去所有时刻的状态都具有依赖性。假如我们要计算时刻$ t=6 $的隐藏层状态$h_{6}$，我们就需要知道当前的输入$x_{6}$和前一时刻的隐藏层状态$h_{5}$，要得到$h_{5}$就要知道$h_{4}$……依次逆推到初始状态的$h_{0}$，这个过程也就表明了循环神经网络对于过去所有信息存在依赖性。

常规的循环神经网络只存在对于过去信息的依赖性，为了利用未来信息与当前信息之间的关系，提出了双向循环神经网络（Bidirectional），具体讲解见第二部分——RNN的变体。




## 1.4、常规RNN存在的问题—长期依赖（Long Term dependencies）的问题

![RNN_6_from_zhihu](/Users/laiye/Desktop/ML/RNN/RNN_6.gif)
![RNN_6_from_zhihu](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN/RNN_6.gif)

$^{[2]}$忽略激活函数的非线性变换，只看上图蓝色箭头的传输。由公式（1）简化可以得出  $$ h_{t} = W_{(hh)} h_{t-1} \qquad (3)$$ 
如果从隐藏层的初识状态$h_{0}$开始计算，可以得到：$$ h_{t} = W_{(hh)}^{t}h_{0} \qquad (4)$$
对（4）中的权重矩阵进行特征分解可得出：$$ h_{t} = Q \cdot \Lambda ^{t} \cdot Q^{t} \cdot h_{0} \qquad (4)$$

RNN最主要的特点在于能够根据上下文进行当前问题的预测。 比如我们要预测“我们用手机打电话”的最后一个词“电话”，可能只需要‘打’字就能完成预测。这种较短的依赖关系并不会给循环神经网络带来问题。但是还有很多任务事需要更长的信息依赖关系，也就是需要更多的上下文来进行当前问题的预测。比如猜测“James从小在中国长大，……，所以James说的一口流利的汉语”的最后一个词，我们一看知道结果应该是‘汉语’，但是因为这个问题要用到很长的上下文才能做出正确的预测，RNN就会出现一些问题。这也就是所说的循环神经网络中存在的长期依赖问题。长期依赖带来问题都是源于梯度，分为梯度消失（Vanishing Gradient）  和梯度爆炸（Exploding Gradient）。

理论上来说，RNN并不限制输入时序信息的长短，也就是说理论上的RNN并不会因长期依赖产生问题。

* 梯度消失（Vanishing Gradient）  
由公式（4）可以看出，当特征值小于1时，连续相乘t次后，就会趋近于0，称为梯度消失现象。

* 梯度爆炸（Exploding Gradient ）  
由公式（4）可以看出，当特征值大于1时，连续相乘t次后，就会趋近于无穷大$\infty$，称为梯度爆炸现象。

不管是梯度消失还是梯度爆炸，在循环神经网络的不断循环的过程中，都会造成隐藏层信息被丢弃的问题，使得隐藏层状态不能有效的向前传递到$h_{t}$。所以为了避免上述两种问题的发生，循环神经网络中的梯度相乘的积要尽可能的保持在1左右，才能够克服循环神经网络中长期依赖带来的梯度消失或者梯度爆炸问题。

## 1.4 RNN的训练——BPTT

## 循环神经网络与前馈神经网络的对比
##与递归神经网络的区别

# 2、RNN变体
为了解决上述循环神经网络中存在的长期依赖问题，经过学者们的不断研究和试验，提出了很多循环神经网络的变体。

1. 渗漏单元（leaky units）

为了使权重矩阵乘积保持在1左右，建立线性自连接单元(linear self-connections），并且这些连接的权重接近为1。线性自连接的隐藏单元称为渗漏单元（leaky unit）。线性自连接单元的权重设置一般有两种方式：（1）手动设定为常数，（2）设为自由变量，学习出来。   
渗漏单元能够在较长时间内积累信息，且不存在梯度爆炸和梯度消失的问题。但是在很多情况下，这些积累的信息一经采用后，让神经网络遗忘掉旧的状态可能是更好的。比如在一个序列信息中又有很多子序列，神经网络计算完一个子序列之后，我们希望它能将状态自动初始为0并继续下一个子序列的计算。基于这种需求，人们提出了能够学习决定何时自动清除历史状态的门控循环神经网络（Gated Recurrent Neural Network）。

2. 门控循环神经网络（Gated Recurrent Neural Network）

门控RNN简单的说就是能够实现线性自连接单元权重的自动调节，神经网络能够学习何时遗忘旧的状态。

- 关于门控循环神经网络‘门’的理解

顾名思义，‘门’的直观意义就是控制信息的输入和输出。循环神经网络中的门控机制就是通过‘门’的输出来控制其他要通过此门的数据，相当于过滤器的作用。

输入：门控依据。  
输出：[0,1]之间的数值，0代表数据全部丢弃，1代表数据全部通过。  
举例说明：    
数据$x=[2, 2, 2, 2]$,门控输出为$g = [0.5, 0.5, 0.5, 0.5 ]$，数据通过门之后输出为$ out  = [ 1, 1, 1, 1]$。

目前比较主流的门控循环神经网络一种是长短期记忆（Long-short term memory），一般称为“LSTM”；另一种就是基于LSTM提出的门控循环单元（Gated recurrent unit），也称为“GRU”。
## 2.1 LSTM（Long-short term memory）

长短期记忆（（Long short-term memory）最早是1997年由Hochreiter 和 Schmidhuber在论文《LONG SHORT-TERM MEMORY》$^{[3]}$中提出的。

![RNN_7_fom_colah](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN/RNN_7.png)
![RNN_7_from_colah](/Users/laiye/Desktop/ML/RNN/RNN_7.png)

在所有的循环神经网络中都有一个如图所示的循环链，链式结构的每个单元将输入的信息进行处理后输出。在常规循环神经网络中的链式结构的单元中只含有一个非线性函数对数据进行非线性转换，比如tanh函数。而在LSTM等RNN变体的循环神经网络中，每个单元在不同位置多加了几个不同的非线性函数，实现不同的功能，解决不同的问题。如下图所示。

![RNN_8_fom_colah](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN/RNN_8.png)
![RNN_8_from_colah](/Users/laiye/Desktop/ML/RNN/RNN_8.png)

LSTM最主要的就是记忆细胞（memory cell ），处于整个单元的水平线上，起到了信息传送带的作用，只几个含有简单的线性操作，能够保证数据流动时保持不变，也就相当于上文说的渗漏单元。如下图所示。

![RNN_9_fom_colah](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN/RNN_9.png)
![RNN_9_from_colah](/Users/laiye/Desktop/ML/RNN/RNN_9.png)

**详解LSTM**
LSTM就是在每个小单元中增加了三个sigmoid函数，实现门控功能，控制数据的流入流出，分别称为遗忘门（forget gate），输入门（input gate）和输出门（output gate）。

1、遗忘门（forget gate)


![RNN_10_fom_colah](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN/RNN_10.png)
![RNN_10_from_colah](/Users/laiye/Desktop/ML/RNN/RNN_10.png)

**主要功能：** 决定记忆细胞中要丢弃的信息，也就是决定前一时刻记忆细胞$C_{t-1}$中有多少信息能够继续传递给当前时刻的记忆细胞$C_{t}$。  
**数学公式：** 
$$ \begin{split}
f_{t} &=\sigma (W_{f} \cdot [h_{t-1}, x_{t}] + b_{f}) \\
&=\sigma (W_{hf} \cdot h_{t-1} + W_{xf} \cdot x_{t} + b_{f}) 
\end{split}$$

2、输入门（input gate）

![RNN_11_fom_colah](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN/RNN_11.png)
![RNN_11_from_colah](/Users/laiye/Desktop/ML/RNN/RNN_11.png)

**主要功能：** 决定有多少新的输入信息能够进入当前时刻的记忆细胞$C_{t}$。  
**数学公式：** 
$$ \begin{split}
i_{t} &=\sigma (W_{i} \cdot [h_{t-1}, x_{t}] + b_{i}) \\
&=\sigma (W_{hi} \cdot h_{t-1} + W_{xi} \cdot x_{t} + b_{i}) 
\end{split}$$

3、输出门（output gate）

![RNN_12_fom_colah](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN/RNN_12.png)
![RNN_12_from_colah](/Users/laiye/Desktop/ML/RNN/RNN_12.png)

**主要功能：** 决定有多少当前时刻的记忆细胞$C_{t}$中的信息能够进入到隐藏层状态$h_{t}$中。  
**数学公式：** 
$$ \begin{split}
o_{t} &=\sigma (W_{o} \cdot [h_{t-1}, x_{t}] + b_{o}) \\
&=\sigma (W_{ho} \cdot h_{t-1} + W_{xo} \cdot x_{t} + b_{o}) 
\end{split}$$

**利用遗忘门和输入门更新记忆细胞：**  
利用tanh函数产生候选记忆细胞$\tilde{C_{t}}$，也就是即将要进入输入门的信息。$$ \tilde{C_{t}} = tanh ( W_{c} \cdot [h_{t-1}, x_{t}] + b_{c})$$

![RNN_13_fom_colah](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN/RNN_13.png)
![RNN_13_from_colah](/Users/laiye/Desktop/ML/RNN/RNN_13.png)

遗忘门的输出与旧状态相乘，决定有多少旧状态的信息能够进入到新的记忆细胞；输入门的输出与候选记忆细胞相乘，决定有多少信息要被更新。二者线性相加得到当前时刻被更新过的记忆细胞。
$$ C_{t} = f_{t} \ast  C_{t-1} + i_{t} \ast \tilde{C_{t}} $$

**利用输出门将信息输出到隐藏层**  

将数据经过tanh函数的处理$(tanh(C_{t}))$，将数据归一化在[-1, 1]区间内。然后输出门的结果与归一化的数据相乘，实现控制输出到隐藏层的数据量。
$$ h_{t} = o_{t} \ast tanh(C_{t}) $$

## 2.2 GRU



    
#参考文献
[1] 《Deep learning》 ····to do
[2] 《知乎——超智能体》···to do
[3]《LONG SHORT-TERM MEMORY》
 
                         
                                                                   
                                                                     
                                                                     
