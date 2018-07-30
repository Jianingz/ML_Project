# 第十章 循环神经网络
# 目录 

10.1、常规循环神经网络简介

	10.1.1 神经网络的简单入门
	10.1.2 循环神经网络的提出背景
	10.1.3 循环神经网络的数学原理
		10.1.3.1 常规循环神经网络的逻辑图
		10.1.3.2 常规神经网络的数学公式
		10.1.3.3 损失函数（loss function）
	10.1.4 循环神经网络的特点
	10.1.5 循环神经网络的类别
	10.1.6 常规循环神经网络存在的问题
	10.1.7 反向传播算法进行梯度求解
		10.1.7.1  反向传播算法（Backpropagation）
		 10.1.7.2 随时间的反向传播过程（Back Propagation Through Time）
		 
10.2、循环神经网络的变体

	10.2.1 渗漏单元和门控机制
		10.2.1.1 渗漏单元（leaky units）
		10.2.1.2 门控机制
	10.2.2 长短期记忆网络LSTM（Long-short term memory）
		10.2.2.1 长短期记忆网络简介
		10.2.2.2 长短期记忆网络详解
	10.2.3 GRU
		10.2.3.1 门控循环神经网络简介
		10.2.3.2 门控循环神经网络详解
		
10.3、循环神经网络的应用

	10.3.1 keras简单介绍
	10.3.2 用LSTM实现MNIST手写数字识别
	10.3.3  LSTM实现简单的问答系统
		10.3.3.1 问答系统简介
		10.3.3.2 基于keras实现简单的问答系统
	10.3.4  LSTM实现简单的文本生成
		10.3.4.1文本生成原理
		10.3.4.2 基于keras实现简单英文文本生成
		10.3.4.3 基于keras实现简单中文文本生成
		
10.4、参考文献
		

# 10.1、常规循环神经网络简介

## 10.1.1 神经网络的简单理解

* 深度学习中‘层’的概念  

每一层就是 $ \vec{y}= a( W \cdot \vec{x} + b)$ ,a:激活函数，y:输出向量，x：输入向量，W：权重矩阵，b：偏移向量。用线性变换$( W \cdot \vec{x} + b)$加上激活函数的非线性变换，将输入空间映射到新的输出空间。

* 训练模型  

通过计算目标函数和预测函数之间的差值——损失函数（loss function）。用梯度下降法(Gradient Descent)减小损失函数,每次减小多少由学习速率（learning rate）决定。

**神经网络三大模块**

1. 结构模块 ——前向 | 反向 
2. 训练模块——初始化 | 训练集 | 验证集
3. 应用模块——预测 | 分析


## 10.1.2提出背景
**1. 处理序列信息**

循环神经网络（Recurrent Neural Network）的提出主要是为了更好的利用序列信息（sequential Information）。在传统的神经网络中，一般假设所有的输入数据和所有的输出数据之间都是相互独立的，即前一个输入和后一个输入之间没有关联。传统神经网络忽略了数据之间的时序性，从输入层到隐藏层再到输出层，层与层之间是全连接的，而在同一层内的各节点之间是没有连接的，这种架构限制了传统神经网络的应用场景，如情感分析，文本预测等需要考虑前后数据关联性的任务，传统神经网络不能很好的解决此类问题。

**窗处理机制** 

随着神经网络的不断发展，传统神经网络增加了窗处理机制来处理时序数据，即将每个时刻视为一个窗口，通过将前后窗口向量并接成一个更大的向量，以此利用前后时刻的信息预测当前信息。

![RNN_5_from_zhihu](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN/RNN_5.png)
![RNN_5_from_zhihu](/Users/laiye/Desktop/ML/RNN/RNN_5.png)

如图所示，要想预测$t_{1}$时刻的信息，传统的神经网络是通过将$t_{0}、t_{1}和t_{2}$时刻的向量并接起来，然后进行预测。
    
* 举例说明：  
任务：“小明做错了事情，老师批评了___。”很明显我们想要的结果是“小明”。这个问题就要考虑前文词语的输入，而且各词语的输入顺序很重要。 

基于此背景，循环神经网络应运而生。循环神经网络的提出最早能够追溯到1990年由EFFREY L. ELMAN发表的文章《Finding Structure in Time》中，这篇文章中论述了如何在时间序列信息中找到一种特定的模式（pattern）或者结构（structure），提出了现在RNN的基本框架。

**2. 具有记忆能力**

循环神经网络是具有记忆能力的架构，它对于之前计算过的输入序列信息都保持一定的记忆。这是因为在RNN中，隐藏层之间采用了循环连接的方式，即隐藏层的输入除了当前t时刻的输入信息的输出$x_{t}$外，还包括前一时刻的隐藏层$ h_{t-1}$的输出信息。也正是因为RNN的隐藏层之间是循环连接的，所以RNN的输入序列长度也是不限制的。

如下图的左边为典型的RNN逻辑图，右边为RNN在时间序列上前向计算的展开图。

![RNN_1](/Users/laiye/Desktop/ML/RNN/RNN_1.jpg)
![RNN_1](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN/RNN_1.png)

图1 RNN逻辑图

## 10.1.3  数学原理
### 10.1.3.1 常规循环神经网络的逻辑图

RNN的数学本质与深度学习中传统网络的数学本质相同，都是通过权重矩阵与输入信息的线性变换加上激活函数的非线性变换，将输入空间映射到新的输出空间中去。


![RNN_2_from_cs224d](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN/RNN_2.png)


![RNN_6_from_zhihu](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN/RNN_6.png)

### 10.1.3.2 常规神经网络的数学公式：

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


### 10.1.3.3 损失函数（loss function）
1. 交叉熵
目前为止，所使用的循环神经网络的损失$L^{t}$都是用训练目标$y^{t}$和输出$o^{t}$之间的交叉熵（cross entropy）计算。
###$$ C\_entropy = - p(y^{t}) log (o^{t})$$
可以看出交叉熵与似然函数表示的损失函数是一样的。

## 10.1.4 循环神经网络的特点

**1. 参数共享性**

>从多层网络出发到循环网络，我们需要利用上世纪80年代机器学习和统计模型早期思想的优点:在模型的不同部分共享参数。参数共享使得模型能够扩展到不 同形式的样本(这里指不同长度的样本)并进行泛化。如果我们在每个时间点都有 一个单独的参数，我们不但不能泛化到训练时没有见过序列长度，也不能在时间上 共享不同序列长度和不同位置的统计强度。当信息的特定部分会在序列内多个位置 出现时，这样的共享尤为重要。$^{[1]}$

卷积神经网络的参数共享体现在：在每个时间步内的卷积核是共享的。
 
循环神经网络的参数共享性体现在：在时间结构上，所有时刻的权重矩阵$W_{hx}、W_{hh}和W_{ho}$是共享的。这是循环神经网络作为突出的优势之一。循环神经网络的循环体现在隐藏层中t时刻的输出正是t+1时刻的输入，且每一层对于输入的信息都采用同样的权重矩阵和计算规则更新，这种循环也使得RNN的参数能够在很深的计算图中进行共享。
 
**2. 长短可变性**

循环神经网络在时间结构上的参数共享特性赋予了RNN能够处理任意长度时间序列的能力。从公式(1)可以看出，只要知道当前时刻的输入信息$x_{t}$和前一时刻隐藏层的输出$h_{t-1}$，又因为循环神经网络的权重矩阵在时间结构上的共享特性，所以RNN可以处理任意长度的时间序列信息，得出当前时刻隐藏层的输出$h_{t}$。

**3. 信息依赖性**

循环神经网络的主要思想就是利用序列信息时间上的依赖特性，从数学公式（1）可以看出RNN对于网络过去所有时刻的状态都具有依赖性。假如我们要计算时刻$ t=6 $的隐藏层状态$h_{6}$，我们就需要知道当前的输入$x_{6}$和前一时刻的隐藏层状态$h_{5}$，要得到$h_{5}$就要知道$h_{4}$……依次逆推到初始状态的$h_{0}$，这个过程也就表明了循环神经网络对于过去所有信息存在依赖性。

常规的循环神经网络只存在对于过去信息的依赖性，为了利用未来信息与当前信息之间的关系，提出了双向循环神经网络（Bidirectional），具体讲解见第二部分——RNN的变体。

## 10.1.5 循环神经网络的类别

循环神经网络主要应用于序列数据的处理，因输入与输出数据之间有时间上的关联性，所以在常规神经网络的基础上，加上了时间维度上的关联性，也就是有了循环神经网络。因此对于循环神经网络而言，它能够记录很长时间的历史信息，即使在某一时刻有相同的输入，但由于历史信息不同，也会得到不同的输出，这也是循环神经网络相比于常规网络的不同之处。
根据输入与输出之间的对应关系，可以将循环神经网络分为以下五大类别：


![RNN_25](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN/RNN_25.png)



常规网络中的输入与输出大多是向量与向量之间的关联，不考虑时间上的联系，而在循环神经网络中，输入与输出之间大多是序列与序列(Sequence-to-Sequence.)之间的联系，也就产生了多种模式。

*  一对一（one to one）：最为简单的反向传播网络
*  一对多（one to many）：可用于图像捕捉（image captioning），将图像转换为文字
*  多对一（many to one）：常用于情感分析（sentiment analysis），将一句话中归为具体的情感类别。
* 多对多（ many to many）：常用于输入输出序列长度不确定时，例如机器翻译（machine translate），实质是两个神经网络的叠加。
*   不确定长度的多对多（many to many）（最右方）：常用于语音识别（speech recognition）中，输入与输出序列等长。

## 10.1.6 常规循环神经网络存在的问题—长期依赖（Long Term dependencies）的问题

![RNN_6_from_zhihu](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN/RNN_6.png)

$^{[2]}$忽略激活函数的非线性变换，只看上图蓝色箭头的传输。由公式（1）简化可以得出  $$ h_{t} = W_{(hh)} h_{t-1} \qquad (3)$$ 
如果从隐藏层的初识状态$h_{0}$开始计算，可以得到：$$ h_{t} = W_{(hh)}^{t}h_{0} \qquad (4)$$
对（4）中的权重矩阵进行特征分解可得出：$$ h_{t} = Q \cdot \Lambda ^{t} \cdot Q^{t} \cdot h_{0} \qquad (5)$$

RNN最主要的特点在于能够根据上下文进行当前问题的预测。 比如我们要预测“我们用手机打电话”的最后一个词“电话”，可能只需要‘打’字就能完成预测。这种较短的依赖关系并不会给循环神经网络带来问题。但是还有很多任务事需要更长的信息依赖关系，也就是需要更多的上下文来进行当前问题的预测。比如猜测“James从小在中国长大，……，所以James说的一口流利的汉语”的最后一个词，我们一看知道结果应该是‘汉语’，但是因为这个问题要用到很长的上下文才能做出正确的预测，RNN就会出现一些问题。这也就是所说的循环神经网络中存在的长期依赖问题。长期依赖带来问题都是源于梯度，分为梯度消失（Vanishing Gradient）  和梯度爆炸（Exploding Gradient）。

理论上来说，RNN并不限制输入时序信息的长短，也就是说理论上的RNN并不会因长期依赖产生问题。

* 梯度消失（Vanishing Gradient）  
由公式（4）可以看出，当特征值小于1时，连续相乘t次后，就会趋近于0，称为梯度消失现象。

* 梯度爆炸（Exploding Gradient ）  
由公式（4）可以看出，当特征值大于1时，连续相乘t次后，就会趋近于无穷大$\infty$，称为梯度爆炸现象。

不管是梯度消失还是梯度爆炸，在循环神经网络的不断循环的过程中，都会造成隐藏层信息被丢弃的问题，使得隐藏层状态不能有效的向前传递到$h_{t}$。所以为了避免上述两种问题的发生，循环神经网络中的梯度相乘的积要尽可能的保持在1左右，才能够克服循环神经网络中长期依赖带来的梯度消失或者梯度爆炸问题。

## 10.1.7 反向传播算法进行梯度求解—BP和BPTT

[BPTT](http://www.cnblogs.com/pinard/p/6509630.html)

### 10.1.7.1  反向传播算法（Backpropagation）

1、反向传播算法要解决的问题

深层神经网络（Deep Neural Network，DNN)由输入层、多个隐藏层和输出层组成，任务分为分类和回归两大类别。如果我们使用深层神经网络做了一个预测任务，预测输出为$\tilde{y}$，真实的为y，这时候就需要定义一个损失函数来评价预测任务的性能，接着进行损失函数的迭代优化使其达到最小值，并得到此时的权重矩阵和偏置值。在神经网络中一般利用梯度下降法（Gradient Descent）迭代求解损失函数的最小值。在深层神经网络中使用梯度下降法迭代优化损失函数使其达到最小值的算法就称为反向传播算法（Back Propagation，BP）。

2、反向传播算法的推导过程

假设深层网络第L层的输出为$a_{L}$:
 $$\begin{split}
 a^{L} &= \sigma(z^{L}) \\
 &= \sigma (W^{L} \cdot a^{L-1}  + b^{L})
\end{split}$$
定义损失函数$J(w,b,x,y)$为：
 $$\begin{split}
J(w,b,x,y) &= \frac{1} {2} \parallel a_{L} - y \parallel _{2} ^{2} \\
&=  \frac{1} {2} \parallel  \sigma(z^{L})  - y \parallel _{2} ^{2}\\
&=  \frac{1} {2} \parallel  \sigma( W^{L} \cdot a^{L-1}  + b^{L} ) - y \parallel _{2} ^{2}\\
\end{split}$$
注解：$a_{L}$为预测输出,$y$为实际值，二者具有相同的维度。$\parallel \cdot \parallel_{2}$ 代表二范数。
对损失函数运用梯度下降法迭代求最小值，分别求解对于权重矩阵$W^{L}$和偏置$b^{L}$的梯度。

**损失函数对权重矩阵的梯度：**
$$\begin{split}
\frac{\partial  J(w,b,x,y)}{\partial  W^{L}} &=  
					\frac{\partial  J(w,b,x,y)}{\partial  a^{L}} \cdot 
 					\frac{\partial  a^{L}}{\partial  z^{L}}  \cdot
 					\frac{\partial  z^{L}}{\partial  w^{L}}   \\
 	&= (a^{L} - y) \bigodot  \sigma^{'}(z^{L}) \ast(a^{(L-1)})^{T}
 \end{split}$$
 
 **损失函数对偏置的梯度：**
$$\begin{split}
\frac{\partial  J(w,b,x,y)}{\partial  b^{L}} &=  
					\frac{\partial  J(w,b,x,y)}{\partial  a^{L}} \cdot 
 					\frac{\partial  a^{L}}{\partial  z^{L}}  \cdot
 					\frac{\partial  z^{L}}{\partial  b^{L}}   \\
 	&= (a^{L} - y) \bigodot  \sigma^{'}(z^{L}) 
 \end{split}$$
 
 其中公式中的符号$ \bigodot$ 代表Hadamard积，即维度相同的两个矩阵中位置相同的对应数相乘后的矩阵。
 
 损失函数对于权重矩阵和偏置的梯度含有共同项$\frac{\partial  J(w,b,x,y)}{\partial  a^{L}} \cdot  \frac{\partial  a^{L}}{\partial  z^{L}} $，令其等于$\delta^{L}$。
 
 可以求得$  \delta^{L}$为
  $$\begin{split}
  \delta^{L} &= \frac{\partial  J(w,b,x,y)}{\partial  a^{L}} \cdot  \frac{\partial  a^{L}}{\partial  z^{L}}  \\
&= (a^{L} - y) \bigodot  \sigma^{'}(z^{L}) 
 \end{split}$$
 
 知道L层的$  \delta^{L}$就可以利用数学归纳法递归的求出L-1，L_2……各层的梯度。
   $$\begin{split}
  \delta^{l} &= \frac{\partial  J(w,b,x,y)}{\partial  a^{l}} \cdot  \frac{\partial  a^{l}}{\partial  z^{l}} \\
  &= \frac{\partial  J(w,b,x,y)}{\partial  a^{L}} \cdot  
 		 \frac{\partial  a^{L}}{\partial  z^{L}} \cdot 
  		\frac{\partial  z^{L}}{\partial  z^{L-l}} \cdot 
  		\frac{\partial  z^{L-1}}{\partial  z^{L-2}}  …… 
  		\cdot \frac{\partial  z^{l+1}}{\partial  z^{l}} 
 \end{split}$$
 又知：
$$ z^{l} = W^{l} \cdot a^{l-1}  + b^{l} $$

所以第$ l $层的梯度$W^{l}、b^{l}$可以表示为 ：
 $$\begin{split}
\frac{\partial  J(w,b,x,y)}{\partial  W^{l}} &=  \delta^{l} (a^{(l-1)})^{T}\\
 \frac{\partial  J(w,b,x,y)}{\partial  b^{l}} &=  \delta^{l}
 \end{split}$$

数学归纳法求：
  $$\begin{split}
 \delta^{l}  &=  \frac{\partial  J(w,b,x,y)}{\partial  a^{l}} \cdot 
 						 \frac{\partial  a^{l}}{\partial  z^{l}} \\
 				& =  \frac{\partial  J(w,b,x,y)}{\partial  a^{l+1}} \cdot 
 						 \frac{\partial  a^{l+1}}{\partial  z^{l+1}} \cdot 
 						 \frac{\partial  z^{l+1}}{\partial  z^{l}} \\
 				& =  \delta^{l+1}  \frac{\partial  z^{l+1}}{\partial  z^{l}} 
\end{split}$$
又知：
$$\begin{split}
z^{l+1} &= W^{l+1} \cdot a^{l}  + b^{l+1} \\ 
	&= W^{l+1} \cdot  \sigma( z^{l})+ b^{l+1}
\end{split}$$
所以可得：
$$\begin{split}
\frac{\partial  z^{l+1}}{\partial  z^{l}} &= ( W^{l+1})^{T} \bigodot  \sigma ^{'}( z^{l})
\end{split}$$
可得：
  $$\begin{split}
 \delta^{l}  & =  \delta^{l+1}  \frac{\partial  z^{l+1}}{\partial  z^{l}}  \\
 				&= \delta^{l+1} ( W^{l+1})^{T}\bigodot  \sigma ^{'}( z^{l})
\end{split}$$

求得了$ \delta^{l}$ 的递推关系之后，就可以依次求得各层的梯度$W^{l}和b^{l}$了。

### 10.1.7.2 随时间的反向传播过程（Back Propagation Through Time）

循环神经网络的特点是利用上下文做出对当前时刻的预测，RNN的循环也正是随时间进行的，采用梯度下降法对循环神经网络的损失函数进行迭代优化求其最小值时也是随时间进行的，所以这个也被称为随时间的反向传播（Back Propagation Through Time，BPTT），区别于深层神经网络中的反向传播（BP）。

![RNN_15_from_zhihu](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN/RNN_15.png)


* 为了更易被读者理解推导过程，如上图所示，我们进行以下定义：  
*  U：输入层的权重矩阵  
*  W：隐藏层的权重矩阵  
*  V：输出层的权重矩阵  
*  t时刻的输入为$x^{(t)}$：同理$x^{(t+1)}$为t+1时刻的输入信息。  
*  t时刻的隐藏层状态为$h^{(t)}$：由$x^{(t)}$和$h^{(t-1)}$共同决定。
$$\begin{split} 
h^{t} &=\sigma(z^{(t)}) \\
&= \sigma (U \cdot x^{(t)} + W \cdot h^{(t-1)} + b)\\
&=tanh (U \cdot x^{(t)} + W \cdot h^{(t-1)} + b)
\end{split}$$
*  t时刻的输出为$o^{(t)}$：只由t时刻的隐藏状态$h^{(t)}$决定。
$$ o^{t} = V \cdot h^{(t)} +c $$  
*  真实的值为$y^{t}$，预测的值为$\hat{y}$。  
$$\begin{split}
 \hat{y} &= \sigma(o^{(t)})  \\
&= softmax(o^{(t)})
\end{split}$$
*  t 时刻的损失函数为$L^{t}$，评价预测的性能也就是量化预测值与真实值之间的差，本篇假设损失函数为交叉熵
$$L^{(t)} = y^{(t)} log( \hat{y^{(t)}} ) + ( 1 - y^{(t)}) log(1- \hat{y^{(t)}} ) $$ 
最终的损失函数为各个时刻损失函数的加和,即
$$ L = \sum_{t=1}^{\tau} L^{(t)}$$

* 注解：

(1) U，V，W为线性共享参数，在循环神经网络的不同时刻不同位置，这三个权重矩阵的值是一样的，这也正是RNN的循环所在。  
(2) 假设损失函数为交叉熵，也就是等价于对数损失函数，隐藏层激活函数为tanh函数，输出层激活函数为softmax函数。

* BPTT的推导

因为我们假设的输出层激活函数为softmax函数，所以得到以下公式：
$$\begin{split}
\frac{\partial{L^{(t)}} }{\partial{o^{(t)}}}  =  \sum_{t=1}^{t=\tau} (\hat{ y^{(t)}}-y^{(t)})
\end{split}$$

假设隐藏层的函数为tanh函数，可得
$$\begin{split}
\frac{\partial{h^{(t)}} }{\partial{h^{(t-1)}}}  &= W^{T} diag(1-h^{(t)} \bigodot h^{(t)})\\
\frac{\partial{h^{(t)}} }{\partial{U}}  &= (x^{(t)})^{T} diag(1-h^{(t)} \bigodot h^{(t)})\\
\frac{\partial{h^{(t)}} }{\partial{W}}  &= (h^{(t-1)})^{T} diag(1-h^{(t)} \bigodot h^{(t)})\\
\frac{\partial{h^{(t)}} }{\partial{b}}  &= diag(1-h^{(t)} \bigodot h^{(t)})\\
\end{split}$$
**损失函数对于c的梯度：**
$$\begin{split}
\frac{\partial{L(U,V, W,b, c)}}{\partial{c}}  &= \sum_{t=1}^{t=\tau}\frac{\partial{L^{(t)}} }{\partial{o^{(t)}}} \frac{\partial{o^{(t)}} }{\partial{c}} \\
&= \sum_{t=1}^{t=\tau}  y^{(t)}-y^{(t)}
\end{split}$$
**损失函数对于V的梯度：**
$$\begin{split}
\frac{\partial{L(U,V, W,b, c)}}{\partial{V}}  &= \sum_{t=1}^{t=\tau}\frac{\partial{L^{(t)}} }{\partial{o^{(t)}}} \frac{\partial{o^{(t)}} }{\partial{V}} \\
&= \sum_{t=1}^{t=\tau}  (\hat{y^{(t)}}-y^{(t)}) (h^{(t)})^{T}
\end{split}$$
**损失函数对于W, U, b的梯度：**
随时间的反向传播算法中，循环神经网络的梯度损失由当前时间步t的梯度和下一时刻t+1的梯度损失两部分决定。
定义损失函数对于隐藏状态$h^{(t)}$的梯度为：
$$\delta ^{(t)} = \frac{\partial{L(U,V, W,b, c)}}{\partial{h^{(t)}}}$$
类似于前文所说的深层神经网络中的反向传播算法，可以由$\delta ^{(t+1)}$递推出$\delta ^{(t)}$，公式如下：

$$\begin{split}
\delta ^{(t)} &=
\frac{\partial{L^{(t)}} }{\partial{o^{(t)}}} \frac{\partial{o^{(t)}} }{\partial{h^{(t)}}} +
\frac{\partial{L^{(t)}} }{\partial{h^{(t+1)}}} \frac{\partial{h^{(t+1)}} }{\partial{h^{(t)}}}  \\
&= V^{T}(\hat{y^{(t)}}-y^{(t)}) +W^{T}\delta ^{(t+1)} diag(1 - h^{(t+
1)} \bigodot h^{(t+1)})
\end{split}$$

注意：  
对于$\delta ^{(\tau)} $因为没有下一时刻的信息了，所以
$$\begin{split}
\delta ^{(\tau)} &=
\frac{\partial{L^{(\tau}} }{\partial{o^{(\tau)}}} \frac{\partial{o^{(\tau)}} }{\partial{h^{(\tau)}}}\\
&= V^{T}(\hat{y^{(\tau)}}-y^{(\tau)}) 
\end{split}$$

在递推出了以上公式后，计算损失函数对于W，U，b的梯度就比较简单了。
$$\begin{split}
\frac{\partial{L(U,V, W,b, c)}}{\partial{U}}  &= \sum_{t=1}^{t=\tau}\frac{\partial{L^{(t)}} }{\partial{h^{(t)}}} \frac{\partial{h^{(t)}} }{\partial{U}} \\
&= \sum_{t=1}^{t=\tau}  \delta^{(t)} (x^{(t)})^{T} diag(1-h^{(t)} \bigodot h^{(t)})\\
\end{split}$$

$$\begin{split}
\frac{\partial{L(U,V, W,b, c)}}{\partial{W}}  &= \sum_{t=1}^{t=\tau}\frac{\partial{L^{(t)}} }{\partial{h^{(t)}}} \frac{\partial{h^{(t)}} }{\partial{W}} \\
&= \sum_{t=1}^{t=\tau}  \delta^{(t)} (h^{(t-1)})^{T} diag(1-h^{(t)} \bigodot h^{(t)})\\
\end{split}$$

$$\begin{split}
\frac{\partial{L(U,V, W,b, c)}}{\partial{b}}  &= \sum_{t=1}^{t=\tau}\frac{\partial{L^{(t)}} }{\partial{h^{(t)}}} \frac{\partial{h^{(t)}} }{\partial{b}} \\
&= \sum_{t=1}^{t=\tau}  \delta^{(t)} diag(1-h^{(t)} \bigodot h^{(t)})\\
\end{split}$$

## 10.1.8 循环神经网络与前馈神经网络的对比

前馈神经网络，也就是我们说的常规网络，只沿着网络向前传递，不存在循环过程，上一层的输出时下一层的输入。

输入和输出：矩阵  
输入矩阵形状：(n_samples, dim_input)  
输出矩阵形状：(n_samples, dim_output)  

在真正训练网络的时候，输入和输出数据就是向量，n\_samples这个维度是为了可以实现一次训练多个样本，也就是说每个样本数量为一个batch_size。然后在多个样本中求出平均梯度来更新权重，这就是所说的Mini-batch 梯度下降法（gradient descent）。
如果n_samples等于1，那么这种更新方式叫做随机梯度下降法（Stochastic Gradient Descent ， SGD)。

循环神经网络，加入了循环也就是说当前时刻的输入除了上一层的输出外，还有自身的输出。循环神经网络的输入和输出维度至少是3维的张量，如果输入是图片等数据，那么张量的维度还会更多，张量也就是所说的tensor。

输入张量形状：(time_steps, n_samples, dim_input)  
输出张量形状：(time_steps, n_samples, dim_output)  

在循环神经网络中=也有Mini-batch gradient descent的训练方式，只不过多了time step这个维度。  
Recurrent 的任意时刻的输入的本质还是单个向量，只不过是将不同时刻的向量按时间顺序输入网络，可以简单理解为矩阵。


# 10.2、循环神经网络的变体
为了解决上述循环神经网络中存在的长期依赖问题，经过学者们的不断研究和试验，提出了很多循环神经网络的变体。

## 10.2.1 渗漏单元和门控机制

### 10.2.1.1 渗漏单元（leaky units）

为了使权重矩阵乘积保持在1左右，建立线性自连接单元(linear self-connections），并且这些连接的权重接近为1。线性自连接的隐藏单元称为渗漏单元（leaky unit）。线性自连接单元的权重设置一般有两种方式：（1）手动设定为常数，（2）设为自由变量，学习出来。   
渗漏单元能够在较长时间内积累信息，且不存在梯度爆炸和梯度消失的问题。但是在很多情况下，这些积累的信息一经采用后，让神经网络遗忘掉旧的状态可能是更好的。比如在一个序列信息中又有很多子序列，神经网络计算完一个子序列之后，我们希望它能将状态自动初始为0并继续下一个子序列的计算。基于这种需求，人们提出了能够学习决定何时自动清除历史状态的门控循环神经网络（Gated Recurrent Neural Network）。

### 10.2.1.2 门控机制

门控RNN简单的说就是能够实现线性自连接单元权重的自动调节，神经网络能够学习何时遗忘旧的状态。

1、关于门控循环神经网络‘门’的理解

顾名思义，‘门’的直观意义就是控制信息的输入和输出。循环神经网络中的门控机制就是通过‘门’的输出来控制其他要通过此门的数据，相当于过滤器的作用。

输入：门控依据。  
输出：[0,1]之间的数值，0代表数据全部丢弃，1代表数据全部通过。  
举例说明：    
数据$x=[2, 2, 2, 2]$,门控输出为$g = [0.5, 0.5, 0.5, 0.5 ]$，数据通过门之后输出为$ out  = [ 1, 1, 1, 1]$。

目前比较主流的门控循环神经网络一种是长短期记忆（Long-short term memory），一般称为“LSTM”；另一种就是基于LSTM提出的门控循环单元（Gated recurrent unit），也称为“GRU”。

## 10.2.2 长短期记忆网络LSTM（Long-short term memory）

###10.2.2.1 长短期记忆网络简介
长短期记忆（（Long short-term memory）最早是1997年由Hochreiter 和 Schmidhuber在论文《LONG SHORT-TERM MEMORY》$^{[3]}$中提出的。

![RNN_7_fom_colah](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN/RNN_7.png)


在所有的循环神经网络中都有一个如图所示的循环链，链式结构的每个单元将输入的信息进行处理后输出。在常规循环神经网络中的链式结构的单元中只含有一个非线性函数对数据进行非线性转换，比如tanh函数。而在LSTM等RNN变体的循环神经网络中，每个单元在不同位置多加了几个不同的非线性函数，实现不同的功能，解决不同的问题。如下图所示。

![RNN_8_fom_colah](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN/RNN_8.png)


LSTM最主要的就是记忆细胞（memory cell ），处于整个单元的水平线上，起到了信息传送带的作用，只几个含有简单的线性操作，能够保证数据流动时保持不变，也就相当于上文说的渗漏单元。如下图所示。

![RNN_9_fom_colah](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN/RNN_9.png)


### 10.2.2.2 长短期记忆网络详解

LSTM就是在每个小单元中增加了三个sigmoid函数，实现门控功能，控制数据的流入流出，分别称为遗忘门（forget gate），输入门（input gate）和输出门（output gate）。

1、遗忘门（forget gate)


![RNN_10_fom_colah](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN/RNN_10.png)


**主要功能：** 决定记忆细胞中要丢弃的信息，也就是决定前一时刻记忆细胞$C_{t-1}$中有多少信息能够继续传递给当前时刻的记忆细胞$C_{t}$。  
**数学公式：** 
$$ \begin{split}
f_{t} &=\sigma (W_{f} \cdot [h_{t-1}, x_{t}] + b_{f}) \\
&=\sigma (W_{hf} \cdot h_{t-1} + W_{xf} \cdot x_{t} + b_{f}) 
\end{split} \qquad (6)$$

2、输入门（input gate）

![RNN_11_fom_colah](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN/RNN_11.png)


**主要功能：** 决定有多少新的输入信息能够进入当前时刻的记忆细胞$C_{t}$。  
**数学公式：** 
$$ \begin{split}
i_{t} &=\sigma (W_{i} \cdot [h_{t-1}, x_{t}] + b_{i}) \\
&=\sigma (W_{hi} \cdot h_{t-1} + W_{xi} \cdot x_{t} + b_{i}) 
\end{split}  \qquad (7) $$

3、输出门（output gate）

![RNN_12_fom_colah](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN/RNN_12.png)


**主要功能：** 决定有多少当前时刻的记忆细胞$C_{t}$中的信息能够进入到隐藏层状态$h_{t}$中。  
**数学公式：** 
$$ \begin{split}
o_{t} &=\sigma (W_{o} \cdot [h_{t-1}, x_{t}] + b_{o}) \\
&=\sigma (W_{ho} \cdot h_{t-1} + W_{xo} \cdot x_{t} + b_{o}) 
\end{split}  \qquad (8)$$

4、利用遗忘门和输入门更新记忆细胞：

利用tanh函数产生候选记忆细胞$\tilde{C_{t}}$，也就是即将要进入输入门的信息。
$$ \begin{split}
\tilde{C_{t}} &= tanh ( W_{c} \cdot [h_{t-1}, x_{t}] + b_{c}) \\
&=  tanh ( W_{hc} \cdot h_{t-1} + W_{xc} \cdot x_{t} + b_{c}) 
\end{split}  \qquad (9)$$

![RNN_13_fom_colah](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN/RNN_13.png)


遗忘门的输出与旧状态相乘，决定有多少旧状态的信息能够进入到新的记忆细胞；输入门的输出与候选记忆细胞相乘，决定有多少信息要被更新。二者线性相加得到当前时刻被更新过的记忆细胞。
$$\begin{split}
 C_{t} &= f_{t} \ast  C_{t-1} + i_{t} \ast \tilde{C_{t}}  \\
 & = f_{t} \ast  C_{t-1} + i_{t} \ast tanh ( W_{hc} \cdot h_{t-1} + W_{xc} \cdot x_{t} + b_{c}) 
 \end{split}  \qquad (10) $$

也正是这一过程实现了历史信息的累积。

5、利用输出门将信息输出到隐藏层

将数据经过tanh函数的处理$(tanh(C_{t}))$，将数据归一化在[-1, 1]区间内。然后输出门的结果与归一化的数据相乘，控制输出到隐藏层的数据量。
$$ h_{t} = o_{t} \ast tanh(C_{t})   \qquad (11) $$


## 10.2.3 门控循环神经网络（Gated Recurrent Neural Network）

### 10.2.3.1 门控循环神经网络简介

在神经网络发展的过程中，几乎所有关于LSTM的文章中对于LSTM的结构都会做出一些变动，也称为LSTM的变体。其中变动较大的是门控循环单元（Gated Recurrent Units），也就是较为流行的GRU。GRU是2014年由Cho, et al在文章《Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation》中提出的，某种程度上GRU也是对于LSTM结构复杂性的优化。LSTM能够解决循环神经网络因长期依赖带来的梯度消失和梯度爆炸问题，但是LSTM有三个不同的门，参数较多，训练起来比较困难。GRU只含有两个门控结构，且在超参数全部调优的情况下，二者性能相当，但是GRU结构更为简单，训练样本较少，易实现。

GRU在LSTM的基础上主要做出了两点改变 ：

（1）GRU只有两个门。GRU将LSTM中的输入门和遗忘门合二为一，称为更新门（update gate），上图中的$z_{(t)}$，控制前边记忆信息能够继续保留到当前时刻的数据量，或者说决定有多少前一时间步的信息和当前时间步的信息要被继续传递到未来；GRU的另一个门称为重置门（reset gate），上图中的$r_{(t)}$，控制要遗忘多少过去的信息。 

（2）取消进行线性自更新的记忆单元（memory cell），而是直接在隐藏单元中利用门控直接进行线性自更新。GRU的逻辑图如上图所示。

 ![RNN_14_fom_colah](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN/RNN_14.png)


### 10.2.3.2 门控循环神经网络详解

1、更新门（update gate）

**主要功能：**   
决定有多少过去的信息可以继续传递到未来。  将前一时刻和当前时刻的信息分别进行线性变换，也就是分别右乘权重矩阵，然后相加后的数据送入更新门，也就是与sigmoid函数相乘，得出的数值在[0, 1]之间。
  
**数学公式：**   

$$\begin{split}
z_{t} &= \sigma(W_{z} \cdot [h_{t-1}, x_{t}]) \\
&= \sigma(W_{hz} \cdot h_{t-1} + W_{xz} \cdot x_{t})
\end{split}  \qquad (12)$$

2、重置门（reset gate）
**主要功能：**

决定有多少历史信息不能继续传递到下一时刻。同更新门的数据处理一样，将前一时刻和当前时刻的信息分别进行线性变换，也就是分别右乘权重矩阵，然后相加后的数据送入重置门，也就是与sigmoid函数相乘，得出的数值在[0, 1]之间。只是两次的权重矩阵的数值和用处不同。
  
**数学公式：**

$$\begin{split}
r_{t} &= \sigma(W_{r} \cdot [h_{t-1}, x_{t}]) \\
&= \sigma(W_{hr} \cdot h_{t-1} + W_{xr} \cdot x_{t})
\end{split}  \qquad (13)$$

3、利用重置门重置记忆信息

GRU不再使用单独的记忆细胞存储记忆信息，而是直接利用隐藏单元记录历史状态。利用重置门控制当前信息和记忆信息的数据量，并生成新的记忆信息继续向前传递。
$$\begin{split}
\tilde{h_{t}} &= tanh (W \cdot [ r_{t} \ast h_{t-1} , x_{t} ]) \\
&= tanh  [ r_{t} \ast  (W_{h} \cdot h_{t-1}) + W_{x}\cdot x_{t} ]
\end{split}   \qquad (14)$$

如公式（14），因为重置门的输出在区间[0, 1]内，所以利用重置门控制记忆信息能够继续向前传递的数据量，当重置门为0时表示记忆信息全部清除，反之当重置门为1时，表示记忆信息全部通过。

4、利用更新门计算当前时刻隐藏状态的输出

隐藏状态的输出信息由前一时刻的隐藏状态信息$h_{t-1}$和当前时刻的隐藏状态输出$h_{t}$，利用更新门控制这两个信息传递到未来的数据量。
$$ h_{t} = z_{t} \ast h_{t-1} + (1 - z_{t} ) \ast \tilde {h_{t}}  \qquad (15) $$

#10.3、循环神经网络的应用
[keras中文文档](http://keras-cn.readthedocs.io/en/latest/for_beginners/concepts/)
##10.3.1、keras简单介绍

**1、张量**

张量——tensor，表示广泛的数据类型，用轴（axis）或者维度表示张量的阶数。
从以下代码就可以看出axis的作用。

```python 
import numpy as np
a = np.array([[1,2],[3,4]])
sum_1 = np.sum(a, axis = 0)
sum_2 = np.sum(a, axis = 1)
print('sum-1',sum_1)
print('sum_2',sum_2)

输出如下：
sum_1 [4 6]
sum_2 [3 7]
```

**2、data_format**

在表示彩色图像时，keras与tensorflow和Theano不同。比如在表示1000个由RGB三通道组成的16*25的图像时，Theano和tensorflow会表示成（1000，3，16，25），也就是‘th’模式的第0个维度表示样本数，第1个维度表示通道数，后两个表示图像的高和宽，称为“channels\_first”模式。 keras中将图像的通道数放在最后一个维度，即（1000，16，25，3），称为‘channels_last’模式。

可以通过代码查看是哪一个模式，在代码中要保持图像模式的一致性。  

```python
from keras import backend as K
K.image_data_format()

输出：'channels_last'
```

**3、Sequential（）模型——序贯模型**

* 构建模型

Sequential模型就是将不同的层进行叠加。通过传入一个列表或者add函数都可以搭建模型。需要注意的是Sequential模型需要在第一层传入输入数据的shape参数，之后的每一层会根据这一参数自行计算出每一层的数据大小，也就是说在之后的每一层中都无需再传入数据的shape参数。

构建简单模型的参考代码如下：

```python
from keras import Sequential
from keras.layers import  Dense, Activation
model = Sequential([
    Dense(32,input_shape = (784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax')
])
# 或者通过add函数
from keras import Sequential
from keras.layers import  Dense, Activation
model = Sequential()
model.add(Dense(32,input_shape = (784,)))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
```

* 编译

使用compile（）函数进行模型的编译，compile需要传入三个参数：

* optimizer ：优化器。对损失函数进行优化时采用的优化函数，例如rmsprop、adagrad。
* loss：损失函数。一般的循环神经网络中采用交叉熵函数也就是categorical_crossentropy，回归问题中多采用平方差mse，或者自己定义损失函数。
* metrics：评估指标。一般设置为正确率accuracy，也可以自己定义函数。

```python 
# 分类问题常用categorical交叉熵，也称为多类的损失函数
model.compile(
    optimizer = 'rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
# 常用binary交叉熵,也称为对数损失函数
model.compile(
    optimizer = 'rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
# 回归问题多用 平方误差作为损失函数
model.compile(optimizer='rmsprop',
              loss='mse')

#或者自定义评估指标
import keras.backend as K
def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])
```
**keras的各层：**

keras的core模块中定义了常用层，包括全连接层，激活层，dropout层等。

（1）Dense layer

相当于全连接层，实现的运算过程就是
$$ out\_dense = activation （ dot（input , weight） + bias）$$

```python
keras.layers.core.Dense(
	units, activation=None, use_bias=True, 
	kernel_initializer='glorot_uniform', bias_initializer='zeros', 
	kernel_regularizer=None, bias_regularizer=None, 
	activity_regularizer=None, kernel_constraint=None, 
	bias_constraint=None)
```
参数说明：  
units：输出维度。  
activation：激活函数。不指定的话就默认没有，即为线性函数。  
use\_bias：是否使用偏置。  
*\_initializer：初始化。  
*\_regularizer：加正则项。  
*\_constraint：加约束项。    
输入：常为（batch\_size，input\_dim）  
输出：常为（batch\_size，units）

（2）Activation layer

激活层，指定激活函数，没有指定视为线性计算。输入不限，如果是第一层要制定输入的shape，输出为与输入shape相同。

```python
keras.layers.core.Activation(activation)
```
参数讲解：  
activation: 即要使用的激活函数，一般为非线性函数，常用函数为tanh，sigmoid，softmax。

（3）Dropout layer

Dropout层为丢弃层，按照设置的rate参数随机丢弃一定比例的数据，作用是防止神经网络训练过程中的过拟合。

```python
keras.layers.core.Dropout(rate, noise_shape=None, seed=None)
```
参数讲解：  
rate：0-1之间的浮点数，控制要随机断开的神经元比例。  
noise\_shape：整数张量，为将要应用在输入上的二值Dropout mask的shape，例如你的输入为(batch\_size, timesteps, features)，并且你希望在各个时间步上的Dropout mask都相同，则可传入noise\_shape=(batch_size, 1, features)。  
seed：整数，使用的随机数种子。  

（4）Recurrent layer

这是循环层的抽象类,在模型中不建议直接使用该层（因为它是抽象类，无法实例化任何对象，一般都是使用它的子类LSTM，GRU或SimpleRNN。

所有的循环层（LSTM,GRU,SimpleRNN）都继承本层，因此下面的参数可以在任何循环层中使用。

```python 
keras.layers.recurrent.Recurrent(
	return_sequences = False, 
	go_backwards=False, 
	stateful=False, 
	unroll=False, 
	implementation=0)
```
参数说明：


（5）LSTM layer

```python 
keras.layers.recurrent.LSTM(
	units,
	activation='tanh', 
	recurrent_activation='hard_sigmoid', 
	use_bias=True, 
	kernel_initializer='glorot_uniform',
	recurrent_initializer='orthogonal', 
	bias_initializer='zeros', 
	unit_forget_bias=True, 
	kernel_regularizer=None, 
	recurrent_regularizer=None, 
	bias_regularizer=None, 
	activity_regularizer=None, 
	kernel_constraint=None,
	recurrent_constraint=None, 
	bias_constraint=None, 
	dropout=0.0, 
	recurrent_dropout=0.0)
```
参数说明：  
units:：输出维度。  
activation ：激活函数。  
recurrent\_activation：循环步中使用的激活函数  
use\_bias ：是否使用偏置。  
dropout：要丢失的数据比例。  
recurrent_dropout：隐藏状态之间要丢失的数据比例     

##10.3.2 用LSTM实现MNIST手写数字识别

就像开始学习编程语言时入门程序是‘Hello World’一样，mnist就是机器学习中的‘Hello World’。

1、MNIST手写数字
本数据库有x_train为60,000个用于训练的28*28的灰度手写数字图片，x_test为10,000个测试图片，如下图所示。y_train，y_test是标记的数字，值为0-9。

![RNN_16_from_zhihu](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN/RNN_16.png)
![RNN_16](/Users/laiye/Desktop/ML/RNN/RNN_16.png)

（1）从keras数据库中加载数据集

```python 
from keras.datasets import mnist
#加载数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('x_train.shape:',x_train.shape)
print('x_test.shape:',x_test.shape)

输出：
('x_train.shape:', (60000, 28, 28))
('x_test.shape:', (10000, 28, 28))
```

（2）处理数据   
因为图片的大小是（28，28），所以将神经网络的输入设置为（28，），也就是一行一行的读入像素值，这样时间序列的值为列数值28。

```python
#时间序列数量
n_step = 28
#每次输入的维度
n_input = 28
#分类类别数
n_classes = 10

#将数据转为28*28的数据（n_samples,height,width）
x_train = x_train.reshape(-1, n_step, n_input)
x_test = x_test.reshape(-1, n_step, n_input)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#标准化数据，因为像素值在0-255之间，所以除以255后数值在0-1之间
x_train /= 255
x_test /= 255

#y_train，y_test 进行 one-hot-encoding，label为0-9的数字，所以一共10类
y_train = keras.utils.to_categorical(y_train, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)
```

（3）构建模型

```python
from keras import Sequential
from keras.layers import LSTM,Dense, Activation
from keras import optimizers

#使用Sequential，简单搭建lstm模型
model = Sequential()

#这个网络中，我们采用LSTM+Dense 层+激活层，优化函数采用Adam，
#损失函数采用交叉熵，评估采用正确率。

#学习率
learning_rate = 0.001
#每次处理的数量
batch_size = 28
#循环次数
epochs = 20
#神经元的数量
n_lstm_out = 128

#LSTM层
model.add(LSTM(
        units = n_lstm_out,
        input_shape = (n_step, n_input)))
#全连接层          
model.add(Dense(units = n_classes))
#激活层
model.add(Activation('softmax'))

#查看各层的几本信息
model.summary()

# 编译
model.compile(
    optimizer = optimizers.Adam(lr = learning_rate),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy'])

#训练
model.fit(x_train, y_train, 
          epochs = epochs, 
          batch_size= batch_size, 
          verbose = 1, 
          validation_data = (x_test,y_test))
#评估
score = model.evaluate(x_test, y_test, 
                       batch_size = batch_size, 
                       verbose = 1)
print('loss:',score[0])
print('acc:',score[1])
```

![RNN_17](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN/RNN_17.png)


```python
import matplotlib.pyplot as plt
%matplotlib inline
num = 20
x_test_reshape = x_test.reshape(x_test.shape[0],n_step,n_input)
prediction = model.predict(x_test[:num])
prediction = prediction.argmax(axis = 1)
plt.figure(figsize = [20,5])
for i in range(num):
    plt.subplot(2,num,i+1)
    plt.imshow(x_test_reshape[i])
    plt.text(0,-5,prediction[i])
    plt.axis = 'off'
```
![RNN_18](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN/RNN_18.png)


# 10.3.3  LSTM实现简单的问答系统
## 10.3.3.1 问答系统简介

本次示例为最为简单的问答系统，即通过语料集训练出模型，之后给模型一个问题，模型预测出问题的答案。

## 10.3.3.2 基于keras实现简单的问答系统

模型逻辑图如下：  
![RNN_20](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN/RNN_20.png)


数据集：Facebook的bAbI数据  
训练集：  

```python 
1 Mary moved to the bathroom.
2 Sandra journeyed to the bedroom.
3 Mary got the football there.
4 John went to the kitchen.
5 Mary went back to the kitchen.
6 Mary went back to the garden.
7 Where is the football? 	garden	3 6
8 Sandra went back to the office.
9 John moved to the office.
10 Sandra journeyed to the hallway.
11 Daniel went back to the kitchen.
12 Mary dropped the football.
13 John got the milk there.
14 Where is the football? 	garden	12 6
15 Mary took the football there.
16 Sandra picked up the apple there.
17 Mary travelled to the hallway.
18 John journeyed to the kitchen.
19 Where is the football? 	hallway	15 17
训练集是对话 + 问题 + 答案的形式，每个问句中以tab键分割问题、答案以及含有答案的句子索引。
```
接下来利用两个循环神经网络实现简单的问答系统。  
（1）获取预处理  
数据在amazoneaws的网站上，如果在运行代码出现下载不成功，就要先把数据集下载下来，然后放到keras的数据集目录下。代码中有具体操作。

```python 
# 获取数据
from keras.utils.data_utils import get_file
import tarfile
try:
    path = get_file('babi-tasks-v1-2.tar.gz', \
                    origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
except:
    print('Error downloading dataset, please download it manually:\n'
          '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz\n'
          '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
    raise
```
（2）数据预处理   
**对文本数据进行向量化，word2vector**  

1. 对文本数据 Tokenize，因为本数据集为英文，分词可直接用空格，如果数据集为中文，需要利用结巴或者其他分词器进行分词。

```python 
#将每个单词分割来
def tokenize(data):
    import re
    # ‘\W’ 匹配所有的字母数字下划线以外的字符
    return [x.strip() for x in re.split(r"(\W+)?", data) if x.strip()]
```

2.  解析对话文本

```python 
# parse_dialog 将所有的对话进行解析，返回tokenize后的(对话,问题,答案)
# 如果 only_supporting为真表明只返回含有答案的对话
def parse_dialog(lines, only_supporting = False):
    data = []
    dialog = []
    for line in lines:
        line = line.strip()
        nid, line = line.split(' ',1)
        nid = int(nid)
        # 标号为1表示新的一段文本的开始，重新记录
        if nid == 1:
            dialog = []
        #含有tab键的说明就是问题，将问题，答案和答案的索引分割开
        if '\t' in line:
            ques, ans, data_idx = line.split('\t')
            ques = tokenize(ques)
            substory = None
            if only_supporting :
                data_idx = list(map(int,data_idx))
                substory = [dialog[ i-1 ] for i in data_idx.split()]
            else:
                substory = [x for x in dialog]
            data.append((substory ,ques, ans))
        else:
            # 不含有tab键的就是对话，tokenize后加入dialog的list
            line = tokenize(line)
            dialog.append(line)
    return data
```

 3. 获得每个对话文本，将tokenize后的每个对话文本放在一个列表中。将（对话，问题，答案）组成相对应的tuple存储。

 ```python 
 #这里的maxlen是控制文本最大长度的，可以利用分位数找出覆盖90%数据的长度，令其为maxlen。
 # 否则序列长度太长，训练时内存不够。
def get_dialog(f, only_supporting = False, max_length = None):
    #将对话完整的提取出来
    data = parse_dialog(f.readlines(),only_supporting = only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(dialog), ques, ans) for (dialog, ques, ans) in data 
                if not max_length or len(flatten(dialog))<max_length]
    return data
 ```
 
 4. 数据长度归一化。找出对话文本的最大单词长度，对所有的对话进行padding，将长度归一化。问题集同此。

 ```python
 def vectorize_dialog(data,wd_idx, dialog_maxlen, ques_maxlen):
    #向量化,返回对应词表的索引号
    import numpy as np
    from keras.preprocessing.sequence import pad_sequences
    dialog_vec = []
    ques_vec = []
    ans_vec = []
    for dialog, ques, ans in data:
        dialog_idx = [wd_idx[w] for w in dialog]
        ques_idx = [wd_idx[w] for w in ques]
        
        ans_zero = np.zeros(len(wd_idx) + 1)
        ans_zero[wd_idx[ans] ] = 1
        
        dialog_vec.append(dialog_idx)
        ques_vec.append(ques_idx)
        ans_vec.append(ans_zero)
        
        #序列长度归一化，分别找出对话，问题和答案的最长长度，然后相对应的对数据进行padding。
    return pad_sequences(dialog_vec, maxlen = dialog_maxlen),\
            pad_sequences(ques_vec, maxlen = ques_maxlen),\
            np.array(ans_vec)
 ```

5. 准备数据，并利用上述函数进行预处理。

```python
#准备数据
train_tar = tar.extractfile(data_path.format('train'))
test_tar = tar.extractfile(data_path.format('test'))
train = get_dialog(train_tar)
test = get_dialog(test_tar)

# 建立词表。词表就是文本中所有出现过的单词组成的词表。
lexicon = set()
for dialog, ques, ans in train + test:
    lexicon |= set(dialog + ques + [ans])
lexicon = sorted(lexicon)
lexicon_size = len(lexicon)+1

#word2vec，并求出对话集和问题集的最大长度，padding时用。
wd_idx = dict((wd, idx+1) for idx, wd in enumerate(lexicon))
dialog_maxlen = max(map(len,(x for x, _, _ in train + test )))
ques_maxlen =  max(map(len,(x for _, x, _ in train + test )))
#计算分位数，在get_dialog函数中传参给max_len
dia_80 = np.percentile(map(len,(x for x, _, _ in train + test )),80)

# 对训练集和测试集，进行word2vec
dialog_train, ques_train, ans_train = vectorize_dialog(train, wd_idx, dialog_maxlen, ques_maxlen)
dialog_test, ques_test, ans_test = vectorize_dialog(test, wd_idx, dialog_maxlen, ques_maxlen)

```

6. 搭建神经网络模型   

因为对话集和问题集之间存在内在联系，所以先分别进行embedding和dropout，在merge到一起后，传入LSTM网络中，最后LSTN网络的输出进入一个softmax函数，完成模型的搭建。

* 对话集的网络搭建。

![RNN_22](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN/RNN_22.png)
 

注：dialog\_maxlen = 149，embedding\_out = 50.   
输入：（dialog\_maxlen，）  即（None，149).     
输出：也就是embedding层的输出，shape为（dialog\_maxlen ，embedding\_out).   

因为对数据进行了长度的归一化处理，所以每个dialog的长度都为dialog\_maxlen ，所以此时输出数据的shape为
（149，50）。  每个列向量就是每个dialog进行embedding后的向量。

```python 
#对话集 构建网络—— embedding + dropout
dialog = Input(shape = (dialog_maxlen, ),dtype='int32')
encodeed_dialog = embeddings.Embedding(lexicon_size, embedding_out)(dialog)
encodeed_dialog = Dropout(0.3)(encodeed_dialog)
```

* 问题集网络搭建  

问题集进入LSTM网络，keras的LSTM默认是只输出hidden_layer的最后一维， 所以LSTM的输出只有一列向量。

LSTM的输出进行RepeatVector，也就是重复dialog_maxlen次，这样encodeed_ques的shape就变为了（dialog_maxlen，lstm_out）。与encodeed_dialog的shape相同，这样才能进行merge层的add。
![RNN_21](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN/RNN_21.png)


```python 
#问题集 embedding + dropout + lstm
question = Input(shape = (ques_maxlen,),dtype= 'int32')
encodeed_ques = embeddings.Embedding(lexicon_size, embedding_out)(question)
encodeed_ques = Dropout(0.3)(encodeed_ques)
encodeed_ques = LSTM(units = lstm_out)(encodeed_ques)
encodeed_ques = RepeatVector(dialog_maxlen)(encodeed_ques)
```

* 对话集和问题集之间是存在内在联系的，将二者通过merge层之后再进行循环神经网络的训练。

![RNN_23](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN/RNN_23.png)


```python
# merge 对话集和问题集的模型 merge后进行 lstm + dropout + dense
merged = Add()([encodeed_dialog, encodeed_ques])
merged = LSTM(units = lstm_out)(merged)
merged = Dropout(0.3)(merged)
preds = Dense(units = lexicon_size, activation = 'softmax')(merged)

model = Model([dialog, question], preds)
```

* 编译

```python
print('compiling........')
model.compile(optimizer='adam',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy']
               )
```

* 训练

```python 
#训练
print('training.......')
model.fit([dialog_train, ques_train], ans_train,
         batch_size = batch_size,
         epochs = epochs,
         verbose = 1,
          validation_split = 0.1
         )
loss , accu = model.evaluate([dialog_test, ques_test], ans_test, 
                             verbose= 1,
                             batch_size = batch_size)
print('%s: %.4f \n %s: %.4f' % ('loss', loss, 'accu', accu))

```

* 预测

```python 
pre = model.predict([dialog_test, ques_test],
              batch_size = batch_size,
              verbose = 1)
#输出测试过程
def get_key(dic,value):
    return [k for k,v in wd_idx.items() if v == value]
import numpy as np
a = pre[0].tolist()
a.index(max(a))
for i in range(len(dialog_test)):
    ques = []
    lis_dia = list(map(lambda x : get_key(wd_idx,x), dialog_test[i]))
    dialog = reduce(lambda x,y :x+' '+y ,(reduce(lambda x, y: x+y,lis_dia)))
    
    lis_ques = (map(lambda x : get_key(wd_idx,x), ques_test[i]))
    ques = reduce(lambda x,y :x+' '+y,(reduce(lambda x, y: x+y,lis_ques)))
    
    ans_idx = np.argmax(ans_test[i])
    pre_idx = np.argmax(pre[i])
    print('%s\n %s ' % ('dialog',dialog))
    print('%s\n %s ' % ('question',ques))
    print('%s\n %s ' % ('right_answer',get_key( wd_idx, ans_idx)))  
    print('%s\n %s\n' % ('pred',get_key(wd_idx, pre_idx)))
```

预测输出示例：

![RNN_19](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN/RNN_19.png)


总的来说就是，对话集数据进行embedding后的输出（矩阵），与问题集进行LSTM及RepeatVector后的输出（矩阵），两个矩阵（shape相同）相对应的拼接成一个大的矩阵，称为mat\_merge；mat\_merge进入LSTM训练，输出最后一个hidden\_state的输出，称为h\_out（一个列向量）。然后h\_out进入全连接层，此时h_out与词表中的每个词计算softmax的概率值，概率最大的即为预测的答案。


## 10.3.4  LSTM实现简单的文本生成

### 10.3.4.1文本生成原理

本章节的文本生成（text  generation）示例基于char RNN，也就是说循环神经网络的输入为一个字一个字的序列，输出为与输入等长的字序列，属于上述中的等长的多对多的结构。

基于字符集的文本生成原理可以这样简单理解：  
（1）将一个长文本序列依次输入到循环神经网络  
（2）对于给定前缀序列的序列数据，对序列中将要出现的下一个字符的概率分布建立模型  
（3）这样就可以每次产生一个新的字符
如下图所示，我们想要从给定序列‘hell’中生成‘hello’。每次输入到循环神经网络中一个字符，并计算其概率分布。所以每个字符的出现概率分布都是基于前面历史序列得到的，如第二个‘l’的概率是通过历史信息‘hel’得出。在输出层可以通过最大似然或者条件随机场等规则选择结果。
再经过不断的迭代和优化训练出文本生成的模型。

![RNN_16](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN/RNN_26.png)



### 10.3.4.2 基于keras实现简单英文文本生成

本示例引用keras的官方示例，原始地址可在keras的GitHub上找到。在此，基于keras的代码，将结合循环神经网络进行进一步详细的注解，有助于刚入门神经网络或者自然语言处理的读者更好的理解神经网络和文本生成，完整代码将会在最后附上。

（1）读取数据。本示例采用尼采的文章数据集，读取文件，并将所有英文字母转为小写字母。

```python
with io.open('nietzsche.txt', encoding='utf-8') as f:
    #read character by character
    text = f.read().lower()
print('corpus length:', len(text))

```

(2)建立字典。得到每个字符对应的索引号，建立{字符：索引号}以及{索引号：字符}两个字典。

```python
chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
```
(3) 切分文本。被切分吼的每个文本段最长为maxlen，并记录每个文本段最大长度后的字符。因为此模型的功能实际上就是依据输入的前maxlen个字符，预测下一个字符。

```python
# cut the text in semi-redundant sequences of maxlen characters
#在maxlen长的半冗余字符中分割文本
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

```
(4)生成输入数据，并对输入数据进行向量化。

```python
print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1
```

（5）建立神经网络。此处采用最为简单的一层LSTM加一层全连接网络，最后加一个softmax的激活函数用于预测计算各个输出概率。采用交叉熵计算损失函数。

```python
# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

```

(6)概率抽样。不是完全选择概率最大的输出，而是根据生成的概率抽样选择最后的预测输出数据。temperature参数可以调节抽样的权重，当temperature较小时，选择概率较大的值输出，反之选择概率较小的。

```python
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

```
(7)每个轮次的输出生成的文本，在训练的过程中可以更直观的看到模型随着训练次数的递增，生成的文本更符合语法规则，模型更加智能化。

```python
def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

```
(8)训练模型。

```python
model.fit(x, y,
          batch_size=128,
          epochs=20,
          callbacks=[print_callback])
```

训练完60个轮次的结果如下：

![RNN_29](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN/RNN_29.png)



### 10.3.4.3 基于keras实现简单中文文本生成

在本示例中，将会通过3000多首的唐诗，利用LSTM网络训练一个古诗生成的模型。

代码引用自[https://github.com/youyuge34/Poems_generator_Keras](https://github.com/youyuge34/Poems_generator_Keras)

**代码解读：**

网络逻辑图如下：

![RNN_24](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN/RNN_24.png)


1、数据处理
自然语言处理中，数据预处理的很多过程是共通的，都是要将文本数据进行编码，变为机器可以识别的词向量或者字向量，比如Word2Vec，one-hot编码等方法。
本次数据预处理步骤如下：
（1）读取古诗数据集，在每首诗的末尾加‘]’，并去除古诗名字只取古诗内容，最后得到一个长文本

```python
#代码引用自[https://github.com/youyuge34/Poems_generator_Keras]
with open('poetry.txt', 'r') as f:
    files_content = ''
    lines = [x  for x in f]
    for line in lines[:10]:
        files_content += (line.strip() + "]").split(":")[-1]
```
（2）统计每个字出现的频率，并去除出现频率较低的字。一般来说如果数据集较小可以不去除。

```python
 # 统计每个字符出现的概率
    words = sorted(list(files_content))
    counted_words = {}
    for word in words:
        if word in counted_words:
            counted_words[word] += 1
        else:
            counted_words[word] = 1
    # 去除出现概率较小的字符。此处设置为2
    word_delet = []
    for key in counted_words:
        if counted_words[key] <= 2:
            word_delet.append(key)
    for key in word_delet:
        del counted_words[key]
```
(3) 按照出现频率排序，保留字符集词典，并在最后添加空格字符。

```python
wordPairs = sorted(counted_words.items(), key=lambda x: -x[1])
    words, _ = zip(*wordPairs)
    print(words)
    words += (" ",)
```
（4） 按照（3）中得到的词典，得到每个字对应的索引号，建立{字符：索引号}以及{索引号：字符}两个字典。因为机器无法识别汉字，只能转换为数字输入，这也是自然语处理中的常规操作。此处还定义了由字符转换为索引号的函数，以便之后使用。

```python
    word2num = dict((c, i) for i, c in enumerate(words))
    num2word = dict((i, c) for i, c in enumerate(words))
    word2numF = lambda x: word2num.get(x, len(words) - 1)
    return word2numF, num2word, words, files_content
```

2、定义配置数据

这里所谓的配置数据指的是每次处理的样本数据量batch_size，模型训练的学习率learning_rate，每次取字符的最大长度maxlen，模型参数存储的文件路径，此处参数主要是权重，model_path。

```python
    model_path = 'poetry_model.h5'
    max_len = 6
    batch_size = 512
    learning_rate = 0.001
```

3、生成训练数据。
（1）训练数据分为输入数据和输出数据，此处生成的是五言古诗。输入任意的6个字符（包含标点），预测输出一首古诗。。

```python
#  max_len=6
x = self.files_content[i: i + self.config.max_len]
y = self.files_content[i + self.config.max_len]
```
（2）将输入输出数据转为向量。

```python
y_vec = np.zeros(
                shape=(1, len(self.words)),
                dtype=np.bool
            )
x_vec = np.zeros(
    shape=(1, self.config.max_len),
    dtype=np.int32
)
y_vec[0, self.word2numF(y)] = 1.0
for t, char in enumerate(x):
    x_vec[0, t] = self.word2numF(char)
    
yield x_vec, y_vec
```

4、建立模型

模型很简单，就是两个lstm的叠加加上一个全连接层。因为keras已经将模型封装的很好了，使用起来很简单。

```python
input_tensor = Input(shape=(self.config.max_len,))
embedd = Embedding(len(self.num2word), 300, input_length=self.config.max_len)(input_tensor)
lstm = (LSTM(512, return_sequences=True))(embedd)
dropout = Dropout(0.6)(lstm)
lstm = LSTM(256)(dropout)
dropout = Dropout(0.6)(lstm)
dense = Dense(len(self.words), activation='softmax')(dropout)
self.model = Model(inputs=input_tensor, outputs=dense)

optimizer = Adam(lr=self.config.learning_rate)
self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
```

5、训练模型

计算epoch，防止输入输出的下标超出文本长度。

```python
number_of_epoch = len(self.files_content) // self.config.batch_size
                      
if not self.model:
    self.build_model()
self.model.summary()
self.model.fit_generator(
    generator=self.data_generator(),
    verbose= 1,
    steps_per_epoch=self.config.batch_size,
    epochs=number_of_epoch)
```

6、输入文本，输出预测的古诗。

（1）概率抽样。不是完全选择概率最大的输出，而是根据生成的概率抽样选择最后的预测输出数据。temperature参数可以调节抽样的权重，当temperature较小时，选择概率较大的值输出，反之选择概率较小的。

```python
    def sample(self, preds, temperature=1.0):
        '''
        当temperature=1.0时，模型输出正常
        当temperature小于1时时，模型输出比较open
        当temperature大于1时，模型输出比较保守
        在训练的过程中可以看到temperature不同，结果也不同
        '''
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

```
（2）预测输出

输入的文本长度如果小于4个字，则自动补全。

```python
with open(self.config.poetry_file, 'r', encoding='utf-8') as f:
            file_list = f.readlines()
random_line = random.choice(file_list)
# 如果给的text不到四个字，则随机补全
if not text or len(text) != 4:
    for _ in range(4 - len(text)):
        random_str_index = random.randrange(0, len(self.words))
        text += self.num2word.get(random_str_index) \
                if self.num2word.get(random_str_index) not in [',', '。','，'] \
                else self.num2word.get(random_str_index + 1)
seed = random_line[-(self.config.max_len):-1]

```
利用sample函数产生概率抽样值，然后选择模型的输出概率，将向量转换为汉字输出。

```python
# text 为输入文本
for c in text:
	# seed 为自定补全的输入文本，长度不小于4个字。
    seed = seed[1:] + c
    for j in range(5):
        x_pred = np.zeros((1, self.config.max_len))
        for t, char in enumerate(seed):
            x_pred[0, t] = self.word2numF(char)

        preds = self.model.predict(x_pred, verbose=0)[0]
        next_index = self.sample(preds, 1.0)
        next_char = self.num2word[next_index]
        seed = seed[1:] + next_char
    res += seed
```
![RNN_27](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN/RNN_27.png)



将模型的参数存储起来，之后可以直接调用模型，进行古诗的生成。示例如下：

![RNN_28](https://github.com/xuman-Amy/ML_project_images/blob/master/RNN/RNN_28.png)


目前此模型可以生成五言古诗，并正确的匹配标点符号，但是语意方面还很不足，读者可以自己尝试增加训练集或者修改模型参数等方式自己尝试更优的方案。完整的示例代码可在github上下载。


#10.4.参考文献
[1] 《Deep learning》 ····to do.   
[2] 《知乎——超智能体》···to do.   
[3]《LONG SHORT-TERM MEMORY》.   

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

7. keras库简单讲解和使用.     
[keras documentation](http://keras-cn.readthedocs.io/en/latest/)
 
                         
                                                                   
                                                                     
                                                                     
