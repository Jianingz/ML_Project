
## 1.1 Python基础

Python是一种广泛使用的高级编程语言，属于通用解释型编程语言，由吉多·范罗苏姆（Guido van Rossum）创造，第一版发布于1991年。可以视之为一种改良（加入一些其他编程语言的优点，如面向对象）的LISP。作为一种解释型语言，Python的设计哲学强调代码的可读性和简洁的语法（尤其是使用空格缩进划分代码块，而非使用大括号或者关键词）。相比于C++或Java，Python让开发者能够用更少的代码表达想法。不管是小型还是大型程序，该语言都试图让程序的结构清晰明了。与Scheme、Ruby、Perl、Tcl等动态类型编程语言一样，Python拥有动态类型系统和垃圾回收功能，能够自动管理内存使用，并且支持多种编程范式，包括面向对象、命令式、函数式和过程式编程。其本身拥有一个巨大而广泛的标准库。

Python 解释器本身几乎可以在所有的操作系统中运行。Python的正式解释器CPython是用C语言编写的、是一个由社群驱动的自由软件，目前由Python软件基金会管理。Python具有动态类型系统和自动内存管理功能。它支持多种编程范例，包括面向对象，命令式，功能性和程序性，并且具有庞大而全面的标准库。

2000年10月16日发布Python 2.0，增加了实现完整的垃圾回收，并且支持Unicode。同时，整个开发过程更加透明，社群对开发进度的影响逐渐扩大。2008年12月3日发布Python 3.0，此版不完全兼容之前的Python原始码。不过，很多新特性后来也被移植到旧的Python 2.6/2.7版本。

Python 是一种解释型语言，所以在开发过程中没有了编译这个环节，它类似于PHP和Perl语言。Python 是一种面向对象语言，所以Python支持面向对象的风格或代码封装在对象的编程技术。Python本身被设计为可扩充的。并非所有的特性和功能都集成到语言核心。Python提供了丰富的API和工具，以便程式设计师能够轻松地使用C、C++、Cython来编写扩充模组。Python编译器本身也可以被集成到其它需要脚本语言的程式内。因此，它对编程的初学者是十分有好的。

Python 的用途十分广泛，它经常被用于Web开发，GUI开发，而且也被用于操作系统的开发，在很多操作系统当中（大多数Linux发行版和Mac OS X），Python是标准的系统组件，我们可以在终端机下直接运行Python。当然，在我们所处的数据爆炸增长的信息时代，Python更是被广泛地用于数据科学领域，尤其是机器学习和深度学习当中。在本书中，我们将会使用到Python以及当中众多科学计算的第三方库。Python是数据科学生态系统中使用的较年轻的编程语言之一（与R相比），但它与R一样经常用于分析处理数据。每个数据科学家和机器学习爱好者都应该有良好的python，R和统计分析软件的基础。


### 1.1.1 Python 基础语法

首先让我们实现在标准输出设备上输出Hello World的简单程序，此次作为开始学习Python编程语言的第一个程序。


```python
print("Hello, world!") #适用于Python 3.0以上版本以及Python 2.6、Python 2.7

print "Hello, world!" #适用于Python 2.6以下版本以及Python 2.6、Python 2.7
```

#### 变量类型

好的，让我们首先深入研究python编程中的一些语法。


```python
int_val = 8
long_val = 21312153213 #在python 3中，long是integer
float_val = 2.16532483154
bool_val = False

print ("变量类型举例：")
print (type(int_val))
print (type(long_val))
print (type(float_val))
print (type(bool_val))

print ("测试变量的类型：")
temp = isinstance(float_val,float)
print (temp)
print (isinstance(float_val,int))
```

    变量类型举例：
    <class 'int'>
    <class 'int'>
    <class 'float'>
    <class 'bool'>
    测试变量的类型：
    True
    False


#### 计算与逻辑值判断

然后我们看看Python的计算与逻辑值判断是什么样子的。


```python
print ("Python计算举例：")
print (8 / 3)
print (float(8) / 3)
print (float(8) / float(3))

print (True and False)  
print (8 == 3) 
print (5 <= 6)  

print (2.0*5.0)
print (37%4)
print (3**3)
```

    Python计算举例：
    2.6666666666666665
    2.6666666666666665
    2.6666666666666665
    False
    False
    True
    10.0
    1
    27


#### 字符串处理

再举例看看字符串的处理。Python 可以使用引号( ' )、双引号( " )、三引号( ''' 或 """ ) 来表示字符串，引号的开始与结束必须的相同类型的。其中三引号可以由多行组成，用来编写多行文本的快捷语法。


```python
print ("字符串举例：\n")
str_val = "字符串使用双引号或单引号。"
str_val_long = '''三引号意味着字符串
多行。'''
str_val_no_newline = '''这样也能跨越多行\
但没有新行。'''

print (str_val)
print (str_val_long)
print (str_val_no_newline)
```

    字符串举例：
    
    字符串使用双引号或单引号。
    三引号意味着字符串
    多行。
    这样也能跨越多行但没有新行。


我们可以通过多种不同的方式来访问字符串。


```python
print (str_val[0]  ) # 第“0”个字符
print (str_val[3:5]) # 第“3”“4”个字符，但是没有“5”
print (str_val[-1] ) # 最后一个字符
print (str_val[-5:]) # 最后五个字符
print (str_val[0:5] + str_val[5:])

str_val[4] = '采'
```

    字
    使用
    。
    或单引号。
    字符串使用双引号或单引号。



    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-44-262047ed1584> in <module>()
          5 print (str_val[0:5] + str_val[5:])
          6 
    ----> 7 str_val[4] = '采' # 这是一个典型的错误
    

    TypeError: 'str' object does not support item assignment


上述的错误是一个典型的错误，字符串一旦设置就是不可变的。


```python
str_val = "Hello, world!"
print (str_val*2) # 重复输出字符串
print ('Python' > 'Java') # 按字母顺序比较字符串 
print (str_val.lower())
print (str_val.upper())
```

    Hello, world!Hello, world!
    True
    hello, world!
    HELLO, WORLD!


#### List列表

列表（List）是python中最强大的工具之一，它创建并实现了大多数抽象类型。list非常像元组的可变版本，它可以保存任何类型的信息。


```python
a_list = [12, 34, "two numbers"]

# 我们可以通过append函数添加到列表中
a_list.append("A string, 作为列表中的新元素附加")

print (a_list)

# 列表中可以包含其他列表
tmp_list = ["a list", "另一个列表", 442]
a_list.append(tmp_list)

print (a_list)

# 我们之前学到的所有索引仍然适用于列表
print (a_list[-1])
print (a_list[-2:])
```

    [12, 34, 'two numbers', 'A string, 作为列表中的新元素附加']
    [12, 34, 'two numbers', 'A string, 作为列表中的新元素附加', ['a list', '另一个列表', 442]]
    ['a list', '另一个列表', 442]
    ['A string, 作为列表中的新元素附加', ['a list', '另一个列表', 442]]


#### Tuple元组

元组（Tuple）是不可变列表，用逗号表示。你可以将任何内容存储在一个元组中，它基本上是一个复杂的对象容器。


```python
a_tuple = 12, 34, "two numbers"
print (a_tuple)

# 可以使用方括号访问元组
print (a_tuple[2])

#不能改变一个元组，它是不可变的
a_tuple[2] = 'three numbers' # 
```

    (12, 34, 'two numbers')
    two numbers



    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-55-74a8ae3830b7> in <module>()
          6 
          7 #不能改变一个元组，它是不可变的
    ----> 8 a_tuple[2] = 'three numbers' #
    

    TypeError: 'tuple' object does not support item assignment


#### Set集合

下面演示集合（Set）的相关操作。


```python
basket = ['apple', 'orange', 'apple', 'pear', 'orange', 'banana']
fruit = set(basket) # 创建一个没有重复的集合
print (fruit)
print ('orange' in fruit )
print ('crabgrass' in fruit)

# 演示来自两个单词的唯一字母的集合操作
a = set('abracadabra')
b = set('alacazam')

# 集合操作
print (a     )   # a中独有的
print (a - b )  # 在a中不在b中
print (a | b )  # 在a或b中
print (a & b )  # 在a和b中都有
print (a ^ b )  # 在a或b中但不能都有
```

    {'banana', 'apple', 'pear', 'orange'}
    True
    False
    {'a', 'b', 'd', 'c', 'r'}
    {'b', 'd', 'r'}
    {'a', 'b', 'd', 'c', 'l', 'm', 'z', 'r'}
    {'a', 'c'}
    {'b', 'l', 'm', 'd', 'z', 'r'}


#### Dictionary字典

最后演示字典（Dictionary）的相关操作。


```python
num_legs = { 'dog': 4, 'cat': 4, 'human': 2 }

print (num_legs)
print (num_legs['dog'])
print (num_legs['human'])

print ('===============')
# 可以添加，更新或删除条目。
num_legs['human'] = 'two'
num_legs['bird'] = 'two and wings'
num_legs[8] = '一个不是字符串的key'

del num_legs['cat']
print (num_legs)
```

    {'dog': 4, 'cat': 4, 'human': 2}
    4
    2
    ===============
    {'dog': 4, 'human': 'two', 'bird': 'two and wings', 8: '一个不是字符串的key'}


Python语言的一些特点需要额外注意。Python 是用缩进来写代码模块的，并不使用大括号 {} 来控制类，函数以及其他逻辑判断。缩进的空白数量是可变的，但是所有代码块语句必须包含相同的缩进空白数量。在 Python 里，标识符由字母、数字、下划线组成，所有标识符可以包括英文、数字以及下划线，但不能以数字开头，而且Python 中的标识符是区分大小写的。

#### Python之禅


```python
import this
```

    The Zen of Python, by Tim Peters
    
    Beautiful is better than ugly.
    Explicit is better than implicit.
    Simple is better than complex.
    Complex is better than complicated.
    Flat is better than nested.
    Sparse is better than dense.
    Readability counts.
    Special cases aren't special enough to break the rules.
    Although practicality beats purity.
    Errors should never pass silently.
    Unless explicitly silenced.
    In the face of ambiguity, refuse the temptation to guess.
    There should be one-- and preferably only one --obvious way to do it.
    Although that way may not be obvious at first unless you're Dutch.
    Now is better than never.
    Although never is often better than *right* now.
    If the implementation is hard to explain, it's a bad idea.
    If the implementation is easy to explain, it may be a good idea.
    Namespaces are one honking great idea -- let's do more of those!


Python创始人Tim Peters的Python之禅

优美胜过丑陋。 # Python 以编写优美的代码为目标

明了优于隐晦。 # 优美的代码应当是明了的，命名规范，风格相似

简单优于复杂。 # 优美的代码应当是简洁的，不要有复杂的内部实现

复杂优于凌乱。 # 如果复杂不可避免，那代码间要减少难懂的关系，要保持接口简洁

扁平胜过嵌套。 # 优美的代码应当是扁平的，不能有太多的嵌套

稀疏优于密集。 # 优美的代码有适当的间隔，不要奢望一行代码解决问题

可读性很重要。 # 优美的代码是可读的

特殊情况也不足以打破规则。

虽然实际的复杂会打破纯粹。

不能包容所有错误。

除非确定需要这样做。

面对模棱两可，不要尝试去猜测。

而是尽量找一种，最好是唯一一种明显的解决方案。

虽然这并不容易，因为你不是 Python 之父。 # 这里的 Dutch 是指 Guido，1991年2月20日，Guido van Rossum发布了Python的0.9.0版。

现在总比没做好。

虽然不做比急于求成好

如果无法向人解释你的方案，那肯定不是一个好注意。

如果很容易向人解释你的方案，那可能是个好主意。

命名空间是一个很棒的主意，我们应当多加使用！


### 1.1.2 数据科学库基础操作

Python已经打包了许多已经打包的库。你只需使用“import”命令导入它们。与此同时，你也可以安装许多第三方库，其中有许多用于数据分析。在接下来的示例中，我们将使用一些内置库以及第三方库。

如果要安装第三方库，则需要安装软件包管理器。有三个主要的途径可以进行安装：

+ Pip, https://pypi.python.org/pypi/pip
+ Homebrew, http://brew.sh
+ Macports, https://www.macports.org

如果你使用的是Windows，那么pip就是很好的选择。如果你不希望安装程序影响系统上安装的其他库，请考虑设置虚拟环境。如果你使用Mac OS X系统，你需要进行探索并确定哪个选择适合你的系统。

当然你可以可下载Anaconda等软件来管理你的安装环境和各种工具包，我们将在后边详细讲解Anaconda的使用。

现在去我们的系统安装Numpy库吧，它的命令很简单。


```python
> pip install numpy
```

#### Numpy库

接下来我们看看Numpy库的简单操作。Numpy库非常适合处理作为多维数组形式的数据，它具有许多与Matlab和Octave相似的功能。在开始之前，我们可以在编程开始时查看Numpy库的版本号，这样可以方便以后再次审阅或执行时了解程序的运行环境。


```python
import numpy as np

print(np.__version__)
```

    1.12.1


我们可以输出一个随机数组成的Numpy矩阵。


```python
x = np.random.rand(5,3)
x
```




    array([[ 0.53910339,  0.97197101,  0.91122752],
           [ 0.08161089,  0.5157216 ,  0.37010068],
           [ 0.43032107,  0.13771763,  0.04753396],
           [ 0.99455492,  0.93211016,  0.91739956],
           [ 0.77690763,  0.02111863,  0.48221205]])



观察矩阵的大小和数据格式。


```python
x.shape
```




    (5, 3)




```python
x.dtype
```




    dtype('float64')



接下来我们进行一些简单的矩阵运算。


```python
y = np.random.rand(3,4)
y
```




    array([[ 0.46807743,  0.47334502,  0.31866803,  0.14190933],
           [ 0.31862305,  0.75068099,  0.30388072,  0.13929306],
           [ 0.04729156,  0.77020571,  0.35080705,  0.48212931]])




```python
z = np.dot(x,y)
z
```




    array([[ 0.60512787,  1.6866547 ,  0.78682331,  0.65122212],
           [ 0.22002364,  0.71082616,  0.31255856,  0.26185417],
           [ 0.24755155,  0.34368327,  0.19565455,  0.1031672 ],
           [ 0.80590575,  1.87707137,  0.9220134 ,  0.71327832],
           [ 0.39318637,  0.75500118,  0.42315656,  0.34568068]])




```python
z = x @ y
z
```




    array([[ 0.60512787,  1.6866547 ,  0.78682331,  0.65122212],
           [ 0.22002364,  0.71082616,  0.31255856,  0.26185417],
           [ 0.24755155,  0.34368327,  0.19565455,  0.1031672 ],
           [ 0.80590575,  1.87707137,  0.9220134 ,  0.71327832],
           [ 0.39318637,  0.75500118,  0.42315656,  0.34568068]])



我们继续做一些简单的索引操作。


```python
x1 = np.array([[1,2,3],
               [4,5,6],
               [7,8,9]])
x1
```




    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])




```python
for row in range(x1.shape[0]):
    print(x1[row,1])
```

    2
    5
    8



```python
x1[:,1]
```




    array([2, 5, 8])




```python
x1[:,1]>3
```




    array([False,  True,  True], dtype=bool)




```python
x1[ x1[:,1]>3 ]
```




    array([[4, 5, 6],
           [7, 8, 9]])




```python
x2 = np.array(range(10))
x2
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
x2.shape
```




    (10,)




```python
x2[x2>5]
```




    array([6, 7, 8, 9])



#### Pandas 库

经过上边的介绍我们可以发现Numpy对于矩阵的处理有着很好的操作方法，现在提出一个问题：如果我们有一个数据矩阵，其中每一行都是对某些特征的观察，并且每列中都显示了特征值，该怎么办？


```python
col_names = ['温度','时长','日期']
data = np.array([[34,2100,1],
                 [30,2200,1],
                 [28,2300,1],
                 [14,0,   3],
                 [20,1000, 5]])
data
```




    array([[  34, 2100,    1],
           [  30, 2200,    1],
           [  28, 2300,    1],
           [  14,    0,    3],
           [  20, 1000,    5]])




```python
data2 = data[data[:,1]>1500]
data2
```




    array([[  34, 2100,    1],
           [  30, 2200,    1],
           [  28, 2300,    1]])



我们用Numpy格式来表示此类数据，可以发现不是很直观，这里我们就要用到Pandas库来对表格数据进行处理与显示了，Pandas库是基于NumPy实现的，其中包括Series和DataFrame两种数据结构。

Series 是一种类似于一维数组的对象，由一组数据（各种numpy数据类型）以及一组与之相关的数据标签（即索引）组成。DataFrame是一个表格型的数据结构，含有一组有序的列，每列可以是不同的值类型（数值、字符串、布尔值等）。DataFrame既有行索引也有列索引，其中的数据是以一个或多个二维块存放的，而不是列表、字典或别的一维数据结构。


```python
import pandas as pd

df = pd.DataFrame(data,columns=col_names)
df
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>温度</th>
      <th>时长</th>
      <th>日期</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>34</td>
      <td>2100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30</td>
      <td>2200</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28</td>
      <td>2300</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20</td>
      <td>1000</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[df.时长>1500]
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>温度</th>
      <th>时长</th>
      <th>日期</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>34</td>
      <td>2100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30</td>
      <td>2200</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28</td>
      <td>2300</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



我们可以查看表格数据的描述。


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5 entries, 0 to 4
    Data columns (total 3 columns):
    温度    5 non-null int64
    时长    5 non-null int64
    日期    5 non-null int64
    dtypes: int64(3)
    memory usage: 200.0 bytes


这里我们可以通过赋值的方式修改表格中的数据。


```python
df.日期[df.日期==1] = 'Mon'
df
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>温度</th>
      <th>时长</th>
      <th>日期</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>34</td>
      <td>2100</td>
      <td>Mon</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30</td>
      <td>2200</td>
      <td>Mon</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28</td>
      <td>2300</td>
      <td>Mon</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20</td>
      <td>1000</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.日期.replace(to_replace=range(7),
               value=['Su','Mon','Tues','Wed','Th','Fri','Sat'],
               inplace=True)
df
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>温度</th>
      <th>时长</th>
      <th>日期</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>34</td>
      <td>2100</td>
      <td>Mon</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30</td>
      <td>2200</td>
      <td>Mon</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28</td>
      <td>2300</td>
      <td>Mon</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14</td>
      <td>0</td>
      <td>Wed</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20</td>
      <td>1000</td>
      <td>Fri</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5 entries, 0 to 4
    Data columns (total 3 columns):
    温度    5 non-null int64
    时长    5 non-null int64
    日期    5 non-null object
    dtypes: int64(2), object(1)
    memory usage: 200.0+ bytes


这里我们可以注意列的类型从int64更改为object。接下来我们可以进行其他的一些基本操作。


```python
df.drop("时长",axis=1,inplace=True) #删除列

df
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>温度</th>
      <th>日期</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>34</td>
      <td>Mon</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30</td>
      <td>Mon</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28</td>
      <td>Mon</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14</td>
      <td>Wed</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20</td>
      <td>Fri</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.drop(0,axis=0,inplace=True) #删除行

df
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>温度</th>
      <th>日期</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>30</td>
      <td>Mon</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28</td>
      <td>Mon</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14</td>
      <td>Wed</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20</td>
      <td>Fri</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.T
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>温度</th>
      <td>30</td>
      <td>28</td>
      <td>14</td>
      <td>20</td>
    </tr>
    <tr>
      <th>日期</th>
      <td>Mon</td>
      <td>Mon</td>
      <td>Wed</td>
      <td>Fri</td>
    </tr>
  </tbody>
</table>
</div>



## 1.2 Python环境

Python的强大之处就在于它宽广的应用领域范围，它能够遍及人工智能、科学计算、Web开发、系统运维、大数据及云计算、金融、游戏开发等多个领域。Python具有d的数量庞大的标准库和第三方库是实现其强大功能的保障。通过对库的引用，我们可以实现对不同领域业务的开发。然而，正是由于数量庞大的库，如何管理这些库以及如何及时更新维护这些库就成为既重要又复杂的工作。在这种背景下，一批库管理工具软件应运而生，Anaconda就是其中非常优秀的代表之一。它就是可以便捷获取包且对包能够进行管理。Anaconda包含了conda、Python在内的超过180个科学包及其依赖项，它包含的科学库包括：conda, numpy, scipy, ipython notebook等等。它的官方网站是：https://www.anaconda.com ，与Python相对应，Anaconda的版本分为Anaconda2和Anaconda3，读者可以自行下载日常常用的版本，提供32位和64位下载。

接下来在哪里编写Python程序代码就成了新的问题。这里我们推荐使用Jupyter Notebook（IPython notebook）。Jupyter Notebook是基于网页的用于交互计算的应用程序。它可被应用于全过程计算：开发、文档编写、运行代码和展示结果。目前Jupyter Notebook已经成为Python教学、计算、科研的一个重要工具。它的优点是可以在网页页面中直接编写代码和运行代码，代码的运行结果也会直接在代码块下显示的程序。如在编程过程中需要编写说明文档，可在同一个页面中直接编写，便于作及时的说明和解释。Jupyter Notebook的输入和输出都是以文档的形式体现的，文档保存的后缀名为.ipynb。读者可以通过使用Anaconda启动Jupyter Notebook来实现Python程序代码的编写。
