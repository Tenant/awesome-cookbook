# Wolfram Alpha

[Transfer Portal](https://www.wolframalpha.com/)

## 1. Mathematics

### 1.1 Algebra

```mathematica
abs() // 绝对值
sqrt() // 根号
log() //  自然对数
pi, lambda, sigma, mu, alpha, beta // 希腊字母输入方法
+infinity // 正无穷

plot f(x) // 画图
factor f(x) // 分解因式
expand f(x) // 展开多项式
simplify f(x) // 化简多项式

complete the square f(x) // 配方，可含参数
partial fractions f(x) // 展开为部分方式
limit (1+1/n)^n as n->infinity // 求极限
D[f(x,y,z),{x,n}] // 求f(x)对x的n阶偏导数
integrate f(x,y,z) dx // 求(fx)对x的不定积分
integrate f(x,y,z) dx from a to b // 求f(x)对x的定积分
sum f(n) for n= a to b // 级数求和

critical points f(x) // 求极值可疑点，即导数为0或不存在的点
inflection points f(x) // 求拐点
local maxima f(x,y) as x from a to b, y from a to b // 求极大值点
local minima f(x,y) // 求极小值点
maximize f(x,y) // 求最大值点
minimize f(x,y) // 求最小值点

Table[f(x),{x,a,b,c}] // 求值表，其中a=起始值，b=终了值，c=步长
distance between (x1,y1)(x2,y2) // 求两点间距离
slope of line through (x1,y1)(x2,y2) // 求过两点直线的距离
equation of a line through (x1,y1)(x2,y2) // 求过两点直线的方程
circle through (x1,y1)(x2,y2)(x3,y3)// 求过三点的圆的方程
parabola through (x1,y1)(x2,y2)(x3,y3) // 求过三点的抛物线方程
```

### 1.2 Linear Algebra

#### 1.2.1 Vector

```mathematica
vector<-3,-4> // Compute properties of a vector
{1/4,-1/2,1} cross{1/3,1,-2/3} // Compute a cross product 
```

#### 1.2.2 Matrix

```mathematica
{{6,-7},{0,3}} // Calculate properties of a matrix
{{2, -1}, {1, 3}} . {{1, 2}, {3, 4}} // Multiply matrices
row reduce {{2, 1, 0, -3}, {3, -1, 0, 1}, {1, 4, -2, -5}} // row reduce a matrix
(inverse {{0,3,4},{3,0,1},{4,1,0}}) . {{1},{4},{3}} // Inverse and multiply
```

#### 1.2.3 Linear Independence

```mathematica
Are (2,-1) and (4,2) linearly independent?
linear independence (a,b,c,d),(e,f,g,h),(i,j,k,l)
```

#### 1.2.4 Vector Spaces

```mathematica
row space {{1, 2, -5}, {-1, 0, -1}, {2, 1, -1}} // Compute the row space of a matrix
{{1, 2, -5}, {-1, 0, -1}, {2, 1, -1}} column space // Compute the column space of a matrix
```

### 1.3 Probability

#### 1.3.1 Probability Distributions

```mathematica
Poisson distribution // Compute properties of Poisson distribution
normal distribution, mean=0, sd=2
```

#### 1.3.2 Bernoulli Trials

```mathematica
probability of 8 successes in 14 trials with p=.6
number of trials until 15th success
streak of 12 successes in 40 trials
```



