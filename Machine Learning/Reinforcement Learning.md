# Reinforcement Learning

**Reference**:

- Books
  - [Reinforcement Learning: An Introduction. Richard S. Sutton, Andrew G. Barto. 2018](../../Shelf/Reinforcement-Learning-An-Introduction.pdf) URL: http://incompleteideas.net/book/the-book-2nd.html
  - [Algorithms for Reinforcement Learning. Csaba Szepesvari. 2009.](../../Shelf/Algorithms-for-Reinforcement-Learning.pdf)
  - [AI textbook Chapters 17, 21. Stuart Russell.]
- Online Courses
  - David Silver
  - http://rail.eecs.berkeley.edu/deeprlcourse/

# Algorithms for Reinforcement Learning

==Csaba Szepesvari==

## 1. Overview

## 2. Markov Decision Processes

### 2.1 Preliminaries

**Metric Space**: In mathematics, a metric space is a set for which distances between all members of the set are defined. 

**Lipschitz**: A mapping $T$ between the metric spaces $(M_1,d_1)$, $(M_2,d_2)$ is called Lipschitz with modulus $L\in \mathbb{R}$ if for any $a, b \in M_1$, $d_2(T(a),T(b)) \le Ld_1(a,b)$. If $T$ is Lipschitz with modulus $L \le 1$, it is called a ==non-expansion==. If $L<1$, the mapping is called a ==contraction==.

**Indicator Function of Event $S$**: $\Vert_{\{S\}}=1$ if $S$ holds and $\Vert_{\{S\}}=0$, otherwise.

### 2.2 Markov Decision Processes

**Transition Probability Kernel $P_0$**: $P_0$ assigns to each state-action pair $(x,a) \in \mathfrak{X} \times \mathfrak{K}$ a probability measure over $\mathfrak{X} \times \mathbb{R}$, which we shall denote by $P_0(\cdot |x,a)$.

**Reward Function $\tau$**: $\mathfrak{X} \times \mathfrak{A} \rightarrow \mathbb{R}$. if $(Y_{(x,a)},R_{(x,a)} \sim P_0 (\cdot | x,a)$, then $\tau(x,a)= \mathbb{E}[R_{(x,a)}]$.

An MDP is called ==finite== if both $\mathfrak{X}$ and $\mathfrak{A}$ are finite, and the rewards are bounded by some quantity $R>0$.

==The goal of MDP is to com up a way of choosing the actions so as to maximize the expected total discounted reward==.

**Cumulative Rewards**: $\mathfrak{R} = \sum_{t=0}^T \gamma ^t R_{t+1}$. Thus, if $\gamma <1$ then rewards far in the future worth exponentially less than the reward received at the first stage. An MDP when the return is defined by this formula is called a ==discounted reward== MDP. When $\gamma=1$, the MDP is called ==undiscounted==.

**Episodic**: An MDP with terminal(or absorbing) states is called episodic.

**The immediate reward of episodic MDP**: 

$$\begin{cases}1 & \text{if } X_t<w^* \text{ and } X_{t+1}=w^* \\ 0 & \text{otherwise }\end{cases}$$

If we set the discount factor to one, the total reward along any trajectory will be one or zero depending on whether the trajectory reaches $w^*$.

### 2.3 Value Functions

**Optimal Value**: $V^*(x)$, of state $x \in \mathfrak{X}$ gives the highest achievable expected return when the process is started from state $x$. A behavior that achieves the optimal values in all states is ==optimal==.

**Optimal Value Function**: The function $V^*: \mathfrak{X}\rightarrow \mathbb{R}$ is called the optimal value function.

**Action-Value Function $Q^\pi$**: $Q^\pi: \mathfrak{X} \times \mathfrak{A} \rightarrow \mathbb{R}$, underlying a policy $\pi \in \Pi_{stat}$ in a MDP.
The optimal value- and action-value functions are connected by the following equations:

$V^*(x) =  \sup_{a \in \mathfrak{A}} Q^*(x,a) \text{, for all } x\in \mathfrak{X} \\ Q^*(x,a) =  r(x,a) + \gamma \sum_{y \in \mathfrak{X}} p(x,a,y)V^*(y) \text{,  for all } x\in \mathfrak{X} \text{, and } a \in \mathfrak{A} \\ \sum_{a \in \mathfrak{A}} \pi(a|x)Q^*(x,a) = V^*(x) \text{, for any } \pi \in \Pi_{\text{stat}}$

==The knowledge of $Q^*$ alone is sufficient for finding an optimal policy. Similarly, knowing $V^*$, $r$, and $P$ also suffices to act optimally==.

**Fixed Point (Invariant Point)**: a fixed point of a function is an element of the function’s domain that is mapped to itself by the function.

### 2.4 Dynamic Programming Algorithms for Solving MDPs

> **Banach fixed-point theorem**:
>
> Definition. Let $(X,d)$ be a metric space. Then a map $T: X \rightarrow X$ is called a ==contraction mapping== on $X$ if there exists $q \in [0,1)$ such that $d(T(x),T(y)) \le qd(x,y)$ for all $x, y \in X$.
>
> Theorem. Let $(X,d)$ be a non-empty ==complete metric space== with a contraction mapping $T:X \rightarrow X$. Then $T$ admits a unique fixed-point $x^*$ in $X$. Furthermore, $x^*$ can be found as follows: start with an arbitrary element $x_0$ in $X$ and define a sequence $\{x_n\}$ by $x_n =T(x_{n-1})$, then $x_n \rightarrow x^*$.

After $k$ iterations, ==policy iteration== gives a policy **not worse** than the policy that is greed w.r.t. the value function computed using $k$ iterations of ==value iteration== if **the two procedures are started with the same initial value function**. However the computational cost of a single step in ==policy iteration== is much higher (because of the policy evaluation step) than that of one update in ==value iteration==.

### 2.5 Value Prediction Problems

==Markov Reward Process(MRP)==, ==Monte-Carlo Method==, ==Temporal Difference Learning==

### 2.5.1 Temporal Difference Learning in Finite State Spaces

##### 2.5.1.1 Tabular TD(0)

**Temporal Difference Error $\delta_{t+1}$**: $\delta_{t+1} = R_{t+1} + \gamma \hat V_t(X_{t+1})- \hat V_t (X_t)$.

**Update**: $\hat V_{t+1}(x) = \hat V_t + \alpha_t \delta_{t+1} \Pi_{X_t = x}, \text{ for all } x \in X$

**Step Sizes**: Any step-size sequence of the form $\alpha_t =ct^{-\eta}$ will work as long as $1/2<\eta \le1$. Of these step-size sequences, $\eta=1$ gives the smallest step-sizes. Asymptotically, this choice will be the best, but from the point of view of the transient behavior of the algorithm choosing $\eta$ closer to $1/2$ will converge faster. ==However==, in practice people often use ==constant== step-sizes.

**Off-Policy Learning**: When learning about one policy, while following another is called off-policy learning.

**Continual Sampling with Restarts from $P_0$**: 

> Being a standard linear ==SA(Stochastic Approximation)== method, the rate of convergence of tabular TD(0) will be of the usual order $O(1/\sqrt t)$. However, the constant factor in the rate will be largely influenced by the choice of the step-size sequence, the properties of the kernel $P_0$ and the value of $\gamma$.

##### 2.5.1.2 Every-Visit Monte-Carlo

$R_T=\sum_{s=t}^{T_{k(t)+1}-1} \gamma ^{s-t} R_{s+1} \\ \hat V_{t+1}(x) = \hat V_t (x) + \alpha_t (R_t - \hat V_t(x)) \Pi_{X_t=x} \text{  for all } x \in X$

**TD(0) or Monte-Carlo?**

##### 2.5.1.3 TD($\lambda$): Unifying Monte-Carlo and TD(0)

**Eligibility Traces $z_{t+1}(x)$**: An eligibility trace is a temporary record of the occurrence of an event, such as the visiting of a state or the taking of an action. The trace marks the memory parameters associated with the event as eligible for undergoing learning changes. Wen a TD error occurs, only the eligible states or actions are assigned credit or blame for the error. Thus, eligibility traces help bridge between events and training information. Like TD methods themselves, eligibility traces are a basic mechanism for temporal credit assignment.

**Update Rule**:

$\delta_{t+1} = R_{t+1} + \gamma \hat V_t (X_{t+1}) - \hat V_t (X_t) \\ z_{t+1}(x)=\Pi_{\{x=X_t\}} + \gamma \lambda z_t(x) \\ \hat V_{t+1}(x) =\hat V_t(x) + \alpha_t \delta_{t+1}z_{t+1}(x) \\ z_0(x)=0$

==The value of $\lambda$ can be changed even during the algorithm, without impacting convergence.==

**Summary**: TD($\lambda$) allows one to estimate ==value functions== in ==MRPs==. It generalizes Monte-Carlo methods, it can be used in ==non-episodic== problems, and it allows for ==bootstrapping==. Further, by ==appropriately tuning $\lambda$== it can converge significantly faster than Monte-Carlo methods or TD(0).

#### 2.5.2 Algorithms for Large State Spaces

When the state space is large (or infinite), it is not feasible to keep a separate value for each state in the memory.In such cases, we often seek an estimate of the values in the form $V_\theta(x)=\theta^T \phi(x) \text{  for all  } x \in X$ where $\theta \in \mathbb{R}^d$ is a vector of parameters and $\phi:X \rightarrow \mathbb{R}^d$ is a mapping of states to d-dimensional vectors. For state $x$, the components $\phi_i(x)$ of the vector $\phi(x)$ are called the features of state $x$ and $\phi$ is called a ==feature extraction method==. The individual functions $\phi_i:X \rightarrow X$ defining the components of $\phi$ are called ==basis functions==.

**Tensor Product Construction**: 

> The vector space $V \times W$ has dimension $dim(V)+dim(W)$, the vector space $Z$ has dimension $|V \times W|$.





















# UCL Course on RL

http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html

















































# Deep Reinforcement Learning

[Video](https://www.youtube.com/watch?v=8jQIKgTzQd4&index=2&list=PLkFD6_40KJIwTmSbCv9OVJB3YaO4sFwkX&t=286s)

==Imitation learning==, ==behavior cloning==, ==Q-learning==, ==policy gradients==, ==advanced model learning==,

**Some evidence in favor of deep learning**:

==Unsupervised learning models of primary cortical receptive fields and receptive field plasticity==

**Some evidence for reinforcement learning**:

==Reinforcement learning in the brain==

**What has proven challenging so far?**:

- Human can learn incredibly quickly
  - DRL methods usually slow
- Humans can reuse past knowledge
  - Transfer learning in DRL is an open problem]
- Not clear what the reward function should be
- Not clear what the role of prediction should be

## 1. What is Reinforcement Learning

> Brach of machine learning concerned with taking sequences of actions.
>
> Usually described in terms of agent interacting with a previously unknown environment, trying to maximize cumulative reward.

***In ==imitation learning== it’s been proved there is no ==reward function==.***

| Scenario                                                    | Observations                | Actions                                  | Reward                                                       |
| ----------------------------------------------------------- | --------------------------- | ---------------------------------------- | ------------------------------------------------------------ |
| Motor control and robotics                                  | Camera images, joint angles | Joint torques                            | Stay balanced, navigate to target locations, serve and protect humans |
| Inventory management                                        | Current inventory levels    | Number of units of each item to purchase | Profit                                                       |
| Classification with Hard attention                          | Current image window        | Where to look                            | +1 for correct classification                                |
| Sequential/structured prediction, e.g., machine translation | Words in source language    | Emit word in target language             | Sentence-level accuracy metric, e.g. BLEU score              |

**What is Deep Reinforcement Learning?**

> Reinforcement learning using neural networks to approximate functions
>
> - Policies
> - Value functions (measure goodness of states or state-action pairs)
> - Dynamics Models (predict next states and rewards)

**How does RL relate to other ML problems?**

> Supervised learning:
>
> - Environment samples input-output pair $(x,y)$
> - Agent predict $\hat y_t = f( x_t )$
> - Agent receives loss $\ell ( y_t, \hat y_t )$
> - Environment asks agent  a question, and then tells it the right answer
>
>
>
> Contextual Bandits:
>
> - Environment samples input $x_t$
> - Agent takes action $\hat y_t = f( x_t )$
> - Agent receives cost $c_t \sim P( c_t | x_t, \hat y_t )$ where $P$ is and unknown probability distribution
> - Environment asks agent a question,and gives agent a noisy score on its answer
> - Application: personalized recommendations
>
>
>
> Reinforcement Learning:
>
> - Environment samples input $x_t \sim P( x_t | x_{t-1}, y_{t-1} )$
>   - Environment is stateful: input depends on your previous actions!
> - Agent takes action $\hat y_t = f( x_t )$
> - Agent receives cost $c_t \sim P( c_t | x_t,\hat y_t )$ where $P$ is a probability distribution unknown to the target
>
> **Summary of differences between RL and supervised learning:**
>
> - You don’t have full access to the function you’re trying to optimize – must query it through interaction
> - Interacting with a stateful world: input $x_t$ depend on your precious actions.

 **Beyond Learning from Reward?**

## 2. Supervised Learning of Behaviors: Deep Learning, Dynamical Systems, and Behavior Cloning

### 2.1 Definition of sequential decision problems

![](http://images-repo.oss-cn-beijing.aliyuncs.com/2018-11-02_110734.png)



### 2.2 Imitation learning: supervised learning for decision  making

[End to End Learning for Self-Driving Cars. Mariusz Bojarski, Davide Del Testa, Daniel Dworakowski, etc. 2016](http://papers-repo.oss-cn-beijing.aliyuncs.com/End%20to%20End%20Learning%20for%20Self-Driving%20Cars.pdf). URL: https://arxiv.org/abs/1604.07316

[Stephane Ross, Geoffrey J. Gordon, J. Andrew Bagnell. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. 2011.](http://papers-repo.oss-cn-beijing.aliyuncs.com/A%20Reduction%20of%20Imitation%20Learning%20and%20Structured%20Prediction%20to%20No-Regret%20Online%20Learning.pdf) 

 **DAgger: Dataset Aggregation:**

> goal: collect training data from $p_{\pi_{\theta}}(o_t)$ instead of $p_{data}(o_t)$
>
> how? just run $\pi_{\theta}(u_t | o_t)$
>
> but need labels $u_t$
>
>  	1. train $\pi_\theta(u_t |o_t)$ from human data $D=\{o_1,u_1,...,o_N,u_N\}$
>  	2. run $\pi_\theta(u_t |o_t)$ to get dataset $D_{\pi}=\{o_1,...,o_M\}$
>  	3. Ask human to label $D_\pi$ with actions $u_t$
>  	4. Aggregate: $D=D +D_\pi$



**Pros and Cons of Imitation learning:**

> Imitation learning is often (but not always) insufficient by itself
>
> - Distribution mismatch problem
>
> Sometimes works well
>
> - Hacks (e.g. left/right images)
> - Samples from a stable trajectory distribution
> - Add more **on-policy** data, e.g. using DAgger

### 2.3 Case studies of recent work in (deep) imitation learning

**Case Study 1**

[Alessandro Giusti, Jerome Guzzi, Dan C. Ciresan, etc. A Machine Learning Approach to Visual Perception of Forest Trails for Mobile Robots. 2016.](http://papers-repo.oss-cn-beijing.aliyuncs.com/A%20Machine%20Learning%20Approach%20to%20Visual%20Perception%20of%20Forest%20Trails%20for%20Mobile%20Robots.pdf) URL: https://ieeexplore.ieee.org/document/7358076

> To acquire such a dataset, we equip a hiker with three head-mounted cameras: one pointing  $30^{\circ}$ to left, one pointing straight ahead, and one pointing $30^{\circ}$ to the right.

> The 17,119 training frames are used as training data. The training set is augmented by synthesizing left/right mirrored versions of each training image. In particular, a mirrored training image of class TR(TL) yields a new training sample for class TL(TR); a mirrored GS training sample yields another GS training sample. Additionally, mild affine distortions($\pm10\%$ translation, $\pm15^{\circ}$ rotation, $\pm10\%$ scaling) are applied to training images to further increase the number of samples.

**Case Study2**

[Shreyansh Daftry, J. Andrew Bagnell, and Martial Hebert. Learning Transferable Policies for Monocular Reactive MAV Control. 2016](http://papers-repo.oss-cn-beijing.aliyuncs.com/Learning%20Transferable%20Policies%20for%20Monocular%20Reactive%20MAV%20Control.pdf) URL: https://arxiv.org/abs/1608.00627

> We build upon a recently proposed Deep Adaption Network(DAN) architecture, which generalizes deep convolutional neural network to the domain adaption scenario. The main idea is to enhance domain transferability in the task-specific layers of the deep neural network by explicitly minimizing the domain discrepancy.
>
> In order to achieve this, the hidden representations of all the task-specific layers are embedded to a reproducing kernel Hilbert space where the mean embedding of target domain distributions can be explicitly matched. As mean embedding matching is sensitive to the kernel choice, an optimal multi-kernel selection procedure is devised to further reduce the domain discrepancy. We use a multiple kernel variant of the maximum mean discrepancies(MK-MMD) metric as the measure of domain discrepancy. It is an effective criterion that compares distributions without initially estimating their density functions. 

**Case Study 3**

[Rouhollah Rahmatizadeh, Pooya Abolghasemi, Aman Behal, etc. Learning real manipulation tasks from virtual demonstraions using LSTM.]()

**Other topics in imitation learning**

- Structured prediction
- Interaction & active learning
- Inverse Reinforcement Learning
  - Instead of copying the demonstration, figure out the goal

### 2.4 What is missing from imitation learning?

- Humans need to provide data, which is typically finite
  - Deep learning works best when data is plentiful
- Humans are not good at providing some kinds of actions
- Humans can learn autonomously; can our machines do the same?
  - Unlimited data from own experience
  - Continuous self-improvement

### 2.5 Cost function for Imitation Learning

$$\begin{align}c(x,u)=-\log p(u=\pi^*(x) | x)\end{align}$$

**The trouble with cost & reward functions**

[Andrei A. Rusu, Mel Vecerik, Thomas Rothorl, etc. Sim-to-Real Robot Learning from Pixels with Progressive Nets. 2018.](http://papers-repo.oss-cn-beijing.aliyuncs.com/Sim-to-Real%20Robot%20Learning%20from%20Pixels%20with%20Progressive%20Nets.pdf) URL: https://arxiv.org/abs/1610.04286

> Progressive nets have been show to produce positive transfer between disparate task such as Atari games by utilizing lateral connections to preciously learnt model. The addition of new capacity for each new task allow specialized input features to be learned, an important advantage for deep RL algorithms which are improved by sharply-tuned perceptual features. An advantage of **progressive nets** compared  with other methods for **transfer learning** or **domain adaption** is that multiple tasks may be learned sequentially, without needing to specify source and target task.

> **Progressive networks** are ideal for simulation-to-real transfer of policies in robot control domains, for multiple reasons. First, features learnt for one task may be transferred to many new tasks without destruction from fine tuning. Second, the columns may be heterogeneous, which may be import for solving different tasks, including different input modalities, or simply to improve learning speed when transferring to the real robot. Third, progressive nets add new capacity, including new input connections, when transferring to new tasks. This is advantageous for bridging the reality gap, to accommodate dissimilar inputs between simulation and real sensor.

## 3. Optimal Control, Trajectory Optimization, and Planning

**Syllabus**

> 1. Making decisions under known dynamics
>    - Definitions & problem statement
> 2. Trajectory optimization: backpropagation through dynamical systems
> 3. Linear dynamics: linear-quadratic regulator(LQR)
> 4. Nonlinear dynamics: differential dynamic programming(DDP) & iterative LQR
> 5. Discrete systems: Monte-Carlo tree search(MCTS)
> 6. Case study: imitation learning from MCTS

**Goals**

> - Understand the terminology and formalisms of optimal control
> - Understand some standard optimal control & planning algorithms

$$\begin{align}\min_{u_1,...,u_T} \sum_{i=1}^Tc(x_t,u_t) \end{align}$$ s.t. $$\begin{align}x_t=f(x_{t-1},u_{t-1})\end{align}$$

usually story: differentiate via backpropagation and optimize!

need $\frac{\partial f}{\partial x_t}$, $\frac{\partial f}{\partial u_t}$, $\frac{\partial c}{\partial u_t}$, $\frac{\partial c}{\partial u_t}$.

in practice, it really helps to use a $2^{nd}$ order method!

**Shooting methods**: optimize over actions only

$$\begin{align}\min_{u_1,...,u_T} c(x_1,u_1)+c(f(x_1,u_1),u_2)+\cdot \cdot \cdot+c(f(f(\cdot \cdot \cdot) \cdot \cdot \cdot),u_T)\end{align}$$

**Collocation**: optimize over actions and states, with constraints

$$\begin{align}\min_{u_1,...,u_T,x_1,...,x_T} \sum_{t=1}^Tc(x_t,u_t)\end{align}$$ s.t. $$\begin{align}x_t=f(x_{t-1},u_{t-1})\end{align}$$

### 3.3 Linear Dynamics: linear-quadratic regulator(LQR)

$$\begin{align}\min_{u_1,...,u_T} c(x_1,u_1)+c(f(x_1,u_1),u_2)+\cdot \cdot \cdot+c(f(f(\cdot \cdot \cdot) \cdot \cdot \cdot),u_T)\end{align}$$

$f(x_t,u_t)=F_t$ $$\left[\begin{matrix}x_t \\ u_t\end{matrix}\right]$$ $+f_t$

The above equation may be time-variant, but it is linear.

$c(x_t,u_t)=\frac{1}{2}$ $$\left[\begin{matrix}x_t^T & u_t^T\end{matrix}\right]$$ $C_t$ $$\left[\begin{matrix}x_t \\ u_t\end{matrix}\right]$$ $+$ $$\left[\begin{matrix}x_t^T & u_t^T\end{matrix}\right]$$ $c_t$

Base case: solve for $u_T$ only

$Q(x_T,u_T)= const+\frac{1}{2}$ $$\left[\begin{matrix}x_T^T & u_T^T\end{matrix}\right]$$ $c_T$ $$\left[\begin{matrix}x_T \\ u_T\end{matrix}\right]$$$+$ $$\left[\begin{matrix}x_T^T & u_T^T\end{matrix}\right]$$ $c_T$

$$\begin{align}\nabla_{u_T} Q(x_T,u_T)=C_{u_T,x_T}x_T+C_{u_T,u_T}u_T+c_{u_T}^T=0\end{align}$$

Set $C_T=$ $$\left[\begin{matrix}C_{x_T,x_T } & C_{x_T,u_T} \\ C_{u_T,x_T} & C_{u_T.u_T}\end{matrix}\right]$$, $c_T=$ $$\left[\begin{align}c_{x_T} \\ c_{u_T}\end{align}\right]$$.

$$\begin{align}u_T=-C^{-1}_{u_T,u_T}(C_{u_T,x_T}x_T+c_{u_T})\end{align}$$

Set $$\begin{align}K_T=-C^{-1}_{u_T,u_T}C_{u_T,x_T}\end{align}$$, $$\begin{align}k_T=-C^{-1}_{u_T,u_T}c_{u_T}\end{align}$$

$u_T=K_Tx_T+k_T$

Since $u_T$ is fully determined by $x_T$, we can eliminate it via substitution!

$$\begin{align}Q_{T-1}=C_{T-1}+F_{T-1}^TV_TF_{T-1}\end{align}$$

$$q_{T-1}=c_{T-1}+F_{T-1}^TV_Tf_{T-1}+F_{T-1}^Tv_T$$

Set $K_{T-1}=-Q^{-1}_{u_{T-1},u_{T-1}}Q_{u_{T-1},x_{T-1}}$, $k_{T-1}=-Q_{u_{T-1},u_{T-1}^{-1}}q_{u_{T-1}}$

$$\begin{align}u_{T-1}=K_{T-1}x_{T-1}+k_{T-1}\end{align}$$

**Backward recursion**:

> for $t=T$ to 1:
>
> ​	$Q_t=C_t+F_t^TV_{t+1}F_t$
>
> ​	$q_t=c_t+F_t^TV_{t+1}f_t+F_t^Tv_{t+1}$
>
> ​	$Q(x_t,u_t)=const+\frac{1}{2}$ $$\left[\begin{matrix}x_t^T & u_t^T\end{matrix}\right]$$ $Q_t$ $$\left[\begin{matrix}x_t \\ u_t\end{matrix}\right]$$ $+$ $$\left[\begin{matrix}x_t^T & u_t^T\end{matrix}\right]$$ $q_t$
>
> ​	$u_t\leftarrow\arg \min_{u_t} Q(x_t,u_t)=K_tx_t+k_t$
>
> ​	$K_t=-Q^{-1}_{u_t,u_t}Q_{u_t,x_t}$
>
> ​	$k_t=-Q^{-1}_{u_t,u_t}q_{u_t}$
>
> ​	$V_t=Q_{x_t,x_t}+Q_{x_t,u_t}K_t+K_t^TQ_{u_t,x_t}+K_t^TQ_{u_t,u_t}K_t$
>
> ​	$v_t=q_{x_t}+Q_{x_t,u_t}k_t+K_t^TQ_{u_t}+K_t^TQ_{u_t,u_t}k_t$
>
> ​	$V(x_t)=const+\frac{1}{2}x_t^TV_tx_t+x_t^Tv_t$

**Forward recursion**:

> for t=1 to T:
>
> ​	$u_t=K_tx_t+k_t$
>
> ​	$x_{t+1}=f(x_t,u_t)$

The $Q(x_t,u_t)$ means the ==total cost from now until end if we take $u_t$ from state $x_t$==, $V(x_t)$ means the ==total cost from now until end from state $x_t$==, namely, $V(x_t)=\min_{u_t} Q(x_t,u_t)$.

$f(x_t,u_t)=F_t$ $$\left[\begin{matrix}x_t^T & u_t^T\end{matrix}\right]$$ $+f_t$


$x_{t+1} \sim p(x_{t+1}|x_t,u_t)$

$p(x_{t+1}|x_t,u_t)=N(F_t$ $$\left[\begin{matrix}x_t \\ u_t\end{matrix}\right]$$ $+f_t,\sum_t)$

Solution: choose actions according to $u_t=K_tx_t+k_t$

> No change to algorithm! We can ignore $\sum_t$ due to symmetry of Gaussians 

### 3.4 Nonlinear dynamics: differential dynamic programming(DDP) & iterative LQR

==Using Taylor expansion==

**Iterative LQR**:

> until convergence:
>
> ​	$F_t=\nabla_{x_t,u_t}f(\hat x_t, \hat u_t)$
>
> ​	$c_t=\nabla_{x_t,u_t}c(\hat x_t, \hat u_t)$
>
> ​	$C_t=\nabla_{x_t,u_t}^2c(\hat x_t, \hat u_t)$
>
> ​	Run LQR backward pass on state $\delta x_t=x_t -\hat x_t$ and action $\delta u_t=u_t-\hat u_t$
>
> ​	Run forward pass with real nonlinear dynamics and $u_t=K_t(x_t-\hat x_t)+k_t$
>
> ​	Update $\hat x_t$ and $\hat u_t$ based on states and actions in forward pass

**Why does this work?**

Compare to Newton’s method for computing $\min_x g(x)$:

> Until convergence:
>
> ​	$g=\nabla_x g(\hat x)$
>
> ​	$H=\nabla_x^2 g(\hat x)$
>
> ​	$\hat x \leftarrow \arg \min_x \frac{1}{2}(x-\hat x)^TH(x-\hat x)+g^T(x-\hat x)$

==Iterative LQR(iLQR)== is the same idea: locally approximate a complex nonlinear function via ==Tayor expansion==.

In fact, iLQR is an approximation of Newton’s method for solving:

$$\begin{align}\min_{u_1,...,u_T} c(x_1,u_1)+c(f(x_1,u_1),u_2)+\cdot \cdot \cdot+c(f(f(\cdot \cdot \cdot) \cdot \cdot \cdot),u_T)\end{align}$$

To get Newton’s method, need to use ***second order*** dynamics approximation:

$$\begin{align}f(x_t,u_t)\simeq f(\hat x_t, \hat u_t)+\nabla_{x_t,uj_t}f(\hat x_t, \hat u_t)\end{align}$$ $$\left[\begin{matrix}\delta x_t\\ \delta u_t\end{matrix}\right]$$ $+\frac{1}{2}(\nabla_{x_t,u_t}^2f(\hat x_t,\hat u_t) \cdot$ $$\left[\begin{matrix}\delta x_t\\ \delta u_t\end{matrix}\right])$$ $$\left[\begin{matrix}\delta x_t\\ \delta u_t\end{matrix}\right]$$

==differential dynamic programming(DDP)==

**Additional Reading**:

[Mayne, Jacobson. Differential Dynamic Programming. 1970]()

> Original differential dynamic programming algorithm.

[Yuval Tassa, Tom Erez, Emanuel Todorov. Synthesis and stabilization of complex behaviors through online trajectory optimization. 2012.](http://papers-repo.oss-cn-beijing.aliyuncs.com/Synthesis%20and%20stabilization%20of%20complex%20behaviors%20through%20online%20trajectory%20optimization.pdf) URL: https://ieeexplore.ieee.org/document/6386025

> Practical guide for implementing non-linear iterative LQR.
>
>

[Sergey Levine, Pieter Abbeel. Learning Neural Network Policies with Guided Policy Search under Unknown Dynamics. 2014.](http://papers-repo.oss-cn-beijing.aliyuncs.com/Learning%20Neural%20Network%20Policies%20with%20Guided%20Policy.pdf)

> Probabilistic formulation and trust region alternative to deterministic line search.

### 3.5 Discrete systems: Monte-Carlo tree search(MCTS)

![](http://images-repo.oss-cn-beijing.aliyuncs.com/2018-11-04_172611.png)

**Additional Reading**:

[A Survey of Monte Carlo Tree Search Methods. Cameron B. Browne, Edward Powley, Daniel Whitehouse, etc. 2012](http://papers-repo.oss-cn-beijing.aliyuncs.com/%5BA%20Survey%20of%20Monte%20Carlo%20Tree%20Search%20Methods.pdf) URL: https://ieeexplore.ieee.org/document/6145622

> Survey of MCTS methods and basic summary.

### 3.6 Case study: imitation learning from MCTS

[Deep Learning for Real-Time Atari Game Play Using Offline Monte-Carlo Tree Search Planning. Xiaoxiao Guo, Satinder Stingh, Honglak Lee, etc. 2014.](http://papers-repo.oss-cn-beijing.aliyuncs.com/5421-deep-learning-for-real-time-atari-game-play-using-offline-monte-carlo-tree-search-planning.pdf) 

> **DAgger**:
>
>  	1. train $\pi_\theta(u_t|o_t)$ from human data $D=\{o_1,u_1,..,o_N,u_N\}$
>  	2. run $\pi_\theta(u_t|o_t)$ to get dataset $D_\pi=\{o_1,...,o_M\}$
>  	3. Choose actions for states in $D_\pi$ using MCTS
>  	4. Aggregate: $D\leftarrow  D \cup D_\pi $
>
> **Why train a policy?**
>
> - In this case, MCTS is too slow for real-time play
> - Other reasons – perception, generalization, etc.

## 4. Learning Dynamical System Models from Data

> Distribution mismatch problem becomes exacerbated as we use more expressive model classes.

**Model-based reinforcement learning version 1.0**:

![](http://images-repo.oss-cn-beijing.aliyuncs.com/2018-11-12_011840.png)

**Case study: model-based policy search with GPs**:

[Learning to Control a Low-Cost Manipulator using Data-Efficient Reinforcement Learning. ]

> 1. run base policy $\pi_0(u_t|x_t)$ (e.g. random policy) to collect $D=\{(x,u,x')_i\}$
> 2. learn GP dynamics model $p(x'|x,u)$ to maximize $\sum_i \log p(x_i'|x_i,u+i)$
> 3. backpropagate through $p(x'|x,u)$ into the policy to optimize $\pi_\theta(u_t|x_t)$
> 4. run $\pi_\theta(u_t|x_t)$. appending the visited tuples $(x,u,x')$ to $D$

![](http://images-repo.oss-cn-beijing.aliyuncs.com/2018-11-12_014515.png)

**Case study: dynamic with recurrent networks**:

[Recurrent Network Models for Human Dynamics. Katerina Fragkiadaki]

**Other related work on learning human dynamics**:

> - Conditional Restricted Boltzmann Machines( Taylor et al.)
> - GPs and GPLVM (Wang et al.)
> - Linear and switching linear dynamical systems (Hsu & Popovic)

The trouble with Global models:

- Planner will seek out regions where the model is erroneously optimistic
- Need to find a very good model in most of the state space to converge on a good solution
- In some tasks, the model is much more complex than the policy

What controller to execute?

> $$\begin{align}p(u_t|x_t)=N(K_t(x_t- \hat x_t)+k_t +\hat u_t,\Sigma_t)\end{align}$$
>
> Set $\sum_t=Q_{u_t,u_t}^{-1}$
>
> $Q(x_t,u_t)$ is the cost to go: total cost we get after taking an action
>
> $Q(x_t,u_t)=const+\frac{1}{2}$ $$\left[\begin{matrix}x_t \\ u_t\end{matrix}\right]^T Q_t $$ $$\left[\begin{matrix}x_t \\ u_t\end{matrix}\right]$$ $+$ $$\left[\begin{matrix}x_t \\ u_t\end{matrix}\right]^T q_t$$
>
> $Q_{u_t,u_t}$ is big if changing $u_t$ changes the Q-value a lot!
>
> If $u_t$ changes Q-value a lot, don’t vary $u_t$ so much
>
> Only act randomly when it minimally affects the cost to go.
>
> > Standard LQR solves $\min \sum_{t=1}^T c(x_t,u_t)$
> >
> > Linear-Gaussian solution solves $\min\sum_{t=1}^T E_{(x_t,u_t) \sim p(x_t,u_t)} [c(x_t,u_t)-H(p(u_t|x_t))] $
> >
> > This is the maximum entropy solution: act as randomly as possible while minimizing cost

**KL-divergences between trajectories**:

> - Not just for trajectory optimization – really important for model-free policy search too!
>
> $$\begin{align}D_{KL}(p(\tau) || \bar p(\tau))=E_{p(\tau)}[\log p(\tau)-\log \bar p(\tau)\end{align}$$
>
> $$p(\tau)=p(x_1) \prod_{t=1}^T p(u_t|x_t)p(x_{t+1}|x_t,u_t) $$
>
> $$\begin{align}\bar p(\tau)=p(x_1) \prod_{t=1}^T \bar p(u_t|x_t) p(x_{t+1}|x_t,u_t)\end{align}$$
>
> $$\begin{align}D_{KL}(p(\tau) || \bar p(\tau))= \sum_{i=1}^T E_{p(x_t,u_t)}[-\log \bar p(u_t|x_t)-H(p(u_t|x_t)]\end{align}$$
>
> If we can get $D_{KL}$ into the cost, we can just use iLQR!
>
> > Digression: Dual Gradient Descent
> >
> > $\min_x f(x) s.t. C(x)=0$
> >
> > $L(x,\lambda)=f(x)+\lambda C(x)$
> >
> > $g(\lambda)=L(x^*(\lambda),\lambda)$
> >
> > $x^*=\arg \min_x L(x,\lambda$
> >
> > $\frac{dg}{d\lambda}=\frac{dL}{d\lambda}(x^*,\lambda)$
>
> > 1. Find $x^* \leftarrow \arg \min_x L(x,\lambda)$
> > 2. Compute $\frac{dg}{d\lambda}=\frac{dL}{d\lambda}(x^*,\lambda)$
> > 3. $\lambda \leftarrow \lambda+\alpha\frac{dg}{d \lambda}$





## A. Coding

### A.1 General

[Video Tutorial](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/1-1-A-RL/)

List of ==RL==:

| 分类                 | 实例                              |
| -------------------- | --------------------------------- |
| 通过价值选择行为     | Q Learning, Sarsa, Deep Q Network |
| 直接选择行为         | Policy Gradients                  |
| 想象环境，并从中学习 | Model based RL                    |

| -------        | ------                              |
| -------------- | ----------------------------------- |
| Model-Free RL  | Q Learning, Sarsa, Policy Gradients |
| Model-Based RL | add a model                         |

| -------                   | ------                                                       |
| ------------------------- | ------------------------------------------------------------ |
| Policy-based RL(基于概率) | 不一定选择概率最高的action，可用于连续模型，如Policy Gradient |
| Value-based RL(基于价值)  | 一定选择价值最大的action，只能用于离散模型，如Q Learning, Sarsa |

> Actor-Critic: Policy-based RL与Value-based RL的结合

| ------                     | ------                                      |
| -------------------------- | ------------------------------------------- |
| Monte-Carlo update         | 基础版Policy Gradient, Monte-Carlo Learning |
| Temporal-Difference update | Q Learning, Sarsa, 升级版Policy Gradients   |

| ------     | ------                     |
| ---------- | -------------------------- |
| On-Policy  | Sarsa, Sarsa(lambda)       |
| Off-Policy | Q Learning, Deep Q Network |



### A.2 Q-Learning

**Pseudo-code**

```pseudocode
Initialize Q(s,a) arbitrarily
Repeat (for each episode):
	Initialize s
	Repeate (for each step of episode):
		Choose a from s using policy drived from Q (e.g. \epsilon-greedy)
		Take action a, observe r, s'
		Q(s,a):=Q(s,a)+\alpha [r+\gamma max_a' Q(s',a')-Q(s,a)]
		s:=s'
	until s is terminal
```



### A.3 Sarsa



### A.4 Deep Q Network

==Experience replay==, ==Fixed Q-targets==

DQN是在Q-Learning主框架上加了装饰后得到的，这些装饰包括：

- 记忆库
- 神经网络计算Q值
- 暂时冻结Q_target值（切断相关性）

