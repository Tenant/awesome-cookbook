# Probabilistic Graphical Models

## 1 Representation

**PGMs**: a model for ==joint probability distribution== over random variables; represent ==dependencies== and ==independencies== between the random variables.

**Types of PGMs**: ==Directed graph==: Bayesian Network(BN), ==Undirected graph==: Markov Random File(MRF).

### 1.1 Bayesian Networks(BNs)

> For any node $C$ (random variables) in a BN, all nodes that are not descendants of  $C$ are independent of $C$, given $C$‘s parents.

**Observation 1**: The graph must be ==acyclic==! BN is defined on ==DAG==.

**Observation 2**: Not every distribution can be represented by a BN satisfying *all the independencies*.

### 1.2 Markov Random Fields(MRF)

> ==Factors== correspond to ==maximal cliques==
>
> The joint distribution is ==the product of all factors normalized by the partition function==.

**Observation 1**: If every node in $B$ and every node in $C$ are not connected in the graph after deleting all nodes in $A$, then $B$ and $C$ are ==conditional independent given $A$==.

## 2. Inference

### 2.1 Definition

**Conditional Probability**: $Pr(X|E=e)$

**MAP Inference**: $\max_{x} Pr(X=x|E=e)$

### 2.2 The hardness of Inference in GM

**Observation 1**: Exact inference in GM is hard: 1). Decision version of exact inference is NPC. 2) Exact inference is #P complete.

**Observation 2**: Approximate inference in GM is hard: $\epsilon-$approximate inference is NP-hard for every $\epsilon$.

**Definition**: $\hat p$ is an $\epsilon$-approximation of $Pr(X=x)$ means that $\frac{Pr(X=x)}{1+\epsilon}\le\hat p \le Pr(X=x)(1+\epsilon)$

**Observation 3**: If one has an $\epsilon$-approximation of $Pr(X=x)$, one can solve the NPC problem $Pr(X=x)>0$.

**Conclusion**: The ==worst-case== complexity of the inferences, both ==exact== and ==approximate== are ==NP-hard==.

### 2.3 Exact Inference Algorithms

Exact Inference Algorithms:

- The variable elimination algorithm.

- Message passing – the sum-product algorithm

- Belief propagation algorithm

### 2.4 Approximate Inference Algorithms

- Variationally Methods

  - (Loopy) belief propagation

    > Loopy belief propagation: just let the belief propagates
    >
    > On clique trees, belief propagation ==converges== after propagating on all edges (two directions)
    >
    > For general ==clique graphs==, it is ==not guaranteed to converge==. Even if it converges, it can converge to a wrong answer.

  - Mean field method

    - Using the first order approximation of the target distribution

- Sampling based methods

  - MCMC

## 3. Learning

### 3.1 Structure Learning

**Note**: the edge $A \rightarrow C$ exists or not *equals to* whether the following equation holds strictly: $P(AC|B)=P(A|B)P(C|B)$.

- Constraint-based structure learning
  - Using hypothesis test to obtain independences
  - Construct the graph
- Score-based structure learning
  - Likelihood scores
  - BIC
  - MDL