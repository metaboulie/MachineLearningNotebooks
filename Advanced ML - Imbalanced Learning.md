# Imbalanced Learning

## Metrics

- Ideal performance metric: the learning is properly biased towards the minority class(es)

- Imbalance-aware performance metrics:

  - G-score / G-mean
  - Matthews Correlation Coefficient

  $$
  \rho_\textit{MCC}=\displaystyle\frac{n\sum_{i=1}^gn_{ii}-\sum_{i=1}^g\hat{n}_in_i}{\sqrt{(n^2-\sum_{i=1}^g\hat{n}_i^2)(n^2-\sum_{i=1}^gn_i^2)}}
  $$

  ​		where $\hat n_i = \sum_{j=1}^gn_{ij}$ is the total number of instances classified as *i*

  - Weighted macro $F_1$ score

    - For imbalanced data sets, give more weights to minority classes
    - $w_1,\dots,w_g \in [0,1] \; \text {such that} \; w_i > w_j \; \text {if} \; n_i < n_j \; \text {and} \; \displaystyle \sum _{i=1}^gw_i = 1$

    $$
    \rho _\textit{wmMETRIC} = \frac 1g\displaystyle\sum_{i=1}^g\rho_{\textit{METRIC}_i}w_i
    $$

    ​		where $\textit{METRIC}_i$ is a class-specific metric such as $\textit{PPV}_i, \; \textit{TPR}_i$ of class *i*

    - Example: $w_i = \displaystyle\frac{n-n_i}{(g-1)n}$ are suitable weights
    - Weighted macro $F_1$ score:

    $$
    \rho_{\textit{wmF}_1} = 2 \cdot \displaystyle\frac{\rho_\textit{wmPPV} \cdot \rho_\textit{wmTPR}}{\rho_\textit{wmPPV}+\rho_\textit{wmTPR}}
    $$

  - Cohen's Kappa or Cross Entropy (Grandini et al. (2021)) treat "predicted" and "true" classes as two discrete random variables 

  

  

## Possible Solutions

| Approach                | Main idea                                                    |
| ----------------------- | ------------------------------------------------------------ |
| Algorithm-level         | Bias classifiers towards minority                            |
| Data-level              | Rebalance classes by resampling                              |
| Cost-sensitive Learning | Introduce different costs for misclassification when learning |
| Ensemble-based          | Ensemble learning plus one of three techniques above         |



## Cost-Sensitive Learning

### Cost Matrix

- Input: cost matrix **C**
- $C(j,k)$ is the cost of classifying class k (true label) as j (predicted label)
- Common heuristic for imbalanced data sets:
  - $C(j,k) = \frac {n_j}{n_k} \quad with n_k \ll n_j$
  - $C(j,k) = 1 \quad with \ n_j \ll n_k$
  - $C(j,k) = 0 \quad\text {for a correct classfication}{}$
- Optimal decisions do not change if
  - **C** is multiplied by positive constant
  - **C** is added with constant shift

### Minimum Expected Cost Principle

- Suppose we have:
  - a cost matrix **C**
  - knowledge of the true posterior $p(\cdot|x)$
- Predict class j with smallest expected costs when marginalizing over true classes:

$$
\mathbb E _{K \sim p(\cdot|x)}(C(j,K)) = \displaystyle \sum _{k=1}^gp(k|x)C(j,k)
$$

- If we trust a probabilistic classifier, we can convert its scores to labels:

$$
h(x) :=  \underset{j=1,\ldots,g}{arg\,min}\displaystyle \sum _{k=1}^g \pi _k(x)C(j,k)
$$

### Optimal Threshold For Binary Case

- Optimal decisions do not change if 
  - **C** is multiplied by positive constant
  - **C** is added with constant shift
- Scale and shift **C** to get simpler **C'** 

|                | True Class |        |
| :------------: | :--------: | :----: |
| **Pred Class** |   $y=1$    | $y=-1$ |
|     $y=1$      | $C'(1,1)$  |   1    |
|     $y=-1$     | $C'(-1,1)$ |   0    |

- Where 
  - $C'(-1,1)=\displaystyle\frac{C(-1,1)-C(-1,-1)}{C(1,-1)-C(-1,-1)}$
  - $C'(1,1)=\displaystyle\frac{C(1,1)-C(-1,-1)}{C(1,-1)-C(-1,-1)}$

- We will predict $y=1$ if 

$$
p(-1|x)C'(1,-1)+p(1|x)C'(1,1) \le p(-1|x)C'(-1,-1)+p(1|x)C'(-1,1) \\
\Rightarrow [1-p(1|x)]\cdot 1 + p(1|x)C'(1,1) \le p(1|x)C'(-1,1) \\
\Rightarrow p(1|x) \ge \displaystyle 
$$

