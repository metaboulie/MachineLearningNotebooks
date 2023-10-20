[toc]



# Machine Learning



## Feature normalization 

(i.e., MinMaxScaler, StandardScaler, …)

then Parallel coordinates plot (with plotly)

## Preprocessing

### Scaling

#### MinMaxScaler

$$
x' = b\frac{a + x - min(x)}{max(x) - min(x)}
$$



### Log and Power Transformations

Another important use of variable transformation is to eliminate  skewness and other distributional features that complicate analysis.  Often the goal is to find a simple transformation that leads to  normality.

A power transformation is generally defined like this:
$$
x_i^\lambda = \left \{
\begin {aligned}
\frac{x_i^\lambda-1}{\lambda GM(x)^{\lambda-1}} && if \ \ \lambda \neq 0 \\
GM(x)lny_i && if \ \ \lambda = 0
\end {aligned}
\right.
$$
where *GM(x)* is the geometric mean of *x*:
$$
GM(x) = (x_1x_2...x_n)^{\frac1n}
$$

####  Box-Cox transformation

$$
x_i^{\lambda} = \left \{
\begin {aligned}
\begin {flalign*}
&\frac {x_i^\lambda - 1} \lambda && \quad if \ \ \lambda \neq 0 \\
&lnx_i && \quad if \ \ \lambda = 0
\end {flalign*}
\end {aligned}
\right.
$$

The value of the parameter $\lambda$ can be via different optimization methods such as the maximum likelihood that the transformed feature is Gaussian.

​    
$$
llf(\hat{\lambda}) - llf(\lambda) < \frac{1}{2}\chi^2(1 - \alpha, 1)
$$
with ``llf`` the log-likelihood function and $\chi^2$ the chi-squared function.

#### Yeo–Johnson transformation

$$
x_i^\lambda = \left \{
\begin {aligned}
\begin {flalign*}
&\frac {(x_i + 1)^\lambda - 1} \lambda && \quad if \ \ \lambda \neq 0,\ \ x \ge 0 \\
&ln(x_i + 1) && \quad if \ \ \lambda = 0,\ \ x \ge 0 \\
&-\frac {(-x_i + 1)^{2 - \lambda} - 1} {2 - \lambda} && \quad if \ \ \lambda \neq 2,\ \ x < 0 \\
&-ln(-x_i + 1)  && \quad if \ \ \lambda = 2,\ \ x < 0
\end {flalign*}
\end {aligned}
\right .
$$

## Similarly 

to preprocessing.OneHotEncoder, the class preprocessing. OrdinalEncoder now supports aggregating infrequent categories into a single output for each feature. The parameters to enable the gathering of infrequent categories are `min_frequency` and `max_categories`. 

## HDBSCAN 

is able to adapt to the multi-scale structure of the dataset without requiring parameter tuning.(DBSCAN assumes that all clusters are in the same density, which is not always the case)

Larger values (of min_cluster_size and min_sample) tend to be more robust with respect to noisy datasets, and min_samples better be tuned after finding a good value for min_cluster_size.

## TargetEncoder 

considers missing values, such as np.nan or None, as another category and encodes them like any other category. Categories that are not seen during fit are encoded with the target mean, i.e. target_mean_.

For the binary classification target, the target encoding is given by:
$$
S_i=\lambda_i\frac{n_{iY}}{n_i}+(1-\lambda_i)\frac{n_Y}{n}
$$
where $ S_i $ is the encoding for category $i$, $n_{iY}$ is the number of observations with $Y=1$ and category $i$, $n_i$ is the number of observations with category $i$, $n_Y$ is the number of observations with $Y=1$, $n$ is the number of observations, and $\lambda_i$ is a shrinkage factor for category $i$. The shrinkage factor is given by:
$$
\lambda_i=\frac{n_i}{m+n_i}
$$
where $m$ is a smoothing factor, which is controlled with the `smooth` parameter in [`TargetEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html#sklearn.preprocessing.TargetEncoder). Large smoothing factors will put more weight on the global mean. When `smooth="auto"`, the smoothing factor is computed as an empirical Bayes estimate: $m=\sigma_i^2/\gamma^2$, where $\sigma_i^2$ is the variance of `y` with category $i$ and $\gamma^2$ is the global variance of `y`.

For continuous targets, the formulation is similar to binary classification:
$$
S_i = \lambda_i\frac{\Sigma_{k\in L_i}Y_k}{n_i}+(1-\lambda_i)\frac{\Sigma_{k=1}^nY_k}{n}
$$
where $L_i$ is the set of observations with category $i$ and $n_i$ is the number of observations with category $i$.

## Learning Curve

In machine learning, the learning curve refers to a graphical representation of the model's performance as the amount of training data increases. It helps to analyze the relationship between the size of the training data and the model's accuracy or error rate.

The learning curve typically consists of two plots: one representing the training set and another for the validation set. As the size of the training set increases, the learning curve shows how the model's performance improves or stabilizes.



- Mini-Batch GD or SGD for some initial values of parameters, then Batch for an accurate result.

# Multi-Layer Perceptron and Backpropagation

## Backpropagation Algorithm in Detail

1. It handles one mini-batch at a time (for example containing 32 instances each), and it goes through the full training set multiple times. Each pass is called an *epoch*.
2. Each mini-batch is passed to the network’s input layer, all the way through till the output layer. This is the *forward pass*: it is exactly like making predictions, except all intermediate results are preserved since they are needed for the backward pass.
3.  Measures the network’s output error with specific loss function.
4. Then it computes how much each output connection contributed to the error. This is done analytically by simply applying the *chain rule* (perhaps the most fun‐ damental rule in calculus).
5. The algorithm then measures how much of these error contributions came from each connection in the layer below, again using the chain rule—and so on until the algorithm reaches the input layer.
6. Finally, the algorithm performs a Gradient Descent step to tweak all the connec‐ tion weights in the network, using the error gradients it just computed.

- It is important to initialize all the hidden layers’ connection weights randomly

- In order for the backpropagation to work properly, the authors made a key change to the MLP’s architecture: they replaced the step function with the logistic/sigmoid function, $σ(z) = 1 / (1 + exp(–z))$. 
- Other *activation functions*:
  1. *The* *hyperbolic tangent function*: $tanh(z) = 2σ(2z) – 1$
  2. *The Rectified Linear Unit function*: $ReLU(z) = max(0, z)$
  3. *The Softplus function*: $softplus(z) = log(1+e^z)$
  4. It turns out that ReLU generally works better in ANNs.

## Regression MLPs

- In general, when building an MLP for regression, you do not want to use any activation function for the output neurons, so they are free to output any range of values. However, if you want to guarantee that the output will always be positive, then you can use the *ReLU* activation function, or the *softplus* activation function in the output layer. Finally, if you want to guarantee that the predictions will fall within a given range of values, then you can use the *logistic function* or the *hyperbolic tangent*, and scale the labels to the appropriate range: 0 to 1 for the *logistic function*, or –1 to 1 for the *hyperbolic tangent*.

- The loss function to use during training is typically the *MSE* but if you have a lot of outliers in the training set, you may prefer to use the *MAE* instead. Alternatively, you can use the *Huber loss*, which is a combination of both.

- The *Huber loss* function is defined as follows: 

  $L(y, f(x)) = \frac12(y - f(x))^2$  if  $|y - f(x)| <= δ$

  $L(y, f(x)) = δ*(|y - f(x)| - \frac12δ)$  if  $|y - f(x)| > δ$

- The ***Huber loss*** is quadratic when the error is smaller than a thres‐ hold δ (typically 1), but linear when the error is larger than δ. This makes it less sensitive to outliers than the mean squared error, and it is often more precise and converges faster than the mean absolute error.

  ### *Typical Regression MLP Architecture*

  | hyperparameter              | Typical value                                                |
  | :-------------------------- | ------------------------------------------------------------ |
  | # input neurons             | One per input feature                                        |
  | # hidden layers             | Depends on the problem. Typically 1 to 5.                    |
  | \# neurons per hidden layer | Depends on the problem. Typically 10 to 100.                 |
  | \# output neurons           | 1 per prediction dimension                                   |
  | Hidden activation           | ReLU (or SELU)                                               |
  | Output activation           | None or ReLU/Softplus (if positive outputs) or Logistic/Tanh (if bounded outputs) |
  | Loss function               | MSE or MAE/Huber (if outliers)                               |

  ### *Typical* *Classification* *MLP Architecture*
  
  | Hyperparameter          | Binary classification | Multilabel binary classification | Multiclass classification |
  | ----------------------- | --------------------- | -------------------------------- | ------------------------- |
  | Input and hidden layers | Same as regression    | Same as regression               | Same as regression        |
  | # output neurons        | 1                     | 1 per label                      | 1 per class               |
  | Output layer activation | Logistic              | Logistic                         | Softmax                   |
  | Loss Function           | Cross-Entropy         | Cross-Entropy                    | Cross-Entropy             |
  

## Building Complex Models Using the Functional API

### *Wide & Deep* neural network.

- It connects all or part of the inputs directly to the output layer.
-  This architecture makes it possible for the neural network to learn both deep patterns (using the deep path) and simple rules (through the short path).
-  Deep networks have a much higher *parameter* *efficiency* than shallow ones: they can model complex functions using exponentially fewer neurons than shallow nets, allowing them to reach much better performance with the same amount of training data.
-  A simpler approach is to pick a model with more layers and neurons than you actually need, then use early stopping to prevent it from overfitting (and other regu‐ larization techniques, such as *dropout*).

### Cases in which you may want to have multiple outputs:

- The task may demand it. For instance, you may want to locate and classify the main object in a picture. This is both a regression tasks and a classification task.
- Similarly, you may have multiple independent tasks based on the same data. For example, you could perform *multitask classification* on pictures of faces, using one output to classify the person’s facial expression (smiling, surprised, etc.) and another output to identify whether they are wearing glasses or not.
- Another use case is as a regularization technique (i.e., a training constraint whose objective is to reduce overfitting and thus improve the model’s ability to generalize). For example, you may want to add an auxiliary output in a neural network architecture to ensure that the underlying part of the network learns something useful on its own, without relying on the rest of the network.
- By default, Keras will compute all the losses and simply add them up to get the final loss used for training.

# Training Deep Neural Networks

## The Vanishing/Exploding Gradients Problems

- We don’t want the signal to die out, nor do we want it to explode and saturate.

### Glorot and He Initialization

#### Equation 11-1. Glorot initialization (when using the sigmoid activation function)

​								  Normal distribution with mean 0 and variance $ \sigma^2 = \frac 1 {fan_{avg}} $

​								  Or a uniform distribution between $ -\gamma $ and $ +\gamma $, with $ \gamma = \sqrt \frac 3 {fan_{avg}} $

- Replace $ fan_{avg} $ with $ fan_{in} $ will get *LeCun initialization*.

##### *Initialization parameters for each type of activation function*

| **Initialization** | **Activation functions**                               | $\sigma^2$(Normal) |
| ------------------ | ------------------------------------------------------ | ------------------ |
| Glorot             | None, tanh, sigmoid, softmax                           | $1/fan_{avg}$      |
| He                 | ReLU, Leaky ReLU, PReLU, RReLU, ELU, GELU, Swish, Mish | $2/fan_{in}$       |
| LeCun              | SELU                                                   | $1/fan_{in}$       |

### Better Activation Functions

- Leaky ReLU: LeakyReLU*α*(*z*) = max(*αz*, *z*), normally setting *α* = 0.2.
- RReLU: *α* is picked randomly in a given range during training and is fixed to an average value during testing.
- PReLU: *α* is authorized to be learned during training.
- PReLU was reported to strongly outperform ReLU on large image datasets, but on smaller datasets it runs the risk of overfitting the training set.

#### Equation 11-2. ELU activation function

$$
ELU_\alpha(z) = \left \{
\begin {aligned}
\begin {flalign*}
&\alpha(exp(z) - 1) && \quad if \ \ z \le 0 \\
&z && \quad if \ \ z > 0
\end {flalign*}
\end {aligned}
\right .
$$

-  *α*  normally set to 1
- SELU: approximately, $ SELU(z) = 1.05 ELU_{1.67}(z) $
- If you build a neural network composed exclusively of a stack of dense layers (i.e., an MLP), and if all hidden layers use the SELU activation function, then the network will self-normalize: the output of each layer will tend to preserve a mean of 0 and a standard deviation of 1 during training, which solves the vanishing/exploding gradients problem.

##### Conditions for self-normalization to happen

- The input features must be standardized: mean 0 and standard deviation 1.
- Every hidden layer’s weights must be initialized using LeCun normal initialization. In Keras, this means setting `kernel_initializer="lecun_normal"`.
- The self-normalizing property is only guaranteed with plain MLPs. If you try to use SELU in other architectures, like recurrent networks or networks with *skip connections* (i.e., connections that skip layers, such as in Wide & Deep nets), it will probably not outperform ELU.
- You cannot use regularization techniques like ℓ1 or ℓ2 regularization, max-norm, batch-norm, or regular dropout.

#### Equation 11-3. GELU activation function

$$
GELU(z) = z\Phi(z)
$$

- Φ is the standard Gaussian cumulative distribution function (CDF): Φ(*z*) corresponds to the probability that a value sampled randomly from a normal distribution of mean 0 and variance 1 is lower than *z*.

$ Swish_\beta(z) = z\sigma(\beta z) $, where σ is the sigmoid function

- GELU is approximately equal to the generalized Swish function using *β* = 1.702
- You can tune β like any other hyperparameter. Alternatively, it’s also possible to make β trainable and let gradient descent optimize it

Mish(*z*) = *z* * tanh(softplus(*z*)), where softplus(*z*) = log(1 + exp(*z*))

### Batch Normalization

Although using He initialization along with ReLU (or any of its variants) can significantly reduce the danger of the vanishing/exploding gradients problems at the beginning of training, it doesn’t guarantee that they won’t come back during training.

#### Equation 11-4. Batch normalization algorithm

$$
\begin {flalign*}
&1. \quad \mu_B = \frac 1 {m_B} \sum^{m_B}_{i=1}x^{(i)} \\
&2. \quad \sigma^2_B = \frac 1 {m_B} \sum^{m_B}_{i=1}(x^{(i)} - \mu_B)^2 \\
&3. \quad \hat x^{(i)} = \frac {x^{(i)} - \mu_B} {\sqrt {\sigma^2_B + \varepsilon}} \\
&4. \quad z^{(i)} = \gamma \otimes \hat x^{(i)} + \beta
\end {flalign*}
$$

- $\mu_B$ is the vector of input means, evaluated over the whole mini-batch B (it contains one mean per input).

- $m_B$ is the number of instances in the mini-batch.
- $σ_B$ is the vector of input standard deviations, also evaluated over the whole mini-batch (it contains one standard deviation per input).
- $ \hat x^{(i)}$ is the vector of zero-centered and normalized inputs for instance $i$.
- ε is a tiny number that avoids division by zero and ensures the gradients don’t grow too large (typically 10–5). This is called a smoothing term.
- γ is the output scale parameter vector for the layer (it contains one scale parameter per input).
- ⊗ represents element-wise multiplication (each input is multiplied by its corresponding output scale parameter).
- β is the output shift (offset) parameter vector for the layer (it contains one offset parameter per input). Each input is offset by its corresponding shift parameter.
- $z^{(i)}$ is the output of the BN operation. It is a rescaled and shifted version of the inputs.
- The technique consists of adding an operation in the model just before or after the activation function of each hidden layer. This operation simply zero-centers and normalizes each input, then scales and shifts the result using two new parameter vectors per layer: one for scaling, the other for shifting.
- If you add a BN layer as the very first layer of your neural network, you do not need to standardize your training set

To sum up, four parameter vectors are learned in each batch-normalized layer: γ (the output scale vector) and β (the output offset vector) are learned through regular backpropagation, and μ (the final input mean vector) and σ (the final input standard deviation vector) are estimated using an exponential moving average. Note that μ and σ are estimated during training, but they are used only after training (to replace the batch input means and standard deviations in Equation 11-4).

You can experiment with adding the BN layers before the activation functions, or after.

The `BatchNormalization` class has quite a few hyperparameters you can tweak. The defaults will usually be fine, but you may occasionally need to tweak the `momentum`. This hyperparameter is used by the `BatchNormalization` layer when it updates the exponential moving averages; given a new value v (i.e., a new vector of input means or standard deviations computed over the current batch), the layer updates the running average $\hat v$ using the following equation:
$$
\hat v \Leftarrow \hat v \times momentum + v \times (1 - momentum)
$$

- A good momentum value is typically close to 1; for example, 0.9, 0.99, or 0.999. You want more 9s for larger datasets and for smaller mini-batches.

### Gradient Clipping

If you observe that the gradients explode during training (you can track the size of the gradients using TensorBoard), you may want to try clipping by value or clipping by norm, with different thresholds, and see which option performs best on the validation set.

```python
optimizer = tf.keras.optimizers.SGD(clipnorm=1.0)
model.compile([...], optimizer=optimizer)
```

This will clip the whole gradient if its $ℓ_2$ norm is greater than the threshold you picked.

## Reusing Pretrained Layers (*Transfer learning*)

- If the input pictures for your new task don’t have the same size as the ones used in the original task, you will usually have to add a preprocessing step to resize them to the size expected by the original model. More generally, transfer learning will work best when the inputs have similar low-level features.
- The output layer of the original model should usually be replaced because it is most likely not useful at all for the new task, and probably will not have the right number of outputs.
- Similarly, the upper hidden layers of the original model are less likely to be as useful as the lower layers, since the high-level features that are most useful for the new task may differ significantly from the ones that were most useful for the original task. You want to find the right number of layers to reuse.
- Try freezing all the reused layers first (i.e., make their weights non-trainable so that gradient descent won’t modify them and they will remain fixed), then train your model and see how it performs. Then try unfreezing one or two of the top hidden layers to let backpropagation tweak them and see if performance improves. The more training data you have, the more layers you can unfreeze. It is also useful to reduce the learning rate when you unfreeze reused layers: this will avoid wrecking their fine-tuned weights.

- Unsupervised pretraining (today typically using autoencoders or GANs rather than RBMs) is still a good option when you have a complex task to solve, no similar model you can reuse, and little labeled training data but plenty of unlabeled training data.
- If you do not have much labeled training data, one last option is to train a first neural network on an auxiliary task for which you can easily obtain or generate labeled training data, then reuse the lower layers of that network for your actual task.

## Faster Optimizers

### Momentum

#### Equation 11-5. Momentum algorithm

$$
\begin {flalign*}
&1. \quad m \Leftarrow \beta m - \eta \nabla_\theta J(\theta) \\
&2. \quad \theta \Leftarrow \theta + m 
\end {flalign*}
$$

- The previous gradients will be recorded in momentum and decay exponentially, thus a history of gradients will used to update weights.
- Due to the momentum, the optimizer may overshoot a bit, then come back, overshoot again, and oscillate like this many times before stabilizing at the minimum. This is one of the reasons it’s good to have a bit of friction ($\beta$) in the system: it gets rid of these oscillations and thus speeds up convergence.

### Nesterov Accelerated Gradient

The Nesterov accelerated gradient (NAG) method, also known as Nesterov momentum optimization, measures the gradient of the cost function not at the local position θ but slightly ahead in the direction of the momentum, at θ + βm.

#### Equation 11-6. Nesterov accelerated gradient algorithm

$$
\begin {flalign*}
&1.\quad m \Leftarrow \beta m - \eta\nabla_\theta J(\theta + \beta m) \\
&2.\quad \theta \Leftarrow \theta + m
\end {flalign*}
$$

### AdaGrad

Consider the elongated bowl problem again: gradient descent starts by quickly going down the steepest slope, which does not point straight toward the global optimum, then it very slowly goes down to the bottom of the valley. It would be nice if the algorithm could correct its direction earlier to point a bit more toward the global optimum.

#### Equation 11-7. AdaGrad algorithm

$$
\begin {flalign*}
& 1. \quad s \Leftarrow s + \nabla_\theta J(\theta) \otimes \nabla_\theta J(\theta) \\
& 2. \quad \theta \Leftarrow \theta - \eta\nabla_\theta J(\theta) \oslash \sqrt {s + \epsilon}
\end {flalign*}
$$

- Each $s_i$ accumulates the squares of the partial derivative of the cost function with regard to parameter $θ_i$. If the cost function is steep along the ith dimension, then $s_i$ will get larger and larger at each iteration.
- The gradient vector is scaled down by a factor of $\sqrt{s+\epsilon}$ (the ⊘ symbol represents the element-wise division, and ε is a smoothing term to avoid division by zero, typically set to $10^{–10}$).
- `AdaGrad` frequently performs well for simple quadratic problems, but it often stops too early when training neural networks: the learning rate gets scaled down so much that the algorithm ends up stopping entirely before reaching the global optimum.**You should not use it to train deep neural networks.**

### RMSProp

The `RMSProp` algorithm⁠ fixes `AdaGrad` by accumulating only the gradients from the most recent iterations, as opposed to all the gradients since the beginning of training. It does so by using exponential decay in the first step (like NAG).

#### Equation 11-8. RMSProp algorithm

$$
\begin {flalign*}
& 1. \quad s \Leftarrow \rho s + (1-\rho)\nabla_\theta J(\theta) \otimes \nabla_\theta J(\theta) \\
& 2. \quad \theta \Leftarrow \theta - \eta\nabla_\theta J(\theta) \oslash \sqrt {s + \epsilon}
\end {flalign*}
$$

The decay rate ρ is typically set to 0.9.

### Adam

Adam, which stands for *adaptive moment estimation*, combines the ideas of momentum optimization and RMSProp: just like momentum optimization, it keeps track of an exponentially decaying average of past gradients; and just like RMSProp, it keeps track of an exponentially decaying average of past squared gradients. These are estimations of the mean and (uncentered) variance of the gradients. The mean is often called the *first moment* while the variance is often called the *second moment*, hence the name of the algorithm.

#### Equation 11-9. Adam algorithm

$$
\begin {flalign*}
& 1. \quad m \Leftarrow \beta_1m - (1-\beta_1)\nabla_\theta J(\theta) \\
& 2. \quad s \Leftarrow \beta_2s + (1-\beta_2)\nabla_\theta J(\theta) \otimes \nabla_\theta J(\theta) \\
& 3. \quad \widehat m \Leftarrow \frac m {1-\beta_1^t} \\
& 4. \quad \widehat s \Leftarrow \frac s {1-\beta_2^t} \\
& 5. \quad \theta \Leftarrow \theta + \eta\widehat m \oslash \sqrt{\widehat s + \epsilon} 
\end {flalign*}
$$

- In this equation, t represents the iteration number (starting at 1).
- Steps 3 and 4 are somewhat of a technical detail: since m and s are initialized at 0, they will be biased toward 0 at the beginning of training, so these two steps will help boost m and s at the beginning of training.
- The momentum decay hyperparameter $\beta_1$ is typically initialized to 0.9, while the scaling decay hyperparameter $\beta_2$ is often initialized to 0.999. As earlier, the smoothing term ε is usually initialized to a tiny number such as $10^{–7}$. 
- You can often use the default value η = 0.001.

### AdaMax

$$
\begin {flalign*}
& 1. \quad m \Leftarrow \beta_1m - (1-\beta_1)\nabla_\theta J(\theta) \\
& 2. \quad s \Leftarrow max(\beta_2s , abs(\nabla_\theta J(\theta)) ) \\
& 3. \quad \widehat m \Leftarrow \frac m {1-\beta_1^t} \\
& 4. \quad \theta \Leftarrow \theta + \eta\widehat m \oslash s
\end {flalign*}
$$

- Nadam and AdamW
- Adaptive optimization methods (including RMSProp, Adam, AdaMax, Nadam, and AdamW optimization) are often great, converging fast to a good solution. However, they can lead to solutions that generalize poorly on some datasets. So when you are disappointed by your model’s performance, try using NAG instead: your dataset may just be allergic to adaptive gradients.

### Learning Rate Scheduling

#### Power scheduling

Set the learning rate to a function of the iteration number t: $η(t) = \frac {η_0}{(1 + t/s)^c}$. The initial learning rate $η_0$, the power c (typically set to 1), and the steps s are hyperparameters. The learning rate drops at each step. After s steps, the learning rate is down to $η_0/2$. After s more steps it is down to $η_0 / 3$, then it goes down to $η_0 / 4$, then $η_0 / 5$, and so on. As you can see, this schedule first drops quickly, then more and more slowly. 

#### Exponential scheduling

Set the learning rate to $η(t) = η_0 0.1^{t/s}$. The learning rate will gradually drop by a factor of 10 every s steps. While power scheduling reduces the learning rate more and more slowly, exponential scheduling keeps slashing it by a factor of 10 every s steps.

#### Piecewise constant scheduling

Use a constant learning rate for a number of epochs (e.g., $η_0 = 0.1$ for 5 epochs), then a smaller learning rate for another number of epochs (e.g., $η_1 = 0.001$ for 50 epochs), and so on.

#### Performance scheduling

Measure the validation error every N steps (just like for early stopping), and reduce the learning rate by a factor of λ when the error stops dropping.

#### 1cycle scheduling

- After training, `history.history["lr"]` gives you access to the list of learning rates used during training.
- To sum up, exponential decay, performance scheduling, and 1cycle can considerably speed up convergence.

## Avoiding Overfitting Through Regularization

### $ℓ_1$ and $ℓ_2$ Regularization

$ℓ_2$ regularization is fine when using SGD, momentum optimization, and Nesterov momentum optimization, but not with Adam and its variants. If you want to use Adam with weight decay, then do not use $ℓ_2$ regularization: use AdamW instead.

### Dropout

There is one small but important technical detail. Suppose p = 75%: on average only 25% of all neurons are active at each step during training. This means that after training, a neuron would be connected to four times as many input neurons as it would be during training. To compensate for this fact, we need to multiply each neuron’s input connection weights by four during training. If we don’t, the neural network will not perform well as it will see different data during and after training. More generally, we need to divide the connection weights by the keep probability (1 – p) during training.

- If you observe that the model is overfitting, you can increase the dropout rate. 
- It can also help to increase the dropout rate for large layers, and reduce it for small ones. Moreover, many state-of-the-art architectures only use dropout after the last hidden layer, so you may want to try this if full dropout is too strong.
- Since dropout is only active during training, comparing the training loss and the validation loss can be misleading. In particular, a model may be overfitting the training set and yet have similar training and validation losses. So, make sure to evaluate the training loss without dropout (e.g., after training).
- If you want to regularize a self-normalizing network based on the SELU activation function (as discussed earlier), you should use alpha dropout.

### Monte Carlo (MC) Dropout

Averaging over multiple predictions with dropout turned on gives us a Monte Carlo estimate that is generally more reliable than the result of a single prediction with dropout turned off. 

- The number of Monte Carlo samples you use (100 in this example) is a hyperparameter you can tweak. The higher it is, the more accurate the predictions and their uncertainty estimates will be. 

# Custom Models and Training with Tensorflow

## Customizing Models and Training Algorithms

- If a function has hyperparameters that need to be saved along with the model, then you will want to subclass the appropriate class, such as `tf.keras.regu⁠larizers.Reg⁠⁠ularizer`, `tf.keras.constraints.Constraint`, `tf.keras.initializers.Ini⁠tializer`, or `tf.keras.layers.Layer` (for any layer, including activation functions).
- You need to call the parent constructor or the `get_config()` method as in `HuberLoss`, if they are defined by the parent class, but don't need if they aren't.
- Note that you must implement the `call()` method for losses, layers (including activation functions), and models, or the `__call__()` method for regularizers, initializers, and constraints. 
- When you define a metric using a simple function, Keras automatically calls it for each batch, and it keeps track of the mean during each epoch, just like we did manually. So the only benefit of our `HuberMetric` class is that the threshold will be saved. But of course, some metrics, like precision, cannot simply be averaged over batches: in those cases, there’s no other option than to implement a streaming metric.

### Custom Layers

If you want to create a custom layer without any weights, the simplest option is to write a function and wrap it in a `tf.keras.layers.Lambda` layer.

To define a custom loss based on model internals, compute it based on any part of the model you want, then pass the result to the `add_loss()` method.

It is also possible to add a custom metric using the model’s `add_metric()` method.

### Computing Gradients

If you try to compute the gradients of a vector, for example a vector containing multiple losses, then TensorFlow will compute the gradients of the vector’s sum. So if you ever need to get the individual gradients (e.g., the gradients of each loss with regard to the model parameters), you must call the tape’s `jacobian()` method: it will perform reverse-mode autodiff once for each loss in the vector (all in parallel by default). 
$$
\curvearrowleft|-v-|\curvearrowright
$$

