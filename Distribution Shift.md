[Toc]

# Distribution Shift

## TOSEARCH

1. ODD distribution (Between Out-of-Distribution and In-Distribution)
2. 

## Essays

1. [Detecting & Correcting for Label Shift w. Black Boxl Predictors](https://arxiv.org/abs/1802.03916)
2. [Failing Loudly](https://arxiv.org/abs/1810.11953)
3. [Importance Weighting & Deep Learning](https://arxiv.org/abs/1812.03372)
4. [Some Trouble (and progress) w. Domain-Adversarial Learning](https://arxiv.org/abs/1903.01689)
5. [Scholkopf et al "On Causal and Anticausal Learning"](https://arxiv.org/abs/1206.6471)

## Phenomena

### A Peculiar Disagreement / Generalization Connection

#### Procedures

1. Train 2 different models in a dataset (by using different data ordering for SGD or using different random initialization)
2. Compute the disagreement between the models

Outcome

The disagreement tracks the test-error quite closely

## Concepts

### Domain Adaptation - Formal Setup

- Probabilities 
  - Source Distribution $p(x, y)$
  - Target Distribution $q(x, y)$
- Data
  - Training Examples $(x_1, y_1), …, (x_n, y_n) \sim p(x, y)$
  - Test Examples $(x'_1, …, x'_n) \sim q(x)$
- Objective
  - Predict well on the test distribution, **WITHOUT** seeing any labels $y_i \sim q(y)$

### Label Shift

- Assume $p(x, y)$ changes, but the conditional $p(x|y)$ is **fixed**, $q(y, x)= q(y)p(x|y)$
- Makes anticausal assumption, (y causes x) [Diseases cause symptoms]
- But how can we estimate $q(y)$ without any samples $y_i \sim q(y)$?

### Covariate Shift

- Assume $p(x, y)$ changes, but the conditional $p(y|x)$ is **fixed**, $q(y, x) = q(x)p(y|x)$
- Implicitly assumes that x causes y [While symptoms don't cause diseases]
- Appealing because we have samples $x_i \sim p(x)$ and $x'_j \sim q(x)$
- Natural to estimate $q(x)/p(x)$ -> use for importance-weighted ERM
-  Under an epidemic, $p(y|x)$ should change

## Methods

### Black Box Shift Estimation (BBSE) 

Assumptions

1. The label shift assumption 
   $$
   p(x|y)=q(x|y), \forall x \in \mathcal X, y \in \mathcal Y
   $$
   
2. For every $y \in \mathcal Y$ with $q(y) > 0$ we require $p(y) > 0$

3. Access to a black box predictor $f : \mathcal X \rightarrow \mathcal Y$ where the expected confusion matrix $C_p(f)$ is invertible 

$$
C_p(f) := p(f(x), y) \in R^{|\mathcal Y| \times |\mathcal Y|}
$$

