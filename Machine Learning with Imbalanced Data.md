# Machine Learning with Imbalanced Data

## Overview

### Nature of the imbalanced classes

- Small sample size
- Class separability

<img src="/Users/chanhuizhihou/Desktop/Images/class-separability.png" style="zoom:33%;" >

- Within-class sub-clusters

<img src="../../../Desktop/Images/within-class-sub-clusters.png" alt="within-class-sub-clusters" style="zoom:33%;" />

### Approaches to work with imbalanced datasets

#### Data-level approaches

> Changing the distribution of the data

- Random Over- / Under-sampling
- Creating new synthetic data
- Removing noise or alternatively, removing easy observations to classify

#### Cost-sensitive approaches

> Different cost to different errors

The cost of misclassifying an instance of the minority class outweighs the cost of misclassifying an instance from the majority

The cost-sensitive learning process seeks to minimize the cost error

### Ensemble approaches

> Combine weak learners

Construct multiple classifiers from the original data and then aggregate their predictions

Combining classifiers generally improve their generalization ability

## Evaluation Metrics

### Macro 

> Take the average

$$
\textit {Macro \ recall} = \frac 1n\sum_{i=1}^n \text {recall}_i
$$

### Weighted

> Take the mean of each metric weighted by the support

$$
\textit {Weighted \ recall} = \frac 1 {\sum {\text {support}_i}} \sum \text {recall}_i \times \text {support}_i
$$



### Micro

> $\mathcal {precision} == \mathcal {recall} == \mathcal {F-score}$

### Macro PR / ROC Curves

### G-Mean

$$
\textit {g-mean} = \sqrt [J] {\displaystyle\prod_{j=1}^J \text {acc}_j}
$$



## Undersampling

### Random

- Random Under-sampling (Fixed)

  But it often outperforms in DL

### Remove Noisy observations

- Edited Nearest Neighbours (ENN)
  1. Trains a 3-KNN on entire dataset
  2. Finds each observation's 3 closest neighbours 
  3. Keeps or removes observation based on neighbours agreement with its class
     - Two selection criteria
       1. **All** neighbours need to agree to retain observation
       2. **Most** neighbours need to agree to retain observation

- Repeated Edited Nearest Neighbours (RENN)
  - Repeat the 3 steps of ENN
    - Until no more observation is removed
    - A maximum number of cycles is reached

-  AIIKNN

   1. Trains 1-KNN on entire dataset
   2. Finds each observation's 1 closest neighbour
   3. Keeps or removes observation based on neighbours agreement with its class
   4. Repeats, but adding 1 K to the KNN, until
      - A maximum number of neighbours is examined
      - The majority class becomes minority
   
- Tomek Links
  - If 2 samples are nearest neighbors, and from a different class
  - Removes the Tomek Link from the majority class
  - **Strength** Remove noise
  - **Weakness** Miss classifies hard cases
  
- Neighbourhood Cleaning Rule (NCR)

  1. Trains a 3-KNN on entire dataset
  2. Finds each observation's 3 closest neighbours 
  3. Flag observations whose most neighbours disagree
  4. Find the 3 neighbours of each observation from the minority class
  5. If most neighbours disagree with the minority class, flag the neighbours

  **Except** Only flag a neighbour in step 5 if it belongs to a class with at least half as many observations as those in the minority

  6. Remove all the flagged observations
  
- Instance Hardness (Fixed)

  1. Train a ML algorithm
  2. Determine the instance hardness

  $$
  \textit{Perc} = (1 - \frac {n_\text{desired}} {n_\text{majority}})\times 100
  $$

  3. Remove observations with high instance hardness 

  - Filter with the same algorithm that I intend to train

### Retain observations in the boundary

- Condensed Nearest Neighbours (CNN)

  1. Get all minority samples in a set $\mathcal C$
  2. Add a sample from the targeted class (class to be under-sampled) in $\mathcal C$ and all other samples of this class in a set $\mathcal S$.
  3. Go through the set $\mathcal S$, sample by sample, and classify each sample using a 1 nearest neighbor rule.
  4. If the sample is misclassified, add it to $\mathcal C$, otherwise do nothing.
  5. Reiterate on $\mathcal S$ until there is no samples to be added.
  6. Repeat for other major classes

  - **Strength** Focus on harder cases
  - **Weakness** Introduces noise

- NearMiss (Fixed)

  - Calculate all distances between instances from majority class and those from minority class

    *Version 1*

    - Retains observations from majority class whose average distance to the K-Nearest observations of the minority class is the smallest

    *Version 2*

    - Retains observations from majority class whose average distance to the K-Farthest observations of the minority class is the smallest

    *Version 3*

    - For each observation in minority class, retains their K-Nearest observations of the majority class, then *Version 1*


### Both

- One Sided Selection 
  1. Create group $\mathcal S$ with all samples from minority
  2. Add 1 observation from the majority to $\mathcal S$ (at random)
  3. Train an 1-KNN on $\mathcal S$
  4. Make predictions on the rest of the majority class observations (together) 
  5. If predictions don't match, pass the samples to $\mathcal S$
  6. In $\mathcal S$, find and remove Tomek Links

### Notice

- Only RUS handles categorical variables out of the box, for all the rest we need to encode the variables first
- Cleaning methods rely on KNN (distance based), so aren't suitable for categorical  and discrete variables
- KNN based algorithms don't scale well, so we need to scale the dataset first

## Oversampling

### Random Over-sampling with smoothing 

1. Take the minority samples and determine their std for each variable
2. Extract a value randomly from $N \sim (0,1)$ for each variable $w$
3. Extract at random 1 observation $x$
4. Determine a shrinkage $\lambda$ e.g. 10
5. Obtain the new sample by $x_i = x_i + w_i \ast \lambda \ast std_i$ 

### Synthetic Minority Over-sampling Technique (SMOTE)

1. Isolate minority samples
2. Trains KNN and finds K Nearest Neighbours to each sample (ususally 5)
3. Determines how many new samples need to be generated
4. Selects randomly from which samples new samples will be generated
5. Selects randomly the neighbour that will be used to extrapolate the sample
6. FInds a random factor

$$
\textit {New sample} = \text {original sample} - \text {factor} \  \ast (\text {original sample} - \text {neighbour})
$$



7. Create the new samples

- SMOTE doesn't contemplate intra-class clusters

### SMOTE-NC

- Extends the functionality of SMOTE to categorical variables

1. Calculate std for each feature and extract the median $m$
2. When calculating the Euclidean distances to find the K neighbours, for categorical features, $x_1 == x_2 \ ?\  0 : m^2$

3. SMOTE
4. Generated values of categorical features are those shown by the majority of the neighbours

### SMOTE-N

- Only work with Nominal (Categorical) variables

- Value Difference Metric (VDM)
  - $N_{a,x}$ is the number of examples in the training set that have value $x$ for variable $a$
  - $N_{a,x, c}$ is the number of examples in the training set that have value $x$ for variable $a$ for a given class $c$
  - $C$ is the number of classes
  - $q$ is a constant, usually 1 or 2

$$
\textit {vdm}_a(x, y) = \displaystyle \sum _{c=1} ^C \Bigg| \frac {N_{a,x,c}} {N_{a,x}} - \frac {N_{a,y,c}} {N_{a,y}} \Bigg|^q = \displaystyle \sum _{c=1} ^C \bigg| P_{a,x,c} - P_{a,y,c} \bigg|^q
$$

- Utilize VDM to calculate distance in each feature
- SMOTE

### ADASYN

1. Determine the number ( $G$ ) of samples to generate 
2. Train KNN on entire dataset
3. Find K-Nearest Neighbours for each sample of minority class
4. Determine the weighting $r=\frac{D}{K}$, where D = neighbours from the majority class
5. Normalise $r$

$$
\mathcal r_\text{norm} = \frac r {\sum r}
$$

6. Calculate the number of synthetic examples that need to be generated for each observation of the minority class
   - $ g_i = r_i \times G $
7. For each minority class example $x_i$, generate $g_i$ synthetic samples

$$
\textit {New sample} = \text {minority sample} - \text {factor} \ast (\text {minority sample} - \text {neighbour})
$$

- The neighbour can be from the majority or the minority class

### Borderline SMOTE

1. Fits KNN to entire dataset
2. Ignores noise: Ignores samples from minority class whose neighbours are all from majority class
3. Ignores safe group: Ignores samples from minority class whose most neighbours are from minority class
4. Selects ==DANGER== group: Selects samples from minority class whose most neighbours are from majority class
5. Fits a KNN using minority class only (all samples)
   - Variant 1: interpolates synthetic samples as SMOTE, between the observations in the danger group and its neighbours from the minority class
   - Variant 2: interpolates synthetic samples as SMOTE, between the observations in the danger group and its neighbours from the majority class, but closer to the danger group

### SVM SMOTE

1. With a SVM find the support vectors from the minority class as templates

2. Train a KNN on entire dataset

3. For each sample in the templates:

   - If most neighbours from the majority class -> interpolation

   $$
   \textit {New sample} = \text {support vector} - \text {factor} \ast (\text {support vector} - \text {neighbour})
   $$

   - If most neighbours from the minority class -> extrapolation
     - The neighbours are selected from closest to farthest in order
     - Another KNN is trained only on the minority class

   $$
   \textit {New sample} = \text {support vector} + \text {factor} \ast (\text {support vector} - \text {neighbour})
   $$

   

### K-Means SMOTE 

> Or other clustering algorithms?



1. Finds clusters on the entire dataset
2. By default select those clusters where more than 50% of observations belong to minority
3. Calculate cluster weights for each selected cluster
   - Calculate Euclidean distance between all samples from minority
   - Calculate mean Euclidean distance (L2 mean)
   - $\textit {density} = ^{N_{observations}} / _{\text{L2 mean} \ \ast \ N_{features}}$
   - $\textit {Sparsity} = ^1 / _ {\text {density}}$
   - $\textit {Cluster Sparsity} = ^{\text {Sparsity}} / {\sum {\text {Sparsity}}}$

4. Calculate the number of synthetic examples that need to be generated for each cluster

$$
\mathcal {g_i} = cs_i \times G
$$

​		with $G$ the total numbers of samples to generate

5. SMOTE

## Ensemble Methods

<img src="../../../Desktop/Images/Ensemble approaches.png" alt="Ensemble approaches" style="zoom: 50%;" />

## Cost Sensitive Learning

> Assigns different costs to different classification errors

let $C(i,j) := \text {cost of assigning an observation of class j to i}$

### How to determine cost

**Heuristic**: Imblance ration, optimization

## Features

1. Fast
2. Flexible API
