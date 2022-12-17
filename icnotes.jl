### A Pluto.jl notebook ###
# v0.19.12

using Markdown
using InteractiveUtils

# ╔═╡ dbd717e2-2881-11ed-096a-5f8228b7f240
begin
	using PlutoUI
	using LinearAlgebra
	using Plots
	
	md"# Class Notes for CIS4526"
end

# ╔═╡ e8c3d4aa-35a3-4b6f-bf38-68a8b0cbe4de
PlutoUI.TableOfContents()

# ╔═╡ cf78886f-e116-4c98-98b2-1309f846f41b
md"# Chapter 1: Introduction of Machine Learning"

# ╔═╡ 7bca21d0-8d31-49a4-9c03-751ab2d9dcb0
md"## What is machine learning (ML)?"

# ╔═╡ 557f06f3-d19b-447d-acfe-b110a6c66541
md"### Machine Learning vs. AI"

# ╔═╡ 103d2317-10d9-4c51-bf3e-ad91d6fbc03c
md"""
**Remark.**

- A sub-field of AI

- AI ≈ ML + ML's applications (NLP, CV, Speech, etc.)
"""

# ╔═╡ cf5859c1-3ee3-4774-9f29-720f5153ef67
md"""
**Remark.**

| AI | Machine Learning |
|----|------------------|
| the goal | a pathway (to AI) |
| make computers behave in  ways that both mimic and go beyond human capabilities | learn patterns from existing data, make predictions on new data |
"""

# ╔═╡ 097dee19-1a1d-4fa6-8fbf-2a1410557a6e
md"### What is an example of AI that is not ML?"

# ╔═╡ 3c4a21c5-6e5d-4cbc-8219-78874b83666d
md"""
**Remark.**
**Expert Systems**

- basically a set number of "if this, then do that" statements.
  It does not learn by itself (so it is not machine learning), and it still can be very useful for use cases like medical diagnosis and treatment.

- This decision tree becomes AI once it is put into a computer.
"""

# ╔═╡ c44aa6bd-0037-4ba8-a8e7-794c7487b0a6
md"### Machine Learning vs. Data Science"

# ╔═╡ 4c677095-bd8e-43ad-a690-b9e1a6c0dc74
md"### Machine Learning vs. Human Learning"

# ╔═╡ 16435bc0-0b50-4e81-8d04-b7b07ba992ee
md"""
**Remark.**

- Human learning:

  Observations → Learning → Knowledge

- Machine learning:

  Data {(x,y)} → Algorithms → Patterns f(x) = y
"""

# ╔═╡ e7712c48-4679-4dab-8404-58152e292a93
md"### Three types of ML"

# ╔═╡ bf8305b5-1e28-463f-86bc-f92e6eecf8f5
md"""
**Remark.**

- supervised learning

- unsupervised learning

- reinforcement learning
"""

# ╔═╡ b439c8bd-a78d-49e2-91f2-0e3f23506b84
md"## What does an ML algorithm look like?"

# ╔═╡ 7a86d245-cf3b-4f41-a544-9e719927f436
md"""
**Remark.**

```python
def model(X, Y):
	train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2)
	model = LogisticRegression()
	trained_model = model.fit(train_X, train_Y)
	predicted_Y = trained_model.predict(test_X)

real_X = [[8.0, 2, 4.5, 1, 0],
		  [7.5, 1, 5.0, 1, 0],
		  [9.0, 2, 3.0, 0, 0],
		  [6.0, 3, 2.5, 1, 0],
		  [8.5, 2, 3.5, 0, 1],
		  [10.0, 1, 2.0, 1, 1],
		  [7.5, 3, 2.5, 1, 0],
		  [8.5, 2, 3.5, 0, 1],
		  [6.0, 2, 1.5, 1, 1],
		  [7.0, 1.0, 3.0, 1, 0]]

real_y = [1, 1, 0, 0, 1, 0, 1, 0, 0, 1]

predicted_Y = model(real_X, real_Y)
accuracy = compare(predicted_Y, real_Y)
```
"""

# ╔═╡ 3c6f81aa-2e6a-413a-850f-afb2995f2c92
md"## What can(not) ML do?"

# ╔═╡ f87e1a07-4a2d-46d3-9b9a-2b1b9f63ef0d
md"### What can ML do?"

# ╔═╡ c7063785-ea15-47ca-94ad-1299a821c966
md"### What can't ML do?"

# ╔═╡ 516360e4-e63f-4df7-a807-d60983ca5d59
md"""
**Remark.**

- To decide if an arbitrary program will halt

- Market analysis

- The task requires long chains of reasoning or complex planning that rely on common sense or background knowledge unknown to the computer

- Explaining its predictions

- Cleaning the data (e.g., restructuring data, free of biases and free of anomalies) in the first place so that it is valuable in a machine learning workflow
"""

# ╔═╡ 02fa00bb-6fc7-4187-a069-032b75b456e7
md"## Recent breakthroughs of ML"

# ╔═╡ 76bca31b-4489-4b23-b182-ddd7aaa75015
md"### AlphaGo"

# ╔═╡ 9a75a73e-15a3-4030-9b05-6722b690a46f
md"""
**Remark.**

- The first program that defeats a professional Go player

- **Reinforcement learning**: machine learning models make a sequence of decisions in an environment in order to maximize the cumulative reward
"""

# ╔═╡ 98f46c15-d796-4e9a-a529-4c002141eeda
md"### AlphaFold"

# ╔═╡ a20021cb-11f9-45a4-be9c-f97065732d8d
md"""
**Remark.**

- A deep learning system that performs predictions of protein structure

- **Deep learning**: trained the program on over 170,000 proteins from a public repository of protein sequences and structures.
"""

# ╔═╡ da29a50c-d22a-437d-8af5-8e0e36f70b94
md"### GPT-3"

# ╔═╡ d65cb3bc-635f-4035-88aa-a0d2829ca80b
md"""
**Remark.**

- (Seemingly) world knowledge

- **Self-supervised learning**: an autoregressive language model (175 billion parameters) that uses deep learning to produce human-like text.
"""

# ╔═╡ 336f7664-44ec-418d-9667-4529862c0aff
md"### DALL⋅E"

# ╔═╡ 40af75c4-05f4-4431-b604-f69a76dff8a2
md"""
**Remark.**

- Draw realistic images and art based on a text description you provide

- **Self-supervised learning**: DALL⋅E is a 12-billion parameter version of GPT-3 trained to generate images from text descriptions, using a dataset of text--image pairs
"""

# ╔═╡ 5c3e598f-800d-4459-a5d7-19c6f9633565
md"### The main facilitators of recent breakthroughs"

# ╔═╡ 26afc9f2-6fb1-4510-97ee-569990ef8f77
md"""
**Remark.**

- Deep learning

- Big data
"""

# ╔═╡ 463aac99-a9eb-4243-851a-f9b16ee1ada1
md"## When to use ML?"

# ╔═╡ 64945175-aa24-4626-bad6-58e8252449a9
md"""
**Remark.**

- A pattern exists

- We cannot pin it down mathematically

- We have data on it
"""

# ╔═╡ ee27fa04-be91-44c0-8e46-9ec2d087a4d9
md"### When to use Deep Learning (DL)"

# ╔═╡ 6579bb80-5776-4633-ae88-da3ecbd68ea5
md"""
**Remark.**

- A pattern exists, but we do not know or we are lazy to manually define them

- We have (not small) data on it

- We have computing resource for it, e.g., GPU
"""

# ╔═╡ 3b28b888-1731-4570-a4b6-b9a47741ee78
md"# Chapter 2: Data Preparation"

# ╔═╡ e468c69c-5c71-4dd3-8d76-03b90e9db7cd
md"## Define Problem"

# ╔═╡ bd5f5fd5-4843-4fe0-b653-f9ee95a3d3a8
md"""
**Remark.**
Is it classification or regression?

- ``f(x) → \{A,B,C\}``

- ``f(x) → [-1,1]``
"""

# ╔═╡ 44cc65e9-b141-4d1b-bc2b-49143c55640a
md"""
**Remark.**
What **features** should be helpful to this problem?
"""

# ╔═╡ 16c36830-7dc7-42fb-867f-428e574464df
md"## Data Collection"

# ╔═╡ 7fbd1f91-eb56-40f0-9f97-a340e45d1edc
md"### Collecting data automatically"

# ╔═╡ bae5ff75-0b84-4621-b6e6-6d3e6259a152
md"""
**Remark.**

- Data means {(input, output)}

- You know the output is available

- You can write computer programs to crawl/extract
"""

# ╔═╡ dc7e9f55-2af4-4137-9815-942a1e7543a3
md"""
**Remark.**
**Common issues:**

- Most AI tasks cannot access data automatically; or

- Most AI tasks cannot access large-scale dataset automatically

- Most AI tasks cannot access large-scale & clean dataset automatically

!!! note "Then we need Human Annotation"
"""

# ╔═╡ 047b5cb4-07c2-45b2-930d-c14560847037
md"### Human annotated data"

# ╔═╡ e82d83b9-4e75-4e87-b491-2c8956529be7
md"""
**Remark.**

- Spend money on hiring people to annotate data

- High-quality data, but quite expensive
"""

# ╔═╡ 69fa269a-adca-435c-8806-1909a771f136
md"## Data Preprocessing"

# ╔═╡ ab86c39c-72ec-47c4-8f28-ea33eeeb6f99
md"### Preprocessing in NLP"

# ╔═╡ 6d2b12c4-91a6-4331-9d6b-5bbdb930c6b9
md"### Common toolkits in NLP Preprocessing"

# ╔═╡ 73e8b017-c3a6-4a5b-bf9c-0231509050b0
md"""
**Remark.**

NLTK:

- Pros:

  - Many third-party extensions

  - Fast sentence tokenization

  - Support the largest number of languages compared to other libraries

- Cons:

  - Does not provide neural network models

  - Slow in general

  - No integrated word embeddings

spaCy:

- Pros:

  - The fastest NLP library

  - Easy to learn and use

  - Object-oriented representation of the results

  - Use neural networks for training some models

  - Provide built-in word embeddings

- Cons:

  - The language size supported is limited

  - Sentence tokenization is slower than NLTK
"""

# ╔═╡ 6e242ba3-22d7-4e9f-8851-e3d6c9d4a094
md"## Data Split (train-dev-test)"

# ╔═╡ 663c3c47-1f41-4842-9214-7ef4c926157e
md"""
!!! info "Training set"

	- The largest set (e.g., 70%)

	- Training the model

!!! warning "Development set"

	- The smallest set (e.g., 10%)

	- Check the model performance quickly, help find the best model setting and model

!!! tip "Test set"

	- The evaluation set (e.g., 20%)

	- Check the performance of the final model
"""

# ╔═╡ b552b2b8-2e8c-4777-8514-9cb233a3c8cf
md"## Estimation of Baseline/Human Performance"

# ╔═╡ b9c2e732-51e5-4658-ace6-bf3a84644229
md"## Advanced Topic: Data Augmentation"

# ╔═╡ 4adc46a2-5102-416d-bb48-26e9cf89daf6
md"# Chapter 3: Input representations -- human-defined features"

# ╔═╡ 41da3074-fa8d-41b8-8a52-620282cd0b36
md"## What is Feature"

# ╔═╡ fd189d77-d09c-404c-930b-c304a72a94fd
md"""
**Definition.**
A **feature** is any measurable input that can be used in a predictive model---basically you can use anything that can describe the input.
"""

# ╔═╡ 4fe5d69c-9893-4484-b14e-c8056d21df3a
md"""
**Remark.**
Features are also sometimes referred to as **variables** or **attributes**.
"""

# ╔═╡ 8f9f4b40-33a9-48b6-9f18-8afd5a6b5ccb
md"""
**Remark.**
In datasets, features appear as columns.
"""

# ╔═╡ 1bba3acd-b301-44f1-a013-6bcb94e41a41
md"### Digitize Non-numeric Feature Values"

# ╔═╡ 8555d610-41e9-472f-83fe-d3c3a296209c
md"""
**Remark.**
You can digitize non-numeric features by:

- Defining some measurable rules.
  E.g., "Name" → ["has Mr", "has Mrs", "has middle name", etc.]

- Defining an option to an ID.
  E.g., {"male": [1,0], "female": [0,1]}.
"""

# ╔═╡ 71907b36-45d6-408d-a241-547a8dba925d
md"### Feature Scaling"

# ╔═╡ 30c3d0ad-825d-4026-a8f9-c287d1e69d7e
md"""
**Remark.**
Different scales lead to a chance that higher weights are given to features with higher magnitude (i.e., model bias).
"""

# ╔═╡ 8890c772-eb26-4c1b-888d-c5b1d72d5534
md"""
**Remark.**
Two mainstream scaling approaches:

- Normalization

- Standardization
"""

# ╔═╡ eef7c272-3ac7-4ca3-aa37-6d48079601f9
md"""
**Definition.**
**Normalization** is when values are scaled from 0 to 1.
The normalized value is given by the formula

$x_\text{norm} = \frac{x - \min(x)}{\max(x) - \min(x)}.$
"""

# ╔═╡ e67ef5c4-4f56-4078-b48c-a23b58e16bbd
md"""
**Definition.**
**Standardization** is when values are centered around the mean with a unit standard deviation.
The standardized value is given by the formula

$x' = \frac{x - \bar{x}}{σ}.$
"""

# ╔═╡ a9898af8-f664-4de1-932b-7fb843b59222
md"""
**Remark.**
In standardization, the values are not restricted to a particular range.
"""

# ╔═╡ dc32ace2-788b-4043-9a08-66517438978f
md"""
**Remark.**
Normalization is good for values that do not follow a Gaussian distribution.
Standardization is good since (1) it's less sensitive to outliers (2) it's helpful in cases where the data follows a aussian distribution (not 100% true)
"""

# ╔═╡ e4fc4f42-a2b8-4ed9-9e87-4e4fb9c3380a
md"## Feature Selection"

# ╔═╡ d30577d4-b522-4499-a5da-88e613f948fb
md"""
**Remark.**
You may define features based on an open-set of input descriptions, but not all features are useful for a target problem.
"""

# ╔═╡ 0225875c-eb1a-497d-ac2a-261b202f8be6
md"""
**Remark.**
Focus on features that are relevant to the problem and avoid features that are **non-informative** (e.g., "breakfast yesterday" → GPA) or **redundant**.
"""

# ╔═╡ f4fd53c9-c0da-4887-b23d-46e95f658cf0
md"""
**Definition.**
**Feature selection** is the process of reducing the number of features when developing a predictive model.
Benefits: it reduces the computing cost and improves the model performance.
"""

# ╔═╡ c35e5392-633b-40db-b8b0-3ec89ea5ae46
md"### Supervised Methods for Feature Selection"

# ╔═╡ d8a395bf-50fb-4978-a6bc-8460f05ae22b
md"""
**Definition.**
The **filter method** of feature selection uses independent techniques/tools (e.g., Chi-square Test, Fisher's Score) to select features by an evaluation criterion to assess the relevance degree (feature, output).
It is independent of subsequent learning algorithms.
"""

# ╔═╡ 07a28400-702d-491c-adea-9a0009a1320b
md"""
**Definition.**
The **wrapper method** of feature selection splits our data into subsets and trains a model using this.
Based on the output of the model, we add and subtract features and train the model again.
"""

# ╔═╡ a783a435-a444-4d10-a821-4c2c82fb8782
md"""
**Remark.**
The wrapper method is computationally expensive which makes it not very popular.
"""

# ╔═╡ fdc18ae1-66a0-4c14-ad0a-fe74f2bbe934
md"""
**Definition.**
The **embedded method** of feature selection combines the qualities of the filter and wrapper methods of feature selection.
"""

# ╔═╡ 4351fe34-0889-4e2b-a6fe-17274e9dd4d8
md"### Unsupervised Methods for Feature Selection"

# ╔═╡ ff1eb1a0-a3de-497e-acf7-ae99aa9a77b6
md"""
**Remark.**
One method of unsupervised feature selection is **removing features with low variance** where the variance of a value is defined by

$\text{var}(x_i) = \frac{1}{n} \sum_{j=1}^n (x_{ij} - \overline{x_i}).$
"""

# ╔═╡ eb86bed0-a3a3-4f0d-903f-4896ddda60b3
md"""
**Remark.**
Another method of unsupervised feature selection is **removing redundant features**.
For example, two highly correlated attributes would be `"is_male"` and `"is_female"`.
One of the columns can be removed.
"""

# ╔═╡ a5607a6d-d568-432b-ac62-5bbabb98f91f
md"## PCA: Principal Component Analysis"

# ╔═╡ 3270cedd-f10e-4eb1-9055-0d4e49e21117
md"""
**Definition.**
**Principal Component Analysis (PCA)** compresses data into lower dimensional representations (PCA finds the principal components of data).
"""

# ╔═╡ ce28b17c-65bd-4864-acaa-3f94a59b237c
md"""
**Definition.**
**Principal components** are the directions where there is the most variance; the angle that can distinguish the data points.
"""

# ╔═╡ a4d22fcf-e1e4-43e6-b910-05f8d90b8ba3
md"""
**Remark.**
PCA uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components.
"""

# ╔═╡ 8e3a92ac-4106-47b1-b5a3-2297e187b422
md"""
**Remark.**
We can use math to find the principal components.
The eigenvector with the highest eigenvalue is the principal component.
"""

# ╔═╡ 95c4e156-949e-4bdf-86a6-3777a3e6d902
md"""
**Example.**
Use math to find the eigenvectors and eigenvalues where

$Mv = λv, \quad M = \begin{bmatrix} -6 & 3 \\ 4 & 5 \end{bmatrix}, \quad v = \begin{bmatrix} 1 \\ 4 \end{bmatrix}, \quad λ = 6$

$Mv = \begin{bmatrix} -6 & 3 \\ 4 & 5 \end{bmatrix} \begin{bmatrix} 1 \\ 4 \end{bmatrix} = \begin{bmatrix} 6 \\ 24 \end{bmatrix}$

$λv = 6 \begin{bmatrix} 1 \\ 4 \end{bmatrix} = \begin{bmatrix} 6 \\ 24 \end{bmatrix}$

First, compute the eigenvalues via ``Mv = λv = λIv \implies Mv - λIv = 0 \implies |M - λI| = 0``.

$\begin{align*}
\left|\begin{bmatrix} -6 & 3 \\ 4 & 5 \end{bmatrix} - λ\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}\right| &= 0 \tag{1} \\
\begin{vmatrix} -6 - λ & 3 \\ 4 & 5 - λ \end{vmatrix} &= 0 \tag{2} \\
(-6 - λ)(5 - λ) - (3)(4) &= 0 \tag{3} \\
λ^2 + λ - 42 &= 0 \tag{4} \\
λ &= -7 \text{ or } 6 \tag{5}
\end{align*}$

Second, compute the eigenvector

$\begin{bmatrix} -6 & 3 \\ 4 & 5 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = 6 \begin{bmatrix} x \\ y \end{bmatrix} \tag{1}$

$\begin{align*}
-6x + 3y &= 6x \\
4x + 5y &= 6y
\end{align*} \tag{2}$

$\begin{align*}
-12x + 3y &= 0 \\
4x - 1y &= 0
\end{align*} \tag{3}$

$v = \begin{bmatrix} 1 \\ 4 \end{bmatrix}$
"""

# ╔═╡ 5e2a016f-d794-48e7-b6d5-6d44a91e235b
# Implement PCA in code
function pca(X)
	n, m = size(X)
	C = (X' * X) ./ (n - 1)

	eigenvals, eigenvecs = eigen(C)

	pca = X * eigenvecs

	return pca
end

# ╔═╡ 51b488a9-4a86-4961-8455-d2ef27dbad19
md"## SVD: Singular Value Decomposition"

# ╔═╡ 9e8537b9-5816-4d96-bbe5-070a24079dc7
md"""
**Definition.**
**Singular Value Decomposition (SVD)** computes the rank matrix which best approximates the original matrix and break it up into three pieces: 

1. A rotation about the origin

2. A rescaling of each coordinate

3. Followed by another rotation about the origin.
"""

# ╔═╡ c0893a95-c170-486f-97cf-140844425526
md"""
**Remark.**
Orthogonal matrix is a rotation, and a diagonal matrix is a rescaling of each coordinate.
"""

# ╔═╡ af921af7-858e-4ec4-81c7-a603bc514c87
md"""
**Remark.**
SVD of a matrix ``M`` is computed via

$M = U Σ V^T.$
"""

# ╔═╡ c5ae5f8f-e282-47e1-8b56-1696c236ed6f
let
	movieratings = [2 5 3
					1 2 1
					4 1 1
					3 5 2
					5 3 1
					4 5 5
					2 4 2
					2 2 5]
	
	svd(movieratings)
end

# ╔═╡ c7d9411a-d867-42b5-bbba-8a05ba9752d4
md"""
**Definition.**
A **singular value** represents how much of the data was captured by the singular vector.
"""

# ╔═╡ f5079220-98f7-4f9b-aacc-135ca7af1144
md"""
**Remark.**
In the SVD example, we see that the first singular vector covers a large part of the structure of the matrix; i.e., a rank-1 matrix would be a pretty good approximation to the whole thing.
"""

# ╔═╡ 041814c2-cfa8-4d2b-bf70-569275bf2ac5
md"### Truncated SVD"

# ╔═╡ cf2bd0d3-52be-4884-b208-da31fc810154
md"""
**Remark.**
Given ``U_r Σ_r`` is the transformed training set with ``r`` features,

$M ≈ M_r = U_r Σ_r V_r^T$

To transform a test set ``X``:

$X_\text{approx} = XV_r$
"""

# ╔═╡ 79eb1d7f-e404-4fa0-bc0c-a85c7100fc93
md"### SVD vs. PCA"

# ╔═╡ 12b39d55-c851-469b-926d-c91c2be80bdd
md"""
**Remark.**
SVD is similar to PCA, but more general.
PCA assumes that input square matrix, SVD doesn't have this assumption.
"""

# ╔═╡ 9a063c8f-7a98-4b31-9b06-5ed9dcf2199e
md"""
**Remark.**
PCA maps the data to lower dimensional.
In order for PCA to do that it should calculate and rank the importance of features/dimensions.
There are two ways to do so:

1. Using eigenvalue and eigenvector in covariance matrix to calculate and rank the importance of features.

2. Using SVD on covariance matrix to calculate and rank the importance of the features SVD (covariance matrix) = [U Σ V']
"""

# ╔═╡ 89d3ef5c-514b-45bb-b57a-6d1299cec274
md"# Chapter 4: Algorithms--Bayesian classifier"

# ╔═╡ 88540572-648b-41f6-ae30-da5afbdf6b8f
md"## What is Probability"

# ╔═╡ 19ceccca-86f9-414f-b72a-3ed99210d2d2
md"### Probability by Frequentist"

# ╔═╡ ceb891a6-ad25-4fcb-b151-c78afc58fe11
md"""
**Definition.**
**Probability** is the frequency of an event occurring in a large (infinite) number of trials.
"""

# ╔═╡ 12008ff8-267b-44a3-a1de-382e51675310
md"""
**Remark.**
Basically, a Frequentist method takes predictions on the underlying truths of the experiment using only data from the current experiment.
"""

# ╔═╡ 0d3aabb0-19ee-472f-83cb-dc849efccff4
md"### Probability by Bayesian"

# ╔═╡ 6a7b2762-a48a-44c6-866d-41a13f129c3d
md"""
**Example.**
Whether the Arctic ice cap will have disappeared by the end of the century? (Events that cannot be repeated)

Initial opinion → new evidence → revise our opinion
"""

# ╔═╡ 4aecc90a-75b1-4df2-8eca-ce2d73d77d61
md"""
**Definition.**
"**Probability** is orderly opinion, and that inference from data is nothing other than the revision of such opinion in the light of relevant new information." --- Eliezer Yudkowsky
"""

# ╔═╡ 4740ecda-4a11-4fc3-8ff5-f57b435609a1
md"""
**Remark.**
Probabilities provide a quantification of belief; updating your beliefs in light of new evidence.
"""

# ╔═╡ 39e7ca7b-1da3-4630-ad1d-1cd6c1a4126e
md"### Why need prior knowledge"

# ╔═╡ 31593502-b853-4ffa-95c0-0e25e92a23d3
md"""
**Remark.**
The Bayesian concept of probability is also more conditional.
It uses prior and posterior knowledge as well as current experiment data to predict outcomes.
Since life doesn’t happen in a vacuum, we often have to make assumptions when running experiments.
Thus, the Bayesian approach attempts to account for previous learnings and current data that could influence the end results.
"""

# ╔═╡ bb9b8343-255a-4b42-a7fa-76d9af26ac30
md"### Frequentist vs. Bayesian"

# ╔═╡ c06c6dec-b508-4a27-8f0c-2642b6fa18fd
md"""
**Remark.**
Frequentists:

- Only repeatable random events have probabilities

- Probability = frequency

- Frequentists don't attach probabilities to hypotheses or to any fixed but unknown values in general

- No prior beliefs

Bayesian:

- Infer probability for events that have never occurred or believe which are not directly observed

- Use probabilities to represent the uncertainty in any event or hypothesis

- Prior beliefs are specified as prior probability
"""

# ╔═╡ 1c3e187f-97c9-4969-bc05-bf90bc35fab8
md"### Basics of Probability"

# ╔═╡ 6acc2d90-1d95-4ba4-895d-6aea53ff7c7e
md"""
**Remark.**

- Probability of ``A``: ``P(A)``

- ``P`` must satisfy three axioms:

  1. ``1 ≥ P(A) ≥ 0`` for every ``A``

  2. ``P(Ω) = 1``, ``Ω`` is the sample space

  3. If ``A_1, A_2, …`` are disjoint then

     $P\left(\bigcup_{i=1}^∞ A_i\right) = \sum_{i=1}^∞ P(A_i)$
"""

# ╔═╡ 9671a7a3-fb78-40e8-85d0-7768ce5a3c05
md"## Joint, Marginal, Conditional Probabilities"

# ╔═╡ b7fef3a4-a9d2-4163-b6f2-e465f606df60
md"""
**Remark.**

- Joint Probability:
  ``P(X,Y)``, Probability of ``X`` and ``Y``.

- Marginal Probability:
  ``P(X) = \sum_Y P(X,Y)``, Probability distribution of a single variable in a joint distribution.

- Conditional Probability:
  ``P(X∣Y)``, Probability of ``X`` given ``Y``.
"""

# ╔═╡ b9932f67-9845-4a56-aefd-5733bcc0ee44
md"### Joint Probability"

# ╔═╡ eae01ecb-5424-4d3d-8a13-29d2d06fa2b6
md"""
**Example.**
Joint probability: ``P(X = \text{minivan}, Y = \text{European}) = 0.1481``.
"""

# ╔═╡ 0e671da4-9a5d-4424-b7a4-b81ab759e4ef
md"### Marginal Probability"

# ╔═╡ add4e352-10a4-4af5-8624-63a6271a5674
md"""
**Example.**
Marginal probability: ``P(X = \text{minivan}) = 0.0741 + 0.1111 + 0.1481 = 0.3333``.
"""

# ╔═╡ c352e06a-af14-48bc-89c7-b308bf183fbd
md"### Conditional Probability"

# ╔═╡ 664eee50-926f-44ce-955b-fa6664f1485b
md"""
**Example.**
Conditional probability: ``P(Y = \text{European} ∣ X = \text{minivan}) =`` ``0.1481 / (0.0741 + 0.1111 + 0.1481) = 0.4433``.
"""

# ╔═╡ 80525897-e876-4ab2-8f0e-b8b88f464a58
md"""
### Relationship between Joint, Conditional, Marginal Probabilities

$\underbrace{P(A,B)}_\text{Joint} = \underbrace{P(A∣B)}_\text{Conditional} \;\; \underbrace{P(B)}_\text{Marginal}$
"""

# ╔═╡ 546e0b09-4b1a-426d-8a17-fc30061d9854
md"""
### Example

- 60% of ML students pass the final ``(P(A))`` and 45% of ML students pass both the final and the midterm ``(P(A,B))``

- Question: What percent of students who passed the final also passed the midterm ``(P(B∣A))``?

  $\begin{align*}
  P(B∣A) &= \frac{P(A,B)}{P(A)} \\
  &= \frac{0.45}{0.6} \\
  &= 0.75
  \end{align*}$
"""

# ╔═╡ f7f4e5a5-0c5f-4ddb-9968-fb8be8cad1ab
md"""
## Bayes' Rule

$P(A ∣ B) = \frac{P(A ∩ B)}{P(B)} = P(A) \frac{P(B ∣ A)}{P(B)} = P(A) \frac{P(B ∣ A)}{P(B ∣ A) + P(B ∣ \overline{A})}$

$\underbrace{P(\text{hypothesis} ∣ \text{evidence})}_\text{Posterior probability} = \underbrace{P(\text{hypothesis})}_\text{Prior probability} \underbrace{\frac{P(\text{evidence} ∣ \text{hypothesis})}{P(\text{evidence})}}_\text{How special is this evidence in the hypothesis space}$
"""

# ╔═╡ 126ebfb3-4cfc-4bc0-bbd4-e1cab157fcda
md"### An Example to Understand Bayes' Rule"

# ╔═╡ 39350f3e-62d6-4616-946c-5391e6ab1490
md"""
**Remark.**

1. Let's estimate how likely a random person has cancer, i.e., **prior probability**.

2. **Evidence**: now, we notice a person has fever

3. **Question**: how likely this person has cancer, i.e., the updated probability
"""

# ╔═╡ 320987dc-51c4-450f-a9f7-fb3af8100b88
md"### An Example to Understand Bayes' Rule"

# ╔═╡ 300d0aa0-f187-4bb7-b36e-df13ee4be5e8
md"""
**Example.**
Suppose we have a pink disc with a circle around it.
Consider the pink disc as "the cancer space" and the ring around it "the world".
Consider a slice of the world space, including the cancer space, as "the fever space"

Three cases:

1. The fever probability in cancer is smaller than the fever probability in the world, so fever is less special in cancer; then updated probability < prior probability.

2. The fever probability in cancer equals to the fever probability in the world, so fever is equally special in cancer; then updated probability == prior probability.

3. The fever probability in cancer is greater than the fever probability in the world, so fever is more special in cancer; then updated probability > prior probability.
"""

# ╔═╡ d6584658-a630-4095-bfd9-1905a0c14e0e
md"### Bayes' Rule"

# ╔═╡ 86976901-429f-4c87-81d2-e98272868b9c
md"""
**Remark.**

$P(A ∣ B) = P(A) \frac{P(B ∣ A)}{P(B)}$

$\overbrace{P(\text{hypothesis} ∣ \text{evidence})}^\text{Posterior probability} = \overbrace{P(\text{hypothesis})}^\text{Prior probability} \frac{\overbrace{P(\text{evidence} ∣ \text{hypothesis})}^\text{likelihood}}{\underbrace{P(\text{evidence})}_\text{a constant number}}$

$\text{Posterior} ∝ \text{prior} × \text{likelihood}$
"""

# ╔═╡ 54771b10-6e45-49b3-adb5-a391b671a80b
md"## Naive Bayes classifier"

# ╔═╡ 09ec58f5-7f88-4558-89e9-61a50f592b96
md"""
**Remark.**
Naive Bayes assumes:

$P(X_1 … X_n ∣ Y) = ∏_i P(X_i ∣ Y)$

i.e., any ``X_i`` and ``X_j`` ``(i ≠ j)`` are conditionally independent given ``Y``.
"""

# ╔═╡ 57a9eba3-2e81-472e-aed3-d8a3563b0582
md"""
**Remark.**
The goal function: ``P(Y ∣ X_1, …, X_n)``

$↓ \quad\text{Based on Bayes' Rule}$

$\begin{align*}
P(Y ∣ X_1,…,X_n) &= P(Y) \frac{P(X_1, …, X_n ∣ Y)}{P(X_1, …, X_n)} \\\\
&↓ \quad \text{Based on ``conditional independence assumption''} \\\\
&∝ P(Y) ⋅ P(X_1 ∣ Y) ⋅ P(X_2 ∣ Y) ⋯ P(X_n ∣ Y)
\end{align*}$

Each probability, i.e., ``P(Y), P(X_1 ∣ Y), …, P(X_n ∣ Y)``, are what we need to learn from the data.
"""

# ╔═╡ 72bba976-8b69-419f-b8bd-37b3201e2cc5
md"### Naive Bayes Algorithm"

# ╔═╡ 4c366d03-ca66-4a14-8793-49d3e0bc6e99
md"""
**Remark.**
Train Naive Bayes (given data D={(X,Y)}):

- For each value ``y_k`` of the output ``Y``:

  - estimate ``π_k = P(Y = y_k)``.

- For each value ``x_{ijk}`` of each feature ``X_i``:

  - estimate ``θ_{ijk} = P(X_i = x_{ijk} ∣ Y = y_k)``

→ Classify ``(X^{new})``:

$\begin{align*}
Y^{new} &← \arg\max_{y_k} P(Y = y_k) \prod_i P(X_i^{new} ∣ Y = y_k) \\
Y^{new} &← \arg\max_{y_k} π_k ∏_i θ_{ijk}
\end{align*}$
"""

# ╔═╡ d0f2049f-0f75-4fb5-a8a3-0793944ffe80
md"### Training Naive Bayes Classifier Using Maximum Likelihood Estimate (MLE)"

# ╔═╡ b0ff0312-8600-4562-b2a6-071d5a01d60f
md"""
**Remark.**

- From the data ``D``, estimate the class priors

  - For each possible value of ``Y``, estimate ``P(Y = y_1), P(Y = y_2), …, P(Y = y_k)``

  - An MLE estimate:

    $π_k = P(Y = y_k) = \frac{\#D\{Y = y_k\}}{|D|}$

- From the data ``D``, estimate the conditional probabilities

  - If every ``X_i`` has values ``x_{i1}, x_{i2}, …, x_{ij}``, then

    $θ_{ijk} = P(X_i = x_{ij} ∣ Y = y_k) = \frac{\#D\{X_i = x_{ij} ∧ Y = y_k\}}{\#D\{Y = y_k\}}$
"""

# ╔═╡ 321bf029-e4ce-4cf6-80a6-653ab0df5720
md"### Issues of MLE"

# ╔═╡ ba63a81b-5065-4a91-bd71-adbe9d36ac5e
md"""
**Remark.**

- Issue #1: Usually features are not conditionally independent

  $P(X_1…X_d ∣ Y) ≠ ∏_iP(X_i∣Y)$

  Nonetheless, NB is the single most used classifier particularly when data is limited, works well

- Issue #2: Typically use MAP estimates instead of MLE since insufficient data may cause MLE to be zero.
"""

# ╔═╡ 7d2cbc23-0dff-4d4a-8c02-dd88a67f7941
md"### Insufficient Data for MLE"

# ╔═╡ 37e7c23b-65ea-4b81-9bc0-f33d470f3fb1
md"""
**Remark.**

- What if you never see a training instance where ``X_1 = a`` when ``Y = b``?

- Thus, no matter what the values that other features ``X_2,…,X_n`` take:

  $P(X_1 = a,X_2,…,X_n ∣ Y) = P(X_1 = a ∣ Y) ∏_{i=2}^n P(X_i ∣ Y) = 0$
"""

# ╔═╡ 4354a5cc-a55f-49b8-b51b-fd1924c573cc
md"### Solution--MAP (Maximum A Posteriori)"

# ╔═╡ b8927499-20e3-4b81-9be9-740379cccc63
md"""
**Remark.**

- MAP estimates by

  - Adding "virtual" data points of size ``m``, and

  - Assuming two priors: ``\hat{P}(Y = b)`` and ``\hat{P}(X_i = a, Y = b)``

    then

    $P(X_i = a ∣ Y = b) = \frac{\#D\{X_i = a, Y = b\} + m ⋅ \hat{P}(X_i = a, Y = b)}{\#D\{Y = b\} + m ⋅ \hat{P}(Y = b)}$
"""

# ╔═╡ c72a211e-ad7f-47a7-b0f4-aa69fa13ad92
md"### MLE vs. MAP"

# ╔═╡ c82d9096-01a6-4b5a-b529-32fe13965452
md"""
**Example.**
In 2018-19 season, Liverpool FC won 30 matches out of 38 matches in Premier league.
Having this data, we’d like to make a guess at the probability that Liverpool FC wins a match in the next season.

- MLE: only based on this data, 30/38=79%
- MAP: assume we know that Liverpool’s winning percentages for the past few seasons were around 50%.
  Is 79% still the best estimation? A value between 50% and 79% would be more realistic, considering the prior knowledge as well as the current data.
"""

# ╔═╡ ca9805d9-9357-490e-b9aa-4c48c920e57d
md"""
**Remark.**

- MLE:
  Choose value that maximizes the probability of observed data

  $\hat{θ}_{MLE}=\arg\max_θ P(D ∣ θ)$

- MAP:
  Choose value that is most probable given observed data and prior belief

  $\begin{align*}
  \hat{θ}_{MAP} &= \arg\max_θP(θ∣D) \\
  &= \arg\max_θ P(D∣θ) P(θ)
  \end{align*}$
"""

# ╔═╡ 9fa42e55-c9c3-4f74-9ac6-f2ba75d4753a
md"# Chapter 5: Algorithms--Logistic Regression"

# ╔═╡ 630a1c82-e3a4-4d39-b864-06d2665228a7
md"## Linear Regression"

# ╔═╡ 1afdb2d4-aa31-454c-ab2c-527d95f16cbd
md"""
**Example.**
Task:

- House square feet (``X``) predicts house price (``Y``)

- A particular data point: ``y = α + βx + ε``

- A line that fits all data points: ``y = w_0 + w_1 x``.
  The goal is to find ``w_0`` and ``w_1`` that fits this data the best.
"""

# ╔═╡ f7a3ddea-6a57-48eb-99f5-1d09f048eeaf
md"""
**Example.**
House has more features than "square feet".

$y(𝐱, 𝐰) = w_0 + w_1 x_1 + … + w_D x_D$

- ``w_0`` is called the "offset" or "bias" or "intercept".

- ``w_1, …, w_D`` are the importance (i.e., "weight" or "coefficient") of each feature.
"""

# ╔═╡ abbf9629-43ad-411d-a7c8-82c8ee38080d
md"""
**Remark.**
When there are multiple features, linear regression is defined as

$\begin{align*}
y(𝐱,𝐰) &= w_0 + w_1 x_1 + … + w_D x_D \\
&= w_0 + \sum_{i=1}^D w_i x_i \\
&= w_0 + 𝐰^T 𝐱
\end{align*}$

where ``𝐰`` is the weight vector and ``𝐱`` is the feature vector and ``y(𝐱, 𝐰)`` can be any continuous value.
"""

# ╔═╡ 671b5fe3-28a4-4cf3-9250-6981c16b1d60
md"### Train Linear Regression--Ordinary Least Squares (OLS)"

# ╔═╡ ff9fc83e-ee89-4a8e-9026-44f01b5059e5
md"""
**Remark.**
The OLS method aims to minimize the sum of square differences between the observed and predicted values.

$\hat{𝐰} = \arg\min_𝐰 \sum_{i=1}^n (y_i - \hat{y}_i)^2$
"""

# ╔═╡ 710e7107-c5f7-4d62-ad33-7660c1843430
md"""
**Remark.**
Let

$𝐗 = \begin{bmatrix} X_{11} & X_{12} & ⋯ & X_{1p} \\ X_{21} & X_{22} & ⋯ & X_{2p} \\ ⋮ & ⋮ & ⋱ & ⋮ \\ X_{n1} & X_{n2} & ⋯ & X_{np} \end{bmatrix}, \quad β = \begin{bmatrix} β_1 \\ β_2 \\ ⋮ \\ β_p \end{bmatrix}, \quad 𝐘 = \begin{bmatrix} y_1 \\ y_2 \\ ⋮ \\ y_n \end{bmatrix}$

We hope:

$\begin{align*}
𝐗 β &= 𝐘 \\
&↓ \\
𝐗^T 𝐗 β &= 𝐗^T 𝐘 \\
&↓ \\
β &= (𝐗^T 𝐗)^{-1} 𝐗^T 𝐘
\end{align*}$
"""

# ╔═╡ 080556d0-abcd-4ba7-b013-c8891b180bf6
md"### Why Linear Regression"

# ╔═╡ 36054985-53c3-490d-9c49-df0cc9346f7f
md"""
**Remark.**
Linear regression models are relatively simple and provide an easy-to-interpret mathematical formula that can generate predictions.
"""

# ╔═╡ 00608dba-4ca5-4e55-8b29-fd867a307eb3
md"""
**Remark.**
Regression analysis tells you what features are statistically significant and which are not.
"""

# ╔═╡ ce7bc9c4-225d-4218-b55f-17771239e366
md"## Binary Logistic Regression"

# ╔═╡ 5f41a7c7-1ec1-46c0-adc7-e3df554d661d
md"### Linear Regression for Binary Data"

# ╔═╡ 811b5dc7-8fff-43d4-bc42-70cf46b3e84e
md"""
**Example.**

- The input image is Dog or Not

- The restaurant review is Positive or Not

- The patient has a disease, Yes or Not

- The two sentences have the same meaning? Paraphrase or Not

- …
"""

# ╔═╡ af5977b0-be59-49f4-bfdb-0fa061f1c79e
md"### Binary Logistic Regression"

# ╔═╡ c56ef9ed-3495-41e8-ad21-2dc7a534609d
md"""
**Remark.**
For binary classification task, we want the class (Yes/No) probability to be in a range [0,1], so we add a nonlinear function ``f()`` on the linear regression:

$P(Y = 1 ∣ 𝐱) = f(𝐰^T 𝐱 + w_0)$

We often make ``f() = σ()``, i.e., sigmoid function:

$P(Y = 1 ∣ 𝐱) = σ(𝐰^T 𝐱 + w_0)$
"""

# ╔═╡ 9a6beed0-cc54-4b10-9118-4f01555871ba
md"""
### Sigmoid Function σ()

$σ(x) = \frac{e^x}{e^x + 1} = \frac{1}{1 + e^{-x}}$
"""

# ╔═╡ b68c461d-debe-43b4-aced-fb5732a359e3
# Sigmoid Function
function logit(x)
	exp.(x) ./ (1 .+ exp.(x))
end

# ╔═╡ fc5721b9-eb8e-446d-9d46-a4737107ae03
let
	x = range(-4,4,100)
	y = logit(x)
	plot(x,y,legend=false)
end

# ╔═╡ a9267d45-09e8-41c6-9caa-510e22fc1db1
md"""
**Remark.**
The **standard logistic function** has another name: sigmoid function,

$σ(x) = \frac{e^x}{e^x + 1} = \frac{1}{1 + e^{-x}}$

Logistic function:

$f(x) = \frac{L}{1 + e^{-k(x - x_0)}}$
"""

# ╔═╡ 768d0c09-b6a3-47c2-9963-6f3392fd69e5
md"""
**Example (Binary Logistic Regression).**
Suppose we have a set of features ``𝐱 = \{x_0, x_1, x_2, …, x_n\}`` and corresponding weights ``𝐰 = \{w_0, w_1, w_2, …, w_n\}``.
We take the sum of each feature-weight product and define it as

$s = \sum_{i = 0}^n x_i w_i.$

The output is the activation function (e.g., sigmoid function) evaluated at ``s``:

$y = σ(s).$
"""

# ╔═╡ 337f5b10-fd08-4692-b7ce-30aadad377ca
md"## Multi-class Logistic Regression"

# ╔═╡ 3cca0baf-2988-4718-b9b5-b23974e1f609
md"""
**Remark.**
Basic idea:

- Find ``K`` scoring function: ``s_1,s_2,…,s_k``

- Classify ``x`` to class ``y = \arg\underset{i}{\max} s_i(x)``
"""

# ╔═╡ eef43fc0-52a6-42c8-9c84-c2792d3a63d4
md"""
**Remark.**
To get a scalar in linear regression we use

$f(x) = 𝐰^T 𝐱$
"""

# ╔═╡ 8c7d8b1c-1f86-4bbe-ad37-aa165f05bd8d
md"""
**Remark.**
To convert scalar to probability in binary logistic regression we use

$P(Y = 1 ∣ 𝐱) = \frac{e^{𝐰^T 𝐱}}{e^{𝐰^T 𝐱} + 1}$
"""

# ╔═╡ ef4bb6eb-1d42-41da-b580-7719bcea5353
md"""
**Remark.**
To convert scalar to probability in ``K``-class logistic regression we use

$P(Y = y_k ∣ 𝐱) = \frac{e^{𝐰^T_k 𝐱}}{\sum_{i=1}^K e^{𝐰_i^T 𝐱}}$

known as a **softmax function** in the form of

$\frac{e^a}{\sum_i e^i}$

where ``𝐗`` represents each data points in a ``N × D`` matrix given each data point has ``D`` features and ``𝐖`` represents a ``K × D`` weight matrix given each class has a weight vector of length ``D``.
"""

# ╔═╡ b720ca3d-bc73-4a8e-967d-7af7d8d490f0
md"### Example: 3-class Logistic Regression with 3 inputs"

# ╔═╡ d691a929-b1bb-4591-8d23-5f7b79619d98
md"""
**Example.**
Suppose we have a set of inputs ``𝐱 = \{x_1,x_2,x_3\}``, a matrix of weights,

$W = \begin{bmatrix} W_1 \\ W_2 \\ W_3 \end{bmatrix} = \begin{bmatrix} W_{1,1} & W_{1,2} & W_{1,3} \\ W_{2,1} & W_{2,2} & W_{2,3} \\ W_{3,1} & W_{3,2} & W_{3,3} \end{bmatrix}$

a set of intercepts ``𝐛 = \{+b_1, +b_2, +b_3\}`` which, when all used in 3-class Logistic Regression, give us some output ``𝐲 = \{y_1, y_2, y_3\}``.

Let ``𝐚 = W^T 𝐱 + 𝐛``, then,

$𝐲 = \text{softmax}(𝐚) \quad\text{ where }\quad y_i = \frac{\exp(a_i)}{\sum_{j = 1}^3 \exp(a_j)}.$

The model computes

$\begin{bmatrix}
y_1 \\ y_2 \\ y_3
\end{bmatrix} = \text{softmax}\begin{pmatrix} 
W_{1,1} x_1 + W_{1,2} x_2 + W_{1,3} x_3 + b_1 \\
W_{2,1} x_1 + W_{2,2} x_2 + W_{2,3} x_3 + b_2 \\
W_{3,1} x_1 + W_{3,2} x_2 + W_{3,3} x_3 + b_3
\end{pmatrix}$

Suppose we have some input image of a cat and are trying to classify it using 3-class (cat, dog, ship) Logistic Regression.
In Logistic Regression, the image will be stretched into a single column, ``𝐱``, which will be used to compute the output ``f(x_i; W,𝐛) = W^T 𝐱 + 𝐛``.
We use softmax to convert the output vector into probabilities.
"""

# ╔═╡ d49a81d1-e01d-4c4e-82d6-17dfbe3caa71
md"### Binary Classification: Binary LR vs. Multi-class LR"

# ╔═╡ fdd1a376-63e9-4c1f-b028-26027dc6dff5
md"""
**Remark.**

|   | Binary LR | 2-Class LR |
|---|-----------|------------|
| #scoring function | 1 | 2 |
| #weights | 1×D | 2×D |
| how to get probability | sigmoid function | softmax function |
| performance | worse | better |
"""

# ╔═╡ 50f09ea6-397d-4618-9643-c886064da6b1
md"## Logistic Regression as Dot-Product"

# ╔═╡ 5e919682-a372-41b4-aff3-8c8a0c85b5fd
md"""
**Remark.**

$\begin{bmatrix}y_1\\y_2\\y_3\end{bmatrix}=\text{softmax}\left(\begin{bmatrix}W_{1,1}&W_{1,2}&W_{1,3}\\W_{2,1}&W_{2,2}&W_{2,3}\\W_{3,1}&W_{3,2}&W_{3,3}\end{bmatrix} ⋅ \begin{bmatrix}x_1\\x_2\\x_3\end{bmatrix}+\begin{bmatrix}b_1\\b_2\\b_3\end{bmatrix}\right)$

The ``W_{i,}`` rows can be treated as feature vectors of classes.
We get similarity through dot product of input

$\begin{bmatrix}x_1 & x_2 & x_3\end{bmatrix}$

with ``W_{i,}`` rows.
"""

# ╔═╡ f8b1b31e-4816-4009-900f-e95d92e236bc
md"# Chapter 6: Algorithms -- Support Vector Machines"

# ╔═╡ 458e9f6a-2a69-4a42-afc6-d8c954b07957
md"## What is Hyperplane"

# ╔═╡ 149293eb-5e66-45e3-9233-eacb111b8e1d
md"""
**Remark.**
A hyperplane is ``p`` dimensions is a flat affine subspace of dimension ``p - 1``.
"""

# ╔═╡ 2282b447-6e2b-4a6d-a44f-057b5cd6a1d5
md"""
**Remark.**
In general the equation for a hyperplane has the form

$β_0 + β_1 X_1 + β_2 X_2 + … + β_p X_p = 0$
"""

# ╔═╡ 8259ee1f-e1e5-4e72-b5a3-c687fd70c9ab
md"""
**Remark.**
In ``p = 2`` dimensions a hyperplane is a line.
"""

# ╔═╡ 1ecf44fa-f0ed-4c22-aea2-35b8150b0ac6
md"""
**Remark.**
If ``β_0 = 0``, the hyperplane goes through the origin, otherwise not.
"""

# ╔═╡ ebf68f55-afd9-44a7-bc42-984afc8f5cfd
md"""
**Remark.**
The vector ``β = (β_1, β_2, …, β_p)`` is called the normal vector -- it points in a direction orthogonal to the surface of a hyperplane.
"""

# ╔═╡ e3afa121-9899-40d4-ad5d-669c4b6b069d
md"### Hyperplane in 2 Dimensions"

# ╔═╡ 009ed35a-49ac-49e2-98cb-e722fcf624f4
md"""
**Example.**
Let ``β_1 = 0.8`` and ``β_2 = 0.6``.
Consider the following parallel hyperplanes:

$β_1 X_1 + β_2 X_2 - 6 = 0$

$β_1 X_1 + β_2 X_2 - 6 = -4$

$β_1 X_1 + β_2 X_2 - 6 = 1.6$

The normal vector ``β = (β_1, β_2)`` is orthogonal to each hyperplane.
"""

# ╔═╡ b666903a-6dea-4a58-9539-04635afd0e9e
md"### Separating Hyperplane"

# ╔═╡ f63a7934-489c-4933-84c6-fca830c764bb
md"""
**Remark.**
If ``f(X) = β_0 + β_1 X_1 + ⋯ + β_p X_p``, then ``f(X) > 0`` for points on one side of the hyperplane, and ``f(X) < 0`` for points on the other.
"""

# ╔═╡ 20590a92-9ff3-419f-92e8-eb351c43fd01
md"""
**Remark.**
If we code the colored points as ``Y_i = +1`` for blue, say, and ``Y_i = -1`` for mauve, then if ``Y_i ⋅ f(X_i) > 0`` for all ``i``, ``f(X) = 0`` defines a separating hyperplane.
"""

# ╔═╡ 4b0fbbeb-293b-4340-941e-1c2e5a764a41
md"## Linear SVM"

# ╔═╡ 888a46c6-c548-4ba5-8055-795e41000ffa
md"### → Maximum Margin"

# ╔═╡ 7c58bca4-3b4c-4b75-8b10-7113e4c7c54a
md"### Linear Classifier"

# ╔═╡ f5d68a19-3cb3-4b47-8260-c3765b24e423
md"""
**Question.**
How would you classify this data?
There are an infinite number of hyperplanes that you could choose...but which is the best?
"""

# ╔═╡ c4bda0c3-c0bd-41f1-a40f-7f1e2bdef39a
let
	v1 = [(first(r), last(r)) for r ∈ eachrow([rand(0:5,5) rand(5:10,5)])]
	v2 = [(first(r), last(r)) for r ∈ eachrow([rand(5:10,5) rand(0:5,5)])]
	scatter(v1, color=:red)
	scatter!(v2, color=:blue, legend=false)
end

# ╔═╡ b6f3a1ca-275c-483b-a75e-545b00c995a6
md"### Classifier Margin"

# ╔═╡ 54c2c096-5107-4479-ba06-62c43c380610
md"""
**Definition.**
**Margin** is how much space the hyperplane can move parallelly before hitting the data-points.
"""

# ╔═╡ 34e441ac-74c9-4f41-b477-c519330eb58f
md"""
**Remark.**
We want to construct a "street"

- with a maximum width

- without touching the buildings

Once we build such a "street" the perfect hyperplane would be the middle line.
"""

# ╔═╡ 67655ea6-b01b-4aa5-8677-31897bfc5e21
md"### Maximum Margin Linear Classifier"

# ╔═╡ d98a2090-eee5-4a72-90c9-efee51b99fec
md"""
**Remark.**
The **maximum margin linear classifier** is the linear classifier with the maximum margin.
This is the simplest kind of SVM (called **Linear SVM**).
"""

# ╔═╡ 6f0a95e8-b091-4151-a5ba-381de21ca390
md"""
**Remark.**
The **support vectors** are on the edge of the best "street".
"""

# ╔═╡ 68a31310-2d63-4d49-bd31-ae321160e508
md"### Why Maximum Margin?"

# ╔═╡ 760c855d-6aa6-40f1-9ae6-bb30c0293877
md"""
**Remark.**
Intuitively this feels safest.
If we've made a small error in the location of the boundary, this gives us least chance of causing a misclassification.
"""

# ╔═╡ 9bb1ceca-70c9-4e10-8f59-8c0a0142ff03
md"""
**Remark.**
The model is robust since it is immune to removal of any non-support-vector datapoints.
"""

# ╔═╡ c515d60e-00a4-4e86-96e3-9d265ab113c2
md"""
**Remark.**
Empirically it works very well.
"""

# ╔═╡ 37950e6c-4a34-4513-ac37-0d90c50eb390
md"### → Optimize Linear SVM"

# ╔═╡ 2db8bf2e-5755-4f63-9b5d-de209aa86a6b
md"### How to find the optimal Hyperplane"

# ╔═╡ 7cf2debb-c558-4bcb-af05-b5a7bc0935a0
md"""
**Remark.**
Linear classifier: ``f(𝐱) = 𝐰𝐱 + b``

The optimal hyperplane satisfies:

$\min_{𝐰,b} \frac{1}{2} 𝐰^T 𝐰$

subject to ``y_i ⋅ (𝐰^T 𝐱_i + b) ≥ 1`` for all ``i``.
"""

# ╔═╡ 98129378-93be-4895-9950-17b9193fecd1
md"""
**Remark.**
Basic knowledge:

- Separating hyperplane ``h(𝐱) = 𝐰𝐱 + b = 0``

- The distance of data point ``x_i`` to the separating hyperplane ``h(𝐱)``:

  $\text{distance}(𝐱_i ∣ h(𝐱)) = \frac{|h(𝐱_i)|}{\|𝐰\|} = \frac{|𝐰𝐱_i + b|}{\|𝐰\|} = \frac{y_i ⋅ (𝐰𝐱_i + b)}{\|𝐰\|}$
"""

# ╔═╡ afb99d1c-aff9-4cb5-8bd5-b5e25ebe9fed
md"""
**Remark.**
Intuitively, this is what we should solve:

$\max_{𝐰,b} \text{margin}(h(𝐱))$

$\text{subject to } y_i ⋅(𝐰^T 𝐱_i + b) > 0 \text{ for all } i$

$\text{margin}(h(𝐱)) = \min_{i=1,…,n} \text{distance}(𝐱_i ∣ h(𝐱))$

using ``\displaystyle \text{distance}(𝐱_i ∣ h(𝐱)) = \frac{y_i ⋅ (𝐰𝐱_i + b)}{\|𝐰\|}`` the above becomes the following:

$\max_{𝐰,b} \text{margin}(h(𝐱))$

$\text{subject to } y_i ⋅(𝐰^T 𝐱_i + b) > 0 \text{ for all } i$

$\text{margin}(h(𝐱)) = \min_{i=1,…,n} \frac{y_i ⋅ (𝐰𝐱_i + b)}{\|𝐰\|}$

scaling the hyperplane parameters so that ``\displaystyle \min_{i=1,…,n} y_i ⋅ (𝐰𝐱_i + b) = 1`` the above becomes the following:

$\max_{w,b} \text{margin}(h(𝐱))$

$\text{subject to } y_i ⋅(𝐰^T 𝐱_i + b) > 0 \text{ for all } i$

$\text{margin}(h(𝐱)) = \frac{1}{\|𝐰\|}$

which means the same thing as

$\min_{𝐰,b} \|𝐰\|$

$\text{subject to } y_i ⋅ (𝐰^T 𝐱_i + b) ≥ 1 \text{ for all } i$

which means the same thing as

$\min_{𝐰,b} 𝐰^T𝐰$

$\text{subject to } y_i ⋅ (𝐰^T 𝐱_i + b) ≥ 1 \text{ for all } i$

This is the simplest format of a constrained optimization problem (quadratic programming), can be solved efficiently.
"""

# ╔═╡ 26b9a0d6-58aa-4f95-9c50-6ea879922a70
md"### → Linear SVM for Non-separable Data"

# ╔═╡ 6ecac6ff-8d9c-49fd-a36a-68a209656734
md"""
**Example.**
Consider a plot where the data on the left are not separable by a linear boundary.
"""

# ╔═╡ d2c7bb32-16ff-421d-b9f6-b13411bbe8b8
md"""
**Remark.**
Sometimes the data are separable, but noisy.
This can lead to a poor solution for the maximal-margin classifier.
"""

# ╔═╡ f75b35bb-5018-433d-b795-31646a1cfb85
md"""
**Remark.**
The SVM maximizes a soft margin.
"""

# ╔═╡ ea847b13-b4b7-474d-9ef0-d1c5c659bf36
md"""
**Remark.**
Minimize: ``𝐰^T 𝐰 + C`` (distance of error points to their correct place)

so we have some error

$\min_{𝐰,b} 𝐰^T 𝐰 + C ∑_{i=1}^n ε_i$

$\text{subject to } y_i ⋅ (𝐰^T 𝐱_i + b) ≥ 1 - ε_i \text{ for all } i$
"""

# ╔═╡ 5a5f401f-a68c-4383-82b8-d593f2abfd7b
md"""
**Remark.**

- ``ε = 0``: data points correctly classified (either on the margin or on the correct side of the margin)

- ``0 < ε ≤ 1``: lie inside the margin, but on the correct side of the decision boundary

- ``ε > 1``: wrong side of the decision boundary are misclassified
"""

# ╔═╡ ca313f8e-b222-43c3-8f91-cd5c3f85d2ea
md"## Nonlinear SVM"

# ╔═╡ 2e9ba193-d51e-4967-bcc5-107a634a23ce
md"### Linear Boundary can fail"

# ╔═╡ b72e3384-01b2-46a5-a5a3-26bfddf2155b
md"""
**Remark.**

- Sometimes a linear boundary simply won't work, no matter what value of ``C``.

- The example on the left is such a case.

- What to do?
"""

# ╔═╡ b61385fe-4126-419f-86c3-4681acc26b66
md"### Nonlinear SVM: Idea"

# ╔═╡ 0b76f9bd-cf9b-411d-9a1e-ef77877ac976
md"""
**Remark.**
The original feature space can be mapped to some higher-dimensional feature space where the training set is separable.
"""

# ╔═╡ 125fa440-073e-4ceb-abb0-17f28a24cdd3
md"### Solution #1: Feature Expansion"

# ╔═╡ 2f16ab06-6623-447d-9c6c-e6f7d9664b3c
md"""
**Remark.**

- Enlarge the space of features by including transformations; e.g., ``{X_1}^2``, ``{X_1}^3``, ``X_1 X_2``, ``X_1 {X_2}^2``, …Hence go from a ``p``-dimensional space to a ``q > p`` dimensional space.

- Fit a support-vector classifier in the enlarged space.

- This results in non-linear decision boundaries in the original space.
"""

# ╔═╡ 2944719a-9a88-4031-8ecc-cd8e5140ebc6
md"""
**Example.**
Suppose we use ``(X_1, X_2, {X_1}^2, X_2, X_1 X_2)`` instead of just ``(X_1,X_2)``.
Then the decision boundary would be of the form:

$β_0 + β_1 X_1 + β_2 X_2 + β_3 {X_1}^2 + β_4 {X_2}^2 + β_5 X_1 X_2$

This leads to nonlinear decision boundaries in the original space (quadratic conic sections).
"""

# ╔═╡ 23f73935-6306-4a63-bdcd-a5c1ff3d773d
md"""
**Remark.**
Cubic Polynomials solution:

- Here we use a basis expansion of cubic polynomials

- From 2 variables to 9

- The support-vector classifier in the enlarged space solves the problem in the lower-dimensional space
"""

# ╔═╡ 72dcae0f-0100-4f1f-9a35-90284e9a80d2
md"### Solution #2: Kernels"

# ╔═╡ 5217080f-219b-40c0-a350-985539ad8fc1
md"""
**Remark.**

- Polynomials (especially high-dimensional ones) get wild rather fast.

- There is a more elegant and controlled way to introduce nonlinearities in support-vector classifiers--through the use of kernels.

- Before we discuss these, we must understand the role of inner products in support-vector classifiers.
"""

# ╔═╡ 2f5a4ba3-62f6-461e-b757-b1e9291b3899
md"""
### Inner Product

$⟨𝐱_i, 𝐱_j⟩ = ∑_{k=1}^D 𝐱_{i,k} ⋅ 𝐱_{j,k}$

Keyword: similarity
"""

# ╔═╡ 4162a41b-1695-4595-89f2-4738e0eecd06
md"### A Quick Example for Kernel"

# ╔═╡ 2df5c299-48db-4bf2-acf0-cdc8e52729e0
md"""
**Remark.**
Simple Example:
``x = (x_1,x_2,x_3)``; ``y = (y_1,y_2,y_3)``.
Then for the function ``ϕ(x) = (x_1x_1,x_1x_2,x_1x_3,x_2x_1,x_2x_2,x_2x_3,x_3x_1,x_3x_2,x_3x_3)``, the kernel is ``K(x,y) = (⟨x,y⟩)^2``,

Suppose ``x = (1,2,3)``; ``y = (4,5,6)``.

**Transform the inner product:**

- Step 1:

$\begin{align*}
ϕ(x) &= (1,2,3,2,4,6,3,6,9) \\
ϕ(y) &= (16,20,24,20,25,30,24,30,36)
\end{align*}$

- Step 2:

$\begin{align*}
⟨ϕ(x),ϕ(y)⟩ &= 16+40+72+40+100+180+72+180+324\\
&=1024
\end{align*}$

These two computation steps can be quite expensive.

**Use Kernel:**

$K(x,y) = (4 + 10 + 18)^2 = 32^2 = 1024$

Same result, but this calculation is so much easier.

> In terms of result,
>
> Kernel = Feature Transformation + Inner Product
"""

# ╔═╡ 8b4cf5cb-82b5-45d8-809a-a8ad8db75b9c
md"### The \"Kernel Trick\""

# ╔═╡ eea21eeb-56b1-42e6-ac34-6b237a126e6a
md"""
**Remark.**
Note: SVMs rely on the inner product between vectors ``K(x_i,x_j) = x_i^T x_j``
"""

# ╔═╡ 289e7a2c-983b-4295-9f9d-3fb327a4436e
md"""
**Remark.**
If every data point is mapped into high-dimensional space via some transformation ``ϕ : x → ϕ(x)``, the inner product becomes ``K(x_i, x_j) = ϕ(x_i)^T ϕ(x_j)``
"""

# ╔═╡ 6dc86a2a-c2d5-44b2-90ad-d63fb4a35784
md"""
**Remark.**
A kernel function is a type of function that is equivalent to an inner product in some feature space.
"""

# ╔═╡ 35fa31d7-a2db-4b8e-9236-a6007e46cbd0
md"""
**Question.**
How do we know if a given function is a kernel function?
"""

# ╔═╡ 6b033d49-8900-4e24-8881-5c1ad6f2d245
md"""
**Example:**
- Input feature is 2D: ``x = [x_1,x_2]``

- ``K(x_i,x_j) = (1 + x_i^T x_j)^2``; is ``K`` a kernel function?

- We need to find a ``ϕ(x)`` function and show that ``K(x_i,x_j)= ϕ(x_i)^T ϕ(x_j)``

$\begin{align*}
K(x_i,x_j) &= (1 + x_i^Tx_j)^2 \\
&= 1 + x_{i,1}^2 x_{j,1}^2 + ⋯ \\
&= [1,x_{i,1}^2,…]^T [1,x_{j,1}^2,…] \\
&= ϕ(x_i)^T ϕ(x_j)
\end{align*}$

- So, this function implicitly maps 2D data to a 6D space and takes the inner product

- Do we have to do this for every function to check if it is a kernel function?
"""

# ╔═╡ 260c7002-80c4-4e9b-a315-07a0e537ca07
md"### What functions are Kernels"

# ╔═╡ 939e7c62-f9c0-4fdf-891d-01fd8dddde0c
md"""
**Mercer's theorem.**
``K(x_i,x_j)`` is a valid kernel if and only if

$\begin{bmatrix}
K(x_1,x_1) & K(x_1,x_2) & ⋯ & K(x_1,x_n) \\
K(x_2,x_1) & K(x_2,x_2) & ⋯ & K(x_2,x_n) \\
⋮ & ⋮ & ⋱ & ⋮ \\
K(x_n,x_1) & K(x_n,x_2) & ⋯ & K(x_i,x_n)
\end{bmatrix} \qquad \text{ is a positive semi-definite matrix}$
"""

# ╔═╡ 04b3498d-ce70-4c13-8b1d-6ae2db571754
md"""
### Popular Kernels

| Kernel | Formula |
|--------|---------|
| Linear | ``⟨𝐱,𝐳⟩ = 𝐱⋅𝐳`` |
| Polynomial | ``(⟨𝐱,𝐳⟩ + v)^d`` |
"""

# ╔═╡ e8650da5-50db-46b4-8784-1a9c23a48b5b
md"### Kernel Compositionality"

# ╔═╡ 5d68fcae-7f86-439f-a8b5-6a9a56b063ea
md"""
**Remark.**
Given valid kernel functions, you can compose them to create new kernel functions
"""

# ╔═╡ d89696db-5f51-4f7b-8591-12e4fd89d4c8
md"### → How kernel helps SVM"

# ╔═╡ 48fa755c-6805-43e4-8235-bfe45344c334
md"""
**Remark.**

SVM originally wants to optimize:

$\min_{𝐰,b} \frac{1}{2} 𝐰^T 𝐰$

$\text{subject to } y_i ⋅ (𝐰^T 𝐱_i + b) ≥ 1 \text{ for all } i$

Bring another data-point wise parameter ``a``:

$L = \min_{𝐰,b} \max_{𝛂 ≥ 0} \frac{1}{2} \|𝐰\|^2 - \sum_j α_j [(𝐰⋅𝐱_j + b) y_j - 1]$

The relation between ``w`` and ``a``:

$\frac{∂L}{∂w} = w - \sum_j α_j y_j x_j \;⟶\; 𝐰 = \sum_j α_j y_j 𝐱_j$
"""

# ╔═╡ d6842622-0220-4bf2-91e7-cc3cf2539a79
md"""
**Remark.**

$f(𝐱) = \text{sign}(𝐰⋅𝐱+b)$

$f(𝐱) = \text{sign}\left(\sum_i α_i y_i (𝐱_i ⋅ 𝐱) + b\right)$

Both function mean:
SVM predicts the label of the new example
"""

# ╔═╡ 6018d647-94ea-422a-80bb-c9c4ee289f6b
md"### SVM vs. Logistic Regression"

# ╔═╡ cc98a6b2-b29d-426d-95c8-06ddd1efb736
md"""
**Remark.**

- When classes are (nearly) separable, SVM does better than LR

- When not, LR and SVM are very similar

- If you wish to estimate probabilities, LR is the choice

- If nonlinear boundaries, kernel SVMs are popular (before Deep Learning)

- In deep learning era, LR is more popular.
"""

# ╔═╡ c2d9a478-f693-4048-a24c-53ce5fb2bc31
md"# Chapter 7: Algorithms -- Deep Learning Basics"

# ╔═╡ 43551f75-17f3-4c33-8f18-287b7f937be7
md"## What is Deep Learning"

# ╔═╡ 1a7b2ba6-537a-4e22-9f0f-5887cd664f63
md"""
**Remark.**
A type of machine learning based on artificial neural networks in which multiple layers of processing are used to extract progressively higher level features from data.
"""

# ╔═╡ 02534887-8c8a-4067-a099-f1806e2f3d1c
md"### Why Deep Learning"

# ╔═╡ de6d3dfc-e596-45a1-b7e2-2a377e60c06b
md"""
**Remark.**

- **Deep learning**: more data, higher performance

- **Older ML**: good performance on small data; performance plateau with more data
"""

# ╔═╡ c4e7640e-e1bd-4f80-a165-0f866cea28de
md"### Deep Learning vs. Traditional ML"

# ╔═╡ c6ce24ea-e754-488a-b6b0-8c8b94fce598
md"""
**Example.**

- **Traditional ML**:

  - Input (Car) → Feature extraction → Classification → Output (Car, Not Car)

- **Deep Learning**:

  - Input (Car) → (Feature extraction + Classification) → Output (Car, Not Car)

  - Deep Learning can extract/learn features automatically!!!
"""

# ╔═╡ 92d0f71f-3b8d-4d56-806c-29b11587258b
md"""
**Remark.**

|   | Deep Learning | Traditional ML |
|---|---------------|----------------|
| Where features come from? | learn automatically | human defined |
| Need data preprocessing? | usually no | yes |
| Need domain expert? | no | yes |
| Interpretability | almost no | yes |
| How to solve a problem | end-to-end | pipeline |
| Training time | usually long | short |
| Hardware | GPU (fast) or CPU (slow) | CPU |
| Task Transferability | relatively strong | limited |
"""

# ╔═╡ 4bf464a1-abf9-4095-bdb9-ee7098cb725e
md"### Why \"Deep\""

# ╔═╡ 62742e7d-e201-45a6-a7cf-7f26caaa314d
md"""
**Remark.**
Geoffrey Hinton co-authored a paper in 2006 titled "A Fast Learning Algorithm for Deep Belief Nets" in which they describe an approach to training "deep" (as in a many layered network) of restricted Boltzmann machines.
"""

# ╔═╡ 4d13f162-00cc-4d29-ab27-e925cbb57c4c
md"""
**Remark.**
Turing Award 2019 winners include Yoshua Bengio, Geoffrey Hinton, and Yann LeCun.
"""

# ╔═╡ d6eebe65-d1be-429b-9b71-07b9fc4715a3
md"### Why Now"

# ╔═╡ 9357503f-0d71-4484-8ba7-81126e870f2a
md"""
**Remark.**
From the 1950's to the 1970's, we had **artificial intelligence** (the engineering of making intelligent machines and programs).
From the 1980's to the 2010's, we had **machine learning** (the ability to learn without being explicitly programmed).
From the 2010's onwards, we have **deep learning** (learning based on deep neural networks).
"""

# ╔═╡ 1890e612-3914-42cb-9f9e-a6b8e8478d95
md"""
**Remark.**
The timeline of AI/ML/DL can also be described as a series of events:

- (1940--1966) **Beginnings**:

  - (1943) Thresholded Logic Unit

  - (1957) Perceptron

  - (1960) Adaline

- (1966--1976) **1st Neural winter**:

  - (1969) XOR Problem

- (1976--1997)

  - (1982--1986) Multilayer Backprop

  - (1989) CNNs

  - (1995) SVMs

  - (1997) LSTMs

- (1997--2006) **2nd Neural Winter**

- (2006--present) **GPU Era**

  - (2006) Deep Nets

  - (2012) Alex Net
"""

# ╔═╡ 5eca0c6a-c9fd-4656-b399-f2ba56e3cda7
md"""
**Remark.**
What was actually wrong with backpropagation in 1986?
We all drew the wrong conclusions about why it failed.
The real reasons were:

- Our labeled datasets were thousands of times too small

- Our computers were millions of times too slow

- We initialized the weights in a stupid way.

- We used the wrong type of non-linearity.
"""

# ╔═╡ da665a8c-16f9-4153-92c4-cb1d46855908
md"### Impact of Deep Learning in Computer Vision"

# ╔═╡ 1c77794b-14e4-4baf-861a-6835ddd24b74
md"""
**Remark.**

- 2012-2014 classification results in ImageNet

  | 2012 Teams | %error |
  |------------|--------|
  | Supervision (Toronto) | 15.3 |
  | ISI (Tokyo) | 26.1 |

  $↓$

  | 2013 Teams | %error |
  |------------|--------|
  | Clarifai (NYU spinoff) | 11.7 |
  | NUS (singapore) | 12.9 |

  $↓$

  | 2014 Teams | %error |
  |------------|--------|
  | GoogLeNet | 6.6 |
  | VGG (Oxford) | 7.3 |

- 2015 results: ResNet under 3.5% error using 150 layers!
"""

# ╔═╡ e0eb3686-c0b8-4c8c-a934-e3709fa9e9f2
md"### Impact of Deep Learning in Speech Recognition"

# ╔═╡ 42e5c894-439f-4d6c-8c3c-9d35f0c7d332
md"""
**Remark.**
Word error rate (%) on switchboard dramatically decreased (from 100% in 1990s to now <10% since 2010s) when DL was introduced.
"""

# ╔═╡ bf3ddab2-7bfb-42c7-aec0-fad614c040cf
md"### Why These Improvements in Performance?"

# ╔═╡ c97a6716-6877-4a96-b674-75f722261d5a
md"""
**Remark.**

- **Features are learned** rather than hand-crafted

- **Deeper** architecture

- **More data** to train DL

- **More computing** (GPUs)

- Better **training tricks**: pretraining, regularization, non-linearities, etc.

However, theoretical understanding of DL remains shallow!
"""

# ╔═╡ 4a948739-bfaf-481e-95d2-413e787af10c
md"## Perceptron"

# ╔═╡ 06c92fdb-70db-498e-82d9-8398c23c3b07
md"""
**Remark.**
Neurons in the brain are connected to each other to form a neural network; a single neuron can receive signals from other neurons.

- The perceptron is a mathematical model of a biological neuron

- The structural building block of deep learning
"""

# ╔═╡ 34e9a2b8-8d1b-4926-8f01-74c2e3797bf7
md"### The Perceptron: Foward Propagation"

# ╔═╡ 699c926b-e98f-4091-a63d-647abe93acbc
md"""
**Remark.**
Inputs + Weights ``(𝐱 * 𝐰)`` are summed ``(∑)`` then nonlinearized (⚡) then outputted ``(\hat{y})``.
In mathematical notation, we have

$\hat{y} = g\left(\sum_{i=1}^m w_i ⋅ x_i\right)$

where ``\hat{y}`` is the output, ``g`` is the non-linear activation function, and the expression ``\sum_{i=1}^m w_i ⋅ x_i`` is the linear combination of inputs.
We could also add a bias ``w_0`` like so:

$\hat{y} = g\left(w_0 + \sum_{i=1}^m w_i ⋅ x_i\right)$

Additionally, we can simply the expression for linear combination:

$\hat{y} = g(w_0 + 𝐰^T 𝐱)$

where

$𝐱 = \begin{bmatrix} x_1 \\ ⋮ \\ x_m \end{bmatrix} \quad\text{and}\quad 𝐰 = \begin{bmatrix} w_1 \\ ⋮ \\ w_m \end{bmatrix}.$
"""

# ╔═╡ 8e8c0ae6-b135-4170-8ce9-7cc1258cc3e4
md"""
**Remark.**
An example of the activation function ``g`` is the sigmoid function,

$σ(x) = \frac{e^x}{e^x + 1} = \frac{1}{1 + e^{-x}}.$
"""

# ╔═╡ b549ad91-6f75-4528-afdc-e57a77265e41
md"### Common Activation Functions"

# ╔═╡ c6ebc38c-b146-4b5d-850f-ccfadab834f7
md"""
**Remark.**
Some common activation functions include:

- Sigmoid:

  $σ(z) = \frac{1}{1 + e^{-z}}$

- Tanh:

  $σ(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$

- ReLU:

  $\text{ReLU}(z) = \begin{cases} z, &z > 0 \\ 0, &\text{otherwise} \end{cases}$

- LeakyReLU(``a = 0.2``)

  $\text{LeakyReLU}(z) = \begin{cases} z, &z > 0 \\ az, &\text{otherwise} \end{cases}$
"""

# ╔═╡ f5f23cc7-8971-4b85-8be0-65b89abf1c25
md"""
**Remark.**
All activation functions are non-linear.
"""

# ╔═╡ 9e482437-b816-41f2-a2b5-cd39bfca9f9a
md"### Importance of Activation Functions"

# ╔═╡ 936effb8-f639-4500-b33e-568a84d32db1
md"""
**Remark.**
The purpose of activation functions is to introduce the non-linearities to the network.

- "What if we want to build a neural network to distinguish red vs. blue points?"

- **Linear activation functions** produce linear decisions no matter the network size

- **Non-linearities** allow us to approximate arbitrarily complex functions
"""

# ╔═╡ 8c6f1649-3228-4186-9913-871a3faf907e
md"### The Perceptron: Simplified"

# ╔═╡ 7b3dec37-8994-4a85-9586-b5443b86ac3a
md"""
**Remark.**

$\hat{y} = g(w_0 + 𝐰^T 𝐱)$

$z = w_0 + \sum_{i=1}^m w_i ⋅ x_i$
"""

# ╔═╡ 950b6497-58fc-423b-98d5-2f643e84f606
md"## Building Neural Networks with Perceptron"

# ╔═╡ 97bf5e20-3024-4913-9983-93fd94d3c2a8
md"### Multi Output Perceptron"

# ╔═╡ 98fede0f-89f4-4a99-94b7-c309511c6fd1
md"""
**Remark.**
Because all inputs are densely connected with all outputs, these layers are called **Dense** layers.
"""

# ╔═╡ ce376e97-aa93-4664-bb22-5c8a94cb6ea2
md"""
**Example (Dense layers).**
For example, we have two outputs, ``y_1 = g(z_1)`` and ``y_2 = g(z_2)``.
Each of these outputs are connected with all of the inputs ``x_1, x_2, …, x_m``.
In mathematical notation, we have

$z_j = w_{0,j} + \sum_{i=1}^m w_{i,j} ⋅ x_i$
"""

# ╔═╡ ff1c1790-d880-4c38-9f84-26edab6d3612
md"### Dense Layer from Scratch"

# ╔═╡ 88238432-40f3-4a83-8fb7-338a31c03ca5
md"""
```python
class MyDenseLayer(tf.keras.layers.Layer):
  def __init__(self, input_dim, output_dim):
    super(MyDenseLayer, self).__init__()

    # Initialize weights and bias
    self.W = self.add_weight([input_dim, output_dim])
    self.b = self.add_weight([1, output_dim])

  def call(self, inputs):
    # Forward propagate the inputs
    z = tf.matmul(inputs, self.W) + self.b

    # Feed through a non-linear activation
    output = tf.math.sigmoid(z)

    return output
```
"""

# ╔═╡ 62d107ce-b957-4bab-9670-eed33bc5afa8
md"### Single Layer Neural Network"

# ╔═╡ 20b190b6-b090-4ee2-b7be-475cd171bf9b
md"""
**Remark.**
In a single layer neural network, we have an input layer ``𝐱 = \{x_1,x_2,…,x_m\}`` and its weights ``𝐖^{(1)}``, a hidden layer ``𝐳 = \{z_1, z_2, z_3, …, z_d\}`` and its weights ``𝐖^{(2)}``, and finally its output layer ``𝐲 = \{\hat{y}_1,\hat{y}_2\}``.
In mathematical notation, we have

$z_j = w_{0,j}^{(1)} + \sum_{i=1}^m w_{i,j}^{(1)} ⋅ x_i$

and

$\hat{y}_j = g\left(w_{0,j}^{(2)} + \sum_{i=1}^d w_{i,j}^{(2)} ⋅ g(z_i)\right).$
"""

# ╔═╡ a6c2720b-83c2-4ac8-a76a-a7517ac79e96
md"### Multi Ouptut Perceptron"

# ╔═╡ 70623b38-f30e-45a4-9835-5d05e3df3344
md"""
```python
import tensorflow as tf

model = tf.keras.Sequential([
	tf.keras.layers.Dense(n),
	tf.keras.layers.Dense(2)
])
```
"""

# ╔═╡ 2a9821f4-b7fe-4f64-8557-f9b57a8ebe19
md"### Deep Neural Network"

# ╔═╡ baa40833-37c8-4946-98fe-da482789f491
md"""
**Remark.**
Each hidden layer in a deep neural network is defined as

$z_{k,i} = w_{0,i}^{(k)} + \sum_{j=1}^{n_{k-1}} w_{j,i}^{(k)} g(z_{k-1}, j)$
"""

# ╔═╡ fd02b5a7-f7a8-49c7-a10e-1568eb29d0d4
md"""
**Remark.**
Using TensorFlow, we have

```python
import tensorflow as tf

model = tf.keras.Sequential([
	tf.keras.layers.Dense(n₁),
	tf.keras.layers.Dense(n₂),
	⋮
	tf.keras.layers.Dense(2)
])
```
"""

# ╔═╡ 16682257-f2b5-4966-af5e-a2d6287968b2
md"## Applying Neural Networks"

# ╔═╡ 184910d6-db53-4631-9371-29fcec2a33fe
md"""
**Remark.**
Task: **Will I pass this course?**

Let's start with a simple two feature model

- ``x_1`` = number of lectures you attend

- ``x_2`` = hours spent on the final project
"""

# ╔═╡ 28e48281-1039-4cab-ae30-609d23e7992a
md"### Example Problem: Will I pass this course?"

# ╔═╡ eda636aa-60a7-4022-ae24-e29065f150fc
md"""
**Example.**
Let ``x_1`` be the number of lectures you attend and ``x_2`` be the hours spent on the final project.
Suppose that, given ``x_1 = 4, x_2 = 5`` we predict 0.1 but the actual result is 1.
"""

# ╔═╡ 11cecba9-4c91-451c-9d27-a17336907097
md"""
### Quantifying Loss

$l(\underbrace{f(x^{(i)};𝐰)}_\text{Predicted}, \underbrace{y^{(i)}}_\text{Actual})$
"""

# ╔═╡ 299dec55-e364-4bfb-bc5d-ec5cf6cf5ca4
md"### Empirical Loss"

# ╔═╡ a067f2b1-b581-4400-b47d-9ba63054acfc
md"""
**Definition.**
The **empirical loss** measures the total loss over our entire dataset.

$\mathcal{L}(𝐰) = \frac{1}{n} \sum_{i=1}^n l(\underbrace{f(x^{(i)}; 𝐰}_\text{Predicted}), \underbrace{y^{(i)}}_\text{Actual})$

Also known as:

- Objective function

- Cost/Loss Function

- Empirical Risk
"""

# ╔═╡ 22a5177e-1866-4230-8ef4-44ab0b561a9f
md"### Binary Cross Entropy"

# ╔═╡ 9c3e4d8a-9972-4f03-af8b-f896a66016ce
md"""
**Definition.**
**Cross entropy loss** can be used with models that output a probability between 0 and 1.

$\mathcal{L}(𝐰) = -\frac{1}{n} \underbrace{y^{(i)}}_\text{Actual} \log(\underbrace{f(x^{(i)};𝐰)}_\text{Predicted}) + (1 - \underbrace{y^{(i)}}_\text{Actual}) \log(1 - \underbrace{f(x^{(i)}; 𝐰)}_\text{Predicted})$
"""

# ╔═╡ f4e267c2-71da-4da4-86e1-f0c1653b6db5
md"### Mean Squared Error Loss"

# ╔═╡ e18d0f0d-2989-465f-ac3f-a520ef669a54
md"""
**Definition.**
**Mean squared error loss** can be used with regression models that output continuous real numbers.

$\mathcal{L}(𝐰) = \frac{1}{n} \sum_{i=1}^n (\underbrace{y^{(i)}}_\text{Actual} - \underbrace{f(x^{(i)};𝐰)}_\text{Predicted})^2$
"""

# ╔═╡ be397193-7162-4efa-8166-a30cc6f612e9
md"## Parameter Initialization"

# ╔═╡ 49575adb-1bc1-4239-a966-9bc8bff34f9a
md"""
**Remark.**

- **Idea #0**: all set to 0. What will happen?

- **Idea #1**: small random numbers

  - Gaussian with 0-mean and 1e-2 std deviation

    - `W = 0.01 * np.random.randn(D,H)`

  - Works well for small neural nets

  - Not so great for larger nets
"""

# ╔═╡ 6192c0c3-2856-44f6-bb9a-83f019c6d0d7
md"### Xavier Initialization (Glorot et al., 2010)"

# ╔═╡ 1f4c8b2b-fd2b-4f00-b705-6f1f5b658f20
md"""
**Remark.**

- Make sure the weights are "just right", not too small, not too big.
  Use number of input (`f_in`) and output (`f_out`)

  ```python
  >>> F_in = 64
  >>> F_out = 32
  >>> limit = np.sqrt(6 / float(F_in))
  >>> W = np.random.uniform(low=-limit, high=limit, size=(F_in, F_out))
  ```


- Glorot/Xavier initialization can also be done with a uniform distribution where we place stronger restrictions on limit:

  ```python
  >>> F_in = 64
  >>> F_out = 32
  >>> limit = np.sqrt(6 / float(F_in + F_out))
  >>> W = np.random.uniform(low=-limit, high=limit, size=(F_in, F_out))
  ```
"""

# ╔═╡ 782ba2ee-4397-41c9-9b95-6453405ae047
md"## Multi-layer Perceptron (MLP) and Its Issues"

# ╔═╡ 42f4af1b-7216-4bc8-b120-201212196819
md"### Multi-layer Perceptron (MLP)"

# ╔═╡ cd585b92-a445-485f-b81b-dbcae9de210d
md"""
**Remark.**
At least 3 layers: **input, hidden, output**.
"""

# ╔═╡ fcb8b589-1776-4f86-a797-cc588482574e
md"### Fully-connected Layers in MLP"

# ╔═╡ 74ce0427-a696-443c-b8ca-f1bfb3aebe25
md"### Issues in MLP"

# ╔═╡ 792bf56b-e160-4e06-9705-9d7a8e691cec
md"""
**Remark.**

- **Too many parameters**:
  the number of total parameters can grow to very high (number of perceptron in layer 1 multiplied by # of p in layer 2 multiplied by # of p in layer 3...)

- **Disregard spatial information**
  It takes flattened vectors as inputs

- **Length limitation of inputs**
"""

# ╔═╡ d1926abb-28bc-4632-8d11-59284077f3eb
md"# Chapter 8: Algorithms -- Convolutional Neural Networks"

# ╔═╡ c8a65e59-937e-4adf-8bf2-0f80580175c4
md"## Motivation"

# ╔═╡ c906256f-4885-4318-ad76-c6204d5f3647
md"### How does MLP work?"

# ╔═╡ bf4173b1-fe25-4744-91cc-3b1e61857d0d
md"""
**Example.**
32 × 32 × 3 image → stretch to 3072 × 1

> **input** (3072 × 1) → ``Wx`` (10 × 3072 weights) → **activation** (10 × 1)

An element of the activation vector is the result of taking a dot product between a row of ``W`` and the input (a 3072-dimensional dot product).
"""

# ╔═╡ ccc65c1e-feea-464e-8acf-6f0ad9405f2d
md"### Issues in MLP"

# ╔═╡ 4ccaaa0f-1bcf-469a-8e28-5b09fc31c3df
md"""
**Remark.**

- **Too many parameters:**
  We need to define a weight/parameter for each input feature

- **Disregard spatial information:**
  In an elephant image, given the nose, we can intuitively guess where the head should be
"""

# ╔═╡ 136ae1ce-a2ef-40d7-bb22-aae7e3fc0685
md"""
**Remark.**

- Can we split the input into subparts, and all subparts share the same parameters? **(reduce parameters)**

- Can we first handle smaller subparts, then combine their results as larger subparts? **(hierarchical structure)**
"""

# ╔═╡ 792d0a82-cf87-4d39-b12c-8028d01d08f5
md"## Convolution"

# ╔═╡ 1a250eff-9fb8-474d-bf7e-dc22f7b5e99d
md"### Convolution Layer"

# ╔═╡ 6ba25d03-4ec3-44b1-bee8-07ef74353c4b
md"""
**Remark.**

32×32×3 image → preserve spatial structure

5×5×3 filter

- **Convolve** the filter with the image, i.e., "slide over the image spatially, computing dot products".

- In MLP, weights should have the same size as the inputs, but convolution uses much smaller size of weights.

- Filters always extend the full depth of the input volume.

- Applying the filter to the image produces 1 number: the result of taking a dot product between the filter and a small 5×5×3 chunk of the image (i.e., 5⋅5⋅3 = 75-dimensional dot product + bias),

  $𝐰^T 𝐱 + b$

- Convolve (slide) over all spatial locations → activation map (28×28×1)

- Consider a second, green filter (we get another activation map).

- For example, if we  had 6 5×5 filters, we'll get 6 separate activation maps… we stack these up to get a "new image" of size 28×28×6!
"""

# ╔═╡ a761d4c2-3fd2-4142-b989-11dadd4c8864
md"### ConvNet"

# ╔═╡ 3e39ea11-dd4a-4366-86d7-c66ce155f049
md"""
**Remark.**
A sequence of Convolution Layers, interspersed with activation functions.
"""

# ╔═╡ dd7fbd56-a447-48b9-88a5-a8a44c3090e0
md"""
**Example (ConvNet).**

32×32×3 (CONV, ReLU, e.g., 6 5×5×3 filters) → 28×28×6 (CONV, RELU, e.g., 10 5×5×6 filters) → 24×24×10 (CONV, RELU) → ⋯
"""

# ╔═╡ b91a7c33-0b8f-427f-ba13-f07a89538dc8
md"""
**Example (ConvNet input dog).**

Dog image → Low-level features → Mid-level features → High-level features → Linearly separable classifier →
"""

# ╔═╡ 609e6940-5283-4886-a77f-7b9c9725bd93
md"""
**Example (ConvNet input car).**

CONV, RELU, CONV, RELU, POOL, CONV, RELU, CONV, RELU, POOL, CONV, RELU, CONV, RELU, POOL, FC (car > truck, airplane > ship > horse)
"""

# ╔═╡ fe1cb0ac-346a-4b4b-a0f5-6dfa26c2f6fd
md"### A Closer Look at Spatial Dimensions"

# ╔═╡ 6e353061-20cd-4000-9bae-98eb26bad0ee
md"""
**Example.**
7×7 input (spatially) assume 3×3 filter.

⟹ 5×5 output

Now, consider 7×7 input (spatially) assume 3×3 filter applied **with stride 2**.

⟹ 3×3 output!

Now, consider 7×7 input (spatially) assume 3×3 filter applied **with stride 3?**.

⟹ Doesn't fit! Cannot apply 3×3 filter on 7×7 input with stride 3.

Output size: **(N - F) / stride + 1**

E.g., N = 7, F = 3:

- Stride 1 ⟹ (7 - 3) / 1 + 1 = 5

- Stride 2 ⟹ (7 - 3) / 2 + 1 = 3

- Stride 3 ⟹ (7 - 3) / 3 + 1 = 2.33
"""

# ╔═╡ ba55f4f5-5e62-4815-a977-15056f6ee7b4
md"### In practice: Common to zero pad the border"

# ╔═╡ 41f93b9e-d5db-47f6-8566-63daedfa7a19
md"""
**Example.**
Input 7×7, 3×3 filter, applied with **stride 1**.
**Pad with 1 pixel** border ⟹ what is the output?

⟹ 7×7 output!

In general, common to see CONV layers with stride 1, filters of size F×F, and zero-padding with (F-1)/2. (will preserve size spatially)

E.g.,

- F = 3 ⟹ zero pad with 1

- F = 5 ⟹ zero pad with 2

- F = 7 ⟹ zero pad with 3
"""

# ╔═╡ b49bde38-6b45-4dee-81c8-e7af7fbe36ff
md"### Potential Issue"

# ╔═╡ 1419ebac-90a0-423d-a260-dea8e96683b9
md"""
**Example.**
32×32 input convolved repeatedly with 5×5 filters shrinks volumes spatially! (32 → 28 → 24…).
**Shrinking too fast is not good, doesn't work well.**
"""

# ╔═╡ d70b8368-49e7-4baa-8c71-cb4ca5789937
md"""
**Example.**
Input volume: 32×32×3, 10 5×5  filters with stride 1, pad 2

Output volume size: (32+2⋅2-5)/1+1 = 32 spatially, so 32×32×10

Number of parameters in this layer: each filter has 5⋅5⋅3 + 1 = 76 params ⟹ 76⋅10 = 760
"""

# ╔═╡ 614c11b2-6b9e-44a7-8e35-ef3c9c5f1b05
md"### Multi-channel Convolution Layer"

# ╔═╡ e881b52c-45a1-4f4b-b7e5-3399ae6da3cf
md"""
**Remark.**
**Channel**: one version of representation for the object.
"""

# ╔═╡ 69da95fe-49e9-4eb0-b463-80e156194b70
md"## Pooling"

# ╔═╡ cd4e2320-1490-40ed-89ce-d512ddf52b30
md"### Pooling Layer"

# ╔═╡ 48583e15-3dd3-4fa1-a6de-32fb2da266f4
md"""
**Remark.**

- Select the most representative features for the next step.

- Operates over each activation map independently.
"""

# ╔═╡ 160ce232-b4d0-4692-add8-92f33014b64a
md"### Max-Pooling (the most popular pooling technique)"

# ╔═╡ b4eddf2a-98ec-4016-bcf5-3ec7a6a7c492
md"""
**Example.**
4×4 has max pool with 2×2 filters and stride 2.
"""

# ╔═╡ 8ed417f8-a03f-417b-b61d-255761bcbbbb
md"## Fully Connected Layer"

# ╔═╡ bb400743-e850-432a-b792-30c02f9e1e17
md"### ConvNet"

# ╔═╡ 407dd2fc-8347-4d80-89e3-5dadc452bb9a
md"""
**Remark.**
What is "FC"?
"""

# ╔═╡ 6ef48dfc-b064-43e6-8ac0-d03c7cd97b6b
md"### Fully Connected Layer (FC Layer)"

# ╔═╡ 6192047f-cfa0-4b60-8af6-d718f8db063c
md"## CNN in NLP"

# ╔═╡ 56657013-c4be-439d-9319-2ba952f01223
md"""
**Remark.**
Take in word vector, apply weights, apply max pooling, then make decision.
"""

# ╔═╡ 18d4c44b-f6eb-4ac3-bba4-fd3a9f37f6c3
md"""
**Remark.**
n×k representation of sentence with static and non-static channels → Convolutional layer with multiple filter widths and feature maps → Max-over-time pooling → Fully-connected layer with dropout and softmax output
"""

# ╔═╡ 176a6470-af74-447f-92de-fcf4afc96f88
md"# Chapter 9: Algorithms -- Recurrent Neural Networks"

# ╔═╡ 2168514b-2d3c-4102-bad9-c973c383447b
md"## (Vanilla) Recurrent Neural Network (RNN)"

# ╔═╡ 089d9285-9a78-4a12-8c4b-072a23a22242
md"""### "Vanilla" Neural Network"""

# ╔═╡ d9c2444c-08b3-4ef5-aea5-7606d41bf463
md"""
**Example.**
One to one:
▭ → ▭ → ▭
"""

# ╔═╡ 0859e9cc-d036-46ff-96c5-30d3d0868b95
md"### Recurrent Neural Networks: Process Sequences"

# ╔═╡ 0ebf7c9a-ed22-4143-a5e6-373d2220364a
md"""
**Remark.**
Types of recurrent neural networks:

- One to one

  - E.g., Vanilla Neural Networks

- One to many

  - E.g., Image Captioning (image → sequence of words)

- Many to one

  - E.g., Sentiment Classification (sequence of words → sentiment)

- Many to many

  - E.g., Machine Translation (sequence of words → sequence of words)

- Many to many

  - E.g., Video classification on frame level
"""

# ╔═╡ b85b4c90-ab1d-4a37-8994-c909b8b8f0b7
md"### Recurrent Neural Networks"

# ╔═╡ e3cc5ea9-0372-4156-af45-db84785bbc26
md"""
**Remark.**
System design:

> x → RNN → y

For y, we usually want to predict a vector at some time step.
"""

# ╔═╡ 9be71560-d07a-4785-9b22-637d4ba5cb4e
md"""
**Remark.**
We can process a sequence of vectors ``𝐱`` by applying a **recurrence formula** at every time step:

$h_t = f_W(h_{t-1}, x_t)$

where ``h_t`` is **new state**, ``f_W`` is **some function with parameters** ``W``, ``h_{t-1}`` is **old state**, and ``x_t`` is **input vector at some time step**.

**Notice: the same function and the same set of parameters are used at every time step.**
"""

# ╔═╡ 035f572d-0437-429b-8772-f12e79c3ae69
md"### (Vanilla) Recurrent Neural Networks"

# ╔═╡ 72ef448b-67f7-480b-aee7-f6643357dbc7
md"""
**Remark.**
The state consists of a single "hidden" vector ``h``:

$h_t = f_W(h_{t-1}, x_t)$

$↓$

$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t)$

$y_t = W_{hy} h_t$
"""

# ╔═╡ 6b606ae3-01e7-40df-810e-dff27cabd6c2
md"### RNN: Computational Graph"

# ╔═╡ 9decb77d-7d56-45dd-a226-a0ef1490ee81
md"""
**Remark.**
Re-use the same weight matrix at every time-step:

> (h0, x1) → fW → (h1, x2) → fW → (h2, x3) → fW → h3 → … → hT

(For each fW, use the same W)
"""

# ╔═╡ d11b21b4-c6ec-4a73-a798-37457d7e41eb
md"### RNN: Computational Graph: Many to Many"

# ╔═╡ 07baa0c8-f50a-40d4-9544-311371bec1be
md"""
**Remark.**
See (RNN: Computational Graph), except each hT produces y1, y2, y3, … yT, and each yT produces L1, L2, L3, …, LT.

> (h0, x1) → fW → (h1→y1→L1, x2,) → fW → (h2→y2→L2, x3) → fW → (h3→y3→L3) → … → (hT→yT→LT)
"""

# ╔═╡ 965a77e4-fec8-40e1-9c0e-9869859c06b1
md"### RNN: Computational Graph: Many to One"

# ╔═╡ fc46f1cc-29fa-4268-b18d-cbebf61549ff
md"""
**Remark.**
See (RNN: Computational Graph), except hT produces y

> (h0, x1) → fW → (h1, x2) → fW → (h2, x3) → fW → h3 → … → (hT→y)
"""

# ╔═╡ 64d2db06-8438-4d2b-8e58-78330545b291
md"### RNN: Computational Graph: One to Many"

# ╔═╡ d6bae6d3-fb6d-4e1b-8903-a8e5b9b15a5d
md"""
**Remark.**
See (RNN: Computational Graph), except there is only one x, and many y's.

> (h0, x) → fW → (h1→y1) → fW → (h2→y2) → fW → (h3→y3) → … → (hT→yT)
"""

# ╔═╡ 2ddcae3a-5de2-4fa7-afda-7a24c33c22fb
md"### Example: Character-level Language Model"

# ╔═╡ 2b1fec6b-1a72-4c98-803b-618b3ce32596
md"""
**Example.**

- Vocabulary: [h,e,l,o]

- Example training sequence: "hello"

The diagram is illustrated:

$\text{Output layer: } \quad \begin{bmatrix} 1.0\\2.2\\-3.0\\4.1\end{bmatrix} \quad \begin{bmatrix} 0.5\\0.3\\-1.0\\1.2\end{bmatrix} \quad \begin{bmatrix} 0.1\\0.5\\1.9\\-1.1\end{bmatrix} \quad \begin{bmatrix} 0.2\\-1.5\\-0.1\\2.2\end{bmatrix}$

$\text{Hidden layer: } \quad \overset{↑}{\begin{bmatrix}0.3\\-0.1\\0.9\end{bmatrix}} → \overset{↑}{\begin{bmatrix}1.0\\0.3\\0.1\end{bmatrix}} → \overset{↑}{\begin{bmatrix}0.1\\-0.5\\-0.3\end{bmatrix}} \overset{W_{hh}}{→} \overset{↑}{\begin{bmatrix}-0.3\\0.9\\0.7\end{bmatrix}}$

$↑ \quad W_{xh}$ 

$\text{Input layer: } \quad \overset{↑}{\underbrace{\begin{bmatrix}1\\0\\0\\0\end{bmatrix}}_\text{``h''}} \quad \overset{↑}{\underbrace{\begin{bmatrix}0\\1\\0\\0\end{bmatrix}}_\text{``e''}} \quad \overset{↑}{\underbrace{\begin{bmatrix}0\\0\\1\\0\end{bmatrix}}_\text{``l''}}\quad \overset{↑}{\underbrace{\begin{bmatrix}0\\0\\1\\0\end{bmatrix}}_\text{``l''}}$

At test-time sample characters one at a time, feed back to model, e.g., use Softmax on output layer to produce sample, then feed this sample back into the model.
"""

# ╔═╡ 609dd371-e997-4c54-8be4-5080563a9152
md"""
**Remark.**

$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t)$
"""

# ╔═╡ dbbafe5a-33bd-44e5-b781-d46522f18e33
md"### Example of RNN Applications"

# ╔═╡ b31d2f7a-534c-4f11-96aa-faa974dc40d3
md"""
**Remark.**
Put in "The Sonnets" poem by William Shakespeare into an RNN model.

The results:

```
tyntd-iafhatawiaoihrdemot  lytdws  e ,tfti, astai f ogoh eoase rrranbyne 'nhthnee e plia tklrgd t o idoe ns,smtt   h ne etie h,hregtrs nigtike,aoaenns lng
```

...train more...


```
"Tmont thithey" fomesscerliund
Keushey. Thom here
sheulke, anmerenith ol sivh I lalterthend Bleipile shuwy fil on asterlome coaniogennc Phe lism thond hon at. MeiDimorotion in ther thize."
```

...train more...

```
Aftair fall unsuch that the hall for Prince Velzonski's that me of
her hearly, and behs to so arwage fiving were to it beloge, pavu say falling misfort how, and Gogition is so overelical and ofter.
```


...train more...

```
"Why do what that day," replied Natasha, and wishing to himself the fact the princess, Princess Mary was easier, fed in had oftened him.
Pierre aking his soul came to the packs and drove up his father-in-law women.
```
"""

# ╔═╡ b1183215-cdfd-4201-96a5-53c27c2ce251
md"### RNN Recap"

# ╔═╡ 259453b5-857f-4f97-8aba-fd8b7b24a3f1
md"""
**Remark.**

$s_t = σ(W s_{t-1} + U x_t), \qquad σ(⋅) = \tanh, \text{ReLU}$

$o_t = \text{softmax}(V s_t)$
"""

# ╔═╡ c9f4ed02-c5de-48a4-ab7b-40855a1118f5
md"### RNN Author"

# ╔═╡ af9946fa-a49c-40c8-8f3b-b4e9f552727c
md"""
**Remark.**

- Jeffrey Elman

  - 1948 -- 2018

  - Ph.D. from UT Austin, 1977

  - Professor at UCSD

  - RNN was called "Elman network"
"""

# ╔═╡ aa1cf0fa-d17d-42d3-90b6-3ffc8a4c3723
md"## Long Short Term Memory"

# ╔═╡ 4e301fbc-7ca7-4a96-96eb-67c8a050a70d
md"### Gating Mechanism relieves Vanishing Gradient"

# ╔═╡ 242a4c45-33bd-4898-a00e-83e4bc30ab7a
md"""
**Remark.**
RNN: keeps temporal sequence information.
"""

# ╔═╡ 4770f4b9-b4cb-4998-8bc9-e08c51bc28e4
md"""
**Example.**
"I grew up in France… I speak fluent *French*."
"""

# ╔═╡ f575b7aa-f84f-4c46-9506-f188eb1d504c
md"""
**Remark.**
Issue: in theory, RNNs can handle such "long-term dependencies," but they cannot in practice → use gates to directly encode the long-distance information.
"""

# ╔═╡ 3aae68af-bf27-451d-9594-0db29fdc9bd4
md"### Long Short-Term Memory (LSTM)"

# ╔═╡ 951926bb-1565-4dd1-996e-31ce788300e7
md"""
**Remark.**
LSTMs (Hochreiter and Schmidhuber, 1997) are explicitly designed to avoid the long-term dependency problem.

$\begin{align*}
𝐢_t &= σ(𝐱_t 𝐔^i + 𝐡_{t-1} 𝐖^i + 𝐛_i) \\
𝐟_t &= σ(𝐱_t 𝐔^f + 𝐡_{t-1} 𝐖^f + 𝐛_f) \\
𝐨_t &= σ(𝐱_t 𝐔^o + 𝐡_{t-1} 𝐖^o + 𝐛_o) \\
𝐪_t &= \tanh(𝐱_t 𝐔^q + 𝐡_{t-1} 𝐖^q + 𝐛_q) \\
𝐩_t &= 𝐟_t * 𝐩_{t-1} + 𝐢_t * 𝐪_t \\
𝐡_t &= 𝐨_t * \tanh(𝐩_t)
\end{align*}$

Three gates: 

- ``𝐢_t``: input gate

- ``𝐟_t``: forget gate

- ``𝐨_t``: output gate

``𝐱_t`` is the current input.

``𝐡_{t-1}`` is the history.
"""

# ╔═╡ 461faabc-d3b6-45a5-bb02-f69fde88e667
md"""
**Remark.**
Gates are a way to optionally let information through → composed of a *sigmoid* ``(σ)`` and a *pointwise multiplication* operation ``(𝐩_t \text{ and } 𝐡_t)``.
For sigmoid we have:

$\begin{cases}
1 &\text{``Completely keep this''} \\
0 &\text{``Completely get rid of this''} \\
\end{cases}$
"""

# ╔═╡ 8bb86d0c-7a63-4758-897f-c9a9e547df9b
md"""
**Remark.**
For ``𝐨_t`` you decide how much to output.
``𝐡_t`` is the final hidden vector you play with.
"""

# ╔═╡ a192d846-a323-4923-bf8e-4b1f15bc836f
md"### LSTM Authors"

# ╔═╡ bae73a44-a2fd-458a-aba8-fcbbf5ab5ce8
md"""
**Remark.**

- Sepp Hochreiter

  - Born in Mühldorf, Germany, 1967

  - Ph.D. from TU Munich

  - Professor at Johannes Kepler University, Austria

- Jürgen Schmidhuber

  - Born in Munich, Germany, 1963

  - Ph.D. from TU Munich

  - Director, Swiss AI Lab, IDSIA, Switzerland
"""

# ╔═╡ ff3f6611-c9dd-4579-b2af-b41f1b2f7773
md"## Gated Recurrent Unit (GRU)"

# ╔═╡ 2171c34e-23ce-4b12-976a-b18a7f31fae8
md"### Gated Recurrent Unit (GRU) (Cho et al., 2014) -- simplified LSTM"

# ╔═╡ 59815622-63bb-47a2-b0a2-b18c5a2194cc
md"""
**Remark.**

> LSTM → GRU
"""

# ╔═╡ 4f5298ff-eba5-4a2f-a10a-7ad1d7db859d
md"### GRU"

# ╔═╡ 929c16fd-ad60-43bc-85eb-534d1f1016eb
md"""
**Remark.**

$\begin{align*}
𝐳 &= σ(𝐱_t 𝐔^z + 𝐡_{t-1} 𝐖^z) \\
𝐫 &= σ(𝐱_t 𝐔^r + 𝐡_{t-1} 𝐖^r) \\
𝐬_t &= \tanh(𝐱_t 𝐔^s + (𝐡_{t-1} ∘ r) 𝐖^s) \\
𝐡_t &= (1 - 𝐳) ∘ 𝐬_t + 𝐳 ∘ 𝐡_{t-1}
\end{align*}$

Two gates:

- ``𝐳``

- ``𝐫``

``𝐬_t`` is the combination of the input with decayed history.

With ``𝐡_t`` we incorporate history again.
"""

# ╔═╡ 872bf814-3e32-4ed1-a33e-da9154d5cb2a
md"### GRU Authors"

# ╔═╡ 20855148-f2eb-4903-a74a-2fc266e6c713
md"""
**Remark.**

- Kyunghyun Cho

  - Born in South Korea

  - Ph.D. from Aalto University, Finland, 2014

  - Associate Professor at NYU
"""

# ╔═╡ 2c25328a-7124-4141-a165-66453391f352
md"### Recap"

# ╔═╡ 8d53df67-ea04-4205-b61f-3108b107a2c6
md"""
**Remark.**

- RNN: 𝐡, 𝐱

- LSTM: 𝐡, 𝐩, 𝐪, 𝐱

- GRU: 𝐡, 𝐬, 𝐱
"""

# ╔═╡ e155b836-9702-43c6-9028-d729653213c4
md"# Chapter 10: Algorithms -- Attention Mechanism and Transformers"

# ╔═╡ 42de6d7b-e26d-4b87-bfcb-f7a7aa44c8d0
md"## Motivation"

# ╔═╡ f5f6f7f2-0f46-48fe-9c87-7242ec994890
md"### The Biological Motivation"

# ╔═╡ 41443739-21c4-493f-a824-7bc501c5e6fb
md"""
**Remark.**

- The retina often has an image of a broader scene, yet human beings rarely use all the available sensory inputs to accomplish specific tasks.

- One pays greater attention to the relevant parts of the image.
"""

# ╔═╡ 5827b975-7b2d-4b03-b4b6-23795e201688
md"### Intuition in Image Captioning"

# ╔═╡ 3649588b-7fbb-4d99-9771-f566e983ff3c
md"""
**Example.**

1. **Input image** (Image of bird flying over water)

2. **Convolutional Feature Extraction** (14×14 Feature Map)

3. **RNN with attention over the image** (LSTM using attention)

4. **Word by word generation** ("A bird flying over a body of water")
"""

# ╔═╡ 32bf76ce-f531-4767-a7a9-731764bf25ff
md"## Attention in RNN"

# ╔═╡ 11f16c8e-e5f5-4b68-a146-0827972512a6
md"### A RNN Language Model"

# ╔═╡ bf080d0b-1093-4ce1-b2fa-d1637df0c7f7
md"""
**Remark.**
Output distribution:

$\hat{y} = \text{softmax}(W_2 h^{(t)} + b_2)$

Hidden states:

$h^{(t)} = f(W_h h^{(t - 1)} + W_e c_t + b_1)$

$h^{(0)} \text{ is initial hidden state!}$

Word embeddings:

$c_1,c_2,c_3,c_4$
"""

# ╔═╡ 91ee95b5-6535-4c28-953f-1d3426cb196a
md"""
**Example.**
"The students opened their"

- ``𝐡^{(0)}`` (``𝐖_h`` →)

- ``c_1`` - "the" (``𝐖_e`` →) ``𝐡^{(1)}`` (``𝐖_h`` →)

- ``c_2`` - "students" (``𝐖_e`` →) ``𝐡^{(2)}`` (``𝐖_h`` →)

- ``c_3`` - "opened" (``𝐖_e`` →) ``𝐡^{(3)}`` (``𝐖_h`` →)

- ``c_4`` - "their" (``𝐖_e`` →) ``𝐡^{(4)}`` (``𝐖_2`` →) ``\hat{y}^{(4)} = P(𝐱^{(5)} ∣ \text{the students opened their})``
"""

# ╔═╡ 2cb743ad-a167-4803-b26d-2087fbf86172
md"""
**Remark.**
*Is this really good?*

**RNN advantages:**

- Can process **any length** input

- **Model size doesn't increase** for longer input

- Computation for step ``t`` can (in theory) use information from **many steps back**

- **Weights are shared** across timesteps

**RNN disadvantages:**

- Recurrent computation is **slow**

- In practice, difficult to access information from **many steps back**
"""

# ╔═╡ f5392e81-4a71-4add-bab9-a391371ffa76
md"### A Bottleneck in RNN"

# ╔═╡ c63da16b-b502-4980-92ff-654e433ebbd4
md"""
**Remark.**
``𝐡^{(4)}`` is the representation of "the students opened their".
In general, the current representation encodes all past information.
"""

# ╔═╡ 685c9694-2d57-455e-9549-1f7e6918bd69
md"""
**Remark.**

> you can't cram the meaning of a whole %&@#&ing sentence into a single $*(&@ing vector!
>
> -- Ray Mooney (NLP professor at UT Austin)
"""

# ╔═╡ aa91b0b4-ffe7-46ca-843c-80a965b19bbc
md"### Idea: what if we use multiple vectors"

# ╔═╡ 0200c462-5642-42de-b5d6-cfa33cc97081
md"""
**Remark.**
``𝐡^{(4)}`` needs to capture all information about "the students opened their"

Instead of this, let's try:

"the students opened their" = ``\{𝐡^{(1)},𝐡^{(2)},𝐡^{(3)},𝐡^{(4)}\}`` (all 4 hidden states!)
"""

# ╔═╡ 5f3065a2-84ef-4d3a-a4e7-2f0624d72764
md"### The Solution: Attention"

# ╔═╡ 23129f27-bd9d-4931-8439-a0bc99e1f620
md"""
**Remark.**

- **Attention mechanisms** (Bahdanau et al., 2015) allow language models to focus on a particular part of the observed context at each time step.

- Originally developed for machine translation, and intuitively similar to *word alignments* between different languages.
"""

# ╔═╡ f5aab038-813c-496c-963f-3cb657ccf380
md"""### How does "Attention Mechanism" work?"""

# ╔═╡ 656ffc3e-6bbc-4595-be8c-8526fff85397
md"""
**Remark.**
In general, we need a single **query vector** and **multiple key vectors**.
We want to **score each query-key pair**.

*In a neural language model, what are the queries and keys?*
"""

# ╔═╡ 54a83ea5-3e2e-4b11-b05d-6268b48eb6ed
md"### Attention Mechanism in Neural Language Model"

# ╔═╡ 9329e350-19c9-4c0d-a617-1f196c6805f3
md"""
**Remark.**

Query 1: Hidden state at current time step

"the → students → opened → their → books"

Dot product with *keys* (encoder hidden states).
Attention scores:

- "the" ⋅ "books" → "the" score 

- "students" ⋅ "books" → "students" score

- "opened" ⋅ "books" → "opened" score

- "their" ⋅ "books" → "their" score

Attention distribution created from attention scores:

"the" > "students" = "opened" = "their"

Attention output created from attention distribution

- An attention output vector

We use the attention distribution to compute a weighted average of the hidden states.

Intuitively, the resulting attention output contains information from hidden states that received high attention scores.

Concatenate (or otherwise compose) the attention output with the current hidden state, then pass through a softmax layer to predict the next word:

"books" + Attention output → ``\hat{y}_1`` → "unwillingly"
"""

# ╔═╡ fcb82330-4c4c-4924-a773-10a3d03b79a5
md"### Attention Mechanism Strengths"

# ╔═╡ e0a2d404-d5ca-4656-80b0-29f59007999d
md"""
**Remark.**

- Attention solves the bottleneck problem

  - Attention allows decoder to look directly at source; bypass bottleneck

- Attention helps with vanishing gradient problem

  - Provides shortcut to faraway states

- Attention provides some interpretability

  - By inspecting attention distribution, we can see what the decoder was focused on

  - We get alignment for free!

  - This is cool because we never explicitly trained an alignment system

  - The network just learned alignment by itself
"""

# ╔═╡ e81f838a-b02e-45a1-92a7-35321a878bd7
md"""### Variants of Scoring "Query-Key" in Attention"""

# ╔═╡ 5c66daf1-7d67-4a30-afb9-cf1d5f005e60
md"""
**Remark.**

- Original formulation: ``a(𝐪, 𝐤) = w_2^T \tanh(W_1[𝐪;𝐤])``

- Bilinear product: ``a(𝐪,𝐤) = 𝐪^T W𝐤``

- Dot product: ``a(𝐪,𝐤) = 𝐪^T 𝐤``

- Scaled dot product: ``\displaystyle a(𝐪,𝐤) = \frac{𝐪^T 𝐤}{\sqrt{|k|}}``
"""

# ╔═╡ 87e00cb6-b1a6-497b-858b-18f669964370
md"## Attention in CNN"

# ╔═╡ bbb983e6-8459-4318-ac95-c5c112861314
md"""
**Remark.**

- Attention results in better performance

- Recurrent systems are still slow

- How about integrating attention mechanisms into CNN?
"""

# ╔═╡ 50c100e9-6993-40a8-a3a5-827aa990f3cc
md"### Attention in Convolutional Neural Networks"

# ╔═╡ 6c6a3802-c1b1-43d6-8b12-a529790c7ed2
md"""
**Remark.**

- Adding attention while keeping the hierarchical architecture

- Each local ``h_i`` takes into account its local context as well as the extra context

- Larger filter size to cover extra context

- What will happen if the two sentences are the same?
"""

# ╔═╡ d2468fde-7c60-43fd-b87a-7d98b57f2594
md"## Self-attention"

# ╔═╡ c1c94275-aad8-4026-8d4f-88b57a1b858f
md"""
**Remark.**
Rationales:

- An input has many cells (e.g., pixels in an image, words in sentences), each cell "pays attention to" its surrounding cells and accumulate their information

- A cell's representation relies on the cell itself as well as its context
"""

# ╔═╡ a4b6c78a-5996-41d8-8b93-2a97d13ad771
md"""
**Example.**

For each p element:
- → Q layer (query)
- → K layer (key)
- → V layer (value)

↑

Layer p: "Nobel" "committee" "awards" "Strickland" "who" "advanced" "optics"
"""

# ╔═╡ 51420ac5-c230-4fc9-8af1-add6f73dee5e
md"""
**Remark.**

- Each word has the opportunity to act as the query

- Whichever word is the query, other words, including the query word, act as keys
"""

# ╔═╡ 0f4290dd-163d-4356-a569-d532a6b38c21
md"""
**Remark.**

- Different attention distributions over the same word sequence
"""

# ╔═╡ 5989a383-8a02-4cbb-af9c-fe94bbe7e748
md"""
**Remark.**
Apply attention distributions over those "value" vectors ``(V)``.
"""

# ╔═╡ d02b9748-963e-460c-805a-a4f388a23c8e
md"""
**Remark.**

- M - Context of "Nobel"

- M - Context of "committee"
"""

# ╔═╡ 59080d5d-3ac6-4d52-8faf-1d3088e0aa67
md"### Multi-head Self-attention"

# ╔═╡ d84ee4c9-78d8-4796-a422-0760a73dfdf2
md"""
**Remark.**
We have collection of contexts: ``\{M_1 … M_H\}`` → Feed Forward → Combine context and word representation
"""

# ╔═╡ 1abb46cb-7323-4146-8c6a-29d7cd4e84af
md"""
**Remark.**

- Layer J: Multi-head self-attention + feed forward

- Layer p: Multi-head self-attention + feed forward

- Layer 1: Multi-head self-attention + feed forward
"""

# ╔═╡ 3893fa74-016b-4ddc-b023-4667d8c05c53
md"## Transformers (Attention is All You Need)"

# ╔═╡ 794fbc5d-33ce-4d25-88e0-b160491ba69a
md"### Attention is All You Need"

# ╔═╡ 0fa3e5da-bd54-429b-ab18-b22e8934524c
md"""
**Remark.**

> The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder.
> The best performing models also connect the encoder and decoder through an attention mechanism.
> **We propose a new simple network architecture, the Transformer**, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.
"""

# ╔═╡ b31409fb-960c-4497-ac70-5f631ae94613
md"### Image Captioning using transformers"

# ╔═╡ d8f1d7a4-4546-4db5-934a-21d5959ed171
md"""
**Remark.**

- **Input**: Image ``𝐈``

- **Output**: Sequence ``𝐲 = \{y_1,y_2,…,y_T\}``

- **Encoder**: ``𝐜 = T_W(𝐳)`` where ``𝐳`` is spatial CNN features and ``T_W(⋅)`` is the transformer encoder.

  Extract spatial features from a pretrained CNN → Features: H × W × D


- **Decoder**: ``y_t = T_D(𝐲_{0:t-1}, 𝐜)`` where ``T_D(⋅)`` is the transformer decoder.
"""

# ╔═╡ 918c1118-9200-4d5a-adbe-6e9db022d0f8
md"### Layer Normalization"

# ╔═╡ bf0f709d-0264-4e48-9e1b-3c5bc8dab12e
md"""
**Remark.**
To overcome the issues of longer training time and instability
"""

# ╔═╡ 8b05ead9-d26e-4608-9ec2-89ed4b48546e
md"### The Transformer Encoder"

# ╔═╡ 28f85d3f-b29d-4835-9933-226a9d291c7a
md"""
**Remark.**
Architecture of an encoder block (component of transformer encoder):

- Residual connection

  ↑

- MLP over each vector individually

  ↑

- LayerNorm over each vector individually

  ↑

- Residual connection

  ↑

- Attention attends over all the vectors

  ↑

- Add positional encoding
"""

# ╔═╡ 2ebf3c76-0d3f-4426-8247-bc9abc50a7ad
md"""
**Remark.**
Transformer Encoder Block:

- Inputs: Set of vectors 𝐱

- Outputs: Set of vectors 𝐲

Self-attention is the only interaction between vectors.

Layer norm and MLP operate independently per vector.

Highly scalable, highly parallelizable, but high memory usage.
"""

# ╔═╡ e3a9d7fc-9c84-4853-9032-64a526186316
md"### The Transformer Decoder"

# ╔═╡ 020d00e4-67b5-4f0f-be84-81aa695f741d
md"""
**Remark.**
Architecture of decoder block (component of transformer decoder):

- Most of the network is the same as the transformer encoder.

- Multi-head attention block attends over the transformer encoder outputs.

- For image captions, this is how we inject image features into the decoder.
"""

# ╔═╡ 2ca9fdc0-cfdc-412b-87a2-4d12db67d65a
md"### Summary"

# ╔═╡ 2fe82dd5-ad59-4ce8-86f3-67f336f9429f
md"""
**Remark.**

- Adding **attention** to RNNs/CNNs allows them to "attend" to different parts of the input at every time step.

- **Transformers** are a type of layer that uses self-attention and layer norm.

  - It is highly scalable and highly parallelizable.

  - Faster training, larger models, better performance across vision and language tasks.

  - They are quickly replacing RNNs, LSTMs, and may even replace convolutions.
"""

# ╔═╡ 9429c73e-87b7-4087-941a-36dc77af576e
md"### Image Captioning using transformers"

# ╔═╡ c570b8be-f280-4521-84c0-36002256a942
md"""
**Remark.**
No recurrence at all.
"""

# ╔═╡ dea301a0-7b2b-4c91-bb07-6275f822a4e6
md"""
**Remark.** *Perhaps we don't need convolutions at all?*
"""

# ╔═╡ f6421a9a-dc72-4ff1-9d99-c11ef0963edf
md"### Image Captioning using ONLY transformers"

# ╔═╡ 4b48abc9-de19-4f46-b8a2-a8a7dc22eff1
md"""
**Remark.**
Transformers from pixels to language.
"""

# ╔═╡ 5198ccf5-60be-4923-b8cd-ffdb12d9cbdb
md"### Summary"

# ╔═╡ ca463cd7-56e8-4b0d-868c-aaf1f4dbf028
md"""
**Remark.**

- Adding **attention** to RNNs/CNNs allows them to "attend" to different parts of the input at every time step.

- **Transformers** are a type of layer that uses self-attention and layer norm.

  - It is highly **scalable** and highly **parallelizable**.

  - **Faster** training, **larger** models, **better** performance across vision and language tasks.

  - They are quickly replacing RNNs, LSTMs, and may even replacec convolutions.
"""

# ╔═╡ ce5ff998-d87d-47d0-bd89-c8dd379281ac
md"# Chapter 11: Training -- Loss Functions"

# ╔═╡ 7813a8d5-2d3f-4cc2-a68e-52bb5ee8f85b
md"## What is Loss Function"

# ╔═╡ 46fc521a-732e-426f-8786-b8d28a273894
md"""
**Remark.**

- **Loss Function**: a function of all parameters that quantifies the difference between model predictions and ground truth across the training data.

- **Model Training**: gradually update model parameters so as to minimize the loss function.
"""

# ╔═╡ 6e28ed29-b69b-4f22-a5b1-9cacbf998cf0
md"""
**Example.**
Suppose: 3 training examples, 3 classes.
With some ``W`` the scores ``f(x,W) = Wx`` are:

$\begin{array}{cccc}
\text{cat} & 3.2 & 1.3 & 2.2 \\
\text{car} & 5.1 & 4.9 & 2.5 \\
\text{frog} & -1.7 & 2.0 & -3.1
\end{array}$

A **loss function** tells how good our current classifier is.

Given a dataset of examples

$\{(x_i,y_i)\}_{i=1}^N$

Where ``x_i`` is image and ``y_i`` is (integer) label.

Loss over the dataset is a sum of loss over examples:

$L = \frac{1}{N} \sum_i L_i(f(x_i, W), y_i)$
"""

# ╔═╡ 12c5cb8f-4dff-48de-bd73-9c4f6b250819
md"## Loss Functions in Classification"

# ╔═╡ 875c70d8-5071-4a66-97b3-47d84c3acc5a
md"### Loss Function in Multi-class Classification"

# ╔═╡ d95a4a78-4f59-4e4e-bc41-0ecfd0762429
md"""
**Remark.**

$P(Y = k ∣ X = x_i) = \frac{e^s k}{\sum_j e^{s_j}} \quad\text{where}\quad s = f(x_i; W)$

Want to maximize the log likelihood, or (for a loss function) to minimize the negative log likelihood of the correct class:

$L_i = -\log P(Y = y_i ∣ X = x_i)$

In summary:

$L_i = -\log\left(\frac{e^{s_{y_i}}}{\sum_j e^{s_j}}\right)$

!!! question "Question 1"
	What's the max/min possible loss ``L_i``?

!!! question "Question 2"
	Usually at initialization ``W`` is so small so all ``s ≈ 0``.
	What is the loss?
"""

# ╔═╡ 6aeeacb3-5b54-49fb-b429-875f8e29b1c5
md"### (Negative Log-likelihood) Loss Function in Multi-class Classification"

# ╔═╡ a93bb539-5ff8-4dd9-a28c-13aaeecac391
md"""
**Remark.**

$P(Y = k ∣ X = x_i) = \underbrace{\frac{e^s k}{\sum_j e^{s_j}}}_\text{Softmax} \quad\text{where}\quad s = f(x_i; W)$

Softmax function converts scores into probabilities.

Want to maximize the log likelihood, or (for a loss function) to minimize the negative log likelihood of the correct class:

$L_i = -\log{P(Y = y_i ∣ X = x_i)}$

In summary:

$L_i = -\log{\left(\frac{e^{s_{y_i}}}{\sum_j e^{s_j}}\right)}$
"""

# ╔═╡ c0cb4145-6369-4877-8273-2fa45ee42c47
md"""
**Example.**

$\underset{\text{raw scores}}{\begin{bmatrix}3.2\\5.1\\-1.7\end{bmatrix}} \overset{\exp}{⟶} \overset{\text{exponential function}}{\begin{bmatrix}24.5\\164.0\\0.18\end{bmatrix}} \overset{\text{normalize}}{⟶} \underset{\text{probabilities}}{\begin{bmatrix}0.13\\0.87\\0.00\end{bmatrix}} ⟶ L_i = -\log_{10}(0.13) = 0.89$

Q1: What's the max/min possible loss ``L_i``

Q2: Usually at initialization ``W`` is small so all ``s ≈ 0``.
What is the loss?

For each training example ``i``, compute the probability of gold class, ``p_i``, then loss = ``-\sum_i \log{p_i}``.
"""

# ╔═╡ afbf6aae-ef77-436f-a8c8-aacb1eee97a1
md"### Pay Attention!!!"

# ╔═╡ 47c1e967-07c5-4505-87c6-6010c48c386b
md"""
**Remark.**
In multi-class classification:

Negative Log-likelihood Loss = Cross Entropy Loss

Negative Log-likelihood Loss:

$\mathcal{L} = -\sum_i \log{\hat{p}(y_i)}$

Cross Entropy Loss:

**Entropy** (of a discrete random variable ``X``): the average level of "uncertainty"/"diversity" to the variable's possible outcomes.

$H(X) := -\sum_{x ∈ X} p(x) \log{p(x)}$

**Cross Entropy** (of two distributions ``p`` and ``\hat{p}``): measures how similar the predicted distribution ``\hat{p}`` towards the gold distribution ``p``.

$H(p,\hat{p}) = -\sum_x p(x) \log{\hat{p}(x)}$

In multi-class classification, ``p = [0,0,…,0,1,0,…,0]``
"""

# ╔═╡ ea18fc6a-9f5b-4cc9-b1a0-a5936b144da9
md"""
**Remark.**

Negative Log-likelihood Loss (not always) = Cross Entropy Loss

!!! note "Given two distributions, cross-entropy always applies; but negative-log-likelihood doesn't."
"""

# ╔═╡ 42f75269-87b3-42c8-85c8-0a5c3db72d92
md"### Calculate Cross Entropy H(P,Q)"

# ╔═╡ ef868b9a-e5c5-47af-9495-d42590668eb8
# Example of calculating cross entropy
let
	# Calculate cross entropy
	function crossentropy(p, q)
		-sum([p[i]*log2(q[i]) for i ∈ 1:length(p)])
	end

	# Define data
	p = [0.10, 0.40, 0.50]
	q = [0.80, 0.15, 0.05]

	# Calculate cross entropy H(P, Q)
	ce_pq = crossentropy(p, q)

	# Calculate cross entropy H(Q, P)
	ce_qp = crossentropy(q, p)

	"H(P,Q)" => ce_pq, "H(Q,P)" => ce_qp
end

# ╔═╡ 6bf77be2-ae30-4d54-bc6f-86b993a5de64
md"## Loss Functions in Regression"

# ╔═╡ 4dd633bf-c30c-4fd5-a635-b02ae6e287d0
md"### Mean Squared Error (MSE or L2) in Regression"

# ╔═╡ ba5de3ce-3d12-4a1b-bc07-6d79c7b3b652
md"""
**Remark.**

- **Regression**: the model predictions and the expected outputs both are continuous values (e.g., house price)

  $\mathcal{L} = \frac{1}{n} \sum_{i=1}^n (y_i - f(x_i))^2$

- **Advantage**: the MSE is great for **ensuring that our trained model has no outlier predictions** with huge errors, since the MSE puts larger weight on these errors due to the squaring part of the function.

- **Disadvantage**: Mean Square Error loss is more sensitive to outliers due to using the square difference (if residual is twice as large, loss is 4 times as large).
"""

# ╔═╡ 7813b028-75f0-46cd-bc3b-2853703b9b5b
md"### Mean Absolute Error (MAE or L1) in Regression"

# ╔═╡ bca965cb-14c2-467d-8207-e64c35cd6aea
md"""
**Remark.**

$\mathcal{L} = \frac{1}{n} \sum_i |y_i - f(x_i)|$

- **Advantage**: More robust, outliers in y are less influential than for L2

- **Disadvantages**: No derivatives when ``y_i = f(x_i)``, **optimization becomes harder**
"""

# ╔═╡ cb189e4c-f882-43d1-a48e-44f095905861
md"### Huber Loss: combine MSE and MAE"

# ╔═╡ cd1cd5d3-94dd-4aa1-bed6-f35d1ad0d3ab
md"""
**Remark.**

$L_δ(y,f(x)) = \begin{cases}
\frac{1}{2}(y-f(x))^2 &\text{for } |y - f(x)| ≤ δ, \\
δ|y-f(x)| - \frac{1}{2} δ^2 &\text{otherwise.}
\end{cases}$

For loss values less than delta, use the MSE; for loss values greater than delta, use the MAE.
"""

# ╔═╡ 603df4eb-52e6-4cb5-8e27-16ffc0d6e135
md"## Loss Functions in Ranking"

# ╔═╡ 6c1fef0d-01e9-443e-8a73-f4df5d139664
md"### Ranking Loss"

# ╔═╡ a8760206-ea8b-46ec-ace1-b4fc86dcc7a4
md"""
**Remark.**
Used when our focus is to make some instances ranked higher than other instances, i.e., webpage ranking

$l_i = \max(0,m-s_p+s_n)$

- ``m``: a margin parameter

- ``s_p``: score for positive instances

- ``s_n``: score for negative instances

In general, we prefer positive instances to get higher scores than negative instances.
"""

# ╔═╡ 6aa50070-3963-4504-a886-893f242973cc
md"""
**Remark.**
When the "similarity" score is expressed by "distance"

$l_i = \max(0,m-s_p+s_n)$

$↓$

$l_i = \max(0,m+d_p-d_n)$

- ``m``: a margin parameter

- ``d_p``: distance for positive instances

- ``d_n``: distance for negative instances
"""

# ╔═╡ 83632ad9-2fce-42d5-b046-b0268cb86032
md"""
!!! note
	In addition to working on two instances (one positive, the other negative), ranking loss is more popularly applied to triples.
"""

# ╔═╡ afa4fc91-a848-488b-87c5-2ef0b36cc2b5
md"""
**Remark.**
Ranking loss tries to learn the representations so that

- Similar items have closer representations

- While dissimilar items have distant representations
"""

# ╔═╡ dee63da6-8a0c-4f8b-b8bf-426694d7ae1c
md"""
**Remark.**

$l_i = \max(0,m+d(x,x^+) - d(x,x^-))$

or

$l_i = \max(0,m-s(x,x^+) - s(x,x^-))$
"""

# ╔═╡ c31c63dd-aaf8-4f52-a479-06be9a19b252
md"""
**Remark.**

$l_i = \max(0,m-s_p+s_n)$
"""

# ╔═╡ 53fd0814-1c1e-433c-960e-fc7ceab43d34
md"""
**Remark.**

- Many problems are ranking, such as binary classification, clustering

- How to select negative pairs influences the performance a lot (more challenging the better)

- Ranking loss has other names in different scenarios/interpretations: Margin loss, Contrastive loss, Triplet loss.
"""

# ╔═╡ d8f2fe6f-9a63-4aac-9aa5-f0a2e81c76c4
md"# Chapter 12: Training -- Gradient Descent"

# ╔═╡ 0f4df7ca-52c8-4a9f-8ad4-e47141469089
md"## Gradient Descent"

# ╔═╡ e56ee78b-e64d-4587-b49d-4a434c5f32ac
md"### Motivation of Gradient Descent"

# ╔═╡ 0e499521-9990-4ef5-a469-82a2ca7a411b
md"""
**Remark.**

- It is good for finding **global minimia/maxima** if the function is **convex**.

- It is good for finding **local minima/maxima** if the function is **not convex**.

- It is used for optimizing many models in Machine Learning

  - It is used in conjunction with

    - Neural Networks

    - Linear Regression

    - Logistic Regression

    - Support Vector Machines
"""

# ╔═╡ 4810516b-c686-425b-b07e-5f08fa906b9b
md"### Non-convex Function (e.g., deep neural nets)"

# ╔═╡ e96cc4dd-b0ee-4201-ab6c-a594e9df6424
md"""
**Illustration.**
Curve with global minimum/maximum and local minimum/maximum.
"""

# ╔═╡ ab52b812-b2e9-4602-a258-1fa75230396e
md"### Quickest Ever Review of Multivariate Calulus"

# ╔═╡ 8c85d9ff-9890-4d7f-bbf9-66487415e856
md"### Derivative"

# ╔═╡ ab0449b6-98c8-4409-9a24-51c4854938c1
md"""
**Remark.**

$f(x) = x^2$

$f'(x) = 2x$

$f''(x) = 2$

- Easy when the function is univariate.
"""

# ╔═╡ 4710788d-f9c7-4d8a-8eee-366a375525db
md"### Chain Rule in Derivative"

# ╔═╡ e38f4cbe-0de2-4876-a441-e66d8710733e
md"""
**Remark.**

$f(x) = A(B(C(D(E(F(G(x)))))))$

$f'(x) = A' ⋅ B' ⋅ C' ⋅ D' ⋅ E' ⋅ F' ⋅ G'(x)$
"""

# ╔═╡ e008c4ed-821a-4e38-bee5-5ec40d73c2b8
md"### Partial Derivative -- Multivariate Functions"

# ╔═╡ db81e6c1-eb5d-477c-b74e-c7f64e34e8c9
md"""
**Remark.**
For multivariate functions (e.g., two variables), **we need partial derivatives: one per dimension**.
"""

# ╔═╡ 4692724a-3a15-4bc5-9bb6-5d0cc8bf9037
md"""
**Remark.**
Examples of multivariate functions:

- ``f(x,y) = x^2 + y^2``

- ``f(x,y) = -x^2 - y^2``

- ``f(x,y) = \cos^2(x) + y^2``

- ``f(x,y) = \cos^2(x) + \cos^2(y)``
"""

# ╔═╡ 385580e7-70e8-4961-8f76-b65e0a9d80fb
md"### Partial Derivative -- Cont'd"

# ╔═╡ b4ad42b0-9d73-45b3-8c91-390d4da6ad3c
md"""
**Remark.**
To visualize the partial derivative of each of the dimensions ``x`` and ``y``, we can imagine a plane that "cuts" out surface along the two dimensions and once again we get the slope of the tangent line.
"""

# ╔═╡ bcd553d8-ed5c-46d3-bbe8-a75896c3c9ad
md"""
**Example.**

- Surface: ``f(x,y) = 9-x^2-y^2``

- Plane: ``y = 1``

- Cut: ``f(x,1) = 8 - x^2``

- Slope/derivative of cut: ``f'(x) = -2x``
"""

# ╔═╡ 58d6547c-507d-434e-a205-9ac68a3495d3
md"### Partial Derivative -- Cont'd 2"

# ╔═╡ b84e49d8-01f6-40d9-a08f-42f92ff0fae9
md"""
**Remark.**
If we partially differentiate a function with respect to ``x``, we pretend ``y`` is constant.

**Task:**
Given ``f(x,y) = 9 - x^2 - y^2`` compute ``\frac{∂f}{∂x}`` and ``\frac{∂f}{∂y}``.
"""

# ╔═╡ 8fcaf7a3-f8dc-45d5-9c35-2fb286318cda
md"### Partial Derivative -- Tangent Plane"

# ╔═╡ 317ee97a-199b-4ff6-8526-fa6a7e5731a1
md"""
**Remark.**
The two tangent lines that pass through a point, define a tangent plane to that point.
"""

# ╔═╡ 428cbc38-7735-4ca7-888b-aa2f2af80e76
md"### Gradient Vector"

# ╔═╡ 7d3f466a-ed5b-4d44-8e95-be8f3c8b43a4
md"""
**Remark.**
The vector that has as coordinates the partial derivatives of the function

$f(x,y) = 9 - x^2 - y^2$

$∇f = \left(\frac{∂f}{∂x},\frac{∂f}{∂y}\right) = (-2x,-2y)$
"""

# ╔═╡ 2a02ba31-4459-4a97-b3cb-92a39105e3f9
md"### Gradient Descent Algorithm & Walkthrough"

# ╔═╡ e5507385-cdc7-4a3e-b918-47538397fc44
md"""
**Remark.**

**Idea:**

- Start somewhere

- Take steps based on the gradient vector

**Convergence:**

- Happens when changes between two steps ``< ε``
"""

# ╔═╡ 98209c74-16d6-4af7-826f-d5a32ba1a711
md"### Gradient Descent Algorithm"

# ╔═╡ 986c8fef-302b-417b-b087-1ae9771ae843
md"""
**Remark.**
Suppose we want to solve

$\min_w \mathcal{L}(w)$

In many machine learning problems, we have that ``\mathcal{L}(w)`` is of the form

$\mathcal{L}(w) = \frac{1}{n} \sum_{i=1}^n l_i(f_w(x_i),y_i)$

Gradient descent (GD):

$w_{t+1} = w_t - α_t ∇ \mathcal{L}(w_t)$
"""

# ╔═╡ 651a6161-f630-4630-b8aa-342577084212
md"### Gradient Descent Code (python)"

# ╔═╡ 8ae5469d-53ee-4618-b494-1dcba2808086
md"""
```python
# From calculation, we expect that local minimum occurs at x=9/4

x_old = 0
x_new = 6 # The algorithm starts at x=6
eps = 0.01 # step size
precision = 0.00001

def f_prime(x):
	return 4 * x**3 - 9 * x**2

while abs(x_new - x_old) > precision:
	x_old = x_new
	x_new = x_old - eps * f_prime(x_old)
print "Local minimum occurs at ", x_new
```
"""

# ╔═╡ 115fbd4a-7c96-43b9-aaaa-03deed5a5429
md"### Potential Issues of Gradient Descent--Convexity"

# ╔═╡ dee11e67-8854-4e5d-a242-84f203ca0570
md"""
**Remark.**
We need a convex function → so there is a global minimum.

$f(x,y) = x^2 + y^2$
"""

# ╔═╡ 9b8d1f9f-76ea-4e40-b0bc-2f37585438de
md"### Potential Issues of Gradient Descent--Convexity (2)"

# ╔═╡ 7e9d3303-4f9e-43c6-9d4d-a2e4e72251c6
md"""
**Remark.**
In real world, we often cannot get convex function, so we can only find local minimum.
"""

# ╔═╡ 1216633f-35db-4763-832e-7f74ce1933bd
md"### Potential Issues of Gradient Descent--Step Size"

# ╔═╡ 67c3666a-0402-416a-8498-1a2189f7fd49
md"""
!!! note
	Bigger steps lead to faster convergence?
"""

# ╔═╡ c834987c-e8f2-43e5-9d06-ae119c74ffb9
md"### Potential Issues of Gradient Descent--Computationally Expensive"

# ╔═╡ 144c994f-0ac3-4b31-88dc-0e7f55fba269
md"""
**Remark.**

$\mathcal{L} =\frac{1}{n}  \sum_{i=1}^n l_i (f_w(x_i),y_i)$

!!! note
	$w_{t+1} = w_t - α_t ∇\mathcal{L}(w_t)$

One practical difficulty is that computing the gradient itself can be **costly, particularly when ``n`` is large.**
"""

# ╔═╡ fff326ae-3707-4862-9509-f8e97adda2e7
md"### Two Solutions to the Issues of Gradient Descent"

# ╔═╡ 3d941738-2678-46d6-99c2-5ddbf7729155
md"""
**Remark.**

$w_{t+1} = w_t - \underbrace{α_t}_{\substack{(2) \text{ adaptive} \\ \text{learning rate}}} \quad \underbrace{∇\mathcal{L}(w_t)}_{\substack{\text{(1) use an} \\ \text{approximation}}}$
"""

# ╔═╡ 575db528-16dc-4928-bf28-f4ec739a00fb
md"## Stochastic Gradient Descent"

# ╔═╡ 47524d32-5763-4d92-88d3-4a30593d04dd
md"""
**Remark.**
Most loss functions in machine learning problems are **separable**:

$\mathcal{L}(w) = \frac{1}{n} \sum_{i=1}^n l_i (f_w(x_i), y_i)$

For example:

- Mean Square Error: ``\displaystyle \mathcal{L} = \frac{1}{n} \sum_{i=1}^n (y_i - f(x_i))^2``

- Negative Log-likelihood Loss: ``\displaystyle = \mathcal{L} = -\sum_{i=1}^n \log \hat{p}(y_i)``
"""

# ╔═╡ 1f7cba77-2220-4538-966b-bc3afd673bfe
md"""
**Remark.**

Vanilla **gradient descent**:

$w_{t+1} = w_t - α_t \underbrace{∇\mathcal{L}(w_t)}_{\text{main computation}}$

The full gradient of the loss is

$∇\mathcal{L}(w) = \frac{1}{n} \sum_{i=1}^n ∇l_i (w)$

**Stochastic gradient descent**:

$∇\mathcal{L}(w) ≈ \frac{1}{|\mathcal{B}|} \sum_{i∈\mathcal{B}}∇l_i(w)$

where ``\mathcal{B} ⊆ \{1,…,n\}`` is a random subset; ``|\mathcal{B}|`` = batch size
"""

# ╔═╡ 6a0b91da-2a43-4eff-954f-26459f3731c5
md"### Stochastic Gradient Descent (SGD) Algorithm"

# ╔═╡ 4c97768a-1d75-402b-b77f-089e26234f18
md"""
**Remark.**
The SGD algorithm is:

- Given ``\{(x_i,y_i) ∣ i = 1, …, n\}``

- Initialize ``w`` (Chapter 7 Deep Learning Basics)

- For ``t = 1,2,3,…``

  - Draw a random subset ``\mathcal{B} ⊆ \{1,…,n\}``

  - Update

    $w_{t+1} = w_t - α_t \frac{1}{|\mathcal{B}|} \sum_{i ∈ \mathcal{B}} ∇ l_i (w)$

If ``|\mathcal{B}| = 1``, then use only one sample at a time.

Some notes:

- **Online learning** algorithm

- Instead of going through the entire dataset on **each iteration, randomly sample and update the model**
"""

# ╔═╡ bc283d66-fe72-4d46-84f4-38b88e6a6bf6
md"### Perspectives of SGD"

# ╔═╡ b4459113-a276-461e-bd05-2457c6922508
md"""
**Remark.**
**In *convex* problems, classical optimization literature have the following observations:**

- SGD offers a trade-off between accuracy and efficiency

- More iterations

- Less gradient computation per iteration

- Noise is a by-product

**Recent studies of SGD for *non-convex* problems (e.g., deep learning) found that**

- SGD for training deep neural networks works

- SGD finds solution **faster**

- SGD **find a better local minima**

- Noise matters
"""

# ╔═╡ 965c4729-0532-4d3f-a884-d0a29ee04fe5
md"### SGD vs. GD"

# ╔═╡ 296ff300-eaf6-4f7f-8906-e823cab3c381
md"""
**Remark.**
Additionally, sampling the data in SGD leads to "noise" that can avoid finding "shallow local minima."
This is good for optimizing non-convex functions.
"""

# ╔═╡ 20514d65-4d3b-4dac-8ab5-0d3126f03b28
md"### What is a good batch size |β| for SGD?"

# ╔═╡ e9705a2b-e9b3-462b-affb-13b7dad024b5
md"""
**Remark.**
There is no definite answer.

- **Look at the validation curve to determine** if you need to increase/decrease the mini-batch size.

- Is typically chosen between 1 and a few hundreds (some people like ``2^x``)

- Larger batch size means larger memory required.
  When you play huge neural networks, such as Transformers, you often do not have many choices due to the GPU memory limitation.
"""

# ╔═╡ b79a8571-6840-4c6c-a809-f5bade9443e1
md"## Adaptive Learning Rate"

# ╔═╡ aa27c1a4-5efa-4275-949c-92505963e1ff
md"""
**Illustration.**

- Steep slope. Value of D is high so take large steps.

- Slope is less steep. Value of D is low so take small steps.
"""

# ╔═╡ bddd0a62-7807-4cb2-9e27-976c4f1c6883
md"""
**Remark.**
So far, we've looked at update steps that look like

$w_{t + 1} = w_t - α_t ∇\mathcal{L}(w_t)$

- Here, the learning rate/step size is a **fixed a priori** for each iteration.

- What if we use a learning rate **that varies depending on the model**?

- This is the idea of an **adaptive learning rate**.
"""

# ╔═╡ 930a1192-15f6-47d2-8afe-0190ef3cee9f
md"### Popular Methods for Adaptive Learning Rate"

# ╔═╡ a5cc6f89-1fd5-4af9-8a18-4732f7b847ef
md"""
**Remark.**

- AdaGrad (Duchi et al., 2011): Adaptive Gradient Algorithm

- RMSProp (Hinton et al., 2018): Root Mean Square Propagation

- Adam (Kingma and Ba, 2015): Adaptive Moment Estimation
"""

# ╔═╡ b0ca47e5-9a61-4444-b3e8-13002c5f54e0
md"### Per-parameter Adaptive Learning Rate Schemes"

# ╔═╡ cc8e4392-d65b-4c9a-9a16-40e4e0ca9cf1
md"""
**Remark.**
Main idea: set the **learning rate per-parameter dynamically** at each iteration based on observed statistics of the past gradients.

$w_{t+1} = w_t - α_t ∇ \mathcal{L}(w_t)$

$↓$

$w_{t+1}^i = w_t^i - α_t^i ∇ \mathcal{L}(w_t^i)$

There are many different schemes in this class
"""

# ╔═╡ eda68dd9-90f8-4e5a-8f27-362dfa53e17c
md"### AdaGrad: One of the first adaptive methods"

# ╔═╡ ac8b10fb-fa63-4194-8cd7-99775ceda837
md"""
**Remark.**
Use **history of sampled gradients** to choose the step size for the next step.

!!! algorithm "Algorithm: AdaGrad"

	- **Input**: initial learning rate ``α``, initial parameters ``w_0``

	- **initialize** ``t \gets 1``

	- **loop**

	  - sample a stochastic gradient ``g_t^i \gets ∇ \mathcal{L}(w_t^i)``

	  - **accumulate past gradients** ``\mathcal{G}_t^i = \mathcal{G}_{t-1}^i + (g_t^i)^2``

	  - update model: for all ``i ∈ \{1,…,d\}``

	    $w_{t+1}^i = w_t^i - \frac{α}{\sqrt{\mathcal{G}_t^i} + ϵ} ⋅ g_t^i$

	  - ``t \gets t + 1``

	- **end loop**
"""

# ╔═╡ 799a635c-cee4-4dbd-9ad2-4e1789469312
md"### Issue of AdaGrad"

# ╔═╡ 93092b86-273e-4b52-a80f-51c9cb9fa54d
md"""
**Remark.**

- What problems might arise when using AdaGrad for non-convex optimization?

  - Think about the step size always decreasing.
    Could this cause a problem?

- If you do think of a problem that might arise, how could you change AdaGrad to fix it?
"""

# ╔═╡ fd7b6013-73cb-4778-99a5-b9c9861857ed
md"### RMSProp--Divide the gradient by a running average of its recent magnitude"

# ╔═╡ a9e0a585-8c4a-415a-b9d1-8b8b360805f5
md"""
!!! algorithm "Algorithm: RMSProp"

	- **Input**: initial learning rate ``α``, initial parameters ``w_0``, weight ``ρ``

	- **initialize** ``t \gets 1``

	- **loop**

	  - sample a stochastic gradient ``g_t^i \gets ∇ \mathcal{L}(w_t^i)``

	  - **accumulate past gradients ``\mathcal{G}_t^i = ρ ⋅ \mathcal{G}_{t-1}^i + (1 - ρ) ⋅ (g_t^i)^2`` for all parameters**

	  - update model: for all ``i ∈ \{1,…,d\}``

	    $w_{t+1}^i = w_t^i - \frac{α}{\sqrt{\mathcal{G}_t^i} + ϵ} ⋅ g_t^i$

	  - ``t \gets t + 1``

	- **end loop**
"""

# ╔═╡ c64d7f92-19ee-408f-8d41-8da41ad56cdf
md"### Adam (biased version)"

# ╔═╡ 4b456b10-51c1-44ea-96b3-e551f116b34a
md"""
!!! algorithm "Algorithm: Adam (biased version)"

	- **Input**: initial learning rate ``α``, initial parameters ``w_0``, weight ``ρ_1,ρ_2``

	- **initialize** ``t \gets 1``

	- **loop**

	  - sample a stochastic gradient ``g_t^i \gets ∇ \mathcal{L}(w_t^i)``

	  - accumulate past gradients in two ways:

	    $\mathcal{F}_t^i = ρ_1 ⋅ \mathcal{F}_{t-1}^i + (1 - ρ_1) ⋅ (g_t^i)^2 \text{ and } \mathcal{G}_t^i = ρ_2 ⋅ \mathcal{G}_{t-1}^i + (1 - ρ_2) ⋅ (g_t^i)^2$

	  - update model: for all ``i ∈ \{1,…,d\}``

	    $w_{t+1}^i = w_t^i - α ⋅ \frac{\mathcal{F}_t^i}{\sqrt{\mathcal{G}_t^i} + ϵ} ⋅ g_t^i$

	  - ``t \gets t + 1``

	- **end loop**
"""

# ╔═╡ 0426433a-c0c1-4e5a-86f8-79b789546381
md"### Where does bias come from?"

# ╔═╡ 3776555e-1519-4d4d-89c9-bd86d3d2e642
md"""
**Remark.**

- Normally in practice ``ρ_2`` is set much closer to ``1`` than ``ρ_1`` (as suggested by the author ``ρ_2 = 0.999, ρ_1 = 0.9``), so the update coefficients ``1 - ρ_2 = 0.001`` is much smaller than ``1 - ρ = 0.1``.

- In the first step of training ``\mathcal{F}_1^i = 0.1 ⋅ g_1^i, \; \mathcal{G}_1^i = 0.001 ⋅ (g_1^i)^2``; the ``\displaystyle \frac{\mathcal{F}_1^i}{\sqrt{\mathcal{G}_1^i} + ϵ} ≈ \frac{0.1}{\sqrt{0.001}}`` term in the parameter update can be **very large** if we use the biased version directly.
"""

# ╔═╡ 2cbbea02-9c33-4b8e-9225-ad3b1663e16d
md"### Adam (unbiased version)"

# ╔═╡ 8156ee09-5a5f-493f-bddb-48884ab10487
md"""
!!! algorithm "Algorithm: Adam (unbiased version)"

	- **Input**: initial learning rate ``α``, initial parameters ``w_0``, weight ``ρ_1,ρ_2``

	- **initialize** ``t \gets 1``

	- **loop**

	  - sample a stochastic gradient ``g_t^i \gets ∇ \mathcal{L}(w_t^i)``

	  - accumulate past gradients in two ways:

	    $\mathcal{F}_t^i = ρ_1 ⋅ \mathcal{F}_{t-1}^i + (1 - ρ_1) ⋅ (g_t^i)^2 \text{ and } \mathcal{G}_t^i = ρ_2 ⋅ \mathcal{G}_{t-1}^i + (1 - ρ_2) ⋅ (g_t^i)^2$

	  - debias by ``\mathcal{F}_t^i \text{ /= } (1 - (ρ_1)^t)`` and ``\mathcal{G}_t^i \text{ /= } (1 - (ρ_2)^t)``

	  - update model: for all ``i ∈ \{1,…,d\}``

	    $w_{t+1}^i = w_t^i - α ⋅ \frac{\mathcal{F}_t^i}{\sqrt{\mathcal{G}_t^i} + ϵ} ⋅ g_t^i$

	  - ``t \gets t + 1``

	- **end loop**
"""

# ╔═╡ 64f923f2-69d0-4d7c-9a96-13a55b0945ff
md"### Is bias correction really a big deal?"

# ╔═╡ 480e5e62-d756-4074-9c70-584503171808
md"""
**Remark.**

- Since it only actually affects the first few steps of training, it seems not a very big issue;

- In the early versions of many popular frameworks (e.g., keras, caffe) only the biased version was implemented.
"""

# ╔═╡ 7a810d30-33c2-46c6-9de0-52a27b53e3e9
md"### Which Adaptive Learning Rate Method Works Best?"

# ╔═╡ 97e4a118-6672-40ac-abe5-49d04f0e2451
md"""
**Remark.**
**Adam** has been widely used as the default SOTA tech to train deep learning systems.
"""

# ╔═╡ 9187264f-25b1-48ff-bc44-e9ec0880e6c3
md"# Chapter 13: Training -- Regularization"

# ╔═╡ b94e3f0a-e387-42bb-bc98-5b9a8fcf746c
md"## What is Regularization"

# ╔═╡ f6acacce-203d-420f-aaef-f4e97d03c246
md"### Generalization"

# ╔═╡ 43f1ee9e-2b9e-4a32-890d-fba92cf12006
md"""
**Remark.**

- An AI problem can be expressed by countless data points;

- Training data is merely a pretty small part of the data;

- A model that works on the training data might not work on the remaining data

- The goal of machine learning is not to fit the training data; instead **we hope the model can handle new data points**.
"""

# ╔═╡ 979751c7-b66a-46f0-9f3b-5c09fb46bdfa
md"### Overfitting"

# ╔═╡ 71f0af91-5772-437e-b8a7-429b22ef3b6b
md"""
**Remark.**
If **a model is over flexible** so that it matches the training data well but is not learning general rules that will work for new data, this is called **overfitting**.
"""

# ╔═╡ a7210602-910a-4090-9373-5718e940d5da
md"""
!!! note "More parameters, more flexible of the model"
"""

# ╔═╡ 0881a661-b6ac-466c-b83e-de8a5b861855
md"### Solution to Overfitting: Regularization"

# ╔═╡ 187f1f82-a089-40fa-b47e-7a7d8e28e126
md"""
**Remark.**

- In general: any method to **prevent overfitting** (or penalize the flexibility of a model)

- Specifically: additional terms in the training loss function to **penalize weights that are large**
"""

# ╔═╡ 862ccff9-b4a1-4714-9083-afb263ec46a2
md"""
!!! danger "How do we define whether weights are large?"
"""

# ╔═╡ 3e042e21-7c53-41e1-9a2e-3e5dedc77901
md"## L2 Regularization"

# ╔═╡ 0e73c006-c7b1-4384-b433-ed38c747f2f7
md"""
**Remark.**
Given weights ``w = \{w_1,w_2,…,w_k\}``, we can define "how large are weights" as:

$\|w\| = \sqrt{\sum_{i=1}^k (w_i)^2}$

- This is called the **L2 norm** of ``w``

- A norm is a measure of a vector's length

- Also called the Euclidean norm
"""

# ╔═╡ 00ce82b3-1272-495f-ac7e-f5cf1d59a6ee
md"""
**Remark.**
New goal for minimization:

$\mathcal{L}(w) + λ ⋅ \|w\|^2$

Where:

- ``\displaystyle \mathcal{L}(𝐰) = \frac{1}{n} \sum_{i=1}^n l(f(x^{(i)}; 𝐰), y^{(i)})``

- By minimizing the term, we prefer solutions where ``w`` is **closer** to ``0``.

- Why squared? Answer: It eliminates the square root; easier to work with mathematically.

- ``λ`` is a hyperparameter that adjusts the tradeoff between having low training loss and having low weights.

  - You may tune it on dev, starting from a small value
"""

# ╔═╡ a730ea1f-06d8-4928-a3cb-8672fb88feb2
md"### Regularization"

# ╔═╡ d21a3dc7-0624-4020-89a0-e1b20e14d0dc
md"""
**Remark.**

- Regularization helps the computational problem so that **gradient descent won't try to make some feature weights grow larger and larger**;

- In logistic regression, probably no practical difference whether your classifier predicts probability .99 or .9999 for a label, but **weights would need to be much larger to reach .9999**.

- This also **helps with generalization** because **it won't give large weight to features unless there is sufficient evidence that they are useful**.
"""

# ╔═╡ 6ae01fc6-ba58-42d0-846f-1d7005a4ae8a
md"""
**Remark.**
More generally:

$\mathcal{L}(w) + λ ⋅ \mathcal{R}(w)$

- ``\mathcal{R}(w)`` is called the **regularization term** or **regularizer** or **penalty**

- The squared L2 norm ``\mathcal{R}(w) = \|w\|^2`` is one kind of penalty, but there are others
"""

# ╔═╡ 840bc8a8-2ab7-4a34-8c00-f32aeaa1a7ef
md"### L2 Regularization"

# ╔═╡ 92e96193-dbaa-4153-8213-453ad6ebd140
md"""
**Remark.**

- L2 is the most common type of regularization

- Logistic regression implementations usually use L2 regularization by default

- L2 regularization can be added to any gradient descent algorithms
"""

# ╔═╡ e01ff30e-0074-41a2-a596-d289d41aa068
md"## L1 Regularization"

# ╔═╡ 866088cd-ea01-47d0-8160-e80a9c5ee1da
md"""
**Remark.**

$\mathcal{L}(w) + λ ⋅ \sum_{i=1}^k |w_i|$

- Often results in many weights being exactly ``0`` (while L2 just makes them small but nonzero)

- You may want to combine L2 and L1?
"""

# ╔═╡ 3dca9acc-65e0-4489-80e7-085da74e4a7b
md"## Dropout regularization"

# ╔═╡ dc286af3-cdaa-441d-a222-13a39b337f6a
md"""
**Remark.**
**Best way to regularize a fixed size model is:**

- Average the predictions of all possible settings of the parameters

- **Impractical** to train many neural networks since it is **expensive** in time and memory

**Dropout is an inexpensive but powerful** method of regularizing a broad family of models.
"""

# ╔═╡ 626658da-1ea8-4e73-b54f-5293bb993b43
md"### Removing Units Creates Networks"

# ╔═╡ 43e83b8a-e6fb-4335-8fb0-e1c3c2f39264
md"""
**Remark.**

- Dropout trains an ensemble of all subnetworks

  - Subnetworks formed by removing non-output units from an underlying base network

- We can effectively remove units by **multiplying its output value by zero** ("mask" technique)
"""

# ╔═╡ df6fb6bb-011c-445c-9d77-ed0598fbbfb4
md"### Dropout Regularization"

# ╔═╡ 12f9d9d8-e136-4fee-824a-31d62844d8cc
md"""
**Illustration.**

(a) A standard neural net with two hidden layers

(b) A thinned net produced by applying dropout, crossed units have been dropped
"""

# ╔═╡ 80cc545d-c4dd-4b45-94de-d62baadabe52
md"""
**Remark.**
Drop hidden and visible units from net, i.e., temporarily remove it from the network with all input/output connections.
Choice of units to drop is random, determined by probability ``p``, chosen by a validation set, or equal to ``0.1``.
"""

# ╔═╡ 98bd6863-f331-4aa2-a347-6d9052aeacf4
md"### Dropout Performance"

# ╔═╡ 12e90822-fd27-412b-978a-884895744c3d
md"""
**Illustration.**
Graph of classification error (%) with respect to number of weight updates shows that dropout significantly reduces classification error.
"""

# ╔═╡ ee14c5f5-b56f-4db9-ae93-62797aa4603e
md"### Dropout as Bagging"

# ╔═╡ 5be00a1b-a1bc-4e6a-89e3-454347f42756
md"""
**Remark.**

- In bagging we define ``k`` different models, construct ``k`` different data sets by sampling from the datasete with replacement, and train model ``i`` on dataset ``i``.

- Dropout aims to approximate this process, but with an exponentially large number of neural networks.
"""

# ╔═╡ 7a58ff8a-64ce-403b-bf66-a1743d629efb
md"### Dropout as an Ensemble Method"

# ╔═╡ 11f12051-631f-41fe-b2ae-a5ad292f200f
md"""
**Remark.**

- Remove non-output units from base network.

- Remaining 4 units yield 16 networks.

- Here many networks have no path from input to output.

- Problem insignificant with large networks.
"""

# ╔═╡ 8fa29ff4-dec9-4a30-b2eb-0f59ec7580f1
md"### Dropout Code Example #1"

# ╔═╡ 1c62630c-bfb5-4231-9ca1-396853a67b12
md"""
```python
class RobertaClassificationHead(nn.Module):

	def __init__(self, bert_hidden_dim, num_labels):
		super(RobertaClassificationHead, self).__init__()
		self.dropout = nn.Dropout(0.1)
		self.out_proj = nn.Linear(bert_hidden_dim, num_labels)

	def forward(self, features):
		x = features
		x = self.dropout(x)
		x = self.dense(x)
		x = torch.tanh(x)
		x = self.dropout(x)
		x = self.out_proj(x)
		return x
```
"""

# ╔═╡ c62361e1-8a0a-441c-9698-93083fad165b
md"### Dropout Code Example #2"

# ╔═╡ 2a72c64a-d79b-470f-bd6e-b57c3c22f1fe
md"""
```python
#create model
model = Sequential()
#add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Dropout(.5, noise_shape=None, seed=None)) #dropout layer
model.add(Conv2D(32, kernel_size=3, activation='relu'))

model.add(Flatten())
model.add(Dense(10, activation='softmax'))
```
"""

# ╔═╡ b1c7bb7f-a263-4575-ad0a-d99e62952c41
md"# Chapter 14: Evaluation Metrics"

# ╔═╡ c8670e69-760c-46ed-8eb6-4c1f5c95f515
md"## Classification Problems: Accuracy, F1"

# ╔═╡ 48b0b7ca-645f-42b2-ad92-ebc4dbcc8547
md"### Accuracy"

# ╔═╡ 233b0379-485b-4f30-a306-38e507c21962
md"""
**Remark.**
Given data ``\{(x_i,y_i) ∣ i = 1,…,n\}``, and the model ``f()``:

$\text{Accuracy} = \frac{\#(f(x_i) == y_i)}{n}$
"""

# ╔═╡ 02a4692e-6ad1-4ceb-80bb-1300cae4dc3a
md"### Issues of Accuracy"

# ╔═╡ 52731845-385c-4ce7-b080-8092dd29bb98
md"""
**Remark.**
Accuracy is misleading when the test data is unbalanced.

!!! note
	**Task**: predict if a person is COVID19 positive?

- In the real world, the **test data is unbalanced** (we can assume only 10% are positive)

- We can use this as prior knowledge to enforce **a model to always predict "negative"**

- Then, this model can get **accuracy 90%** even without learning anything.

!!! danger "Majority voting leads to unreliable performance"
"""

# ╔═╡ 0c50d692-87b3-417c-8011-31bfcd787fbc
md"### F1"

# ╔═╡ ecfdf429-b8c2-4f83-8d82-fb661b11d058
md"""
**Remark.**

!!! note
	**Task**: predict if a person is COVID19 positive?

Let ``G`` be "all positive people" (the golden set), ``M`` be "positive people predicted by model" (the predicted set), and ``O`` be the overlap of ``G`` and ``M``.

- Precision ``(P)``: ``O / M``

- Recall ``(R)``: ``O / G``

$\text{F1} = 2 ⋅ \frac{\text{precision} ⋅ \text{recall}}{\text{precision} + \text{recall}} = \frac{2 ⋅ P ⋅ R}{P + R}$

"G O M" 

P = O / M

R = O / G

F1 = 2PR / (P + R)
"""

# ╔═╡ 68051c7b-0248-401d-82a1-8ab9d4a5ccbd
md"""
**Remark.**

- Precision ``(P)``: **in all your predictions, how many are correct**

- Recall ``(R)``: **in all of the targets, how many are found**

- F1 is a **tradeoff between precision and recall**.

When **precision > recall**:

$\text{F1} = \text{precision} ⋅ \frac{2 ⋅ \text{recall}}{\text{precision} + \text{recall}} < \text{precision}$

$\text{F1} = \text{recall} ⋅ \frac{2 ⋅ \text{precision}}{\text{precision} + \text{recall}} > \text{recall}$

- Once we have **two 0/1 vectors (one gold, one predicted)**, we can compute F1
"""

# ╔═╡ 1f4e7172-5108-42ef-8c3b-08baac1946bf
md"### Extend Binary Classification to Multi-label Classification"

# ╔═╡ cc23045b-394e-4c74-b840-d9db52b879bc
md"### Multi-label Classification"

# ╔═╡ a126774b-cd0a-4fa7-8828-2eab53346d76
md"""
**Remark.**
We can find 0/1 vectors from different angles.
"""

# ╔═╡ 3cad956a-e134-4070-a1ac-20b205f5d558
md"### Macro F1"

# ╔═╡ 6365c361-8a9e-4ef7-8c8b-876fb3b49008
md"""
**Remark.**

(i) each column (i.e., per label)

**Macro F1:** compute F1 for **each label**, then average
"""

# ╔═╡ a797d907-ca04-419e-a8d0-8f74927e4297
md"""
**Remark.**

- Some labels may have more actual occurrences in the dataset

- How to **pay more attention to labels that are more frequent**?
"""

# ╔═╡ f34958e5-4ba3-4530-97b6-6d3987b95f18
md"### Weighted-averaged F1"

# ╔═╡ df324a0f-694b-4a4d-a2a1-b53397b8301b
md"""
**Remark.**
The weighted-averaged F1 score is calculated by taking the mean of all per-class F1 scores while considering each class's support.
"""

# ╔═╡ d1c49ef0-1cf2-4834-8b82-d9b3eaa9e970
md"### Micro F1"

# ╔═╡ a9f7fb81-5250-4b69-ba78-a5ba627e19ab
md"""
**Remark.**
(ii) concatenate all rows/cols as a long 0/1 vector
"""

# ╔═╡ 9cefbb4e-28e0-456a-8626-217d530663af
md"### Sample-wise F1"

# ╔═╡ 7346500e-8ee7-4cb5-bdda-e99f2f5bfe26
md"""
**Remark.**
(iii) treat each row (i.e., each sample's predictions) as a 0/1 vector.
Compute F1 per sample, then report the average.
"""

# ╔═╡ f8c76558-2cc2-4571-856b-54ef5d318bbe
md"### Sklearn for F1 Calculation"

# ╔═╡ e64cda1d-d6c5-4e7a-ad2d-f90a741fa108
md"""
```
sklearn.metrics.f1_score
```
"""

# ╔═╡ 899c9f77-6c49-432f-b12b-6911ac9bcf3f
md"## Text Generation: BLEU, Rouge"

# ╔═╡ ecd44d7a-f5eb-4ebd-a2ed-3721288beb3d
md"### Text Generation Examples"

# ╔═╡ 107f71b2-a375-4628-9e21-f36fe53c5314
md"### BLEU (Papineni et al., 2002)"

# ╔═╡ e2cd252f-338e-48cf-8998-c48fa6841c64
md"""
**Remark.**

- N-gram overlap between machine translation output and reference translation

- Compute **precision** for *n-grams of size 1 to 4*

- Add brevity penalty (for too short translations)

$\text{BLEU} = \min\left(1, \frac{\text{output-length}}{\text{reference-length}}\right) \left(\prod_{i=1}^4 \text{precision}_i\right)^{1/4}$
"""

# ╔═╡ 4d7a53ca-a7c4-4971-b2d2-c7d3aa918919
md"### ROUGE (Lin, 2004)"

# ╔═╡ b1ca1568-1181-4bfd-b132-507e3f11b55a
md"""
**Example.**

- Model generated summary:

!!! tip "the cat was found under the bed"

- Gold summary:

!!! tip "the cat was under the bed"

To get a good quantitative value, we can compute the **precision** and **recall** using the **overlap**.
"""

# ╔═╡ 98f9e5d4-536f-47ba-b66d-eee3a7e45a26
md"""
**Remark.**
**What to overlap?**

- ROUGE-N:

  - N = 1: compare the overlap between unigrams (i.e., single words)

  - N = 2: compare the overlap between bigrams (i.e., a phrase with two consecutive words)

  - N = 3: compare the overlap between trigrams (i.e., a phrase with three consecutive words)

- ROUGE-L:

  - L: Longest Common Subsequence (it does not require consecutive matches but in-sequence matches)

- ROUGE-S:

  - S: compare the overlap between skip-grams

!!! tip "For each evaluation metric, we can compute precision, recall and F1"
"""

# ╔═╡ 0596357c-2e82-4768-aae3-48fb5ca3a279
md"""
!!! tip ""

	- Usually the summaries are forced to be concise (i.e., by length limit), then you could consider using just the recall, since precision is of less concern in this scenario.

	- So, ROUGE is **Recall-Oriented** evaluation.
"""

# ╔═╡ 0ae0b117-4ad8-46fd-8b07-99a3ddfbe47d
md"### BLEU vs. ROUGE"

# ╔═╡ a7d0fdc8-fff5-4737-aafd-cc81d4b31473
md"""
**Remark.**
**Common:**

- Both considered the overlap

**Difference:**

- BLEU focuses on **precision**

- ROUGE focuses on **recall** (note the ROUGE library also reports precision, F1)
"""

# ╔═╡ 8756f750-188f-4b91-9e27-0e59d65e03bc
md"## Ranking Problems: MRR, MAP, NDCG"

# ╔═╡ 5064b566-ecee-46c0-b4ba-2e34b13a96b9
md"### Ranking Problems"

# ╔═╡ c9d3c395-9ad5-48c0-93af-f59ac90cf302
md"""
**Illustration.**
Google search page ranking.
Word cloud ranking.
"""

# ╔═╡ 4f9f3b7d-e8d0-4ab9-8213-1999b4ccc03b
md"### MRR (Mean Reciprocal Rank)"

# ╔═╡ 232e89ad-e481-40e4-96d2-a004f6d9d64f
md"""
**Remark.**
For each user ``u``:

- Generate list of recommendations

- Find rank ``k_u`` of its first relevant recommendation (the first rec has rank 1)

- Compute reciprocal rank ``\frac{1}{k_u}``

Overall algorithm performance is mean reciprocal rank:

$\text{MRR}(U) = \frac{1}{|U|} \sum_{u ∈ U} \frac{1}{k_u}$
"""

# ╔═╡ db293503-f9ab-41f0-91ea-39e3c209642b
md"""
**Remark.**
**MRR Pros:**

- This method is **simple** to compute and is easy to interpret.

- This method puts a high focus on the first relevant element of the list.
  It is best suited for targeted searches such as users asking for the "best item for me".

**MRR Cons:**

- It focuses on a single item from the list, **ignoring the rest of the list**

- It gives **a list with a single relevant item just as much weight as a list with many relevant items.**

- This might not be good for users that want a list of related items to browse.
"""

# ╔═╡ 039fa96f-41a3-4651-ba3b-b532dc01d885
md"### MAP (Mean Average Precision)"

# ╔═╡ ef350652-3180-45ad-a1db-b4833cf7694b
md"""
**Remark.**

- We want to evaluate the whole list of recommended items up to a specific cut-off ``N``

- Precision @ ``N``
"""

# ╔═╡ 5488c60f-ee91-455a-8426-276d2d8628c3
md"""
**Remark.**
**MRR Pros:**

- Compared with MRR, **it considers multiple relevant items**.

- **Value putting highly relevant documents high up the recommended lists**.

**MRR Cons:**

- This metrics shines for binary (relevant/non-relevant) ratings.
  However, **it is not fit for fine-grained numerical ratings**.

- With fine-grained ratings, for example on a scale from 1 to 5 stars, the evaluation would need first to threshold the ratings to make binary relevancies.
  One option is to consider only ratings bigger than 4 as relevant.
  This introduces bias in the evaluation metric because of the manual threshold.
"""

# ╔═╡ a6ce9556-974a-4dd4-ba67-edd451cbdab4
md"### NDCG (Normalized Discounted Cumulative Gain)"

# ╔═╡ ad0fb083-460b-4a66-af22-928405d625c3
md"""
**Remark.**

- Cumulative Gain: ``\displaystyle \text{CG}_p = \sum_{i=1}^p rel_i``

- Discounted Cumulative Gain: ``\displaystyle \text{DCG}_p = \sum_{i=1}^p \frac{rel_i}{\log_2(i + 1)}``

- **Ideal** Discounted Cumulative Gain (IDCG):
  compute DCG on the ground-truth list

- **NDCG**: ``\displaystyle \text{NDCG}_p = \frac{\text{DCG}_p}{\text{IDCG}_p}``
"""

# ╔═╡ 57237519-ebd3-489c-a770-491a4e63dc9c
md"# Chapter 15: Ensemble Methods"

# ╔═╡ 7cf5143e-41d6-481c-8ae5-ca3c6c6a58ee
md"## Bias-Variance Decomposition"

# ╔═╡ b7fa334a-a763-4d81-8425-ac94731e20ae
md"### Recall Statistics Basics"

# ╔═╡ f939634f-bca3-4f4c-8d51-40c88282a9c2
md"""
**Remark.**

- **Expectation** or mean of ``Z``:

  $\bar{Z} = \text{E}[Z] = \sum_i z_i ⋅ p(z_i)$

- **Variance** of ``Z``:

  $\text{Var}[Z] = \text{E}[(Z - \bar{Z})^2] = \text{E}[Z^2] - \bar{Z}^2$

!!! note

    Here it defines the relationship between **Expectation** and **Variance**
"""

# ╔═╡ 8168b488-16b8-4113-be6f-d17383b6b020
md"### Bias-Variance Tradeoff: Intuition"

# ╔═╡ afcdb8a5-5aeb-4a8d-89d1-97f463409c0f
md"""
**Remark.**

- Model **too simple**: does not fit the data well -- A **biased** solution

- Model **too complex**: small changes to the data, solution changes a lot -- a **high-variance** solution
"""

# ╔═╡ 17acab28-d952-4d9c-9952-edc03e917b7b
md"## Bias-Variance-Noise Decomposition"

# ╔═╡ 33470d25-31dd-4f72-a393-fcc1fa48d185
md"""
**Remark.**

**Notations**:

- Dataset: ``\{(x,y)\}``

- Gold function: ``g(x)``, usually ``y = g(x) + ε``

- Function we are optimizing: ``f(x)``

- Sum-squared error:

  $\text{sum\_error} = \sum_i (y - f(x_i))^2$

  Given a new data point ``x`` what is the expected prediction error?

  $\text{E}[\text{error}] = \text{E}[(y - f(x_i))^2]$
"""

# ╔═╡ b268f146-797d-4ffc-af6e-8856aa18f43f
md"""
**Remark.**

$\begin{align*}
\text{E}[\text{error}] &= \text{E}[(y - f(x_i))^2] \\
&= \text{E}[y^2 - 2y f(x) + f^2(x)] \\
&= \text{E}[y^2] - 2\text{E}[y]\text{E}[f(x)] + \text{E}[f^2(x)] \\
&= \text{E}[(y - g(x))^2] + g^2(x) + \text{E}[(f(x) - \bar{f}(x))^2] + \bar{f}^2(x) - 2g(x) \bar{f}(x) \\
&= (g(x) - \bar{f}(x))^2 + \text{E}[(f(x) - \bar{f}(x))^2] + \text{E}[(y - g(x))^2] \\
&= \text{bias}^2 + \text{variance} + \text{noise}
\end{align*}$
"""

# ╔═╡ 5337dbf7-d894-45b9-ac74-8a3e147544e8
md"""
**Remark.**

$\text{E}[\text{error}] = \text{E}[(y - f(x_i))^2]$

**Bias**: difference between _expected prediction_ and _truth_

- Measures **how well you approximate the true solution**

- **Decreases with more complex model**

- **Model with high bias pays very little attention to the training data** and oversimplifies the model.
  It always leads to high error on training and test data (i.e., underfitting).

**Variance**: comes from the sampling of training data, and how it affects the learning algorithm

- Measures **how twisty is the function's curve**

- **Decreases with simpler model**

- Model with high variance pays a lot of attention to training data and **does not generalize on unseen data** (i.e., overfitting)

**Noise**: is often called **irreducible error**

- it depends on data and so it is not possible to eliminate it, regardless of what algorithm is used
"""

# ╔═╡ 891ad36a-c228-4dba-9d3e-e58bfab6720a
md"### Graphical View of Bias-Variance Trade-Off"

# ╔═╡ ddc4bfc6-f6cf-4fb7-b1a1-8f748850f050
md"""
**Illustration.**

- Low variance, Low Bias: Bullseye

- Low variance, High Bias: Concentrated, but not centered

- High variance, Low Bias: Spread on center

- High variance, High Bias: Spread and not centered
"""

# ╔═╡ 46919066-a1cb-4e0e-b8f0-611e160cdc0e
md"""
**Illustration.**
Plot of error with respect to model complexity.
The optimal model complexity is where variance and squared bias intersect.
"""

# ╔═╡ 00d264df-89be-45ff-83dc-484090871cef
md"## Decrease Variance--Bagging (bootstrap with aggregating)"

# ╔═╡ 53dfe27d-c540-43b4-ae85-db51d9fbcb82
md"### Ensemble Learning"

# ╔═╡ 471303af-e55a-4416-b4ab-b68d37515eda
md"""
**Remark.**

- Ensemble learning gives credence to the idea of the "wisdom of crowds," which suggests that the **decision-making of a larger group of people is typically better than that of an individual expert.**
  Similarly, ensemble learning refers to **a group (or ensemble) of base learners, or models, which work collectively to achieve a better final prediction.**

- A **single model, also known as a base or weak learner**, may not perform well individually **due to high variance or high bias**.
  However, when weak learners are aggregated, they can form a strong learner, as their combination **reduces bias or variance**, yielding better model performance.
"""

# ╔═╡ edec6623-7bde-40e8-b279-b0c45313e9cb
md"""
**Remark.**
**Ensemble Methods can also be divided into two groups:**

- **Parallel Learners**, where base models are generated in parallel.
  This exploits the independence between models by averaging out the mistakes (e.g. **Bagging**).

- **Sequential Learners**, where different models are generated sequentially and the mistakes of previous models are learned by their successors.
  This aims at exploiting the dependency between models by giving the mislabeled examples higher weights (e.g. **Boosting**)
"""

# ╔═╡ 5ab79d7e-afe9-4d31-ab78-e6ea7e8c9f07
md"### Bagging (Bootstrap with Aggregating)"

# ╔═╡ a516f178-60d4-44f2-97a5-97e441f95c68
md"""
**Remark.**
**Bootstrap** method is a resampling technique by **sampling a dataset with replacement**.

**Main Assumption of Bagging**:

- Combining many unstable predictors to produce a ensemble (stable) predictor.

- Unstable Predictor: **small changes in training data produce large changes in the model.** e.g. Neural Nets, Decision Trees
"""

# ╔═╡ b18f044c-2e1b-4297-987b-46f6bf6764d0
md"### Bagging Algorithm"

# ╔═╡ 962d83f3-6a0b-4e5a-84dd-6181c455d795
md"""
**Remark.**

- Bagging **reduces variance** by averaging

- Bagging **has little effect on bias**
"""

# ╔═╡ ba4da7c8-00bb-4584-9887-32e1322a3789
md"## Decrease Bias--Boosting"

# ╔═╡ 67378081-7005-4b0c-9b22-327f2d7fe42f
md"### Boosting"

# ╔═╡ 97f1347c-898b-4524-8d0f-970ae602a896
md"""
**Remark.**

- A series of models are constructed sequentially

- With each new model iteration, **the weights of the misclassified data in the previous model are increased**.
  This redistribution of weights helps the algorithm identify the parameters that it needs to focus on to improve its performance.

- Each classifier votes to obtain a final outcome
"""

# ╔═╡ 8ac061f1-85f4-4079-89ba-96ac46bb5ded
md"""
**Remark.**

- **AdaBoost** (Adaptive boosting)

  - This method operates iteratively, identifying misclassified data points and adjusting their weights to minimize the training error.
    The model continues optimize in a sequential fashion until it yields the strongest predictor.

  - AdaBoost uses an iterative approach to learn from the mistakes of weak classifiers, and turn them into strong ones.
"""

# ╔═╡ 2f06c009-bdb6-4c06-ad41-14efa127b681
md"### AdaBoost"

# ╔═╡ 39ac7277-6efc-483f-98fb-c26ca78db546
md"""
**Algorithm.**

- Initial sample-wise weights: ``𝐰``

- Predictions:

  $\hat{y}_j = \max_k \left(\sum_{i=1}^M α_i ⋅ I(f^i(x_j) = k)\right)$
"""

# ╔═╡ a0734dfc-c083-4456-80b3-ae732c10648b
md"### Advantages and Disadvantages of AdaBoost"

# ╔═╡ 4731b14f-ac57-4f73-a258-4ce525cb3ca7
md"""
**Remark.**

- It is intuitive and easier to use

- AdaBoost is sensitive to Noisy data and outliers (highly recommended to eliminate them before using AdaBoost).

- AdaBoost is relatively slow
"""

# ╔═╡ be9c0893-a37c-43db-8dfb-b6388e68c108
md"### Bagging vs. Boosting"

# ╔═╡ e6c1f1e2-ed75-4f03-b4ef-4d826777d19e
md"""
**Table.**

| | Bagging | Boosting |
|---|---|---|
| Dataset | Different subsets of the same data | the same data but with different sample importance |
| How to train base learners | parallel | sequentially |
| When to use | reduce variance | reduce bias |
"""

# ╔═╡ b0bf2fe3-020d-4c08-8660-e696efdf867c
md"# Chapter 16: Typical Classification Challenges -- Class Imbalance"

# ╔═╡ 9dd97fce-ba94-4f16-938e-db94853a49dd
md"## Class Imbalance Problem"

# ╔═╡ d1819b4b-648f-48c4-a700-97fb8bfd995f
md"### What is Class Imbalance Problem"

# ╔═╡ c51b20ea-4165-49c9-aff9-fe48b4e01b5a
md"""
**Remark.**
**Problem**:

- When there are ≥ 2 classes, some classes have **clearly more** examples than other classes

- **Most ML algorithms assume the training data is balanced**

**Consequence**:

- Influence the system performance

- The system may have **difficulties in learning the concept related to the minority classes.**
"""

# ╔═╡ 0790b2b0-5089-4de2-bab5-b4db3ec24111
md"""
!!! info "WAIT BUT WHY"

	Most ML learns by updating parameters based on the loss on training examples; therefore, the majority classes will have more impact on the final model.
"""

# ╔═╡ 574aec30-a3e6-4126-ae02-07ecc4791b5d
md"### Real-world Class Imbalance Problems"

# ╔═╡ 8f4f0416-8bb3-4fa6-a3cc-f88ef2d1655d
md"""
**Illustration.**

- Image recognition

- Online reviews

- Cancer/Not Cancer image data set
"""

# ╔═╡ 7314ec0f-31b5-4b70-bf5e-ac1ca1abb9e4
md"## Methods of Learning from Imbalanced Data"

# ╔═╡ bc7cbebb-6698-46ad-8765-d99aadd68bdc
md"### Sampling Method (Data Level)"

# ╔═╡ cc96ca34-2ae7-4d96-a791-b8a44f384d9b
md"""
**Remark.**

- **Undersampling (downsampling):**
  randomly **removes samples of majority class**

- **Oversampling (upsampling):**
  **adds new samples in minority class**, such as duplicate examples; merge two "cat" images as a new example of "cat"; concatenate two "positive" reviews as a new "positive" review, etc.
"""

# ╔═╡ f3172887-d58c-4e50-b55c-164434bc0156
md"""
!!! danger "Undersampling"

	Discard data is often not good idea
"""

# ╔═╡ 91184b08-2897-4741-b5d6-e6175fcc5a43
md"""
!!! danger "Oversampling"

	- not easy (otherwise class imbalance problem would not exist)

	- and may bring noisy data
"""

# ╔═╡ 18112d1e-d0ae-4c53-bd0d-39ccf9131041
md"### Cost-sensitive (Algorithm Level)"

# ╔═╡ 7af2f58e-1bcb-4aae-813e-f45e458661db
md"""
**Remark.**

- Bagging and Boosting

- Change the Cost Function:

  $\mathcal{L}(𝐰) = \frac{1}{n} \sum_{i=1}^n l(f(x^{(i)}; 𝐰), y^{(i)})$

  - Cost-sensitive learning: **penalizing wrong classification of the minority class more than wrong classifications of the majority class**

    One option:

    $\mathcal{L}(𝐰) = \frac{1}{n} \sum_{i=1}^n l(f(x^{(i)};𝐰), y^{(i)}) ⋅ \frac{\text{\#majority\_label}}{\#y^i}$
"""

# ╔═╡ a26439a7-07db-47a4-af01-a0bc36808294
md"# Chapter 17: Typical Classification Challenges -- Multi-class vs. Multi-label"

# ╔═╡ 0b280f99-fc2a-409f-be20-1d7a173f7bba
md"## Definition: Multi-class, Multi-label"

# ╔═╡ eadf2823-fd2e-42e2-8cf9-20aff12b75ec
md"### Multi-class Classification"

# ╔═╡ 51cb4c97-a24e-4ecd-8b1b-e5bf04c72e7f
md"""
**Remark.**

- 1 out of N classes is correct.

In conventional classification, class labels are **mutually exclusive**.
"""

# ╔═╡ 962180ba-42ae-4802-8568-af3d7fb82b8d
md"### Multi-label Classification"

# ╔═╡ 74656e49-87e8-42aa-9959-aa4b92aea3ea
md"""
**Remark.**

- ≥ 1 out of N classes are correct.

**Exploiting label dependencies (correlations)** has become a major concern in multi-label research
"""

# ╔═╡ 35b7b249-94af-45de-9c8d-3700502b9308
md"### Multi-class Classification = Single-label Classification"

# ╔═╡ 3ec04b5a-80a2-4e5a-baf5-f3f3018517b7
md"""
**Remark.**
Note that conventional **multi-class classification is a special case of multi-label classification**, not the other way around...
"""

# ╔═╡ b29c36e7-2914-4e58-b076-c8cf5040a43a
md"## Solution for Multi-label"

# ╔═╡ 0c78b962-d95e-4121-a818-17e2a596f02c
md"### How did we handle multi-class?"

# ╔═╡ 7b51ef6a-56e7-4de3-b486-9a218a8decb1
md"""
**Remark.**

$L_i = -\log\left(\frac{e^{s_{y_i}}}{\sum_j e^{s_j}}\right)$

In multi-class, we do **softmax across all classes**, and negative log-likelihood tries to **increase the probability of the only correct label**.

In other words, we compare the scores among all labels in multi-class and pick the label with the highest score.

!!! danger "Warning"
	Doesn't apply to multi-label since we want to increase the probabilities of all possible labels

	We should not compare scores among labels
"""

# ╔═╡ 84e22ce4-10f4-41aa-981c-299da19e6080
md"### Solution for multi-label classification"

# ╔═╡ 7e18f490-97cd-423c-adac-50a811e54a8c
md"""
**Remark.**
For each label, ask "if this label is correct for the input"

So, we convert multi-label into binary classification for each label

For each label, we hope its probability to be

- **high if the label is correct**

- **low if the label is incorrect**

!!! danger ""
	- How to get a probability score?
	- How to increase/decrease the probability score?

!!! note ""
	Q1: How to get a probability for each label in binary classification?

- Binary Logistic Regression

  $P(Y = 1 ∣ 𝐱) = σ(𝐰^T 𝐱 + w_0)$

- 2-class Logistic Regression ``(K = 2)``

  $P(Y = y_k ∣ 𝐱) = \frac{e^{𝐰_k^T 𝐱}}{\sum_{i=1}^K e^{𝐰_i^T 𝐱}}$

!!! note ""
	Q2: How to increase/decrease the target probability score?

- Binary Logistic Regression

- 2-class Logistic Regression (K=2)
"""

# ╔═╡ ba074713-0e56-45f8-b029-2791f8e5ed8e
md"## Challenges in Multi-label Classification"

# ╔═╡ 16e1cb97-d724-490f-8a5b-f1b4a251c5e8
md"""
**Illustration.**

- Imbalance

(Figure) Imbalanced data of movie genres (there are much more drama movies compared to the rest of the genres).
"""

# ╔═╡ 90c5df5f-1a38-404f-9cda-820969d15a65
md"""
**Illustration.**

- Few-shot (some labels have too few examples)
"""

# ╔═╡ b2000719-dfbc-4608-8c3a-5f48360f7b28
md"""
**Illustration.**

- Noise (sometimes because of low Inter-Annotator Agreement)
"""

# ╔═╡ d0f07594-0240-4254-baab-f0245f425916
md"## Evaluations in Multi-label Classification"

# ╔═╡ 7be8444b-c30d-4060-b3f0-032469dd7d98
md"""
**Remark.**
We can find 0/1 vectors from different angles.
"""

# ╔═╡ 2f3fdc4e-b012-4ef3-b428-c1fe53e7f778
md"### Macro F1"

# ╔═╡ 5d9e3330-a419-4cd7-850a-44c7fdd03223
md"""
**Remark.**
(i) each column (i.e., per label) **Macro F1**: compute F1 for **each label**, then average
"""

# ╔═╡ 427b6e28-5812-4c41-98f4-669ebd113797
md"""
**Remark.**

- Some labels may have more actual occurrences in the dataset

- How to **pay more attention to labels that are more frequent?**
"""

# ╔═╡ f51bfc4e-24dc-4030-9523-21723fad2ed9
let
	airplane_f1 = 0.67
	boat_f1 = 0.40	
	car_f1 = 0.67

	total = (airplane_f1 + boat_f1 + car_f1) / 3
end

# ╔═╡ 03d5c242-58f1-474e-a818-9ad013d42df1
md"### Weighted-average F1"

# ╔═╡ 153c1095-a5a5-4379-a94a-8ee677a63a6a
md"""
**Remark.**
**The weighted-average F1 score is calculated by taking the mean of all per-class F1 scores while considering each class's support.**
"""

# ╔═╡ dfaa7c44-5676-4327-81ba-9cd04b2c2b2f
let
	airplane_f1 = 0.67
	airplane_support_proportion = 0.3
	
	boat_f1 = 0.40
	boat_support_proportion = 0.1
	
	car_f1 = 0.67
	car_support_proportion = 0.6

	total = (airplane_f1 * airplane_support_proportion) + (boat_f1 * boat_support_proportion) + (car_f1 * car_support_proportion)
end

# ╔═╡ 9b020e9e-7344-44de-8f15-8e9baebe8aa1
md"### Micro F1"

# ╔═╡ 9c8ba689-e3d5-4d9a-a3ff-de2aaa68ae40
md"""
**Remark.**
(ii) Concatenate all rows/cols as a long 0/1 vector
"""

# ╔═╡ 1d9fb4aa-1969-4bed-9565-7a8549c8336b
md"### Sample-wise F1"

# ╔═╡ f55e3719-3ef7-45d2-a2a5-58ee65380dd9
md"""
**Remark.**
(iii) treat each row (i.e., each sample's predictions) as a 0/1 vector Compute F1 per sample, then report the average
"""

# ╔═╡ 4f11e61a-675c-4c73-925b-8b71730e71b7
md"### Sklearn for F1 Calculation"

# ╔═╡ a8196bcc-fc44-4209-aefd-0c91d145be17
md"""
```python
sklearn.metrics.f1_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
```
"""

# ╔═╡ 6c616147-4482-4e07-a7b2-d07febbac2cc
md"# Chapter 18: Typical Classification Challenges -- Few-shot Machine Learning"

# ╔═╡ 25acc017-0c61-4e31-8e9b-b6591effb937
md"## What is Few-Shot Learning"

# ╔═╡ 7eac0960-b3c9-4742-ab5f-acd08305f0f7
md"""
**Remark.**

- A system can easily can recognize a concept if it has many examples in the training set; however, in reality, **some concepts may have only a few examples**

- Then, **how to recognize concepts that only appear a couple of times in the training set**?
"""

# ╔═╡ 3f7b1398-b84a-4561-8010-031d0c292a2e
md"""
**Remark.**

- **Case #1**: in the training set, **only a part of classes are "few-shot"** (data unbalance)

- **Case #2**: in the training set, **all classes are "few-shot"**
"""

# ╔═╡ 245ff794-d1fa-4166-a687-ad5f6a4345cd
md"### Terminology in Few-shot Learning"

# ╔═╡ 7ffd79df-aa24-4fb3-a300-27a08b85fbc3
md"""
**Remark.**

- Few-shot classes

- **K-shot**: ``K`` examples per class

- **Support set** -- The set containin all class examples (i.e., training set)

- **Query set** -- query inputs that your model is asked to predict the label (i.e., test set)
"""

# ╔═╡ 39fbf8aa-45b7-4e49-bfbb-04e67a586ef8
md"## Methods for Few-shot Learning"

# ╔═╡ 83c4bd88-8d87-4d88-8738-b100d519dcb0
md"### Data augmentation (raw, mixup)"

# ╔═╡ b6e484ff-0055-404d-b5b3-951099edc9d7
md"#### Method #1: more data (i.e., data augmentation)"

# ╔═╡ 00a995da-4ea9-488e-92f7-bf4d15b41189
md"""
**Remark.**

- Add data points **explicitly**

  - **Oversampling (upsampling)**:
    **adds new samples in minority class**, such as duplicate examples; merge two "cat" images as a new example of "cat"; concatenate two "positive" reviews as a new "positive" review, etc.

- Add data points **implicitly**: **Mixup** (Zhang et al., 2018)
"""

# ╔═╡ fd64471e-28e4-413c-ab21-4bd9ed225e93
md"#### Mixup (Zhang et al., 2018)"

# ╔═╡ 2a2b282d-4d13-41e1-9651-93d2a9599228
md"""
**Remark.**
Given a pair of samples ``(x_i,y_i)`` and ``(x_j,y_j)`` from the original mini-batch (``x``: input, ``y``: the one-hot label), Mixup generates a synthetic sample as follows:

$\begin{align*}
\hat{x}_{ij} &= β x_i + (1 - β) x_j \\
\hat{y}_{ij} &= β y_i + (1 - β) y_j \\
\end{align*}$

where ``β`` is a mixing scalar.
The generated synthetic data are then fed into the model for training to minimize the loss function.
"""

# ╔═╡ d79af8f6-926e-4ba5-b457-c71c07710223
md"""
**Remark.**
**"Implicitly"**:
-- those steps are conducted on hidden vectors; we cannot see a concrete data point
"""

# ╔═╡ 6674f5c8-3185-4f20-a882-c0855a2565ab
md"#### Mixup Algorithm in Pytorch"

# ╔═╡ 89d94abb-fd83-45ba-b4a7-09b290b0130d
md"""
```python
# y1, y2 should be one-hot vectors

for (x1, y1), (x2, y2) in zip(loader1, loader2):
	lam = numpy.random.beta(alpha, alpha)
	x = Variable(lam * x1 + (1. - lam) * x2)
	y = Variable(lam * y1 + (1. - lam) * y2)
	optimizer.zero_grad()
	loss(net(x), y).backward()
	optimizer.step()
```
"""

# ╔═╡ 30c03bb7-f20a-441d-a21e-10562767f577
md"### Better Representation Learning"

# ╔═╡ 25207360-eaf6-47e9-9b5f-1e9d29ac120a
md"#### Method #2: Better Representation Learning"

# ╔═╡ 7e047927-140e-4dce-8ee9-14b5e339b5d6
md"""
**Remark.**

- In conventional classifications, **examples' representations are compared with the classes' representation** -- "few examples" means the classes cannot update their representations enough times

- In testing, instead of comparing a query with the class, **how about comparing the query with the ``K`` examples?** (let's assume all examples and queries get high-quality representations)
"""

# ╔═╡ 187eaad5-d931-439c-9ced-6f29adab3303
md"""
**Remark.**
We **convert few-shot classification into binary classification**; each new class (positive vs. negative) has many labeled examples.
"""

# ╔═╡ b8d437d2-1d7c-4e74-b226-628f9c1c97ee
md"""
**Remark.**
In testing, the **query compares with the ``K`` examples** of all classes, and picks the best class.
"""

# ╔═╡ 694d1e90-cdb6-4b9d-94d8-a01650933d22
md"### Look for New Supervision (Pretraining, Meta-learning)"

# ╔═╡ e3bb2963-743b-4885-af6a-030b1304306f
md"#### Method #3: Look for new supervision"

# ╔═╡ 8f140a1d-e1f3-44b4-ad7f-1417d2d287e8
md"""
**Remark.**

!!! note ""
	If the ``K`` examples cannot provide enough supervision, can we obtain supervision from related tasks/datasets?

- **Pretrain** the system on related tasks/datasets first, then fine-tune on the few-shot learning dataset

- **Meta-learning**: pretraining on lots of related tasks, optimizing the few-shot learning specifically
"""

# ╔═╡ 1df83a54-d041-423f-9f95-3374bf67da95
md"""
**Remark.**

- Why pretraining works?

!!! info ""
	- Knowledge from prior tasks are helpful for downstream tasks

	- The model parameters can get more familiar with the target problem before touching its poor training set
"""

# ╔═╡ 1c4de6e9-a0c8-46cc-8a70-eff2448392ba
md"#### Pretraining is everywhere"

# ╔═╡ 93a79363-e32a-4376-819a-985436fa1b0f
md"#### Meta Learning"

# ╔═╡ e339d855-9bcc-4d36-a80b-7744f6b0f569
md"""
**Remark.**
To solve a target few-shot, **meta learning optimizes on training tasks also by the few-shot setup**: -- **if the model can handle many ``K``-shot tasks, it should also handle a new ``K``-shot task**
"""

# ╔═╡ e6c4b87a-200b-46d8-9ee7-bd8527147910
md"""
**Remark.**
Training:

- randomly pick a training task

- randomly **sample ``K`` as support set, and a subset as query set**

- **build model on support set, compute loss on query set**

- update the model
"""

# ╔═╡ 3d950a1e-1da4-476a-8b3e-7c2a6e34ffe6
md"""
**Remark.**
Testing:

- for the target ``K``-shot task

- the model predicts for each test instance given the ``K`` supporting examples
"""

# ╔═╡ 4420141f-72d6-46d3-afdf-64bc867f7299
md"#### Meta Learning Paradigm #1 -- metric learning"

# ╔═╡ 1a4b1618-c463-49d5-81a1-b63824452804
md"""
**Remark.**

- **Metric learning:**
  -- **learns a distance function between data points** so that it classifies test instances by comparing them to the ``K`` labeled examples.
  -- The "distance function" often consists of two parts: one is an **embedding function** which encodes any instances into a representation space, the other is a **similarity metric**, such as cosine similarity or Euclidean distance, to calculate how close two instances are in the space.
  -- A typical algorithm "**Prototypical Network**" (Snell et al. 2017)
"""

# ╔═╡ 6a9b437f-16b0-4938-92ee-d45ba7756075
md"## Pay Attention to the Evaluations of Few-shot"

# ╔═╡ 12b3d774-7a11-4ab6-b38e-c321dfc4e07c
md"""
**Remark.**

- **Case #1**: in the training set, **only a part of classes are "few-shot"** (data unbalance)
"""

# ╔═╡ 764c9301-d438-462e-959f-37ceeeab60b0
md"""
**Remark.**

- Usually, the test set contains both few-shot classes as well as frequent classes; **an overall performance/improvement is not reliable**
"""

# ╔═╡ e62fca9c-268c-4f41-8788-2090b2737dfa
md"""
**Remark.**

- An overall performance/improvement is not reliable

  -- because **the improvement may come from frequent classes, and the system performance on few-shot classes maybe is decreasing...**

  -- **report performance for frequent classes and few-shot classes separately**

  -- **ideally, we want the system to improve both types of classes**
"""

# ╔═╡ 9fbad1b0-c77c-4240-9443-a813ea4c739b
md"## Advanced Topic: Life-long Few-shot Learning"

# ╔═╡ ed098f71-aa5c-469b-b437-967113f9bbae
md"### One of AI dreams: The universal AI system"

# ╔═╡ edbdf0fd-b2a1-46c4-b637-68b4d5735575
md"""
**Remark.**

- A **single** system

- New **tasks keep coming**

- **Limited task-specific supervision** (zero or k examples)

- The system can **remain unchanged or keep learning** the new knowledg
"""

# ╔═╡ 3cc58568-2cb8-4ebc-8832-aaa6f8dcdfb2
md"### Example: Life-long few-shot text classification"

# ╔═╡ 5033c232-0595-4737-863e-0ed578071dac
md"""
**Remark.**

- New labels keep coming, **each with a few examples**.

- The system **keeps learning** from the new concepts **without retraining** from scratch.

- The common issue of **catastrophic forgetting**.
"""

# ╔═╡ a0447e66-ba2a-4110-98ab-35bf28ebdd63
md"### Incremental few-shot learning (Xia et al., 2021)"

# ╔═╡ ffd2c599-ca36-4b75-9cfa-d28d5fd0a5f9
md"""
**Remark.**
Definition #1

- **Base classes**: rich annotations

- **Multi-round new classes**: 1~5 examples

Definition #2

- **The system starts from scratch**

- Multi-round new classes: 1~5 examples
"""

# ╔═╡ d7759b12-9dc3-4fd5-b08f-b99c7bd434e7
md"# Chapter 19: Typical Classification Challenges -- Zero-shot Machine Learning"

# ╔═╡ 3df4a62f-2521-408b-9f87-429ac47ac7c8
md"## Formulation of Zero-shot Learning"

# ╔═╡ 0fc4efb2-1b55-42e6-bb85-477deac83d45
md"""
### Formulation of Easy "zero-shot text classification"
"""

# ╔═╡ 6f9f1588-1e2d-4e51-bed5-fa29036b042a
md"""
**Remark.**
Multi-way classifier ``f(⋅) : X → Y``, **where ``Y`` is ``O``**.
"""

# ╔═╡ 02de3efb-4c01-4ceb-a40c-6248c30ca732
md"""
### Formulation of Hard "zero-shot text classification"
"""

# ╔═╡ c2e7e4bb-f187-4d50-b365-fdd38a553dfe
md"""
**Remark.**
Multi-way classifier ``f(⋅) : X → Y``, **where ``Y`` includes all labels**.
"""

# ╔═╡ 0a481de4-9b04-4217-9c82-d103ea67f427
md"### When zero-shot learning is needed"

# ╔═╡ 9f731224-4247-42e2-88ea-3ebffd4559d2
md"""
**Remark.**
E.g.,

- A piece of text can be assigned with any open-domain open-form labels; impossible to annotate for each label
"""

# ╔═╡ 56f006bd-861c-4c88-aea7-2b52f21d4cc8
md"""
**Remark.**
E.g.,

- In online learning, the system needs to **handle the new tasks immediately**

- **Too costly (money, human effort) to annotate** for the new task
"""

# ╔═╡ 6ca7209f-d61e-47ae-8857-3f4258df336b
md"### Cross-task generalization"

# ╔═╡ 12179ea8-f1be-4e91-8b4a-a1ef8b3a68f4
md"""
**Remark.**
Learn from a set of data-rich tasks to solve unseen tasks.
"""

# ╔═╡ 543a1441-c941-42ee-8874-eb0d3caf5aa8
md"## Why Traditional Classifiers do not Work for Zero-shot Learning"

# ╔═╡ 6bc8950e-3a1d-4729-b790-e39f5412f13c
md"""
**Remark.**
"**Training**" of traditional supervised classifiers:

1. If collecting **all seen labels**, **index them as [0, 1, 2, …, n-1]**

2. Build a supervised classifier **on the n classes**
"""

# ╔═╡ b49bba8c-da43-465f-86ed-977e2463b91f
md"""
**Remark.**
"**Testing**" of traditional supervised classifiers:

1. **Pick the label with the highest probability**.
   **But, who can get probabilities? Only seen labels**.
"""

# ╔═╡ f41accd9-af8f-45a8-b4e4-44334ab57616
md"""
**Remark.**
"**Training"** of traditional supervised classifiers:

1. If collecting **all seen+unseen labels**, **index them as [0, 1, 2, …, n-1]**

   **Only the parameters of seen labels will be updated.**
   **In other words, the representations of seen and unseen labels are not comparable.**
"""

# ╔═╡ 31444da3-a091-4d3a-8f25-b612de0ac19b
md"""
**Remark.**
"**Testing**" of traditional supervised classifiers:

- **Although unseen classes join the modeling process, they can not get high probabilities because their parameters are not optimized specific to the task.**
"""

# ╔═╡ ad79b200-ad92-4f35-b2ed-ce2ae5dcd857
md"## Solution for Zero-shot Learning"

# ╔═╡ 811bd9b8-7574-4fdb-90f3-866e8c4a7d63
md"### Representation Learning of Labels"

# ╔═╡ 760ef092-8bc7-413f-9bde-592637b9eb5b
md"""
**Remark.**
In **human learning**:

- humans recognize unseen labels because **humans understand their semantics**

So, in **machine learning**:

- we **should not convert labels into indices**;

- we should **learn the semantic representations of seen&unseen labels and make sure they are comparable**
"""

# ╔═╡ 817072cd-ac1e-48e7-ad8a-cadb3f1f8782
md"""
!!! info "Learn the semantic representations of seen&unseen labels and make sure they are comparable"
"""

# ╔═╡ 6571a3a6-0a14-4cc9-89e1-fc1d503cb8df
md"""
!!! info "We can learn their representations without caring about if they are seen or unseen"

	Approaches: word2vec, tf-idf, etc.
"""

# ╔═╡ c552a6d7-74d7-45dc-b403-f28bff97aaab
md"""
**Remark.**
Conventional supervised classification only care about the representation learning of inputs.
"""

# ╔═╡ 7e6b20ca-3064-4b41-8e31-8144a714d7e5
md"""
**Remark.**
Zero-shot learning emphasizes the representation learning of inputs as well as labels.
"""

# ╔═╡ 1c9942f0-d657-4f2f-97d5-f3332b77309f
md"### Transfer Unseen Tasks into Seen Tasks"

# ╔═╡ e67374d8-23b7-4be6-8098-c8db305b5a7e
md"""
**Remark.**

- Machine Learning consists of many subtasks

- We often study subtasks separately

- In fact, **some tasks share the same reasoning process**, e.g., "topic classification" and "textual entailment'

- This provides an opportunity to **use one task to solve another one**
"""

# ╔═╡ 08f586ca-c2f3-4f5d-a22f-4ae733f3a95d
md"#### Formulate text classification as entailment"

# ╔═╡ 89f78f44-70f9-4988-a3f6-58fc355cda17
md"""
**Example.**
`health`, **`anger`**, `news`, `sad`, `…`


Premise (anger):

> My car was smashed last night.

Hypothesis:

- Label name: This text expresses **`anger`**.

- Definition of label name: This text expresses **a strong emotion that is oriented toward some real or supposed grievance**.
"""

# ╔═╡ ee8b1439-626c-43e3-9eaf-c5ace926794d
md"#### Formulate text classification as language model"

# ╔═╡ 2b56a15a-03c5-465c-af42-3e71b3a99a51
md"""
**Example.**
`health`, **`anger`**, `news`, `sad`, `…`

> My car was smashed last night.

$↓$

> My car was smashed last night.
> So I am very ____
"""

# ╔═╡ 35f5f96b-bd5f-4444-b428-84e96ea56fc0
md"## Advanced Topic: Learning from Task Instructions"

# ╔═╡ d572fe72-1700-45a8-a054-36304f9f5c2b
md"### Learning from task instructions"

# ╔═╡ 37144bcf-d4ad-4488-b95d-3de6b5265a02
md"""
**Remark.**

- A paradigm to achieve cross-task generalization
"""

# ╔═╡ a9154e81-3fef-4ca8-b63e-8d1187d9a255
md"### How do humans learn?"

# ╔═╡ 235f630f-3364-45c2-8132-5b6ba30c800d
md"""
**Example.**

> Count on by tens to add.
> Then, fill in each blank.

- Task instruction

  - how to "count", and

  - where to "fill"

- Very few training examples

  - only one here
"""

# ╔═╡ 6ffcee72-e684-4265-aeb9-26e48c589dbc
md"### A typical form of human learning"

# ╔═╡ 7138096a-1ea0-404a-a0af-f63be9dd50fe
md"""
**Remark.**

- Detailed instruction

- Very few examples (≤ 10)
"""

# ╔═╡ 862711d2-4ae2-4f58-8800-241934c84efc
md"### Mainstream form of machine learning"

# ╔═╡ 0b447d6a-f10a-429f-ad4c-e0d4c6131f83
md"""
**Remark.**

- No instructions

- Large training sets

- Even few-shot learning often uses 1000s of examples
"""

# ╔═╡ 4d188df7-521e-4f29-85d1-4c4ceb7425a1
md"""
!!! danger "Motivation"

	How can task instructions benefit machine learning?
"""

# ╔═╡ c3603b58-c4b6-4242-b2a7-c40d18b83d9e
md"### What is task instruction?"

# ╔═╡ 44f76e70-c260-4726-b718-e97cf0751134
md"""
**Remark.**

- Description of an aspect of the task

  - e.g., "find the most appropriate emotion from {angry, happy, surprised, …}"

- Description of the solution

  - e.g., "Count on by tens to add. Then, fill in each blank"

- Description of properties of training instances

  - e.g., **task**: "what is an elephant?"

    **instruction**: "elephants have long trunks and big ears …"
"""

# ╔═╡ bd11ef76-0128-4bff-b8ed-1c081362052b
md"### Why use task instructions?"

# ╔═╡ 68a6b311-d335-48a8-b6cb-baa35a37b964
md"""
**Remark.**

- A typical task representation/supervision

- Cheap

- Friendly to task generalization

- User-friendly
"""

# ╔═╡ 1f22691f-4540-488d-b495-b8e1e9f35c15
md"### Example of task instructions"

# ╔═╡ 845d09de-b4d2-49f8-85ef-c9881b073ea6
md"# Chapter 20: Flat Clustering Algorithms"

# ╔═╡ fc14945d-7896-4d6a-9a83-b8034914dee7
md"## Clustering Basics"

# ╔═╡ 6739b85a-af92-4332-a328-c7c13b36b6b0
md"### Clustering"

# ╔═╡ 635e3ba5-4bf6-4503-9c3a-3092d24b7142
md"""
**Remark.**

- **Unsupervised learning**

- Requires data, but no labels

- **Detect patterns** e.g. in

  - Group emails or search results

  - Customer shopping patterns

  - Regions of images

- But: can get gibberish
"""

# ╔═╡ 14d93155-fb66-49ce-b2b4-44a1ada27182
md"""
**Remark.**
**Basic idea:**

- **group together similar instances**

  e.g., 2D point patterns

**What could "similar" mean?**

- One option: small Euclidean distance (squared)

  $\text{dist}(𝐱,𝐲) = \|𝐱 - 𝐲\|^2$

- Clustering results are crucially dependent on the i) **representations** of data points, and ii) **measure of similarity (or distance)**
"""

# ╔═╡ 3e3b3673-ae70-42a1-a169-5a05a342a357
md"### Clustering Algorithms"

# ╔═╡ 5eef2b95-90e8-4825-889e-187d79e87841
md"""
**Remark.**
**Flat algorithms:**

- K-means

- Mixture of Gaussian

- Spectral Clustering

**Hierarchical algorithms:**

- Bottom up -- agglomerative

- Top down -- divisive
"""

# ╔═╡ f461b0e8-d6a1-4566-b581-71509e387d7b
md"### Clustering Examples"

# ╔═╡ a11cef97-788f-4c8d-a122-154089b6192b
md"""
**Example.**
**Image segmentation**

- Break up the image into meaningful or perceptually similar regions
"""

# ╔═╡ 72c160ca-ada7-4d7e-8090-3dad88461f24
md"## K-means Clustering"

# ╔═╡ a0b33768-8e62-410e-a38f-07ae440c7f99
md"""
**Remark.**
An iterative clustering algorithm

- Initialize: Pick K random points as cluster centers

- Alternate:

  - Assign data points to the closest cluster center

  - Update the cluster center as the average of its assigned points

- Stop when no points' assignments change
"""

# ╔═╡ 5d0beb88-ba76-415f-a6b5-4c40c6189207
md"### K-means clustering: example"

# ╔═╡ 0ae8a531-7e4d-43fc-92e8-abd85d1881f8
md"""
**Remark.**
Iterative Step 1:

- Assign data points to closest cluster center

Iterative Step 2:

- Change the cluster center to the average of the assigned points

Repeat until convergence
"""

# ╔═╡ 10c29c69-03c2-4b57-8092-3a5186b78f34
md"### Properties of K-means algorithm"

# ╔═╡ b75f682d-beb2-4dcd-ac2f-ff456dea4335
md"""
**Remark.**

- Guaranteed to converge in a finite number of iterations

Running time per iteration:

- Assign data points to closest cluster center: O(KN)

- Change the cluster center to the average of its assigned points: O(N)
"""

# ╔═╡ 37fa0170-f397-4736-890b-a861cc70f1d4
md"### What properties should a distance measure have?"

# ╔═╡ e0a53309-023c-4e13-a117-b38702e7f059
md"""
**Remark.**

- Symmetric

  - D(A,B) = D(B,A)

  - otherwise, we can say A looks like B but B does not look like A

- Positivity, self-similarity

  - D(A,B) ≥ 0 and D(A,B) = 0 iff A = B

  - otherwise, there will be different data points we cannot tell apart

- Triangle inequality

  - D(A,B) + D(B,C) ≥ D(A,C)

  - otherwise, one can say "A is like B and B is like C, but A is not like C at all"
"""

# ╔═╡ 406a9c4d-42b4-44aa-aea8-a2033ecd1c98
md"### K-means clustering: results"

# ╔═╡ e9124f68-51cf-4f78-af87-81f46e800647
md"""
**Illustration.**
Higher ``K`` approaches original image.
"""

# ╔═╡ 2195b859-cce2-44f0-987f-cc9191336a84
md"### When K-means does not work"

# ╔═╡ ed69c39d-bf74-4480-9cbe-1b10348d9a2f
md"""
**Illustration.**
Cluster within a ring.
"""

# ╔═╡ 1c00618c-53c9-4de7-ae54-e5cab1467a24
md"### K-means summary"

# ╔═╡ 9109f4f3-dea1-4d18-8953-b527a5974757
md"""
**Remark.**
**Strengths:**

- Relatively simple to implement

- Scales to large data sets

- Guarantees convergence

- Can warm-start the positions of centroids

- Easily adapts to new examples

**Weakness:**

- choose K manually

- dependent on the seed data points

- influenced by outliers

- works only if the clusters are spherical
"""

# ╔═╡ ae321ad5-3770-432e-99bb-6d723587139d
md"## Gaussian Mixture Model"

# ╔═╡ 024cc93c-bd93-4a9a-9aa8-f02e6654e74f
md"### Gaussian (Normal) Distribution"

# ╔═╡ 99f73253-290a-43c1-8fef-4c22ae42b695
md"""
$f(x) = \frac{1}{σ\sqrt{2π}} e^{-\frac{1}{2} \left(\frac{x - μ}{σ}\right)^2}$
"""

# ╔═╡ 1fcf84ed-abd6-4f13-ac5c-41ddc060611d
md"### Gaussian Mixture Model"

# ╔═╡ ce51da33-c127-48bf-8082-c75d5e20aaf5
md"""
**Remark.**
Motivation

- what if the clusters aren't spherical?

Let's instead treat clustering as a distribution modeling problem.

- let's fit a mixture model, where each data point belongs to a different component

- e.g., in a mixture of Gaussians, **each data point comes from one of several different Gaussian distributions**.
  (We don't need to use Gaussians -- we can pick whatever distribution best represents our data)
"""

# ╔═╡ 23e32823-f493-48c5-90ef-f5bdbeec6c04
md"### Expectation-Maximization (EM) algorithm"

# ╔═╡ da4a68c9-716a-4e98-85d0-9bac9fa68a4b
md"### Initialization for GMM"

# ╔═╡ b7ca48ec-eff1-43d5-adb6-ad3ce455f6b7
md"""
**Remark.**
The initial parameters or weights

- **can be chosen randomly** or

- could be chosen via some heuristic methods (e.g., **by the results of the k-means algorithm**)
"""

# ╔═╡ c917c81e-747b-43e9-995a-43a9e35dc9a7
md"### K-means vs. GMM"

# ╔═╡ c0afd5fa-ceb1-41ec-b0d3-32e2592b29f6
md"""
**Remark.**

- GMM is a lot more flexible in terms of cluster covariance

- K-means is actually a special case of GMM in which each cluster's covariance along all dimensions approaches 0.
  This implies that a point will get assigned only to the cluster closest to it.

- GMM supports mixed membership -- Depending on the task, mixed membership may be more appropriate (e.g., news articles can belong to multiple topic clusters) or not (e.g., organisms can belong to only one species)

- Both need the predefined K value
"""

# ╔═╡ 904422dd-cb19-4e69-aab1-42993af08693
md"## Spectral Clustering"

# ╔═╡ 119a55e8-93cc-4314-b827-fdd00ab71cbd
md"### Data clustering"

# ╔═╡ e2dad27f-dfb0-407b-9280-4146d7b65774
md"""
**Remark.**
Two different criteria:

- compactness, e.g., k-means, GMM

- connectivity, e.g., spectral clustering
"""

# ╔═╡ 826c73f1-8ce8-4d79-8a74-8914b81fd7fa
md"### Spectral clustering"

# ╔═╡ 4cd9ebd7-99ff-4542-9ccd-a700b120ee46
md"""
**Remark.**
**Input**: similarity matrix ``W``, number K of clusters to construct

- Build similarity graph

- Compute the first K eigenvectors of ``v_1,v_2,…,v_k`` of the matrix ``W``

- Building a new matrix ``V ∈ R^{N×K}`` with the eigenvectors as columns

- Interpret the rows of ``V`` as new data points

- Clustering the data points ``Z_i`` with the K-means algorithm
"""

# ╔═╡ 8278aacc-41e3-422e-bdcd-5f5ea7e8354b
md"### Similarity graph construction"

# ╔═╡ 7ef28ed1-0026-495f-8aa6-12a6f1ac7aa9
md"""
**Remark.**
E.g. Gaussian kernel similarity function

$W_{ij} = e^{\frac{\|x_i - x_j\|^2}{2σ^2}}$
"""

# ╔═╡ 4526f19d-fe02-4855-9794-db7787f957da
md"### Why spectral clustering works"

# ╔═╡ 89f32f0b-8d19-4c35-b365-253431a4dce8
md"""
**Remark.**
Data are projected into a lower-dimensional space (the spectral/eigenvector domain) where they are easily separable, say using k-means.
"""

# ╔═╡ 1bb95cf5-864f-49a5-a58b-f51a9a8d7dca
md"### K-means vs. Spectral clustering"

# ╔═╡ 2aa7b7a7-4fab-43d7-bc3c-c183d00568db
md"""
**Remark.**
Applying k-means to eigenvectors allows us to find cluster with non-convex boundaries.
"""

# ╔═╡ b40c7f0b-4597-458a-868d-1e8e859b929d
md"### Spectral clustering summary"

# ╔═╡ 8c08e1e1-0f97-47d1-8c76-69709d3aa95d
md"""
**Remark.**

- Algorithms that cluster points using eigenvectors of matrices derived from the data

- Useful in hard non-convex clustering problems

- Obtain data representation in the low-dimensional space that can be easily clustered

- Variety of methods that use eigenvectors to derive clusters

- Empirically very successful
"""

# ╔═╡ 15e54924-4e5a-41b7-9fee-28a159671d26
md"# Chapter 21: Hierarchical Clustering Algorithms"

# ╔═╡ c7a0632c-0b3e-41a8-b4ec-1800ff18bc49
md"## Hierarchical Clustering"

# ╔═╡ 0d80f018-c78c-4d13-8131-5efc3fb5f3b5
md"### Two Types of Clustering"

# ╔═╡ 12504545-7574-449a-89c6-838d1ce55b8f
md"""
**Remark.**

- **Flat/partitional algorithms**: Construct various partitions and then evaluate them by some criterion (Chapter 20)

- **Hierarchical algorithms**: Create a hierarchical decomposition of the set of objects using some criterion (focus of this chapter)
"""

# ╔═╡ 0513f330-5fbd-4b79-bc97-d9dbc8550816
md"### Hierarchical Clustering"

# ╔═╡ 12afb21f-7733-4fcf-93cf-a3d2073b4357
md"""
**Remark.**
**Hierarchical vs. Flat**

- Flat methods generate a single partition into ``k`` clusters.
  The number ``k`` of clusters has to be determined by the user ahead of time.

- Hierarchical methods generate a hierarchy of partitions, i.e.

  - a partition ``P_1`` into 1 clusters (the entire collection)

  - a partition ``P_2`` into 2 clusters

  - …

  - a partition ``P_n`` into ``n`` clusters (each object forms its own cluster)

- **It is then up to the user to decide which of the partitions reflects actual sub-populations in the data.**
"""

# ╔═╡ 9299cf57-28f9-4ec5-9dc4-0082e9d0508c
md"""
**Remark.**
**Note:**
A sequence of partitions is called "hierarchical" if each cluster in a given partition is the union of clusters in the next larger partition.

**Top:** hierarchical sequence of partitions

**Bottom:** non hierarchical sequence
"""

# ╔═╡ ecdc07ca-3d21-40bd-9b63-4432febf3fee
md"### Why Hierarchical Clustering"

# ╔═╡ da3839f0-d149-4bba-ae79-8db979bad453
md"""
**Remark.**
Lots of data are hierarchical in nature.
For example,

- Image types

- Product categories

- Nouns
"""

# ╔═╡ 94e17553-e4ed-4e5c-a598-0ec235cca96d
md"### Other motivations for hierarchical clustering"

# ╔═╡ 298e7538-daf5-400f-b5ca-a456e922deb1
md"""
**Remark.**

- **Avoid choosing # clusters beforehand**

- Dendrograms help **visualize** different clustering granularities
"""

# ╔═╡ 866c521d-31db-496e-ae9e-3666439c8fc8
md"### Hierarchical Clustering"

# ╔═╡ 130f16f5-ab11-4c53-a9ec-68b9d8f75224
md"""
**Remark.**
Hierarchical methods again come in two varieties, **agglomerative** and **divisive**.

**Agglomerative methods**:

- Start with partition ``P_n``, where each object forms its own cluster.

- Merge the two closest clusters, obtaining ``P_{n-1}``.

- Repeat merge until only one cluster is left.

**Divisive methods**:

- Start with ``P_1``.

- Split the collection into two clusters that are as homogenous (and as different from each other) as possible.

- Apply splitting procedure recursively to the clusters.
"""

# ╔═╡ 676a8ab2-5fba-4688-a0e8-1130f8678f06
md"## Agglomerative Clustering"

# ╔═╡ bfe2d57d-daa5-42dc-86cb-3df09a144dee
md"""
**Remark.**

- **Agglomerative clustering**:

  - First merge very similar instances

  - Incrementally build larger clusters out of smaller clusters

- **Algorithm**:

  - Initially, **each instance in its own cluster**

  - Repeat:

    - **Compute distances between all clusters**

    - **Merge two closest clusters into a new cluster**

  - Stop when there's only one cluster left

- **Produces a family of clusterings represented by**
"""

# ╔═╡ 737a13ce-0e37-42cb-a8a9-202865d46d08
md"### Agglomerative Clustering: example"

# ╔═╡ d45b4721-5836-4a7e-b448-fe1d8fc0f961
md"""
**Remark.**
We begin with a distance matrix which contains the distances between every data point in our dataset.
"""

# ╔═╡ a7732a20-5c57-431b-a5c8-9bb1442ced09
md"""
**Remark.**
Starting with each item in its own cluster, find the best pair to merge into a new cluster.
Repeat until all clusters are fused together.
"""

# ╔═╡ 54696221-d753-4453-bda9-c9adcd3dcfdc
md"### Agglomerative Clustering"

# ╔═╡ 725e28cb-3d28-4781-b538-6018b25530df
md"""
**Remark.**

- How should we define "closest" for clusters with multiple elements?

- Many options

  - **Closest pair** (single-link clustering)

  - **Farthest pair** (complete-link clustering)

  - **Average of all pairs**

  - **Distance between centers**

- Different choices create different clustering behaviors
"""

# ╔═╡ 0bbb473b-4e22-4bbc-b7e9-5146171df9af
md"""
### Illustration of "Closest Pair"
"""

# ╔═╡ 6d258980-7204-439e-a036-575066da4e5d
md"### Behavior of Agglomerative Clustering"

# ╔═╡ d30709a5-3fc6-493a-995b-5ff38384963d
md"### Python Code of Agglomerative Clustering"

# ╔═╡ 2cd48325-4fe4-4ccf-a4be-eaf35a33a551
md"""
```python
from sklearn.cluster import AgglomerativeClustering
import numpy as np

# randomly chosen dataset
x = np.array([[1, 2], [1, 4], [1, 0],
			  [4, 2], [4, 4], [4, 0]])

# here we need to mention the number of clusters
# otherwise the result will be a single cluster
# containing all the data
clustering = AgglomerativeClustering(n_clusters = 2).fit(X)

# print the class labels
print(clustering.labels_)
```
"""

# ╔═╡ 9fd208db-1928-482e-b124-aa6d2db27257
md"## Divisive Clustering"

# ╔═╡ 50248d12-21da-45de-8673-34a47a0fe003
md"### Divisive in pictures -- level 1"

# ╔═╡ e4df9ac1-97fc-45de-a9e5-deb190372806
md"### Divisive in pictures -- level 2"

# ╔═╡ 911a622c-a14e-47b8-99e6-5ca21f90f54f
md"### Divisive: Recursive k-means"

# ╔═╡ 8dc3535f-dcbd-4817-aa2d-d6a851f06851
md"### Divisive choices to be made"

# ╔═╡ c116e645-482f-4a14-80b7-0cf00744510e
md"""
**Remark.**

- Which algorithm to recurse

- How many clusters per split

- When to split vs. stop

  - Max cluster size:
    -- number of points in cluster falls below threshold

  - Max cluster radius:
    -- distance to furthest point falls below threshold

  - Specified # clusters:
    -- split until pre-specified # clusters is reached
"""

# ╔═╡ 2dc7cad1-1589-47d4-b9de-da972f934dbb
md"## Summary"

# ╔═╡ 9e225f7b-d833-49da-8ae3-5a0299fea126
md"### Agglomerative vs. Divisive"

# ╔═╡ 2e4d3f47-81e4-42bb-a5a1-fc29e36c9e05
md"""
**Remark.**

- **Divisive clustering is more efficient**.
  **Time complexity of a naive agglomerative clustering can be brought down to ``O(n^2)``.**
  Whereas for divisive clustering given a fixed number of top levels, using an efficient flat algorithm like K-Means, **divisive algorithms are linear in the number of patterns and clusters**.

- **Divisive algorithm is also more accurate**.
  **Agglomerative clustering makes decisions by considering the local patterns**, whereas **divisive clustering takes into consideration the global distribution** of data when making top-level partitioning decisions.
"""

# ╔═╡ ef836347-5dce-4138-970d-1a9430868b0f
md"### Summary of Hierarchical Clustering"

# ╔═╡ c4af92fb-e9be-433d-8372-2a461a6d4ff1
md"""
**Remark.**

- No need to specify the number of clusters in advance.

- Hierarchical structure maps nicely onto human intuition for some domains.

- They do not scale well: time complexity of at least ``O(n^2)``, where ``n`` is the number of total objects.

- Like any heuristic search algorithsm, local optima are a problem.

- Interpretation of results is (very) subjective.
"""

# ╔═╡ b972d605-36d8-44e6-b066-569647f83337
md"# Chapter 22: Clustering Evaluations"

# ╔═╡ 3108657a-92c4-45b2-b0a8-3564483d41c9
md"## External Measures"

# ╔═╡ f004934a-fb6e-42dc-98b5-daa93e747e59
md"### Matching-based measures"

# ╔═╡ a61fd9bf-c8d2-4650-be91-5da2ef6bb721
md"### Entropy-based measures"

# ╔═╡ e62a7294-458f-48a1-9109-84a8685a7f4a
md"### Pairwise-based measures"

# ╔═╡ 3bf34436-b6df-4bc4-abe4-416c8b9f3adb
md"# Chapter 23: Pretraining and Transfer Learning"

# ╔═╡ c1b1b2f4-1f4d-4a80-8720-f31412a53813
md"## What is Transfer Learning"

# ╔═╡ 6fde46e6-881d-4e09-be60-e8e799ab553b
md"### Real-life challenges in AI"

# ╔═╡ 6707ccc5-6a06-4f16-ba9e-ebf240d75926
md"""
**Remark.**

- Deep learning methods are **data-hungry**

  e.g., >50K data items needed for training

- The **distributions** of the _source_ and _target_ data **must be the same**

- Labeled data in the target domain may be limited

- This problem is typically addressed with **transfer learning**
"""

# ╔═╡ 4e7ca352-eb72-4eac-934b-9d5add2032c9
md"## Self-supervision"

# ╔═╡ 6cdd4c66-61dd-4a62-b9bf-1d03951254c0
md"## Transfer Learning by Supervised Pretraining"

# ╔═╡ 626679bd-78fb-428d-8adb-42328da7c382
md"### Computer Vision"

# ╔═╡ 091301ee-4e37-413f-ba62-89dfd198d808
md"### Natural Language Processing"

# ╔═╡ b3640804-d086-46b1-9b20-17584567501a
md"# Chapter 24: Adversarial Training"

# ╔═╡ a83c3670-95cb-41bc-b9cc-87a539b12c5f
md"## Adversarial Examples/Attacks"

# ╔═╡ 89b4342b-6f48-4650-9480-bd23461dc76c
md"### Adversarial Examples"

# ╔═╡ aaee2a54-bcbf-4c93-90d1-ffb240eb40d4
md"""
**Remark.**

- We've touched upon two ways an algorithm can fail to generalize:

  - overfitting the training data

  - dataset bias (overfit the idosyncrasises of a dataset)

- But algorithms can also be vulnerable to adversarial examples, which **are specialised inputs created with the purpose of confusing a neural network, resulting in the misclassification of a given input.**
  **These notorious inputs are indistinguishable to the human eye, but cause the network to fail to identify the contents of the input.**
"""

# ╔═╡ ca8bf083-b78e-4c98-8ded-cd800bebbe3e
md"""
**Remark.**

- There are several types of such attacks, however, here the focus is on the **fast gradient sign method** attack, which is a white box attack whose goal is to ensure misclassification.

- A **white box attack** is where the attacker has complete access to the model being attacked.
"""

# ╔═╡ 87b6a86e-ef1b-4170-b327-765d8fa55083
md"""
**Example.**

- Here, starting with the image of a panda, the attacker adds small perturbations (distortions) to the original image, which results in the model labelling this image as a gibbon, with high confidence.

- **Who to add these perturbations?**
"""

# ╔═╡ 34ccf23a-5e67-4933-9994-6137cfaf20df
md"### Fast gradient sign method"

# ╔═╡ b08d5dee-84ee-4240-93dc-56d25daaeb6a
md"""
**Remark.**

- The fast gradient sign method works by **using the gradients of the neural network to create an adversarial example**.

- For an input image ``(x)``, the method uses the gradients of the loss with respect to the input image **to create a new image ``(adv\_x)`` that maximises the loss**.
  This new image is called the **adversarial image**.

This can be summarised using the following expression:

$adv\_x = x + ϵ * \text{sign}(∇_x J(θ, x, y))$
"""

# ╔═╡ 48787394-9010-4f03-8a49-1bd0ff2ca68b
md"""
**Remark.**

$adv\_x = x + ϵ * \text{sign}(∇_x J(θ, x, y))$

- An intriguing property here, is the fact that the **gradients are taken with respect to the input image**.

- This is done because the objective is to create an image that maximises the loss.

- A method to accomplish this is to **find how much each pixel in the image contributes to the loss value**, and add a perturbation accordingly.

- In addition, since the model is no longer being trained (thus the gradient is not taken with respect to the trainable variables, i.e., the model parameters), and so **the model parameters remain constant. The only goal is to fool an already trained model**.
"""

# ╔═╡ 867a5210-0777-4c62-a52a-8d66abf5bf69
md"### Code example for adversarial attacks"

# ╔═╡ 2db1e66c-75bc-4dd9-b6d2-926fa20fdacf
md"""
**Remark.**

- "Can we train classifiers with **Flickr photos**, as they have already been collected and annotated, and hope the classifiers still work well on **mobile camera images**?" [Gonq et al., CVPR 2012]

- Object classifiers optimized on benchmark dataset **often exhibit much worse accuracy when evaluated on another one"** [Gonq et al., ICML 2013, Torralba et al., CVPR 2011, Perronnin et al., CVPR 2010]

- "Hot topic" - Visual domain adaptation [Tutorial CVPR'12, ICCV'13]
"""

# ╔═╡ 9124bffb-58f2-4c92-b96b-e82a99ddfc7e
md"### Implementation of fast gradient sign method"

# ╔═╡ 89646197-5a1e-4fdd-bd84-c839df8c97b5
md"""
```python
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False

pretrained_model = tf.keras.applications.MobileNetV2(include_top=True,
								  					 weights='imagenet')
pretrained_model.trainable = False

# ImageNet labels
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

# Read original image and preprocess it
image_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/')
image_raw = tf.io.read_file(image_path)
image = tf.image.decode_image(image_raw)

image = preprocess(image)
image_probs = pretrained_model.predict(image)

# Let's have a look at the image
plt.figure()
plt.imshow(image[0] * 0.5 + 0.5)  # To change [-1, 1] to [0, 1]
_, image_class, class_confidence = get_imagenet_label(image_probs)
plt.title('{} : {:.2f}% Confidence'.format(image_class, class_confidence*100))
plt.show()

# Create the adversarial image (first, create perturbations)
loss_object = tf.keras.losses.CategoricalCrossentropy()

def create_adversarial_pattern(input_image, input_label):
  with tf.GradientTape() as tape:
    tape.watch(input_image)
    prediction = pretrained_model(input_image)
    loss = loss_object(input_label, prediction)

  # Get the gradients of the loss w.r.t to the input image
  gradient = tape.gradient(loss, input_image)

  # Get the sign of the gradients to create the perturbation
  signed_grad = tf.sign(gradient)

  return signed_grad

# The resulting perturbations can also be visualised.

# Get the input label of the image
labrador_retriever_index = 208
label = tf.one_hot(labrador_retriever_index, image_probs.shape[-1])
label = tf.reshape(label, (1, image_probs.shape[-1]))

perturbations = create_adversarial_pattern(image, label)
plt.imshow(perturbations[0] * 0.5 + 0.5);  # To change [-1, 1] to [0, 1]

# Let's try this out for different values of epsilon and observe the resultant image

def display_images(image, description):
  _, label, confidence = get_imagenet_label(pretrained_model.predict(image))
  plt.figure()
  plt.imshow(image[0]*0.5+0.5)
  plt.title('{} \n {} : {:.2f}% Confidence'.format(description,
												   label, confidence*100))
  plt.show()

epsilons = [0, 0.01, 0.1, 0.15]
descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
			    for eps in epsilons]

for i, eps in enumerate(epsilons):
  adv_x = image + eps*perturbations
  adv_x = tf.clip_by_value(adv_x, -1, 1)
  display_images(adv_x, descriptions[i])
```
"""

# ╔═╡ c691af71-44ca-4a4b-abcf-522a14054709
md"### Adversarial Examples"

# ╔═╡ b585a567-22f5-4438-b016-3481a8787a38
md"""
**Remark.**

- Here are adversarial examples constructed for a (variational) autoencoder

- Right = reconstructions of the images on the left

- This is **a security threat** if a web service uses an autoencoder to compress images: you share an image with your friend, and it decompresses to something entirely different
"""

# ╔═╡ 15e1ce20-d3bb-4ec4-9bca-cd084e4f84e4
md"### Adversarial Attacks"

# ╔═╡ 9e741db2-9802-46f1-a10f-4fdb555d300b
md"""
**Remark.**

- The paper which introduced adversarial examples (in 2013) was titled "Intriguing Properties of Neural Networks."

- Now they're regarded as a serious security threat.

  - **Nobody has found a reliable method yet to defend against them**.

  - Adversarial examples transfer to different networks trained on a disjoint subset of the training set!

  - **Blackbox attack: You don't need access to the original network; you can train up a new network to match its predictions, and then construct adversarial examples for that**.

    - Attack carried out against proprietary classification networks accessed using prediction APIs (MetaMind, Amazon, Google)
"""

# ╔═╡ 64b433ce-051e-4e28-b14e-576a8a1b331b
md"## Generative Adversarial Networks (GANs)"

# ╔═╡ 316993be-b3c0-40bf-88ed-66777d9a5b30
md"""
**Remark.**

- **Generative modeling**: **automatically discovering and learning the regularities or patterns in input data** in such a way that the model can be used **to generate new examples** that plausibly could have been drawn from the original dataset.
  (**How to design a loss function to evaluate the generation?--there are no obvious error metrics to use, since we are not told what kinds of patterns to look for**)

- **GANs**: are a clever way of **training a generative model by framing the problem as a supervised learning problem** with _two sub-models_:

  - the **generator** model that we train to **generate new examples**, and

  - the **discriminator** (i.e., adversary) model that tries to **classify examples as either real (from the domain) or fake (generated)**.

  - The **two models are trained together** in a zero-sum game, adversarial, **until the discriminator model is fooled about half the time, meaning the generator model is generating plausible examples**.
"""

# ╔═╡ 7643048b-add1-4ac0-9ddb-23adb3ac4542
md"### Generator"

# ╔═╡ 3d3ae07d-9b2a-4831-b22a-727f8e436e75
md"""
**Illustration.**

[ Random Input Vector ] → [ Generator Model ] → [ Generated Example ]
"""

# ╔═╡ b1b1a243-91eb-4d7b-bfc2-203b967c5ca1
md"### Discriminator"

# ╔═╡ 8defdfb1-0db8-4c69-ab49-a8928d18d854
md"""
**Remark.**

- The discriminator model **takes an example from the domain as input** (real or generated) and predicts a binary class label of real or fake (generated).

- The real example comes from the training dataset.
  The generated examples are output by the generator model.

- **After the training process, the discriminator model is discarded as we are interested in the generator**.
"""

# ╔═╡ 953c04f0-3d80-4d6d-b8f1-ac9d5c0f6dad
md"### GANs as a Two Player Game"

# ╔═╡ a44f0ddb-9fa6-4771-b3d6-0542506e16ff
md"""
**Remark.**

- The two models, the generator and discriminator, are trained together.
  The **generator generates a batch of samples, and these, along with real examples from the domain, are provided to the discriminator and classified as real or fake**.

- We can think of the generator as being like a **counterfeiter, trying to make fake money**, and the discriminator as being like **police, trying to allow legitimate money and catch counterfeit money**.
  To succeed in this game, the counterfeiter must learn to make money that is indistinguishable from genuine money, and the generator network must learn to create samples that are drawn from the same distribution as the training data.
  (NIPS 2016 Tutorial: Generative Adversarial Networks, 2016.)
"""

# ╔═╡ fa8e217d-8552-4671-9ee0-92455b114ee4
md"### GANs update the discriminator"

# ╔═╡ 8cd4ffaa-25ec-4b62-bba9-be421e56b05b
md"### GANs overall optimization"

# ╔═╡ 8f4a7a58-dd06-42d5-bebd-17dfff186513
md"### Conditional GANs"

# ╔═╡ 98da17be-fcf7-498d-9b85-a076d998fae3
md"""
**Remark.**

- An important extension to the GAN is in their use for conditionally generating an output.

- GANs can be extended to a conditional model if **both the generator and discriminator are conditioned on some extra information L**.

  - L could be any kind of auxiliary information, such as class labels or data from other modalities.
    We can perform the conditioning by feeding L into both the discriminator and generator as an additional input layer.
    (Conditional Generative Adversarial Nets, 2014.)

- In this way, a conditional GAN can be used to **generate examples from a domain of a given type**.
"""

# ╔═╡ bf57ed37-6832-4eb2-a6bb-2b66288492c7
md"### GANs vs. Conditional GANs"

# ╔═╡ c2a4c4d9-4d76-48ae-b59c-ed94902a9b62
md"""
**Illustration.**

- (a) GAN architecture

- (b) CGAN architecture
"""

# ╔═╡ 5a6bf12d-cdd5-4e66-82e3-4a8d9aaa1726
md"### Conditional GANs"

# ╔═╡ bf6e3506-ca86-4d8c-8f30-c66f72dade55
md"""
**Remark.**

- Taken one step further, the GAN models can **be conditioned on** an example from the domain, **such as an image**.

- This allows for applications of GANs such as **text-to-image, translation, or image-to-image translation**.

- This allows for some of the more impressive applications of GANs, such as **style transfer, photo colorization, transforming photos from summer to winter or day to night**, and so on.
"""

# ╔═╡ 619d71e8-85ae-4a07-8475-d609c8657292
md"### GANs Applications"

# ╔═╡ a23fe5f9-2a66-4d84-95b1-a537c40e592d
md"""
**Remark.**

- GANs for **data augmentation**

- **Image Super-Resolution**.
  The ability to generate high-resolution versions of input images.

- **Creating Art**.
  The ability to generate new and artistic images, sketches, painting, and more.

- **Image-to-Image Translation**.
  The ability to translate photographs across domains, such as day to night, summer to winter, and more.
"""

# ╔═╡ 9287b3c9-a81f-4797-873b-fca0aea6e2f1
md"## Adversarial Training"

# ╔═╡ d2988200-34f4-4301-88e0-c945d9c35778
md"""
**Remark.**

- **Main idea**:
  Adding noise, randomness, or adversarial loss in optimization.

- **Goal**:
  **make the trained model more robust**.
"""

# ╔═╡ 22fe58cf-24cb-4a9f-9f7e-8232b08499c4
md"### Adversarial Training in NLP--a simple example"

# ╔═╡ 86a860f7-0b06-4216-985e-825518e96b51
md"# Chapter 25: Reinforcement Learning"

# ╔═╡ eef1de1c-f0b5-44c8-abcd-f6bffd6a2193
md"## What is Reinforcement Learning"

# ╔═╡ cd010bcf-72ce-4ada-bc4d-8fe7b594b377
md"### So far... Supervised Learning"

# ╔═╡ 1f90deb6-17a8-4aa7-b25e-bc5dc126c13d
md"""
**Remark.**

- **Data**: (x, y)

  x is data, y is label

- **Goal**: Learn a *function* to map x → y

- **Examples**: Classification, regression, object detection, semantic segmentation, image captioning, etc.
"""

# ╔═╡ 1203ac77-0b13-4919-af2f-2597c0fcd13e
md"### So far... Unsupervised Learning"

# ╔═╡ ce06cac2-054f-4345-819b-249e03338fb6
md"""
**Remark.**

- **Data**: x

  Just data, no labels!

- **Goal**: Learn some underlying hidden *structure* of the data

- **Examples**: Clustering, dimensionality reduction, feature learning, density estimation, etc.
"""

# ╔═╡ 641012c8-6d68-408f-b10f-84dc5f32782d
md"### Today: Reinforcement Learning"

# ╔═╡ 0c03ec0d-a05c-4c29-a253-0d7ce2d75f0a
md"""
**Remark.**

- Problems involving an **agent** interacting with an **environment**, which provides numeric **reward** signals

- **Goal**: Learn how to take actions in order to maximize reward
"""

# ╔═╡ 43455e7b-607d-4bba-b62f-ae14c2ed70f8
md"### Cart-Pole Problem"

# ╔═╡ 271ce7f9-0dfb-4609-8abe-6c7d37fc3867
md"""
**Remark.**

- **Objective**: Balance a pole on top of a movable cart

- **State**: angle, angular speed, position, horizontal velocity

- **Action**: horizontal force applied on the cart

- **Reward**: 1 at each time step if the pole is upright
"""

# ╔═╡ 570585e8-d80f-4645-a8c8-6e80baaea2ce
md"### Atari Games"

# ╔═╡ 188038f3-0acf-4b46-8765-015c5ec847a4
md"""
**Remark.**

- **Objective**: Complete the game with the highest score

- **State**: Raw pixel inputs of the game state

- **Action**: Game controls e.g. Left, Right, Up, Down

- **Reward**: Score increase/decrease at each time step
"""

# ╔═╡ aca25cab-d7e7-4f08-b5f7-86cc01113f91
md"### Go"

# ╔═╡ 24e25bcb-c7ce-40c2-b931-1e29817b30ea
md"""
**Remark.**

- **Objective**: Win the game!

- **State**: Position of all pieces

- **Action**: Where to put the next piece down

- **Reward**: 1 if win at the end of the game, 0 otherwise
"""

# ╔═╡ cf25644b-74e3-4510-81b1-36776f69f2a8
md"## Markov Decision Processes"

# ╔═╡ 5187cd18-fa3a-4911-91e2-e7e38761f3dd
md"### How to mathematically formalize the RL problem?"

# ╔═╡ 2594edac-f75f-4a80-aab3-d020ff6eba4b
md"""
**Illustration.**
Agent (action ``a_t`` →) Environment. Environment (state ``s_t``, reward ``r_t``, next state ``s_{t+1}`` →) Agent.
"""

# ╔═╡ 966bbe33-1982-42ff-8841-3dd35900efd2
md"### Markov Decision Process"

# ╔═╡ eb2ef545-a694-4964-a673-ca0612b82438
md"""
**Remark.**

- Mathematical formulation of the RL problem

- **Markov property**: Current state completely characterises the state of the world

Defined by: ``(\mathcal{S}, \mathcal{A}, \mathcal{R}, ℙ, γ)``

- ``\mathcal{S}``: set of possible states

- ``\mathcal{A}``: set of possible actions

- ``\mathcal{R}``: distribution of reward given (state, action) pair

- ``ℙ``: transition probability i.e. distribution over next state given (state, action) pair

- ``γ``: discount factor
"""

# ╔═╡ 84045a7c-6e97-4bad-867e-bb3a22ebec0e
md"""
**Remark.**

- At time step ``t = 0``, environment samples initial state ``s_0 ∼ p(s_0)``

- Then, for ``t = 0`` until done:

  - Agent selects action ``a_t``

  - Environment samples reward ``r_t ∼ R(⋅∣s_t,a_t)``

  - Environment samples next state ``s_{t+1} ∼ P(⋅ ∣ s_t, a_t)``

  - Agent receives reward ``r_t`` and next state ``s_{t+1}``

- A policy ``π`` is a function from ``\mathcal{S}`` to ``\mathcal{A}`` that specifies what action to take in each state

- **Objective**: find policy ``π^*`` that maximizes cumulative discounted reward: ``\sum_{t≥0} γ^t r_t``
"""

# ╔═╡ 95562c13-aa2b-424b-907a-d158b9787a1e
md"### A simple MDP: Grid World"

# ╔═╡ 272fb642-3b34-457d-b6c4-1e9c0a7735ca
md"""
**Remark.**

- actions = { 1. right, 2. left, 3. up, 4. down }

- states = grid

- Set a negative "reward" for each transition (e.g. ``r = -1``)

- **Objective**: reach one of terminal states (greyed out) in least number of actions
"""

# ╔═╡ 5ce027b1-9a42-499f-b762-4d25baf1cbfe
md"""
**Illustration.**
Random Policy, Optimal Policy
"""

# ╔═╡ 5f912c0b-0c9b-4300-90cb-50a46d965b35
md"### The optimal policy ``π^*``"

# ╔═╡ 8d36e42f-9e6b-4c4f-99e5-b08eb20cd01f
md"""
**Remark.**
We want to find optimal policy ``π^*`` that maximizes the sum of rewards.

How do we handle the randomness (initial state, transition probability...)?
Maximize the **expected sum of rewards!**

Formally: ``\displaystyle π^* = \arg\max_π{𝔼} \left[\sum_{t≥0} γ^t r_t ∣ π\right]`` with ``s_0 ∼ p(s_0)``, ``a_t ∼ π(⋅ ∣ s_t)``, ``s_{t+1} ∼ p(⋅ ∣ s_t, a_t)``
"""

# ╔═╡ ea1070bb-3c0f-43e2-9a34-54ad000e00f2
md"## Q-Learning"

# ╔═╡ 2512548f-8f84-42c8-bb8a-1c2b93c80108
md"""
**Remark.**
**Game**:

- Imagine a board with 5×5; we can start in any white cell.

- The **goal** will be to travel and **reach the green cell (5,5)** using as few steps as possible.

- We can travel either **Up, Down, Left or Right**.

- We cannot fall out of the board, i.e. move into the red cells; otherwise, we die and we lose the game.
"""

# ╔═╡ 029d6d56-860f-42c6-953b-9d65f82586c6
md"### (Immediate) Reward table"

# ╔═╡ 2420ca5e-7658-4b32-9e46-03c0ab85d2ef
md"""
**Remark.**

- We can start with assigning a "distance" value to all the cells.
"""

# ╔═╡ 8b982174-66a7-4f9e-81ff-bbc21bee6d04
md"""
**Remark.**

- But Q-learning and reinforcement learning in general is about selecting an action that gives us the maximum reward overall.
  And here **reward is the inverse of distance**.

- Also, going off the grid here means we have lost and so should be penalized.
"""

# ╔═╡ 43e5f780-1eb5-4645-ba8f-33cbced20c5a
md"### (Immediate) Reward table: other examples"

# ╔═╡ 799d96c2-1ab0-488c-ac6d-caa45ece806a
md"""
**Remark.**

> Reward when reach step closer to goal = +1
>
> Reward when hit obstacle = -1
>
> Reward when idle = 0
"""

# ╔═╡ 4779ee5e-ebed-467f-8724-ffb148161a5f
md"### Q-learning table (aka Q-Table)"

# ╔═╡ bd7b6929-48ca-4119-adca-976e8f5a3cf6
md"""
**Remark.**

- Q-table is a 2d matrix of status-by-action

- Q-values are what our algorithm should derive

- Q-values can be all zeros or randomly initialized in the beginning.
"""

# ╔═╡ 4a3b7003-d290-4b9a-a85e-b9d1b07c9e80
md"### Bellman equation"

# ╔═╡ f6b52ffd-269f-4135-a5f3-82df747fbd91
md"""
**Remark.**

$Q^{new}(s_t,a_t) \gets (1 - α) ⋅ Q(s_t,a_t) + α ⋅ (r_t + γ ⋅ \max_a Q(s_{t+1},a_t))$

- ``Q(s_t,a_t)``: current value

- ``α`` : learning rate

- ``r_t``: reward

- ``γ``: discount factor

- ``\max_a Q(s_{t+1},a_t)``: estimate of optimal future value
"""

# ╔═╡ fa7e7564-0447-472d-a0bd-275731992fb2
md"### Exploration vs. Exploitation"

# ╔═╡ f85af1c7-5b19-42b4-8342-c074b90ef6c9
md"""
**Remark.**

**Issue**:
Perhaps during the first few tries the algorithm finds a particular action for a given state rewarding.
**If it keeps selecting the max reward action all the time time**, without trying anything else and perhaps some other untried action has a better reward than this.

**Exploration**:
allow the algorithm to select a random action with a probability

!!! note "Epsilon-Greedy Exploration Strategy:"

    $Action = \begin{cases}
    \max_a Q(s,a), &R > ε \\
    Random \; a, &R ≤ ε
    \end{cases}$

    ``R`` is random number between ``0`` and ``1``, ``ε`` is the exploration factor between ``0`` and ``1``.
    If ``ε`` is ``0.1``, then ``10\%`` of the times, the algorithm will select a random action to explore corresponding rewards.
"""

# ╔═╡ e6347331-5c3e-4cdb-bbda-6eac91b61343
md"### Q-table based Q-learning implementation"

# ╔═╡ f00a8609-0569-4f9f-861c-5553b40966d0
md"""
```python
class Game:
	rewards = None
	positionCol = None
	positionRow = None

learning_rate=1
discount=0
random_explore=0.1

for i in range(1000):
	game = Game()
```
"""

# ╔═╡ e6ce7a39-eff2-4e1f-8166-115dee02962c
md"### Q-learning algorithm"

# ╔═╡ 5efa6c70-c17d-4f65-9c70-fd80409b4d1e
md"""
**Illustration.**

Initialize Q-table → Choose an Action → Perform an Action → Measure Reward → Update Q-table → Choose an Action

(A number of iterations - result a good Q-table)
"""

# ╔═╡ 657d07c3-7194-492c-872f-691e197afdea
md"## Deep Q-Learning"

# ╔═╡ 8e2ec07b-7023-41d1-b5db-37ce8a939e1f
md"""
**Remark.**

- **Vanilla Q-Learning**: A table maps each **state-action** pair to its corresponding **Q-value**

- **Deep Q-Learning**: A Neural Network maps input **states** to **(action, Q-value)** pairs
"""

# ╔═╡ 6713e9b4-52a4-49e9-96fe-ca03e6397d94
md"""
!!! note

    A core difference between Deep Q-Learning and Vanilla Q-Learning is the implementation of the Q-table.
    Critically, Deep Q-Learning **replaces the regular Q-table with a neural network**.
"""

# ╔═╡ 05e7cb15-101a-43fb-9b86-734db7446b6c
md"""
**Illustration.**
Neural network.

- Input States [.1, .5, -1, 2]

- Output states [8, 5]

  Each output node represents an action.
  The value inside an output node is the action's q-value.
"""

# ╔═╡ ab92b4f0-b969-4045-8ebe-f56bfdb2cad5
md"### Deep Q-learning architecture example"

# ╔═╡ 9f7b4e6f-34e6-40e8-9152-25284d728d53
md"""
**Illustration.**
``Q(s,a;θ)``: neural network with weights ``θ``.
Current state ``s_t``: 84×84×4 stack of last 4 frames (after RGB→grayscale conversion, downsampling, and cropping)

- Input: state ``s_t``

- Familiar conv layers, FC layer

- Last FC layer has 4-d output (if 4 actions), corresponding to ``Q(s,a_1), Q(s_t,a_2), Q(s_t,a_3), Q(s_t,a_4)``

Number of actions between 4-18 depending on Atari game

A single feedforward pass to compute Q-values for all actions from the current state ⟹ efficient!
"""

# ╔═╡ 4c57e400-eab1-4290-9a6e-0fef12980299
md"### Challenges in Deep RL as Compared to Deep Learning"

# ╔═╡ a9467c2a-8b9d-4fbd-a20f-06525e5fa06c
md"""
**Remark.**

- Chasing a nonstationary target

- Updates are correlated within a trajectory

!!! tip "The target is continuously changing with each iteration"
"""

# ╔═╡ c146e9b6-be90-4ee3-a022-0b7fa511dc81
md"### Solution: two neural networks"

# ╔═╡ a6d72ae2-1781-43ac-8de3-3ab28dd69c12
md"""
**Remark.**
**2 neural networks:**

- Have the **same architecture but different weights**.

- Every N steps, the **weights from the main network are copied to the target network**.

- Using both of these networks leads to **more stability** in the learning process and helps the algorithm to learn more effectively.
"""

# ╔═╡ 8bdb73ef-c86b-4f33-8055-180681a0b178
md"### Full deep Q-Learning process"

# ╔═╡ 5f199652-cb3d-44a5-a726-9ae0628da4d2
md"""
**Remark.**

1. Preprocess and **feed the state s to the prediction model**, which will return the Q-values of all possible actions in the state

2. **Select an action** using the epsilon-greedy policy;

3. **Perform this action** in a state s and move to a new state s' to **receive a reward**.
   We **store this transition in our replay buffer as <s,a,r,s'>**

4. Next, sample some random batches of transitions from the replay buffer and **calculate the loss**

   $Loss = (r + γ\max_{a'} Q(s',a';θ') - Q(s,a;θ))^2$

   which is just the squared difference between target ``Q`` and predicted ``Q``

5. Perform **gradient descent with respect to our prediction network parameters** in order to minimize this loss

6. After every C iterations, **copy our prediction network weights to the target network weights**

7. Repeat these steps for M number of episodes
"""

# ╔═╡ 58b4a7b9-a359-4343-83f0-43c2db158bfb
md"# Chapter 26: Trustworthy Machine Learning"

# ╔═╡ 361cb208-765b-4f7e-a233-1836a950e911
md"## What is Trustworthy ML"

# ╔═╡ 5b7e9525-18fc-4630-a48a-d9ca7364ea8f
md"""
**Remark.**
As ML systems are increasingly being deployed in real-world applications
"""

# ╔═╡ 8a47ae00-2226-4e3b-bf18-bfe56d11254b
md"## Explainable ML"

# ╔═╡ 5f43acc4-06f0-40ca-8e77-80eea5cfc67a
md"## Fairness in ML"

# ╔═╡ cd27aba7-8207-4dd1-aad5-09839abf1fa3
md"## Privacy-preserving ML"

# ╔═╡ 7cbd9c20-4f70-4294-b47d-ceee9e39483d
md"## Robust ML"

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
Plots = "~1.35.5"
PlutoUI = "~0.7.48"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.0"
manifest_format = "2.0"
project_hash = "cb3ea6ae02cd694efde5bdb98553263d714f1867"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "84259bb6172806304b9101094a7cc4bc6f56dbc6"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.5"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e7ff6cadf743c098e08fca25c91103ee4303c9bb"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.6"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "1fd869cc3875b57347f7027521f561cf46d1fcd8"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.19.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "d08c20eef1f2cbc6e60fd3612ac4340b89fea322"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.9"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "3ca828fe1b75fa84b021a7860bd039eaea84d2f2"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.3.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.DataAPI]]
git-tree-sha1 = "46d2680e618f8abd007bce0c3026cb0c4a8f2032"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.12.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "c36550cb29cbe373e95b3f40486b9a4148f89ffd"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.2"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "00a9d4abadc05b9476e937a5557fcce476b9e547"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.69.5"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "bc9f7725571ddb4ab2c4bc74fa397c1c5ad08943"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.69.1+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "fb83fbe02fe57f2c068013aa94bcdf6760d3a7a7"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.74.0+1"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "Dates", "IniFile", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "a97d47758e933cd5fe5ea181d178936a9fc60427"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.5.1"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "49510dfcb407e572524ba94aeae2fced1f3feb0f"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.8"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "f377670cda23b6b7c1c0b3893e37451c5c1a2185"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.5"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "ab9aa169d2160129beb241cb2750ca499b4e90e9"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.17"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "94d9c52ca447e23eac0c0f074effbcd38830deb5"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.18"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "5d4d2d9904227b8bd66386c1138cf4d5ffa826bf"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "0.4.9"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "03a9b9718f5682ecb107ac9f7308991db4ce395b"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.7"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "3c3c4a401d267b04942545b1e964a20279587fd7"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.3.0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e60321e3f2616584ff98f0a4f18d98ae6f89bbb3"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.17+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.40.0+0"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "6c01a9b494f6d2a9fc180a08b182fcb06f0958a0"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.4.2"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "SnoopPrecompile", "Statistics"]
git-tree-sha1 = "21303256d239f6b484977314674aef4bb1fe4420"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.1"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SnoopPrecompile", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "0a56829d264eb1bc910cf7c39ac008b5bcb5a0d9"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.35.5"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "efc140104e6d0ae3e7e30d56c98c4a927154d684"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.48"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
deps = ["SnoopPrecompile"]
git-tree-sha1 = "d12e612bba40d189cead6ff857ddb67bd2e6a387"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.1"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase", "SnoopPrecompile"]
git-tree-sha1 = "9b1c0c8e9188950e66fc28f40bfe0f8aac311fe0"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.7"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "90bc7a7c96410424509e4263e277e43250c05691"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "f94f779c94e58bf9ea243e77a37e16d9de9126bd"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SnoopPrecompile]]
git-tree-sha1 = "f604441450a3c0569830946e5b33b78c928e1a85"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.1"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "8a75929dcd3c38611db2f8d08546decb514fcadf"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.9"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.URIs]]
git-tree-sha1 = "e59ecc5a41b000fa94423a578d29290c7266fc10"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "58443b63fb7e465a8a7210828c91c08b92132dff"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.14+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "868e669ccb12ba16eaf50cb2957ee2ff61261c56"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.29.0+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9ebfc140cc56e8c2156a15ceac2f0302e327ac0a"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+0"
"""

# ╔═╡ Cell order:
# ╟─e8c3d4aa-35a3-4b6f-bf38-68a8b0cbe4de
# ╟─dbd717e2-2881-11ed-096a-5f8228b7f240
# ╟─cf78886f-e116-4c98-98b2-1309f846f41b
# ╟─7bca21d0-8d31-49a4-9c03-751ab2d9dcb0
# ╟─557f06f3-d19b-447d-acfe-b110a6c66541
# ╟─103d2317-10d9-4c51-bf3e-ad91d6fbc03c
# ╟─cf5859c1-3ee3-4774-9f29-720f5153ef67
# ╟─097dee19-1a1d-4fa6-8fbf-2a1410557a6e
# ╟─3c4a21c5-6e5d-4cbc-8219-78874b83666d
# ╟─c44aa6bd-0037-4ba8-a8e7-794c7487b0a6
# ╟─4c677095-bd8e-43ad-a690-b9e1a6c0dc74
# ╟─16435bc0-0b50-4e81-8d04-b7b07ba992ee
# ╟─e7712c48-4679-4dab-8404-58152e292a93
# ╟─bf8305b5-1e28-463f-86bc-f92e6eecf8f5
# ╟─b439c8bd-a78d-49e2-91f2-0e3f23506b84
# ╟─7a86d245-cf3b-4f41-a544-9e719927f436
# ╟─3c6f81aa-2e6a-413a-850f-afb2995f2c92
# ╟─f87e1a07-4a2d-46d3-9b9a-2b1b9f63ef0d
# ╟─c7063785-ea15-47ca-94ad-1299a821c966
# ╟─516360e4-e63f-4df7-a807-d60983ca5d59
# ╟─02fa00bb-6fc7-4187-a069-032b75b456e7
# ╟─76bca31b-4489-4b23-b182-ddd7aaa75015
# ╟─9a75a73e-15a3-4030-9b05-6722b690a46f
# ╟─98f46c15-d796-4e9a-a529-4c002141eeda
# ╟─a20021cb-11f9-45a4-be9c-f97065732d8d
# ╟─da29a50c-d22a-437d-8af5-8e0e36f70b94
# ╟─d65cb3bc-635f-4035-88aa-a0d2829ca80b
# ╟─336f7664-44ec-418d-9667-4529862c0aff
# ╟─40af75c4-05f4-4431-b604-f69a76dff8a2
# ╟─5c3e598f-800d-4459-a5d7-19c6f9633565
# ╟─26afc9f2-6fb1-4510-97ee-569990ef8f77
# ╟─463aac99-a9eb-4243-851a-f9b16ee1ada1
# ╟─64945175-aa24-4626-bad6-58e8252449a9
# ╟─ee27fa04-be91-44c0-8e46-9ec2d087a4d9
# ╟─6579bb80-5776-4633-ae88-da3ecbd68ea5
# ╟─3b28b888-1731-4570-a4b6-b9a47741ee78
# ╟─e468c69c-5c71-4dd3-8d76-03b90e9db7cd
# ╟─bd5f5fd5-4843-4fe0-b653-f9ee95a3d3a8
# ╟─44cc65e9-b141-4d1b-bc2b-49143c55640a
# ╟─16c36830-7dc7-42fb-867f-428e574464df
# ╟─7fbd1f91-eb56-40f0-9f97-a340e45d1edc
# ╟─bae5ff75-0b84-4621-b6e6-6d3e6259a152
# ╟─dc7e9f55-2af4-4137-9815-942a1e7543a3
# ╟─047b5cb4-07c2-45b2-930d-c14560847037
# ╟─e82d83b9-4e75-4e87-b491-2c8956529be7
# ╟─69fa269a-adca-435c-8806-1909a771f136
# ╟─ab86c39c-72ec-47c4-8f28-ea33eeeb6f99
# ╟─6d2b12c4-91a6-4331-9d6b-5bbdb930c6b9
# ╟─73e8b017-c3a6-4a5b-bf9c-0231509050b0
# ╟─6e242ba3-22d7-4e9f-8851-e3d6c9d4a094
# ╟─663c3c47-1f41-4842-9214-7ef4c926157e
# ╟─b552b2b8-2e8c-4777-8514-9cb233a3c8cf
# ╟─b9c2e732-51e5-4658-ace6-bf3a84644229
# ╟─4adc46a2-5102-416d-bb48-26e9cf89daf6
# ╟─41da3074-fa8d-41b8-8a52-620282cd0b36
# ╟─fd189d77-d09c-404c-930b-c304a72a94fd
# ╟─4fe5d69c-9893-4484-b14e-c8056d21df3a
# ╟─8f9f4b40-33a9-48b6-9f18-8afd5a6b5ccb
# ╟─1bba3acd-b301-44f1-a013-6bcb94e41a41
# ╟─8555d610-41e9-472f-83fe-d3c3a296209c
# ╟─71907b36-45d6-408d-a241-547a8dba925d
# ╟─30c3d0ad-825d-4026-a8f9-c287d1e69d7e
# ╟─8890c772-eb26-4c1b-888d-c5b1d72d5534
# ╟─eef7c272-3ac7-4ca3-aa37-6d48079601f9
# ╟─e67ef5c4-4f56-4078-b48c-a23b58e16bbd
# ╟─a9898af8-f664-4de1-932b-7fb843b59222
# ╟─dc32ace2-788b-4043-9a08-66517438978f
# ╟─e4fc4f42-a2b8-4ed9-9e87-4e4fb9c3380a
# ╟─d30577d4-b522-4499-a5da-88e613f948fb
# ╟─0225875c-eb1a-497d-ac2a-261b202f8be6
# ╟─f4fd53c9-c0da-4887-b23d-46e95f658cf0
# ╟─c35e5392-633b-40db-b8b0-3ec89ea5ae46
# ╟─d8a395bf-50fb-4978-a6bc-8460f05ae22b
# ╟─07a28400-702d-491c-adea-9a0009a1320b
# ╟─a783a435-a444-4d10-a821-4c2c82fb8782
# ╟─fdc18ae1-66a0-4c14-ad0a-fe74f2bbe934
# ╟─4351fe34-0889-4e2b-a6fe-17274e9dd4d8
# ╟─ff1eb1a0-a3de-497e-acf7-ae99aa9a77b6
# ╟─eb86bed0-a3a3-4f0d-903f-4896ddda60b3
# ╟─a5607a6d-d568-432b-ac62-5bbabb98f91f
# ╟─3270cedd-f10e-4eb1-9055-0d4e49e21117
# ╟─ce28b17c-65bd-4864-acaa-3f94a59b237c
# ╟─a4d22fcf-e1e4-43e6-b910-05f8d90b8ba3
# ╟─8e3a92ac-4106-47b1-b5a3-2297e187b422
# ╟─95c4e156-949e-4bdf-86a6-3777a3e6d902
# ╠═5e2a016f-d794-48e7-b6d5-6d44a91e235b
# ╟─51b488a9-4a86-4961-8455-d2ef27dbad19
# ╟─9e8537b9-5816-4d96-bbe5-070a24079dc7
# ╟─c0893a95-c170-486f-97cf-140844425526
# ╟─af921af7-858e-4ec4-81c7-a603bc514c87
# ╠═c5ae5f8f-e282-47e1-8b56-1696c236ed6f
# ╟─c7d9411a-d867-42b5-bbba-8a05ba9752d4
# ╟─f5079220-98f7-4f9b-aacc-135ca7af1144
# ╟─041814c2-cfa8-4d2b-bf70-569275bf2ac5
# ╟─cf2bd0d3-52be-4884-b208-da31fc810154
# ╟─79eb1d7f-e404-4fa0-bc0c-a85c7100fc93
# ╟─12b39d55-c851-469b-926d-c91c2be80bdd
# ╟─9a063c8f-7a98-4b31-9b06-5ed9dcf2199e
# ╟─89d3ef5c-514b-45bb-b57a-6d1299cec274
# ╟─88540572-648b-41f6-ae30-da5afbdf6b8f
# ╟─19ceccca-86f9-414f-b72a-3ed99210d2d2
# ╟─ceb891a6-ad25-4fcb-b151-c78afc58fe11
# ╟─12008ff8-267b-44a3-a1de-382e51675310
# ╟─0d3aabb0-19ee-472f-83cb-dc849efccff4
# ╟─6a7b2762-a48a-44c6-866d-41a13f129c3d
# ╟─4aecc90a-75b1-4df2-8eca-ce2d73d77d61
# ╟─4740ecda-4a11-4fc3-8ff5-f57b435609a1
# ╟─39e7ca7b-1da3-4630-ad1d-1cd6c1a4126e
# ╟─31593502-b853-4ffa-95c0-0e25e92a23d3
# ╟─bb9b8343-255a-4b42-a7fa-76d9af26ac30
# ╟─c06c6dec-b508-4a27-8f0c-2642b6fa18fd
# ╟─1c3e187f-97c9-4969-bc05-bf90bc35fab8
# ╟─6acc2d90-1d95-4ba4-895d-6aea53ff7c7e
# ╟─9671a7a3-fb78-40e8-85d0-7768ce5a3c05
# ╟─b7fef3a4-a9d2-4163-b6f2-e465f606df60
# ╟─b9932f67-9845-4a56-aefd-5733bcc0ee44
# ╟─eae01ecb-5424-4d3d-8a13-29d2d06fa2b6
# ╟─0e671da4-9a5d-4424-b7a4-b81ab759e4ef
# ╟─add4e352-10a4-4af5-8624-63a6271a5674
# ╟─c352e06a-af14-48bc-89c7-b308bf183fbd
# ╟─664eee50-926f-44ce-955b-fa6664f1485b
# ╟─80525897-e876-4ab2-8f0e-b8b88f464a58
# ╟─546e0b09-4b1a-426d-8a17-fc30061d9854
# ╟─f7f4e5a5-0c5f-4ddb-9968-fb8be8cad1ab
# ╟─126ebfb3-4cfc-4bc0-bbd4-e1cab157fcda
# ╟─39350f3e-62d6-4616-946c-5391e6ab1490
# ╟─320987dc-51c4-450f-a9f7-fb3af8100b88
# ╟─300d0aa0-f187-4bb7-b36e-df13ee4be5e8
# ╟─d6584658-a630-4095-bfd9-1905a0c14e0e
# ╟─86976901-429f-4c87-81d2-e98272868b9c
# ╟─54771b10-6e45-49b3-adb5-a391b671a80b
# ╟─09ec58f5-7f88-4558-89e9-61a50f592b96
# ╟─57a9eba3-2e81-472e-aed3-d8a3563b0582
# ╟─72bba976-8b69-419f-b8bd-37b3201e2cc5
# ╟─4c366d03-ca66-4a14-8793-49d3e0bc6e99
# ╟─d0f2049f-0f75-4fb5-a8a3-0793944ffe80
# ╟─b0ff0312-8600-4562-b2a6-071d5a01d60f
# ╟─321bf029-e4ce-4cf6-80a6-653ab0df5720
# ╟─ba63a81b-5065-4a91-bd71-adbe9d36ac5e
# ╟─7d2cbc23-0dff-4d4a-8c02-dd88a67f7941
# ╟─37e7c23b-65ea-4b81-9bc0-f33d470f3fb1
# ╟─4354a5cc-a55f-49b8-b51b-fd1924c573cc
# ╟─b8927499-20e3-4b81-9be9-740379cccc63
# ╟─c72a211e-ad7f-47a7-b0f4-aa69fa13ad92
# ╟─c82d9096-01a6-4b5a-b529-32fe13965452
# ╟─ca9805d9-9357-490e-b9aa-4c48c920e57d
# ╟─9fa42e55-c9c3-4f74-9ac6-f2ba75d4753a
# ╟─630a1c82-e3a4-4d39-b864-06d2665228a7
# ╟─1afdb2d4-aa31-454c-ab2c-527d95f16cbd
# ╟─f7a3ddea-6a57-48eb-99f5-1d09f048eeaf
# ╟─abbf9629-43ad-411d-a7c8-82c8ee38080d
# ╟─671b5fe3-28a4-4cf3-9250-6981c16b1d60
# ╟─ff9fc83e-ee89-4a8e-9026-44f01b5059e5
# ╟─710e7107-c5f7-4d62-ad33-7660c1843430
# ╟─080556d0-abcd-4ba7-b013-c8891b180bf6
# ╟─36054985-53c3-490d-9c49-df0cc9346f7f
# ╟─00608dba-4ca5-4e55-8b29-fd867a307eb3
# ╟─ce7bc9c4-225d-4218-b55f-17771239e366
# ╟─5f41a7c7-1ec1-46c0-adc7-e3df554d661d
# ╟─811b5dc7-8fff-43d4-bc42-70cf46b3e84e
# ╟─af5977b0-be59-49f4-bfdb-0fa061f1c79e
# ╟─c56ef9ed-3495-41e8-ad21-2dc7a534609d
# ╟─9a6beed0-cc54-4b10-9118-4f01555871ba
# ╠═b68c461d-debe-43b4-aced-fb5732a359e3
# ╠═fc5721b9-eb8e-446d-9d46-a4737107ae03
# ╟─a9267d45-09e8-41c6-9caa-510e22fc1db1
# ╟─768d0c09-b6a3-47c2-9963-6f3392fd69e5
# ╟─337f5b10-fd08-4692-b7ce-30aadad377ca
# ╟─3cca0baf-2988-4718-b9b5-b23974e1f609
# ╟─eef43fc0-52a6-42c8-9c84-c2792d3a63d4
# ╟─8c7d8b1c-1f86-4bbe-ad37-aa165f05bd8d
# ╟─ef4bb6eb-1d42-41da-b580-7719bcea5353
# ╟─b720ca3d-bc73-4a8e-967d-7af7d8d490f0
# ╟─d691a929-b1bb-4591-8d23-5f7b79619d98
# ╟─d49a81d1-e01d-4c4e-82d6-17dfbe3caa71
# ╟─fdd1a376-63e9-4c1f-b028-26027dc6dff5
# ╟─50f09ea6-397d-4618-9643-c886064da6b1
# ╟─5e919682-a372-41b4-aff3-8c8a0c85b5fd
# ╟─f8b1b31e-4816-4009-900f-e95d92e236bc
# ╟─458e9f6a-2a69-4a42-afc6-d8c954b07957
# ╟─149293eb-5e66-45e3-9233-eacb111b8e1d
# ╟─2282b447-6e2b-4a6d-a44f-057b5cd6a1d5
# ╟─8259ee1f-e1e5-4e72-b5a3-c687fd70c9ab
# ╟─1ecf44fa-f0ed-4c22-aea2-35b8150b0ac6
# ╟─ebf68f55-afd9-44a7-bc42-984afc8f5cfd
# ╟─e3afa121-9899-40d4-ad5d-669c4b6b069d
# ╟─009ed35a-49ac-49e2-98cb-e722fcf624f4
# ╟─b666903a-6dea-4a58-9539-04635afd0e9e
# ╟─f63a7934-489c-4933-84c6-fca830c764bb
# ╟─20590a92-9ff3-419f-92e8-eb351c43fd01
# ╟─4b0fbbeb-293b-4340-941e-1c2e5a764a41
# ╟─888a46c6-c548-4ba5-8055-795e41000ffa
# ╟─7c58bca4-3b4c-4b75-8b10-7113e4c7c54a
# ╟─f5d68a19-3cb3-4b47-8260-c3765b24e423
# ╟─c4bda0c3-c0bd-41f1-a40f-7f1e2bdef39a
# ╟─b6f3a1ca-275c-483b-a75e-545b00c995a6
# ╟─54c2c096-5107-4479-ba06-62c43c380610
# ╟─34e441ac-74c9-4f41-b477-c519330eb58f
# ╟─67655ea6-b01b-4aa5-8677-31897bfc5e21
# ╟─d98a2090-eee5-4a72-90c9-efee51b99fec
# ╟─6f0a95e8-b091-4151-a5ba-381de21ca390
# ╟─68a31310-2d63-4d49-bd31-ae321160e508
# ╟─760c855d-6aa6-40f1-9ae6-bb30c0293877
# ╟─9bb1ceca-70c9-4e10-8f59-8c0a0142ff03
# ╟─c515d60e-00a4-4e86-96e3-9d265ab113c2
# ╟─37950e6c-4a34-4513-ac37-0d90c50eb390
# ╟─2db8bf2e-5755-4f63-9b5d-de209aa86a6b
# ╟─7cf2debb-c558-4bcb-af05-b5a7bc0935a0
# ╟─98129378-93be-4895-9950-17b9193fecd1
# ╟─afb99d1c-aff9-4cb5-8bd5-b5e25ebe9fed
# ╟─26b9a0d6-58aa-4f95-9c50-6ea879922a70
# ╟─6ecac6ff-8d9c-49fd-a36a-68a209656734
# ╟─d2c7bb32-16ff-421d-b9f6-b13411bbe8b8
# ╟─f75b35bb-5018-433d-b795-31646a1cfb85
# ╟─ea847b13-b4b7-474d-9ef0-d1c5c659bf36
# ╟─5a5f401f-a68c-4383-82b8-d593f2abfd7b
# ╟─ca313f8e-b222-43c3-8f91-cd5c3f85d2ea
# ╟─2e9ba193-d51e-4967-bcc5-107a634a23ce
# ╟─b72e3384-01b2-46a5-a5a3-26bfddf2155b
# ╟─b61385fe-4126-419f-86c3-4681acc26b66
# ╟─0b76f9bd-cf9b-411d-9a1e-ef77877ac976
# ╟─125fa440-073e-4ceb-abb0-17f28a24cdd3
# ╟─2f16ab06-6623-447d-9c6c-e6f7d9664b3c
# ╟─2944719a-9a88-4031-8ecc-cd8e5140ebc6
# ╟─23f73935-6306-4a63-bdcd-a5c1ff3d773d
# ╟─72dcae0f-0100-4f1f-9a35-90284e9a80d2
# ╟─5217080f-219b-40c0-a350-985539ad8fc1
# ╟─2f5a4ba3-62f6-461e-b757-b1e9291b3899
# ╟─4162a41b-1695-4595-89f2-4738e0eecd06
# ╟─2df5c299-48db-4bf2-acf0-cdc8e52729e0
# ╟─8b4cf5cb-82b5-45d8-809a-a8ad8db75b9c
# ╟─eea21eeb-56b1-42e6-ac34-6b237a126e6a
# ╟─289e7a2c-983b-4295-9f9d-3fb327a4436e
# ╟─6dc86a2a-c2d5-44b2-90ad-d63fb4a35784
# ╟─35fa31d7-a2db-4b8e-9236-a6007e46cbd0
# ╟─6b033d49-8900-4e24-8881-5c1ad6f2d245
# ╟─260c7002-80c4-4e9b-a315-07a0e537ca07
# ╟─939e7c62-f9c0-4fdf-891d-01fd8dddde0c
# ╟─04b3498d-ce70-4c13-8b1d-6ae2db571754
# ╟─e8650da5-50db-46b4-8784-1a9c23a48b5b
# ╟─5d68fcae-7f86-439f-a8b5-6a9a56b063ea
# ╟─d89696db-5f51-4f7b-8591-12e4fd89d4c8
# ╟─48fa755c-6805-43e4-8235-bfe45344c334
# ╟─d6842622-0220-4bf2-91e7-cc3cf2539a79
# ╟─6018d647-94ea-422a-80bb-c9c4ee289f6b
# ╟─cc98a6b2-b29d-426d-95c8-06ddd1efb736
# ╟─c2d9a478-f693-4048-a24c-53ce5fb2bc31
# ╟─43551f75-17f3-4c33-8f18-287b7f937be7
# ╟─1a7b2ba6-537a-4e22-9f0f-5887cd664f63
# ╟─02534887-8c8a-4067-a099-f1806e2f3d1c
# ╟─de6d3dfc-e596-45a1-b7e2-2a377e60c06b
# ╟─c4e7640e-e1bd-4f80-a165-0f866cea28de
# ╟─c6ce24ea-e754-488a-b6b0-8c8b94fce598
# ╟─92d0f71f-3b8d-4d56-806c-29b11587258b
# ╟─4bf464a1-abf9-4095-bdb9-ee7098cb725e
# ╟─62742e7d-e201-45a6-a7cf-7f26caaa314d
# ╟─4d13f162-00cc-4d29-ab27-e925cbb57c4c
# ╟─d6eebe65-d1be-429b-9b71-07b9fc4715a3
# ╟─9357503f-0d71-4484-8ba7-81126e870f2a
# ╟─1890e612-3914-42cb-9f9e-a6b8e8478d95
# ╟─5eca0c6a-c9fd-4656-b399-f2ba56e3cda7
# ╟─da665a8c-16f9-4153-92c4-cb1d46855908
# ╟─1c77794b-14e4-4baf-861a-6835ddd24b74
# ╟─e0eb3686-c0b8-4c8c-a934-e3709fa9e9f2
# ╟─42e5c894-439f-4d6c-8c3c-9d35f0c7d332
# ╟─bf3ddab2-7bfb-42c7-aec0-fad614c040cf
# ╟─c97a6716-6877-4a96-b674-75f722261d5a
# ╟─4a948739-bfaf-481e-95d2-413e787af10c
# ╟─06c92fdb-70db-498e-82d9-8398c23c3b07
# ╟─34e9a2b8-8d1b-4926-8f01-74c2e3797bf7
# ╟─699c926b-e98f-4091-a63d-647abe93acbc
# ╟─8e8c0ae6-b135-4170-8ce9-7cc1258cc3e4
# ╟─b549ad91-6f75-4528-afdc-e57a77265e41
# ╟─c6ebc38c-b146-4b5d-850f-ccfadab834f7
# ╟─f5f23cc7-8971-4b85-8be0-65b89abf1c25
# ╟─9e482437-b816-41f2-a2b5-cd39bfca9f9a
# ╟─936effb8-f639-4500-b33e-568a84d32db1
# ╟─8c6f1649-3228-4186-9913-871a3faf907e
# ╟─7b3dec37-8994-4a85-9586-b5443b86ac3a
# ╟─950b6497-58fc-423b-98d5-2f643e84f606
# ╟─97bf5e20-3024-4913-9983-93fd94d3c2a8
# ╟─98fede0f-89f4-4a99-94b7-c309511c6fd1
# ╟─ce376e97-aa93-4664-bb22-5c8a94cb6ea2
# ╟─ff1c1790-d880-4c38-9f84-26edab6d3612
# ╟─88238432-40f3-4a83-8fb7-338a31c03ca5
# ╟─62d107ce-b957-4bab-9670-eed33bc5afa8
# ╟─20b190b6-b090-4ee2-b7be-475cd171bf9b
# ╟─a6c2720b-83c2-4ac8-a76a-a7517ac79e96
# ╟─70623b38-f30e-45a4-9835-5d05e3df3344
# ╟─2a9821f4-b7fe-4f64-8557-f9b57a8ebe19
# ╟─baa40833-37c8-4946-98fe-da482789f491
# ╟─fd02b5a7-f7a8-49c7-a10e-1568eb29d0d4
# ╟─16682257-f2b5-4966-af5e-a2d6287968b2
# ╟─184910d6-db53-4631-9371-29fcec2a33fe
# ╟─28e48281-1039-4cab-ae30-609d23e7992a
# ╟─eda636aa-60a7-4022-ae24-e29065f150fc
# ╟─11cecba9-4c91-451c-9d27-a17336907097
# ╟─299dec55-e364-4bfb-bc5d-ec5cf6cf5ca4
# ╟─a067f2b1-b581-4400-b47d-9ba63054acfc
# ╟─22a5177e-1866-4230-8ef4-44ab0b561a9f
# ╟─9c3e4d8a-9972-4f03-af8b-f896a66016ce
# ╟─f4e267c2-71da-4da4-86e1-f0c1653b6db5
# ╟─e18d0f0d-2989-465f-ac3f-a520ef669a54
# ╟─be397193-7162-4efa-8166-a30cc6f612e9
# ╟─49575adb-1bc1-4239-a966-9bc8bff34f9a
# ╟─6192c0c3-2856-44f6-bb9a-83f019c6d0d7
# ╟─1f4c8b2b-fd2b-4f00-b705-6f1f5b658f20
# ╟─782ba2ee-4397-41c9-9b95-6453405ae047
# ╟─42f4af1b-7216-4bc8-b120-201212196819
# ╟─cd585b92-a445-485f-b81b-dbcae9de210d
# ╟─fcb8b589-1776-4f86-a797-cc588482574e
# ╟─74ce0427-a696-443c-b8ca-f1bfb3aebe25
# ╟─792bf56b-e160-4e06-9705-9d7a8e691cec
# ╟─d1926abb-28bc-4632-8d11-59284077f3eb
# ╟─c8a65e59-937e-4adf-8bf2-0f80580175c4
# ╟─c906256f-4885-4318-ad76-c6204d5f3647
# ╟─bf4173b1-fe25-4744-91cc-3b1e61857d0d
# ╟─ccc65c1e-feea-464e-8acf-6f0ad9405f2d
# ╟─4ccaaa0f-1bcf-469a-8e28-5b09fc31c3df
# ╟─136ae1ce-a2ef-40d7-bb22-aae7e3fc0685
# ╟─792d0a82-cf87-4d39-b12c-8028d01d08f5
# ╟─1a250eff-9fb8-474d-bf7e-dc22f7b5e99d
# ╟─6ba25d03-4ec3-44b1-bee8-07ef74353c4b
# ╟─a761d4c2-3fd2-4142-b989-11dadd4c8864
# ╟─3e39ea11-dd4a-4366-86d7-c66ce155f049
# ╟─dd7fbd56-a447-48b9-88a5-a8a44c3090e0
# ╟─b91a7c33-0b8f-427f-ba13-f07a89538dc8
# ╟─609e6940-5283-4886-a77f-7b9c9725bd93
# ╟─fe1cb0ac-346a-4b4b-a0f5-6dfa26c2f6fd
# ╟─6e353061-20cd-4000-9bae-98eb26bad0ee
# ╟─ba55f4f5-5e62-4815-a977-15056f6ee7b4
# ╟─41f93b9e-d5db-47f6-8566-63daedfa7a19
# ╟─b49bde38-6b45-4dee-81c8-e7af7fbe36ff
# ╟─1419ebac-90a0-423d-a260-dea8e96683b9
# ╟─d70b8368-49e7-4baa-8c71-cb4ca5789937
# ╟─614c11b2-6b9e-44a7-8e35-ef3c9c5f1b05
# ╟─e881b52c-45a1-4f4b-b7e5-3399ae6da3cf
# ╟─69da95fe-49e9-4eb0-b463-80e156194b70
# ╟─cd4e2320-1490-40ed-89ce-d512ddf52b30
# ╟─48583e15-3dd3-4fa1-a6de-32fb2da266f4
# ╟─160ce232-b4d0-4692-add8-92f33014b64a
# ╟─b4eddf2a-98ec-4016-bcf5-3ec7a6a7c492
# ╟─8ed417f8-a03f-417b-b61d-255761bcbbbb
# ╟─bb400743-e850-432a-b792-30c02f9e1e17
# ╟─407dd2fc-8347-4d80-89e3-5dadc452bb9a
# ╟─6ef48dfc-b064-43e6-8ac0-d03c7cd97b6b
# ╟─6192047f-cfa0-4b60-8af6-d718f8db063c
# ╟─56657013-c4be-439d-9319-2ba952f01223
# ╟─18d4c44b-f6eb-4ac3-bba4-fd3a9f37f6c3
# ╟─176a6470-af74-447f-92de-fcf4afc96f88
# ╟─2168514b-2d3c-4102-bad9-c973c383447b
# ╟─089d9285-9a78-4a12-8c4b-072a23a22242
# ╟─d9c2444c-08b3-4ef5-aea5-7606d41bf463
# ╟─0859e9cc-d036-46ff-96c5-30d3d0868b95
# ╟─0ebf7c9a-ed22-4143-a5e6-373d2220364a
# ╟─b85b4c90-ab1d-4a37-8994-c909b8b8f0b7
# ╟─e3cc5ea9-0372-4156-af45-db84785bbc26
# ╟─9be71560-d07a-4785-9b22-637d4ba5cb4e
# ╟─035f572d-0437-429b-8772-f12e79c3ae69
# ╟─72ef448b-67f7-480b-aee7-f6643357dbc7
# ╟─6b606ae3-01e7-40df-810e-dff27cabd6c2
# ╟─9decb77d-7d56-45dd-a226-a0ef1490ee81
# ╟─d11b21b4-c6ec-4a73-a798-37457d7e41eb
# ╟─07baa0c8-f50a-40d4-9544-311371bec1be
# ╟─965a77e4-fec8-40e1-9c0e-9869859c06b1
# ╟─fc46f1cc-29fa-4268-b18d-cbebf61549ff
# ╟─64d2db06-8438-4d2b-8e58-78330545b291
# ╟─d6bae6d3-fb6d-4e1b-8903-a8e5b9b15a5d
# ╟─2ddcae3a-5de2-4fa7-afda-7a24c33c22fb
# ╟─2b1fec6b-1a72-4c98-803b-618b3ce32596
# ╟─609dd371-e997-4c54-8be4-5080563a9152
# ╟─dbbafe5a-33bd-44e5-b781-d46522f18e33
# ╟─b31d2f7a-534c-4f11-96aa-faa974dc40d3
# ╟─b1183215-cdfd-4201-96a5-53c27c2ce251
# ╟─259453b5-857f-4f97-8aba-fd8b7b24a3f1
# ╟─c9f4ed02-c5de-48a4-ab7b-40855a1118f5
# ╟─af9946fa-a49c-40c8-8f3b-b4e9f552727c
# ╟─aa1cf0fa-d17d-42d3-90b6-3ffc8a4c3723
# ╟─4e301fbc-7ca7-4a96-96eb-67c8a050a70d
# ╟─242a4c45-33bd-4898-a00e-83e4bc30ab7a
# ╟─4770f4b9-b4cb-4998-8bc9-e08c51bc28e4
# ╟─f575b7aa-f84f-4c46-9506-f188eb1d504c
# ╟─3aae68af-bf27-451d-9594-0db29fdc9bd4
# ╟─951926bb-1565-4dd1-996e-31ce788300e7
# ╟─461faabc-d3b6-45a5-bb02-f69fde88e667
# ╟─8bb86d0c-7a63-4758-897f-c9a9e547df9b
# ╟─a192d846-a323-4923-bf8e-4b1f15bc836f
# ╟─bae73a44-a2fd-458a-aba8-fcbbf5ab5ce8
# ╟─ff3f6611-c9dd-4579-b2af-b41f1b2f7773
# ╟─2171c34e-23ce-4b12-976a-b18a7f31fae8
# ╟─59815622-63bb-47a2-b0a2-b18c5a2194cc
# ╟─4f5298ff-eba5-4a2f-a10a-7ad1d7db859d
# ╟─929c16fd-ad60-43bc-85eb-534d1f1016eb
# ╟─872bf814-3e32-4ed1-a33e-da9154d5cb2a
# ╟─20855148-f2eb-4903-a74a-2fc266e6c713
# ╟─2c25328a-7124-4141-a165-66453391f352
# ╟─8d53df67-ea04-4205-b61f-3108b107a2c6
# ╟─e155b836-9702-43c6-9028-d729653213c4
# ╟─42de6d7b-e26d-4b87-bfcb-f7a7aa44c8d0
# ╟─f5f6f7f2-0f46-48fe-9c87-7242ec994890
# ╟─41443739-21c4-493f-a824-7bc501c5e6fb
# ╟─5827b975-7b2d-4b03-b4b6-23795e201688
# ╟─3649588b-7fbb-4d99-9771-f566e983ff3c
# ╟─32bf76ce-f531-4767-a7a9-731764bf25ff
# ╟─11f16c8e-e5f5-4b68-a146-0827972512a6
# ╟─bf080d0b-1093-4ce1-b2fa-d1637df0c7f7
# ╟─91ee95b5-6535-4c28-953f-1d3426cb196a
# ╟─2cb743ad-a167-4803-b26d-2087fbf86172
# ╟─f5392e81-4a71-4add-bab9-a391371ffa76
# ╟─c63da16b-b502-4980-92ff-654e433ebbd4
# ╟─685c9694-2d57-455e-9549-1f7e6918bd69
# ╟─aa91b0b4-ffe7-46ca-843c-80a965b19bbc
# ╟─0200c462-5642-42de-b5d6-cfa33cc97081
# ╟─5f3065a2-84ef-4d3a-a4e7-2f0624d72764
# ╟─23129f27-bd9d-4931-8439-a0bc99e1f620
# ╟─f5aab038-813c-496c-963f-3cb657ccf380
# ╟─656ffc3e-6bbc-4595-be8c-8526fff85397
# ╟─54a83ea5-3e2e-4b11-b05d-6268b48eb6ed
# ╟─9329e350-19c9-4c0d-a617-1f196c6805f3
# ╟─fcb82330-4c4c-4924-a773-10a3d03b79a5
# ╟─e0a2d404-d5ca-4656-80b0-29f59007999d
# ╟─e81f838a-b02e-45a1-92a7-35321a878bd7
# ╟─5c66daf1-7d67-4a30-afb9-cf1d5f005e60
# ╟─87e00cb6-b1a6-497b-858b-18f669964370
# ╟─bbb983e6-8459-4318-ac95-c5c112861314
# ╟─50c100e9-6993-40a8-a3a5-827aa990f3cc
# ╟─6c6a3802-c1b1-43d6-8b12-a529790c7ed2
# ╟─d2468fde-7c60-43fd-b87a-7d98b57f2594
# ╟─c1c94275-aad8-4026-8d4f-88b57a1b858f
# ╟─a4b6c78a-5996-41d8-8b93-2a97d13ad771
# ╟─51420ac5-c230-4fc9-8af1-add6f73dee5e
# ╟─0f4290dd-163d-4356-a569-d532a6b38c21
# ╟─5989a383-8a02-4cbb-af9c-fe94bbe7e748
# ╟─d02b9748-963e-460c-805a-a4f388a23c8e
# ╟─59080d5d-3ac6-4d52-8faf-1d3088e0aa67
# ╟─d84ee4c9-78d8-4796-a422-0760a73dfdf2
# ╟─1abb46cb-7323-4146-8c6a-29d7cd4e84af
# ╟─3893fa74-016b-4ddc-b023-4667d8c05c53
# ╟─794fbc5d-33ce-4d25-88e0-b160491ba69a
# ╟─0fa3e5da-bd54-429b-ab18-b22e8934524c
# ╟─b31409fb-960c-4497-ac70-5f631ae94613
# ╟─d8f1d7a4-4546-4db5-934a-21d5959ed171
# ╟─918c1118-9200-4d5a-adbe-6e9db022d0f8
# ╟─bf0f709d-0264-4e48-9e1b-3c5bc8dab12e
# ╟─8b05ead9-d26e-4608-9ec2-89ed4b48546e
# ╟─28f85d3f-b29d-4835-9933-226a9d291c7a
# ╟─2ebf3c76-0d3f-4426-8247-bc9abc50a7ad
# ╟─e3a9d7fc-9c84-4853-9032-64a526186316
# ╟─020d00e4-67b5-4f0f-be84-81aa695f741d
# ╟─2ca9fdc0-cfdc-412b-87a2-4d12db67d65a
# ╟─2fe82dd5-ad59-4ce8-86f3-67f336f9429f
# ╟─9429c73e-87b7-4087-941a-36dc77af576e
# ╟─c570b8be-f280-4521-84c0-36002256a942
# ╟─dea301a0-7b2b-4c91-bb07-6275f822a4e6
# ╟─f6421a9a-dc72-4ff1-9d99-c11ef0963edf
# ╟─4b48abc9-de19-4f46-b8a2-a8a7dc22eff1
# ╟─5198ccf5-60be-4923-b8cd-ffdb12d9cbdb
# ╟─ca463cd7-56e8-4b0d-868c-aaf1f4dbf028
# ╟─ce5ff998-d87d-47d0-bd89-c8dd379281ac
# ╟─7813a8d5-2d3f-4cc2-a68e-52bb5ee8f85b
# ╟─46fc521a-732e-426f-8786-b8d28a273894
# ╟─6e28ed29-b69b-4f22-a5b1-9cacbf998cf0
# ╟─12c5cb8f-4dff-48de-bd73-9c4f6b250819
# ╟─875c70d8-5071-4a66-97b3-47d84c3acc5a
# ╟─d95a4a78-4f59-4e4e-bc41-0ecfd0762429
# ╟─6aeeacb3-5b54-49fb-b429-875f8e29b1c5
# ╟─a93bb539-5ff8-4dd9-a28c-13aaeecac391
# ╟─c0cb4145-6369-4877-8273-2fa45ee42c47
# ╟─afbf6aae-ef77-436f-a8c8-aacb1eee97a1
# ╟─47c1e967-07c5-4505-87c6-6010c48c386b
# ╟─ea18fc6a-9f5b-4cc9-b1a0-a5936b144da9
# ╟─42f75269-87b3-42c8-85c8-0a5c3db72d92
# ╠═ef868b9a-e5c5-47af-9495-d42590668eb8
# ╟─6bf77be2-ae30-4d54-bc6f-86b993a5de64
# ╟─4dd633bf-c30c-4fd5-a635-b02ae6e287d0
# ╟─ba5de3ce-3d12-4a1b-bc07-6d79c7b3b652
# ╟─7813b028-75f0-46cd-bc3b-2853703b9b5b
# ╟─bca965cb-14c2-467d-8207-e64c35cd6aea
# ╟─cb189e4c-f882-43d1-a48e-44f095905861
# ╟─cd1cd5d3-94dd-4aa1-bed6-f35d1ad0d3ab
# ╟─603df4eb-52e6-4cb5-8e27-16ffc0d6e135
# ╟─6c1fef0d-01e9-443e-8a73-f4df5d139664
# ╟─a8760206-ea8b-46ec-ace1-b4fc86dcc7a4
# ╟─6aa50070-3963-4504-a886-893f242973cc
# ╟─83632ad9-2fce-42d5-b046-b0268cb86032
# ╟─afa4fc91-a848-488b-87c5-2ef0b36cc2b5
# ╟─dee63da6-8a0c-4f8b-b8bf-426694d7ae1c
# ╟─c31c63dd-aaf8-4f52-a479-06be9a19b252
# ╟─53fd0814-1c1e-433c-960e-fc7ceab43d34
# ╟─d8f2fe6f-9a63-4aac-9aa5-f0a2e81c76c4
# ╟─0f4df7ca-52c8-4a9f-8ad4-e47141469089
# ╟─e56ee78b-e64d-4587-b49d-4a434c5f32ac
# ╟─0e499521-9990-4ef5-a469-82a2ca7a411b
# ╟─4810516b-c686-425b-b07e-5f08fa906b9b
# ╟─e96cc4dd-b0ee-4201-ab6c-a594e9df6424
# ╟─ab52b812-b2e9-4602-a258-1fa75230396e
# ╟─8c85d9ff-9890-4d7f-bbf9-66487415e856
# ╟─ab0449b6-98c8-4409-9a24-51c4854938c1
# ╟─4710788d-f9c7-4d8a-8eee-366a375525db
# ╟─e38f4cbe-0de2-4876-a441-e66d8710733e
# ╟─e008c4ed-821a-4e38-bee5-5ec40d73c2b8
# ╟─db81e6c1-eb5d-477c-b74e-c7f64e34e8c9
# ╟─4692724a-3a15-4bc5-9bb6-5d0cc8bf9037
# ╟─385580e7-70e8-4961-8f76-b65e0a9d80fb
# ╟─b4ad42b0-9d73-45b3-8c91-390d4da6ad3c
# ╟─bcd553d8-ed5c-46d3-bbe8-a75896c3c9ad
# ╟─58d6547c-507d-434e-a205-9ac68a3495d3
# ╟─b84e49d8-01f6-40d9-a08f-42f92ff0fae9
# ╟─8fcaf7a3-f8dc-45d5-9c35-2fb286318cda
# ╟─317ee97a-199b-4ff6-8526-fa6a7e5731a1
# ╟─428cbc38-7735-4ca7-888b-aa2f2af80e76
# ╟─7d3f466a-ed5b-4d44-8e95-be8f3c8b43a4
# ╟─2a02ba31-4459-4a97-b3cb-92a39105e3f9
# ╟─e5507385-cdc7-4a3e-b918-47538397fc44
# ╟─98209c74-16d6-4af7-826f-d5a32ba1a711
# ╟─986c8fef-302b-417b-b087-1ae9771ae843
# ╟─651a6161-f630-4630-b8aa-342577084212
# ╟─8ae5469d-53ee-4618-b494-1dcba2808086
# ╟─115fbd4a-7c96-43b9-aaaa-03deed5a5429
# ╟─dee11e67-8854-4e5d-a242-84f203ca0570
# ╟─9b8d1f9f-76ea-4e40-b0bc-2f37585438de
# ╟─7e9d3303-4f9e-43c6-9d4d-a2e4e72251c6
# ╟─1216633f-35db-4763-832e-7f74ce1933bd
# ╟─67c3666a-0402-416a-8498-1a2189f7fd49
# ╟─c834987c-e8f2-43e5-9d06-ae119c74ffb9
# ╟─144c994f-0ac3-4b31-88dc-0e7f55fba269
# ╟─fff326ae-3707-4862-9509-f8e97adda2e7
# ╟─3d941738-2678-46d6-99c2-5ddbf7729155
# ╟─575db528-16dc-4928-bf28-f4ec739a00fb
# ╟─47524d32-5763-4d92-88d3-4a30593d04dd
# ╟─1f7cba77-2220-4538-966b-bc3afd673bfe
# ╟─6a0b91da-2a43-4eff-954f-26459f3731c5
# ╟─4c97768a-1d75-402b-b77f-089e26234f18
# ╟─bc283d66-fe72-4d46-84f4-38b88e6a6bf6
# ╟─b4459113-a276-461e-bd05-2457c6922508
# ╟─965c4729-0532-4d3f-a884-d0a29ee04fe5
# ╟─296ff300-eaf6-4f7f-8906-e823cab3c381
# ╟─20514d65-4d3b-4dac-8ab5-0d3126f03b28
# ╟─e9705a2b-e9b3-462b-affb-13b7dad024b5
# ╟─b79a8571-6840-4c6c-a809-f5bade9443e1
# ╟─aa27c1a4-5efa-4275-949c-92505963e1ff
# ╟─bddd0a62-7807-4cb2-9e27-976c4f1c6883
# ╟─930a1192-15f6-47d2-8afe-0190ef3cee9f
# ╟─a5cc6f89-1fd5-4af9-8a18-4732f7b847ef
# ╟─b0ca47e5-9a61-4444-b3e8-13002c5f54e0
# ╟─cc8e4392-d65b-4c9a-9a16-40e4e0ca9cf1
# ╟─eda68dd9-90f8-4e5a-8f27-362dfa53e17c
# ╟─ac8b10fb-fa63-4194-8cd7-99775ceda837
# ╟─799a635c-cee4-4dbd-9ad2-4e1789469312
# ╟─93092b86-273e-4b52-a80f-51c9cb9fa54d
# ╟─fd7b6013-73cb-4778-99a5-b9c9861857ed
# ╟─a9e0a585-8c4a-415a-b9d1-8b8b360805f5
# ╟─c64d7f92-19ee-408f-8d41-8da41ad56cdf
# ╟─4b456b10-51c1-44ea-96b3-e551f116b34a
# ╟─0426433a-c0c1-4e5a-86f8-79b789546381
# ╟─3776555e-1519-4d4d-89c9-bd86d3d2e642
# ╟─2cbbea02-9c33-4b8e-9225-ad3b1663e16d
# ╟─8156ee09-5a5f-493f-bddb-48884ab10487
# ╟─64f923f2-69d0-4d7c-9a96-13a55b0945ff
# ╟─480e5e62-d756-4074-9c70-584503171808
# ╟─7a810d30-33c2-46c6-9de0-52a27b53e3e9
# ╟─97e4a118-6672-40ac-abe5-49d04f0e2451
# ╟─9187264f-25b1-48ff-bc44-e9ec0880e6c3
# ╟─b94e3f0a-e387-42bb-bc98-5b9a8fcf746c
# ╟─f6acacce-203d-420f-aaef-f4e97d03c246
# ╟─43f1ee9e-2b9e-4a32-890d-fba92cf12006
# ╟─979751c7-b66a-46f0-9f3b-5c09fb46bdfa
# ╟─71f0af91-5772-437e-b8a7-429b22ef3b6b
# ╟─a7210602-910a-4090-9373-5718e940d5da
# ╟─0881a661-b6ac-466c-b83e-de8a5b861855
# ╟─187f1f82-a089-40fa-b47e-7a7d8e28e126
# ╟─862ccff9-b4a1-4714-9083-afb263ec46a2
# ╟─3e042e21-7c53-41e1-9a2e-3e5dedc77901
# ╟─0e73c006-c7b1-4384-b433-ed38c747f2f7
# ╟─00ce82b3-1272-495f-ac7e-f5cf1d59a6ee
# ╟─a730ea1f-06d8-4928-a3cb-8672fb88feb2
# ╟─d21a3dc7-0624-4020-89a0-e1b20e14d0dc
# ╟─6ae01fc6-ba58-42d0-846f-1d7005a4ae8a
# ╟─840bc8a8-2ab7-4a34-8c00-f32aeaa1a7ef
# ╟─92e96193-dbaa-4153-8213-453ad6ebd140
# ╟─e01ff30e-0074-41a2-a596-d289d41aa068
# ╟─866088cd-ea01-47d0-8160-e80a9c5ee1da
# ╟─3dca9acc-65e0-4489-80e7-085da74e4a7b
# ╟─dc286af3-cdaa-441d-a222-13a39b337f6a
# ╟─626658da-1ea8-4e73-b54f-5293bb993b43
# ╟─43e83b8a-e6fb-4335-8fb0-e1c3c2f39264
# ╟─df6fb6bb-011c-445c-9d77-ed0598fbbfb4
# ╟─12f9d9d8-e136-4fee-824a-31d62844d8cc
# ╟─80cc545d-c4dd-4b45-94de-d62baadabe52
# ╟─98bd6863-f331-4aa2-a347-6d9052aeacf4
# ╟─12e90822-fd27-412b-978a-884895744c3d
# ╟─ee14c5f5-b56f-4db9-ae93-62797aa4603e
# ╟─5be00a1b-a1bc-4e6a-89e3-454347f42756
# ╟─7a58ff8a-64ce-403b-bf66-a1743d629efb
# ╟─11f12051-631f-41fe-b2ae-a5ad292f200f
# ╟─8fa29ff4-dec9-4a30-b2eb-0f59ec7580f1
# ╟─1c62630c-bfb5-4231-9ca1-396853a67b12
# ╟─c62361e1-8a0a-441c-9698-93083fad165b
# ╟─2a72c64a-d79b-470f-bd6e-b57c3c22f1fe
# ╟─b1c7bb7f-a263-4575-ad0a-d99e62952c41
# ╟─c8670e69-760c-46ed-8eb6-4c1f5c95f515
# ╟─48b0b7ca-645f-42b2-ad92-ebc4dbcc8547
# ╟─233b0379-485b-4f30-a306-38e507c21962
# ╟─02a4692e-6ad1-4ceb-80bb-1300cae4dc3a
# ╟─52731845-385c-4ce7-b080-8092dd29bb98
# ╟─0c50d692-87b3-417c-8011-31bfcd787fbc
# ╟─ecfdf429-b8c2-4f83-8d82-fb661b11d058
# ╟─68051c7b-0248-401d-82a1-8ab9d4a5ccbd
# ╟─1f4e7172-5108-42ef-8c3b-08baac1946bf
# ╟─cc23045b-394e-4c74-b840-d9db52b879bc
# ╟─a126774b-cd0a-4fa7-8828-2eab53346d76
# ╟─3cad956a-e134-4070-a1ac-20b205f5d558
# ╟─6365c361-8a9e-4ef7-8c8b-876fb3b49008
# ╟─a797d907-ca04-419e-a8d0-8f74927e4297
# ╟─f34958e5-4ba3-4530-97b6-6d3987b95f18
# ╟─df324a0f-694b-4a4d-a2a1-b53397b8301b
# ╟─d1c49ef0-1cf2-4834-8b82-d9b3eaa9e970
# ╟─a9f7fb81-5250-4b69-ba78-a5ba627e19ab
# ╟─9cefbb4e-28e0-456a-8626-217d530663af
# ╟─7346500e-8ee7-4cb5-bdda-e99f2f5bfe26
# ╟─f8c76558-2cc2-4571-856b-54ef5d318bbe
# ╟─e64cda1d-d6c5-4e7a-ad2d-f90a741fa108
# ╟─899c9f77-6c49-432f-b12b-6911ac9bcf3f
# ╟─ecd44d7a-f5eb-4ebd-a2ed-3721288beb3d
# ╟─107f71b2-a375-4628-9e21-f36fe53c5314
# ╟─e2cd252f-338e-48cf-8998-c48fa6841c64
# ╟─4d7a53ca-a7c4-4971-b2d2-c7d3aa918919
# ╟─b1ca1568-1181-4bfd-b132-507e3f11b55a
# ╟─98f9e5d4-536f-47ba-b66d-eee3a7e45a26
# ╟─0596357c-2e82-4768-aae3-48fb5ca3a279
# ╟─0ae0b117-4ad8-46fd-8b07-99a3ddfbe47d
# ╟─a7d0fdc8-fff5-4737-aafd-cc81d4b31473
# ╟─8756f750-188f-4b91-9e27-0e59d65e03bc
# ╟─5064b566-ecee-46c0-b4ba-2e34b13a96b9
# ╟─c9d3c395-9ad5-48c0-93af-f59ac90cf302
# ╟─4f9f3b7d-e8d0-4ab9-8213-1999b4ccc03b
# ╟─232e89ad-e481-40e4-96d2-a004f6d9d64f
# ╟─db293503-f9ab-41f0-91ea-39e3c209642b
# ╟─039fa96f-41a3-4651-ba3b-b532dc01d885
# ╟─ef350652-3180-45ad-a1db-b4833cf7694b
# ╟─5488c60f-ee91-455a-8426-276d2d8628c3
# ╟─a6ce9556-974a-4dd4-ba67-edd451cbdab4
# ╟─ad0fb083-460b-4a66-af22-928405d625c3
# ╟─57237519-ebd3-489c-a770-491a4e63dc9c
# ╟─7cf5143e-41d6-481c-8ae5-ca3c6c6a58ee
# ╟─b7fa334a-a763-4d81-8425-ac94731e20ae
# ╟─f939634f-bca3-4f4c-8d51-40c88282a9c2
# ╟─8168b488-16b8-4113-be6f-d17383b6b020
# ╟─afcdb8a5-5aeb-4a8d-89d1-97f463409c0f
# ╟─17acab28-d952-4d9c-9952-edc03e917b7b
# ╟─33470d25-31dd-4f72-a393-fcc1fa48d185
# ╟─b268f146-797d-4ffc-af6e-8856aa18f43f
# ╟─5337dbf7-d894-45b9-ac74-8a3e147544e8
# ╟─891ad36a-c228-4dba-9d3e-e58bfab6720a
# ╟─ddc4bfc6-f6cf-4fb7-b1a1-8f748850f050
# ╟─46919066-a1cb-4e0e-b8f0-611e160cdc0e
# ╟─00d264df-89be-45ff-83dc-484090871cef
# ╟─53dfe27d-c540-43b4-ae85-db51d9fbcb82
# ╟─471303af-e55a-4416-b4ab-b68d37515eda
# ╟─edec6623-7bde-40e8-b279-b0c45313e9cb
# ╟─5ab79d7e-afe9-4d31-ab78-e6ea7e8c9f07
# ╟─a516f178-60d4-44f2-97a5-97e441f95c68
# ╟─b18f044c-2e1b-4297-987b-46f6bf6764d0
# ╟─962d83f3-6a0b-4e5a-84dd-6181c455d795
# ╟─ba4da7c8-00bb-4584-9887-32e1322a3789
# ╟─67378081-7005-4b0c-9b22-327f2d7fe42f
# ╟─97f1347c-898b-4524-8d0f-970ae602a896
# ╟─8ac061f1-85f4-4079-89ba-96ac46bb5ded
# ╟─2f06c009-bdb6-4c06-ad41-14efa127b681
# ╟─39ac7277-6efc-483f-98fb-c26ca78db546
# ╟─a0734dfc-c083-4456-80b3-ae732c10648b
# ╟─4731b14f-ac57-4f73-a258-4ce525cb3ca7
# ╟─be9c0893-a37c-43db-8dfb-b6388e68c108
# ╟─e6c1f1e2-ed75-4f03-b4ef-4d826777d19e
# ╟─b0bf2fe3-020d-4c08-8660-e696efdf867c
# ╟─9dd97fce-ba94-4f16-938e-db94853a49dd
# ╟─d1819b4b-648f-48c4-a700-97fb8bfd995f
# ╟─c51b20ea-4165-49c9-aff9-fe48b4e01b5a
# ╟─0790b2b0-5089-4de2-bab5-b4db3ec24111
# ╟─574aec30-a3e6-4126-ae02-07ecc4791b5d
# ╟─8f4f0416-8bb3-4fa6-a3cc-f88ef2d1655d
# ╟─7314ec0f-31b5-4b70-bf5e-ac1ca1abb9e4
# ╟─bc7cbebb-6698-46ad-8765-d99aadd68bdc
# ╟─cc96ca34-2ae7-4d96-a791-b8a44f384d9b
# ╟─f3172887-d58c-4e50-b55c-164434bc0156
# ╟─91184b08-2897-4741-b5d6-e6175fcc5a43
# ╟─18112d1e-d0ae-4c53-bd0d-39ccf9131041
# ╟─7af2f58e-1bcb-4aae-813e-f45e458661db
# ╟─a26439a7-07db-47a4-af01-a0bc36808294
# ╟─0b280f99-fc2a-409f-be20-1d7a173f7bba
# ╟─eadf2823-fd2e-42e2-8cf9-20aff12b75ec
# ╟─51cb4c97-a24e-4ecd-8b1b-e5bf04c72e7f
# ╟─962180ba-42ae-4802-8568-af3d7fb82b8d
# ╟─74656e49-87e8-42aa-9959-aa4b92aea3ea
# ╟─35b7b249-94af-45de-9c8d-3700502b9308
# ╟─3ec04b5a-80a2-4e5a-baf5-f3f3018517b7
# ╟─b29c36e7-2914-4e58-b076-c8cf5040a43a
# ╟─0c78b962-d95e-4121-a818-17e2a596f02c
# ╟─7b51ef6a-56e7-4de3-b486-9a218a8decb1
# ╟─84e22ce4-10f4-41aa-981c-299da19e6080
# ╟─7e18f490-97cd-423c-adac-50a811e54a8c
# ╟─ba074713-0e56-45f8-b029-2791f8e5ed8e
# ╟─16e1cb97-d724-490f-8a5b-f1b4a251c5e8
# ╟─90c5df5f-1a38-404f-9cda-820969d15a65
# ╟─b2000719-dfbc-4608-8c3a-5f48360f7b28
# ╟─d0f07594-0240-4254-baab-f0245f425916
# ╟─7be8444b-c30d-4060-b3f0-032469dd7d98
# ╟─2f3fdc4e-b012-4ef3-b428-c1fe53e7f778
# ╟─5d9e3330-a419-4cd7-850a-44c7fdd03223
# ╟─427b6e28-5812-4c41-98f4-669ebd113797
# ╠═f51bfc4e-24dc-4030-9523-21723fad2ed9
# ╟─03d5c242-58f1-474e-a818-9ad013d42df1
# ╟─153c1095-a5a5-4379-a94a-8ee677a63a6a
# ╠═dfaa7c44-5676-4327-81ba-9cd04b2c2b2f
# ╟─9b020e9e-7344-44de-8f15-8e9baebe8aa1
# ╟─9c8ba689-e3d5-4d9a-a3ff-de2aaa68ae40
# ╟─1d9fb4aa-1969-4bed-9565-7a8549c8336b
# ╟─f55e3719-3ef7-45d2-a2a5-58ee65380dd9
# ╟─4f11e61a-675c-4c73-925b-8b71730e71b7
# ╟─a8196bcc-fc44-4209-aefd-0c91d145be17
# ╟─6c616147-4482-4e07-a7b2-d07febbac2cc
# ╟─25acc017-0c61-4e31-8e9b-b6591effb937
# ╟─7eac0960-b3c9-4742-ab5f-acd08305f0f7
# ╟─3f7b1398-b84a-4561-8010-031d0c292a2e
# ╟─245ff794-d1fa-4166-a687-ad5f6a4345cd
# ╟─7ffd79df-aa24-4fb3-a300-27a08b85fbc3
# ╟─39fbf8aa-45b7-4e49-bfbb-04e67a586ef8
# ╟─83c4bd88-8d87-4d88-8738-b100d519dcb0
# ╟─b6e484ff-0055-404d-b5b3-951099edc9d7
# ╟─00a995da-4ea9-488e-92f7-bf4d15b41189
# ╟─fd64471e-28e4-413c-ab21-4bd9ed225e93
# ╟─2a2b282d-4d13-41e1-9651-93d2a9599228
# ╟─d79af8f6-926e-4ba5-b457-c71c07710223
# ╟─6674f5c8-3185-4f20-a882-c0855a2565ab
# ╟─89d94abb-fd83-45ba-b4a7-09b290b0130d
# ╟─30c03bb7-f20a-441d-a21e-10562767f577
# ╟─25207360-eaf6-47e9-9b5f-1e9d29ac120a
# ╟─7e047927-140e-4dce-8ee9-14b5e339b5d6
# ╟─187eaad5-d931-439c-9ced-6f29adab3303
# ╟─b8d437d2-1d7c-4e74-b226-628f9c1c97ee
# ╟─694d1e90-cdb6-4b9d-94d8-a01650933d22
# ╟─e3bb2963-743b-4885-af6a-030b1304306f
# ╟─8f140a1d-e1f3-44b4-ad7f-1417d2d287e8
# ╟─1df83a54-d041-423f-9f95-3374bf67da95
# ╟─1c4de6e9-a0c8-46cc-8a70-eff2448392ba
# ╟─93a79363-e32a-4376-819a-985436fa1b0f
# ╟─e339d855-9bcc-4d36-a80b-7744f6b0f569
# ╟─e6c4b87a-200b-46d8-9ee7-bd8527147910
# ╟─3d950a1e-1da4-476a-8b3e-7c2a6e34ffe6
# ╟─4420141f-72d6-46d3-afdf-64bc867f7299
# ╟─1a4b1618-c463-49d5-81a1-b63824452804
# ╟─6a9b437f-16b0-4938-92ee-d45ba7756075
# ╟─12b3d774-7a11-4ab6-b38e-c321dfc4e07c
# ╟─764c9301-d438-462e-959f-37ceeeab60b0
# ╟─e62fca9c-268c-4f41-8788-2090b2737dfa
# ╟─9fbad1b0-c77c-4240-9443-a813ea4c739b
# ╟─ed098f71-aa5c-469b-b437-967113f9bbae
# ╟─edbdf0fd-b2a1-46c4-b637-68b4d5735575
# ╟─3cc58568-2cb8-4ebc-8832-aaa6f8dcdfb2
# ╟─5033c232-0595-4737-863e-0ed578071dac
# ╟─a0447e66-ba2a-4110-98ab-35bf28ebdd63
# ╟─ffd2c599-ca36-4b75-9cfa-d28d5fd0a5f9
# ╟─d7759b12-9dc3-4fd5-b08f-b99c7bd434e7
# ╟─3df4a62f-2521-408b-9f87-429ac47ac7c8
# ╟─0fc4efb2-1b55-42e6-bb85-477deac83d45
# ╟─6f9f1588-1e2d-4e51-bed5-fa29036b042a
# ╟─02de3efb-4c01-4ceb-a40c-6248c30ca732
# ╟─c2e7e4bb-f187-4d50-b365-fdd38a553dfe
# ╟─0a481de4-9b04-4217-9c82-d103ea67f427
# ╟─9f731224-4247-42e2-88ea-3ebffd4559d2
# ╟─56f006bd-861c-4c88-aea7-2b52f21d4cc8
# ╟─6ca7209f-d61e-47ae-8857-3f4258df336b
# ╟─12179ea8-f1be-4e91-8b4a-a1ef8b3a68f4
# ╟─543a1441-c941-42ee-8874-eb0d3caf5aa8
# ╟─6bc8950e-3a1d-4729-b790-e39f5412f13c
# ╟─b49bba8c-da43-465f-86ed-977e2463b91f
# ╟─f41accd9-af8f-45a8-b4e4-44334ab57616
# ╟─31444da3-a091-4d3a-8f25-b612de0ac19b
# ╟─ad79b200-ad92-4f35-b2ed-ce2ae5dcd857
# ╟─811bd9b8-7574-4fdb-90f3-866e8c4a7d63
# ╟─760ef092-8bc7-413f-9bde-592637b9eb5b
# ╟─817072cd-ac1e-48e7-ad8a-cadb3f1f8782
# ╟─6571a3a6-0a14-4cc9-89e1-fc1d503cb8df
# ╟─c552a6d7-74d7-45dc-b403-f28bff97aaab
# ╟─7e6b20ca-3064-4b41-8e31-8144a714d7e5
# ╟─1c9942f0-d657-4f2f-97d5-f3332b77309f
# ╟─e67374d8-23b7-4be6-8098-c8db305b5a7e
# ╟─08f586ca-c2f3-4f5d-a22f-4ae733f3a95d
# ╟─89f78f44-70f9-4988-a3f6-58fc355cda17
# ╟─ee8b1439-626c-43e3-9eaf-c5ace926794d
# ╟─2b56a15a-03c5-465c-af42-3e71b3a99a51
# ╟─35f5f96b-bd5f-4444-b428-84e96ea56fc0
# ╟─d572fe72-1700-45a8-a054-36304f9f5c2b
# ╟─37144bcf-d4ad-4488-b95d-3de6b5265a02
# ╟─a9154e81-3fef-4ca8-b63e-8d1187d9a255
# ╟─235f630f-3364-45c2-8132-5b6ba30c800d
# ╟─6ffcee72-e684-4265-aeb9-26e48c589dbc
# ╟─7138096a-1ea0-404a-a0af-f63be9dd50fe
# ╟─862711d2-4ae2-4f58-8800-241934c84efc
# ╟─0b447d6a-f10a-429f-ad4c-e0d4c6131f83
# ╟─4d188df7-521e-4f29-85d1-4c4ceb7425a1
# ╟─c3603b58-c4b6-4242-b2a7-c40d18b83d9e
# ╟─44f76e70-c260-4726-b718-e97cf0751134
# ╟─bd11ef76-0128-4bff-b8ed-1c081362052b
# ╟─68a6b311-d335-48a8-b6cb-baa35a37b964
# ╟─1f22691f-4540-488d-b495-b8e1e9f35c15
# ╟─845d09de-b4d2-49f8-85ef-c9881b073ea6
# ╟─fc14945d-7896-4d6a-9a83-b8034914dee7
# ╟─6739b85a-af92-4332-a328-c7c13b36b6b0
# ╟─635e3ba5-4bf6-4503-9c3a-3092d24b7142
# ╟─14d93155-fb66-49ce-b2b4-44a1ada27182
# ╟─3e3b3673-ae70-42a1-a169-5a05a342a357
# ╟─5eef2b95-90e8-4825-889e-187d79e87841
# ╟─f461b0e8-d6a1-4566-b581-71509e387d7b
# ╟─a11cef97-788f-4c8d-a122-154089b6192b
# ╟─72c160ca-ada7-4d7e-8090-3dad88461f24
# ╟─a0b33768-8e62-410e-a38f-07ae440c7f99
# ╟─5d0beb88-ba76-415f-a6b5-4c40c6189207
# ╟─0ae8a531-7e4d-43fc-92e8-abd85d1881f8
# ╟─10c29c69-03c2-4b57-8092-3a5186b78f34
# ╟─b75f682d-beb2-4dcd-ac2f-ff456dea4335
# ╟─37fa0170-f397-4736-890b-a861cc70f1d4
# ╟─e0a53309-023c-4e13-a117-b38702e7f059
# ╟─406a9c4d-42b4-44aa-aea8-a2033ecd1c98
# ╟─e9124f68-51cf-4f78-af87-81f46e800647
# ╟─2195b859-cce2-44f0-987f-cc9191336a84
# ╟─ed69c39d-bf74-4480-9cbe-1b10348d9a2f
# ╟─1c00618c-53c9-4de7-ae54-e5cab1467a24
# ╟─9109f4f3-dea1-4d18-8953-b527a5974757
# ╟─ae321ad5-3770-432e-99bb-6d723587139d
# ╟─024cc93c-bd93-4a9a-9aa8-f02e6654e74f
# ╟─99f73253-290a-43c1-8fef-4c22ae42b695
# ╟─1fcf84ed-abd6-4f13-ac5c-41ddc060611d
# ╟─ce51da33-c127-48bf-8082-c75d5e20aaf5
# ╟─23e32823-f493-48c5-90ef-f5bdbeec6c04
# ╟─da4a68c9-716a-4e98-85d0-9bac9fa68a4b
# ╟─b7ca48ec-eff1-43d5-adb6-ad3ce455f6b7
# ╟─c917c81e-747b-43e9-995a-43a9e35dc9a7
# ╟─c0afd5fa-ceb1-41ec-b0d3-32e2592b29f6
# ╟─904422dd-cb19-4e69-aab1-42993af08693
# ╟─119a55e8-93cc-4314-b827-fdd00ab71cbd
# ╟─e2dad27f-dfb0-407b-9280-4146d7b65774
# ╟─826c73f1-8ce8-4d79-8a74-8914b81fd7fa
# ╟─4cd9ebd7-99ff-4542-9ccd-a700b120ee46
# ╟─8278aacc-41e3-422e-bdcd-5f5ea7e8354b
# ╟─7ef28ed1-0026-495f-8aa6-12a6f1ac7aa9
# ╟─4526f19d-fe02-4855-9794-db7787f957da
# ╟─89f32f0b-8d19-4c35-b365-253431a4dce8
# ╟─1bb95cf5-864f-49a5-a58b-f51a9a8d7dca
# ╟─2aa7b7a7-4fab-43d7-bc3c-c183d00568db
# ╟─b40c7f0b-4597-458a-868d-1e8e859b929d
# ╟─8c08e1e1-0f97-47d1-8c76-69709d3aa95d
# ╟─15e54924-4e5a-41b7-9fee-28a159671d26
# ╟─c7a0632c-0b3e-41a8-b4ec-1800ff18bc49
# ╟─0d80f018-c78c-4d13-8131-5efc3fb5f3b5
# ╟─12504545-7574-449a-89c6-838d1ce55b8f
# ╟─0513f330-5fbd-4b79-bc97-d9dbc8550816
# ╟─12afb21f-7733-4fcf-93cf-a3d2073b4357
# ╟─9299cf57-28f9-4ec5-9dc4-0082e9d0508c
# ╟─ecdc07ca-3d21-40bd-9b63-4432febf3fee
# ╟─da3839f0-d149-4bba-ae79-8db979bad453
# ╟─94e17553-e4ed-4e5c-a598-0ec235cca96d
# ╟─298e7538-daf5-400f-b5ca-a456e922deb1
# ╟─866c521d-31db-496e-ae9e-3666439c8fc8
# ╟─130f16f5-ab11-4c53-a9ec-68b9d8f75224
# ╟─676a8ab2-5fba-4688-a0e8-1130f8678f06
# ╟─bfe2d57d-daa5-42dc-86cb-3df09a144dee
# ╟─737a13ce-0e37-42cb-a8a9-202865d46d08
# ╟─d45b4721-5836-4a7e-b448-fe1d8fc0f961
# ╟─a7732a20-5c57-431b-a5c8-9bb1442ced09
# ╟─54696221-d753-4453-bda9-c9adcd3dcfdc
# ╟─725e28cb-3d28-4781-b538-6018b25530df
# ╟─0bbb473b-4e22-4bbc-b7e9-5146171df9af
# ╟─6d258980-7204-439e-a036-575066da4e5d
# ╟─d30709a5-3fc6-493a-995b-5ff38384963d
# ╟─2cd48325-4fe4-4ccf-a4be-eaf35a33a551
# ╟─9fd208db-1928-482e-b124-aa6d2db27257
# ╟─50248d12-21da-45de-8673-34a47a0fe003
# ╟─e4df9ac1-97fc-45de-a9e5-deb190372806
# ╟─911a622c-a14e-47b8-99e6-5ca21f90f54f
# ╟─8dc3535f-dcbd-4817-aa2d-d6a851f06851
# ╟─c116e645-482f-4a14-80b7-0cf00744510e
# ╟─2dc7cad1-1589-47d4-b9de-da972f934dbb
# ╟─9e225f7b-d833-49da-8ae3-5a0299fea126
# ╟─2e4d3f47-81e4-42bb-a5a1-fc29e36c9e05
# ╟─ef836347-5dce-4138-970d-1a9430868b0f
# ╟─c4af92fb-e9be-433d-8372-2a461a6d4ff1
# ╟─b972d605-36d8-44e6-b066-569647f83337
# ╟─3108657a-92c4-45b2-b0a8-3564483d41c9
# ╟─f004934a-fb6e-42dc-98b5-daa93e747e59
# ╟─a61fd9bf-c8d2-4650-be91-5da2ef6bb721
# ╟─e62a7294-458f-48a1-9109-84a8685a7f4a
# ╟─3bf34436-b6df-4bc4-abe4-416c8b9f3adb
# ╟─c1b1b2f4-1f4d-4a80-8720-f31412a53813
# ╟─6fde46e6-881d-4e09-be60-e8e799ab553b
# ╟─6707ccc5-6a06-4f16-ba9e-ebf240d75926
# ╟─4e7ca352-eb72-4eac-934b-9d5add2032c9
# ╟─6cdd4c66-61dd-4a62-b9bf-1d03951254c0
# ╟─626679bd-78fb-428d-8adb-42328da7c382
# ╟─091301ee-4e37-413f-ba62-89dfd198d808
# ╟─b3640804-d086-46b1-9b20-17584567501a
# ╟─a83c3670-95cb-41bc-b9cc-87a539b12c5f
# ╟─89b4342b-6f48-4650-9480-bd23461dc76c
# ╟─aaee2a54-bcbf-4c93-90d1-ffb240eb40d4
# ╟─ca8bf083-b78e-4c98-8ded-cd800bebbe3e
# ╟─87b6a86e-ef1b-4170-b327-765d8fa55083
# ╟─34ccf23a-5e67-4933-9994-6137cfaf20df
# ╟─b08d5dee-84ee-4240-93dc-56d25daaeb6a
# ╟─48787394-9010-4f03-8a49-1bd0ff2ca68b
# ╟─867a5210-0777-4c62-a52a-8d66abf5bf69
# ╟─2db1e66c-75bc-4dd9-b6d2-926fa20fdacf
# ╟─9124bffb-58f2-4c92-b96b-e82a99ddfc7e
# ╟─89646197-5a1e-4fdd-bd84-c839df8c97b5
# ╟─c691af71-44ca-4a4b-abcf-522a14054709
# ╟─b585a567-22f5-4438-b016-3481a8787a38
# ╟─15e1ce20-d3bb-4ec4-9bca-cd084e4f84e4
# ╟─9e741db2-9802-46f1-a10f-4fdb555d300b
# ╟─64b433ce-051e-4e28-b14e-576a8a1b331b
# ╟─316993be-b3c0-40bf-88ed-66777d9a5b30
# ╟─7643048b-add1-4ac0-9ddb-23adb3ac4542
# ╟─3d3ae07d-9b2a-4831-b22a-727f8e436e75
# ╟─b1b1a243-91eb-4d7b-bfc2-203b967c5ca1
# ╟─8defdfb1-0db8-4c69-ab49-a8928d18d854
# ╟─953c04f0-3d80-4d6d-b8f1-ac9d5c0f6dad
# ╟─a44f0ddb-9fa6-4771-b3d6-0542506e16ff
# ╟─fa8e217d-8552-4671-9ee0-92455b114ee4
# ╟─8cd4ffaa-25ec-4b62-bba9-be421e56b05b
# ╟─8f4a7a58-dd06-42d5-bebd-17dfff186513
# ╟─98da17be-fcf7-498d-9b85-a076d998fae3
# ╟─bf57ed37-6832-4eb2-a6bb-2b66288492c7
# ╟─c2a4c4d9-4d76-48ae-b59c-ed94902a9b62
# ╟─5a6bf12d-cdd5-4e66-82e3-4a8d9aaa1726
# ╟─bf6e3506-ca86-4d8c-8f30-c66f72dade55
# ╟─619d71e8-85ae-4a07-8475-d609c8657292
# ╟─a23fe5f9-2a66-4d84-95b1-a537c40e592d
# ╟─9287b3c9-a81f-4797-873b-fca0aea6e2f1
# ╟─d2988200-34f4-4301-88e0-c945d9c35778
# ╟─22fe58cf-24cb-4a9f-9f7e-8232b08499c4
# ╟─86a860f7-0b06-4216-985e-825518e96b51
# ╟─eef1de1c-f0b5-44c8-abcd-f6bffd6a2193
# ╟─cd010bcf-72ce-4ada-bc4d-8fe7b594b377
# ╟─1f90deb6-17a8-4aa7-b25e-bc5dc126c13d
# ╟─1203ac77-0b13-4919-af2f-2597c0fcd13e
# ╟─ce06cac2-054f-4345-819b-249e03338fb6
# ╟─641012c8-6d68-408f-b10f-84dc5f32782d
# ╟─0c03ec0d-a05c-4c29-a253-0d7ce2d75f0a
# ╟─43455e7b-607d-4bba-b62f-ae14c2ed70f8
# ╟─271ce7f9-0dfb-4609-8abe-6c7d37fc3867
# ╟─570585e8-d80f-4645-a8c8-6e80baaea2ce
# ╟─188038f3-0acf-4b46-8765-015c5ec847a4
# ╟─aca25cab-d7e7-4f08-b5f7-86cc01113f91
# ╟─24e25bcb-c7ce-40c2-b931-1e29817b30ea
# ╟─cf25644b-74e3-4510-81b1-36776f69f2a8
# ╟─5187cd18-fa3a-4911-91e2-e7e38761f3dd
# ╟─2594edac-f75f-4a80-aab3-d020ff6eba4b
# ╟─966bbe33-1982-42ff-8841-3dd35900efd2
# ╟─eb2ef545-a694-4964-a673-ca0612b82438
# ╟─84045a7c-6e97-4bad-867e-bb3a22ebec0e
# ╟─95562c13-aa2b-424b-907a-d158b9787a1e
# ╟─272fb642-3b34-457d-b6c4-1e9c0a7735ca
# ╟─5ce027b1-9a42-499f-b762-4d25baf1cbfe
# ╟─5f912c0b-0c9b-4300-90cb-50a46d965b35
# ╟─8d36e42f-9e6b-4c4f-99e5-b08eb20cd01f
# ╟─ea1070bb-3c0f-43e2-9a34-54ad000e00f2
# ╟─2512548f-8f84-42c8-bb8a-1c2b93c80108
# ╟─029d6d56-860f-42c6-953b-9d65f82586c6
# ╟─2420ca5e-7658-4b32-9e46-03c0ab85d2ef
# ╟─8b982174-66a7-4f9e-81ff-bbc21bee6d04
# ╟─43e5f780-1eb5-4645-ba8f-33cbced20c5a
# ╟─799d96c2-1ab0-488c-ac6d-caa45ece806a
# ╟─4779ee5e-ebed-467f-8724-ffb148161a5f
# ╟─bd7b6929-48ca-4119-adca-976e8f5a3cf6
# ╟─4a3b7003-d290-4b9a-a85e-b9d1b07c9e80
# ╟─f6b52ffd-269f-4135-a5f3-82df747fbd91
# ╟─fa7e7564-0447-472d-a0bd-275731992fb2
# ╟─f85af1c7-5b19-42b4-8342-c074b90ef6c9
# ╟─e6347331-5c3e-4cdb-bbda-6eac91b61343
# ╟─f00a8609-0569-4f9f-861c-5553b40966d0
# ╟─e6ce7a39-eff2-4e1f-8166-115dee02962c
# ╟─5efa6c70-c17d-4f65-9c70-fd80409b4d1e
# ╟─657d07c3-7194-492c-872f-691e197afdea
# ╟─8e2ec07b-7023-41d1-b5db-37ce8a939e1f
# ╟─6713e9b4-52a4-49e9-96fe-ca03e6397d94
# ╟─05e7cb15-101a-43fb-9b86-734db7446b6c
# ╟─ab92b4f0-b969-4045-8ebe-f56bfdb2cad5
# ╟─9f7b4e6f-34e6-40e8-9152-25284d728d53
# ╟─4c57e400-eab1-4290-9a6e-0fef12980299
# ╟─a9467c2a-8b9d-4fbd-a20f-06525e5fa06c
# ╟─c146e9b6-be90-4ee3-a022-0b7fa511dc81
# ╟─a6d72ae2-1781-43ac-8de3-3ab28dd69c12
# ╟─8bdb73ef-c86b-4f33-8055-180681a0b178
# ╟─5f199652-cb3d-44a5-a726-9ae0628da4d2
# ╟─58b4a7b9-a359-4343-83f0-43c2db158bfb
# ╟─361cb208-765b-4f7e-a233-1836a950e911
# ╟─5b7e9525-18fc-4630-a48a-d9ca7364ea8f
# ╟─8a47ae00-2226-4e3b-bf18-bfe56d11254b
# ╟─5f43acc4-06f0-40ca-8e77-80eea5cfc67a
# ╟─cd27aba7-8207-4dd1-aad5-09839abf1fa3
# ╟─7cbd9c20-4f70-4294-b47d-ceee9e39483d
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
