### A Pluto.jl notebook ###
# v0.19.12

using Markdown
using InteractiveUtils

# ╔═╡ 24fec450-459d-11ed-1f65-5199b11ca026
md"""
# Quiz 2
"""

# ╔═╡ d42ade60-a9cf-4f22-963a-1a6c93ecf870
md"""
### Problem 1
90% of ML students pass the final and 45% pass both the final and the midterm.


_**Solution.**_
``P(A) = 90\%``, ``P(A ∪ B) = 45\%``.
Find ``P(B ∣ A) = \frac{P(A ∪ B)}{P(A)}``.
"""

# ╔═╡ e4967d94-699b-4fd5-9d63-c73331a274da
md"""
### Problem 2
"""

# ╔═╡ 521d739f-b037-4ae7-b350-d23a0cb14ca8
md"""
### Problem 3
Given the following training set ``f(X_1, X_2) → Y``, the predicted label of the new example ("long", "white") by Naive Bayesian Classifier should be…

Hint:
P(dog ∣ (long, brown)) ≈ P(dog) * P(long ∣ dog) * P(brown ∣ dog).
P(not\_dog ∣ (long, brown)) ≈ P(not\_dog) * P(long ∣ not\_dog) * P(brown ∣ not\_dog).
"""

# ╔═╡ ea394a41-50a4-46bc-9eb0-676c0ab51ba1
md"""
### Problem 4
Logistic Regression is for

A: regression problem

B: clustering problem

C: classification problem

D: data preprocessing problem

_**Solution.**_ **(C)**
"""

# ╔═╡ 7da2e3d0-198a-4f30-a2af-2e0c06d855f5
md"""
### Problem 5

Which of the following equation is for Logistic Regression

A: ``σ(𝐰^T 𝐱 + b)``

B: ``𝐰 + 𝐱``

C: ``𝐰^T 𝐱 + b``

D: ``\frac{wx + b}{\sum wx + b}``

_**Solution.**_ **(A)**
"""

# ╔═╡ 1bb0715e-d0f0-4e95-9ccb-ae5e4f8c064d
md"""
### Problem 6

In multi-class Logistic Regression,

A: Sigmoid function

B: ``\frac{\text{score}}{\text{max of score}}``

C: Softmax

D: ``\frac{\text{score}}{\sum \text{score}}``

_**Solution.**_ **(C)**
"""

# ╔═╡ abf1d942-99d5-47ec-9c99-dc1699ee4714
md"""
### Problem 7

A hyperplane ``f(x)`` that can perfectly separate binary datapoints ``\{(x_i,y_i)\}`` means

A: ``f(x_i) = 0``

B: ``f(x_i) < 0``

C: ``f(x_i) > 0``

D: ``y_i ⋅ f(x_i) > 0``

_**Solution.**_ **(D)**
"""

# ╔═╡ 1e3b664a-24bc-44ce-a2b4-49bcc8b2e790
md"""
### Problem 8

Which of the following statement is correct for Support Vector Machine (SVM)

A: SVM is Logistic Regression plus Sigmoid function

B: SVM is only for regression problem

C: The hyperplane in SVM is only determined by support vectors

D: The hyperplane in SVM is determined by all data points

_**Solution.**_ **(C)**
"""

# ╔═╡ 41baecfb-8bb8-4988-9819-a6c7217b5e92
md"""
### Problem 9

Which of the following statement is **incorrect** for Kernel:

A: Kernel is equivalent with feature transformation plus inner product

B: Kernel helps SVM handle non-linearly separable datapoints

C: Using Kernel means we do not need to project lower-dimensional feature vectors into higher-dimension feature vectors explicitly

D: Kernel is used in Logistic Regression

_**Solution.**_ **(C)**
"""

# ╔═╡ a5d8076d-b214-49ab-bd4d-1b65b78fbd00
md"""
### Problem 10

Which of the following equation is for inner product:

A: ``⟨𝐱, 𝐲⟩ = \sum_i 𝐱_i ⋅ 𝐲_i``

B: ``⟨𝐱, 𝐲⟩ = \sum_i 𝐱_i + 𝐲_i``

C: ``⟨𝐱, 𝐲⟩ = \sum_i 𝐱_i ⋅ 𝐲_i``

D: ``⟨𝐱, 𝐲⟩ = 𝐱 + 𝐲``

_**Solution.**_ **(A)**
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.0"
manifest_format = "2.0"
project_hash = "da39a3ee5e6b4b0d3255bfef95601890afd80709"

[deps]
"""

# ╔═╡ Cell order:
# ╟─24fec450-459d-11ed-1f65-5199b11ca026
# ╟─d42ade60-a9cf-4f22-963a-1a6c93ecf870
# ╟─e4967d94-699b-4fd5-9d63-c73331a274da
# ╟─521d739f-b037-4ae7-b350-d23a0cb14ca8
# ╟─ea394a41-50a4-46bc-9eb0-676c0ab51ba1
# ╟─7da2e3d0-198a-4f30-a2af-2e0c06d855f5
# ╟─1bb0715e-d0f0-4e95-9ccb-ae5e4f8c064d
# ╟─abf1d942-99d5-47ec-9c99-dc1699ee4714
# ╟─1e3b664a-24bc-44ce-a2b4-49bcc8b2e790
# ╟─41baecfb-8bb8-4988-9819-a6c7217b5e92
# ╟─a5d8076d-b214-49ab-bd4d-1b65b78fbd00
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
