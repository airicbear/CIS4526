### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ be2ce2e8-2495-11ed-1e02-ab78d3aa7dcb
begin
	using PlutoUI
	
	md"""
	# Foundations of Machine Learning 2nd Edition

	By Mehryar Mohri, Afshin Rostamizadeh, and Ameet Talwalkar
	"""
end

# ╔═╡ 3ac26ceb-55ce-4921-9834-fbfc59ebcac4
PlutoUI.TableOfContents()

# ╔═╡ f33fecbe-ddb7-45eb-93a0-0623eae31026
md"# 1 Introduction"

# ╔═╡ 1237a6ee-40f5-48c8-a73b-53de8c5bed64
md"""
**Remark.**
This chapter presents a preliminary introduction to machine learning, including an overview of some key learning tasks and applications, basic definitions and terminology, and the discussion of some general scenarios.
"""

# ╔═╡ af751cc3-e77d-43b3-ac62-e272807a6c90
md"## 1.1 What is machine learning?"

# ╔═╡ a560df55-4384-4f27-a324-df2f143fe87b
md"""
**Remark.**
Machine learning can be broadly defined as computational methods using experience to improve performance or to make accurate predictions.
"""

# ╔═╡ 617fd12e-0473-448f-9edf-1ad734b3863a
md"## 1.2 What kind of problems can be tackled using machine learning?"

# ╔═╡ 2a71aa26-d6e5-4304-92e9-df2e3ee926f7
md"""
**Remark.**
Predicting the label of a document, also known as document classification, is by no means the only learning task.
Machine learning admits a very broad set of practical applications, which include the following:

- Text or document classification

- Natural language process (NLP)

- Speech processing applications

- Computer vision applications

- Computational biology applications

- Many other problems such as fraud detection for credit card, telephone or insurance companies, network intrusion, learning to play games such as chess, backgammon, or Go, unassisted control of vehicles such as robots or cars, medical diagnosis, the design of recommendation systems, search engines, or information extraction systems, are tackled using machine learning techniques.
"""

# ╔═╡ 34324852-077c-45ae-9f4e-e1f27557b172
md"## 1.3 Some standard learning tasks"

# ╔═╡ 2fef69e5-cdd9-4599-bd1c-b6966f09a47a
md"""
**Remark.**
The following are some standard machine learning tasks that have been extensively studied:

- Classification

- Regression

- Ranking

- Clustering

- Dimensionality reduction or manifold learning
"""

# ╔═╡ 1412aca0-a682-4601-93ce-af5d55c83793
md"## 1.4 Learning stages"

# ╔═╡ 8b0ca042-8fc9-4345-9df1-ec2740a4fc1a
md"""
**Remark.**
Here, we will use the canonical problem of spam detection as a running example to illustrate some basic definitions and describe the use and evaluation of machine learning algorithms in practice, including their different stages.
"""

# ╔═╡ 11b533a0-dee2-48cf-9e87-917350b4122b
md"""
**Remark.**
Spam detection is the problem of learning to automatically classify email messages as either SPAM or non-SPAM.
The following is a list of definitions and terminology commonly used in machine learning:

- Examples

- Features

- Labels

- Hyperparameters

- Training sample

- Validation sample

- Test sample

- Loss function

- Hypothesis set
"""

# ╔═╡ 89442e18-7b56-4381-9327-c25560490a7a
md"## 1.5 Learning scenarios"

# ╔═╡ 5a594d69-fa82-4a63-8ce1-731c9e6e672f
md"""
**Remark.**
We next briefly describe some common machine learning scenarios.
These scenarios differ in the types of training data available to the learner, the order and method by which training data is received and the test data used to evaluate the learning algorithm.

- Supervised learning

- Unsupervised learning

- Semi-supervised learning

- Transductive inference

- On-line learning

- Reinforcement learning

- Active learning
"""

# ╔═╡ c619a62c-758a-4d17-8821-cc981af4bc01
md"## 1.6 Generalization"

# ╔═╡ 1f033fe4-4017-498b-a57f-8390ac143023
md"""
**Remark.**
Machine learning is fundamentally about *generalization*.
As an example, the standard supervised learning scenario consists of using a finite sample of labeled examples to make accurate predictions about unseen examples.
The problem is typically formulated as that of selecting a function out of a *hypothesis set*, that is a subset of the family of all functions.
The function selected is subsequently used to label all instances, including unseen examples.
"""

# ╔═╡ 34534d10-7181-4095-bb0d-f1fc4a66f325
md"# 2 The PAC Learning Framework"

# ╔═╡ df1c31bc-64ae-48a3-a859-8d17ef6a087c
md"""
**Remark.**
Several fundamental questions arise when designing and analyzing algorithms that learn from examples:
What can be learned efficiently?
What is inherently hard to learn?
How many examples are needed to learn successfully?
Is there a general model of learning?
In this chapter, we begin to formalize and address these questions by introducing the *Probably Approximately Correct* (PAC) learning framework.
The PAC framework helps define the class of learnable concepts in terms of *complexity*, and the time and space complexity of the learning algorithm, which depends on the cost of the computational representation of the concepts.
"""

# ╔═╡ d4487b0e-488c-4b80-bc0c-62ba70fe2220
md"""
**Remark.**
We first describe the PAC framework and illustrate it, then present some general learning guarantees within this framework when the hypothesis set used is finite, both for the *consistent* case wherer the hypothesis set used contains the concept to learn and for the opposite *inconsistent* case.
"""

# ╔═╡ 741b2f4b-9732-4b16-8fbf-73613f865621
md"## 2.1 The PAC learning model"

# ╔═╡ 6fa0d564-fb66-4ea5-abc1-89919b7565d8
md"""
**Remark.**
We first introduce several definitions and the notation needed to present the PAC model, which will also be used throughout much of this book.
"""

# ╔═╡ be179889-e379-4317-a213-13d81e2e7ad4
md"## 2.2 Guarantees for finite hypothesis sets --- consistent case"

# ╔═╡ 7862516c-08bc-4586-af9b-0ae8efac213b
md"## 2.3 Guarantees for finite hypothesis sets --- inconsistent case"

# ╔═╡ f12b1e38-07d5-4ac3-b81b-59b36a3c06ea
md"## 2.4 Generalities"

# ╔═╡ 2d90b03d-fbb3-4444-93b7-04347db8dbf6
md"### 2.4.1 Deterministic versus stochastic scenarios"

# ╔═╡ 27654b80-be7a-4859-883f-217bdeb0b2c6
md"### 2.4.2 Bayes error and noise"

# ╔═╡ e2f12266-4fb5-45e2-b819-71dc48f85a2d
md"## 2.5 Chapter notes"

# ╔═╡ 65fb44ca-6bd2-4e28-93c5-9d90326a25e9
md"## 2.6 Exercises"

# ╔═╡ f4c58f74-58e9-4e96-ba74-3b89e7e3ee77
md"# 3 Rademacher Complexity and VC-Dimension"

# ╔═╡ 9ed597f6-13ba-4e0f-9a4f-acf9af408eb6
md"# 4 Model Selection"

# ╔═╡ c8fb6415-1a7f-4e2a-92f6-be4aafd05584
md"# 5 Support Vector Machines"

# ╔═╡ 14115f68-3879-49ba-afa2-212583c07440
md"# 6 Kernel Methods"

# ╔═╡ 9b220f58-c8aa-474f-8a4f-d48c730fc15e
md"# 7 Boosting"

# ╔═╡ a385b3bb-2cb5-451a-a449-1b172e48d278
md"# 8 On-Line Learning"

# ╔═╡ 6725a749-93ac-4b9b-8167-40bcfaf8287d
md"# 9 Multi-Class Classification"

# ╔═╡ cb5cbc13-6ce6-4468-bd5e-b7c4ab6406d3
md"# 10 Ranking"

# ╔═╡ f2b9afc6-4b60-4eba-9f30-63ae80cf0d8e
md"# 11 Regression"

# ╔═╡ a9cbadc1-0cba-4ab9-9499-0765ac944126
md"# 12 Maximum Entropy Models"

# ╔═╡ 5de41a99-c592-42b4-9e0e-8de5386f8ea8
md"# 13 Conditional Maximum Entropy Models"

# ╔═╡ 469edda4-cee1-441e-b668-955c9fd4e80c
md"# 14 Algorithmic Stability"

# ╔═╡ 5b074de5-a13e-4ad1-81c1-859ec49e6649
md"# 15 Dimensionality Reduction"

# ╔═╡ cbfc5461-02a6-4fcb-a8bb-2830aa4f40a4
md"# 16 Learning Automata and Languages"

# ╔═╡ 33d0a252-6ef0-4e51-bf83-acb5dae9dfda
md"# 17 Reinforcement Learning"

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
PlutoUI = "~0.7.39"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.0"
manifest_format = "2.0"
project_hash = "6ff2529dffd0652d0349be095d4d180abf958f56"

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

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

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

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

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

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "0044b23da09b5608b4ecacb4e5e6c6332f833a7e"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.2"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "8d1f54886b9037091edf146b517989fc4a09efec"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.39"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╟─3ac26ceb-55ce-4921-9834-fbfc59ebcac4
# ╟─be2ce2e8-2495-11ed-1e02-ab78d3aa7dcb
# ╟─f33fecbe-ddb7-45eb-93a0-0623eae31026
# ╟─1237a6ee-40f5-48c8-a73b-53de8c5bed64
# ╟─af751cc3-e77d-43b3-ac62-e272807a6c90
# ╟─a560df55-4384-4f27-a324-df2f143fe87b
# ╟─617fd12e-0473-448f-9edf-1ad734b3863a
# ╟─2a71aa26-d6e5-4304-92e9-df2e3ee926f7
# ╟─34324852-077c-45ae-9f4e-e1f27557b172
# ╟─2fef69e5-cdd9-4599-bd1c-b6966f09a47a
# ╟─1412aca0-a682-4601-93ce-af5d55c83793
# ╟─8b0ca042-8fc9-4345-9df1-ec2740a4fc1a
# ╟─11b533a0-dee2-48cf-9e87-917350b4122b
# ╟─89442e18-7b56-4381-9327-c25560490a7a
# ╟─5a594d69-fa82-4a63-8ce1-731c9e6e672f
# ╟─c619a62c-758a-4d17-8821-cc981af4bc01
# ╟─1f033fe4-4017-498b-a57f-8390ac143023
# ╟─34534d10-7181-4095-bb0d-f1fc4a66f325
# ╟─df1c31bc-64ae-48a3-a859-8d17ef6a087c
# ╟─d4487b0e-488c-4b80-bc0c-62ba70fe2220
# ╟─741b2f4b-9732-4b16-8fbf-73613f865621
# ╟─6fa0d564-fb66-4ea5-abc1-89919b7565d8
# ╟─be179889-e379-4317-a213-13d81e2e7ad4
# ╟─7862516c-08bc-4586-af9b-0ae8efac213b
# ╟─f12b1e38-07d5-4ac3-b81b-59b36a3c06ea
# ╟─2d90b03d-fbb3-4444-93b7-04347db8dbf6
# ╟─27654b80-be7a-4859-883f-217bdeb0b2c6
# ╟─e2f12266-4fb5-45e2-b819-71dc48f85a2d
# ╟─65fb44ca-6bd2-4e28-93c5-9d90326a25e9
# ╟─f4c58f74-58e9-4e96-ba74-3b89e7e3ee77
# ╟─9ed597f6-13ba-4e0f-9a4f-acf9af408eb6
# ╟─c8fb6415-1a7f-4e2a-92f6-be4aafd05584
# ╟─14115f68-3879-49ba-afa2-212583c07440
# ╟─9b220f58-c8aa-474f-8a4f-d48c730fc15e
# ╟─a385b3bb-2cb5-451a-a449-1b172e48d278
# ╟─6725a749-93ac-4b9b-8167-40bcfaf8287d
# ╟─cb5cbc13-6ce6-4468-bd5e-b7c4ab6406d3
# ╟─f2b9afc6-4b60-4eba-9f30-63ae80cf0d8e
# ╟─a9cbadc1-0cba-4ab9-9499-0765ac944126
# ╟─5de41a99-c592-42b4-9e0e-8de5386f8ea8
# ╟─469edda4-cee1-441e-b668-955c9fd4e80c
# ╟─5b074de5-a13e-4ad1-81c1-859ec49e6649
# ╟─cbfc5461-02a6-4fcb-a8bb-2830aa4f40a4
# ╟─33d0a252-6ef0-4e51-bf83-acb5dae9dfda
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
