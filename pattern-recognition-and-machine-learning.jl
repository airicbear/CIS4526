### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ 5858da92-2497-11ed-02bd-358df40aab94
begin
	using PlutoUI

	md"""
	# Pattern Recognition and Machine Learning

	By Christopher M. Bishop
	"""
end

# ╔═╡ 7ddfd1a9-14ef-478b-ae6b-6e0677222639
PlutoUI.TableOfContents()

# ╔═╡ 6e290f04-b4b9-468c-9be4-2fcebf928ef2
md"# 1 Introduction"

# ╔═╡ 8238c50b-de47-4545-b725-94bd46503d63
md"""
**Remark.**
The problem of searching for patterns in data is a fundamental one and has a long and successful history.
"""

# ╔═╡ eebbd0cf-4ac7-4664-bda5-948f8ac99106
md"## 1.1 Example: Polynomial Curve Fitting"

# ╔═╡ e74eb618-d30d-4d62-bfd6-70ba171fef48
md"## 1.2 Probability Theory"

# ╔═╡ 19ab052d-b322-4307-9854-55fd6b2f0f71
md"## 1.3 Model Selection"

# ╔═╡ 44ed8c4a-aa7d-4610-9a16-7c917293e567
md"## 1.4 The Curse of Dimensionality"

# ╔═╡ b933201e-13bf-446f-8721-3dcea73dfdae
md"## 1.5 Decision Theory"

# ╔═╡ 2cfec58a-90c5-4101-82e8-33ccfbb617da
md"## 1.6 Information Theory"

# ╔═╡ f21c4a07-c251-4651-8992-57ed9c83dd55
md"# 2 Probability Distributions"

# ╔═╡ 1831cc5d-56a5-4305-a874-7678b3611b5c
md"## 2.1 Binary Variables"

# ╔═╡ d776f1b4-5bec-459b-b42b-4469505f8864
md"## 2.2 Multinomial Variables"

# ╔═╡ d9b1145b-15f2-403c-8483-fa6dafca10c1
md"## 2.3 The Gaussian Distribution"

# ╔═╡ 5dd1a76e-cdf6-4be3-99db-b233241a76d1
md"## 2.4 The Exponential Family"

# ╔═╡ 3179d4fa-fc6a-4e9a-8a91-4b08e013de3a
md"## 2.5 Nonparametric Methods"

# ╔═╡ 896a07f3-7ce0-47cc-ae23-c3f6ed9c980f
md"# 3 Linear Models for Regression"

# ╔═╡ 5b03d15b-a097-46cc-9dab-811fbc6e0d44
md"## 3.1 Linear Basis Function Models"

# ╔═╡ 6b124afe-dee8-41d0-b3bf-ff0a4edd5f42
md"## 3.2 The Bias-Variance Decomposition"

# ╔═╡ 18f9c711-32ab-4953-a166-3e2f7f7ef5aa
md"## 3.3 Bayesian Linear Regression"

# ╔═╡ 0e992c43-beb9-46af-ab03-5bc8edd53eaf
md"## 3.4 Bayesian Model Comparison"

# ╔═╡ 32047173-3d0d-4bd9-b2f2-799397e6059e
md"## 3.5 The Evidence Approximation"

# ╔═╡ b040310c-41dd-494b-8a0f-7f3c136768a1
md"## 3.6 Limitations of Fixed Basis Functions"

# ╔═╡ 698e68be-2e9a-474f-b9c9-a26760e7c4f4
md"# 4 Linear Models for Classification"

# ╔═╡ a0541733-9c31-4faa-bb80-fbb6459f5f67
md"## 4.1 Discriminant Functions"

# ╔═╡ 855c1ad2-4688-495e-82b8-1d0e3fb2fd6c
md"## 4.2 Probabilistic Generative Models"

# ╔═╡ a0aa4ed4-a7b0-437d-9729-8ba13da221b3
md"## 4.3 Probabilistic Discriminative Models"

# ╔═╡ 32b36b7d-8993-4e88-ba4e-39dac03fe5d4
md"## 4.4 The Laplace Approximation"

# ╔═╡ b3bf6b33-ac3e-43c4-894f-7a6af6a52c55
md"## 4.5 Bayesian Logistic Regression"

# ╔═╡ 62d3c536-9c7c-4b1f-9883-8f19a133c60c
md"# 5 Neural Networks"

# ╔═╡ fa537b9b-08c8-4745-98be-827fac7d84cc
md"## 5.1 Feed-forward Network Functions"

# ╔═╡ fadc4cc4-4bfa-42b7-b2b9-0090bb548492
md"## 5.2 Network Training"

# ╔═╡ 8b6436b4-871e-4cc2-adab-e2b96e5afee2
md"## 5.3 Error Backpropagation"

# ╔═╡ d4be159e-b3d8-4219-bc98-32454e492653
md"## 5.4 The Hessian Matrix"

# ╔═╡ ce800802-cafe-47b2-8bc8-3b78f5209efa
md"## 5.5 Regularization in Neural Networks"

# ╔═╡ a4a42099-790c-4236-ae3f-c5f2a6089a7b
md"## 5.6 Mixture Density Networks"

# ╔═╡ e0176195-581a-459c-809e-106badf22698
md"## 5.7 Bayesian Neural Networks"

# ╔═╡ 3d805fb2-4d0e-4665-a680-b7e155844b82
md"# 6 Kernel Methods"

# ╔═╡ 0ca250eb-522c-481b-8133-43c6324cdad2
md"## 6.1 Dual Representations"

# ╔═╡ ad30cd7e-f62b-497c-903b-e938eb15a39b
md"## 6.2 Constructing Kernels"

# ╔═╡ 1ada959f-fbaa-4aaa-81f2-ca1ea1a8ee99
md"## 6.3 Radial Basis Function Networks"

# ╔═╡ 13fc8f22-6765-43c0-bf1a-d067a8a9a170
md"## 6.4 Gaussian Processes"

# ╔═╡ ccab518f-4363-4515-ba91-d71bb383c592
md"# 7 Sparse Kernel Machines"

# ╔═╡ 06b3cfbb-53a7-4554-90b1-286ebfeb8dc4
md"# 8 Graphical Models"

# ╔═╡ feda4dfb-63d9-408d-80d8-8079c4783f07
md"# 9 Mixture Models and EM"

# ╔═╡ df8cec1f-150e-49a0-8874-112d378c65ae
md"# 10 Approximate Inference"

# ╔═╡ bd0c8210-df81-45e3-8a98-066632e36d03
md"# 11 Sampling Methods"

# ╔═╡ 4541b521-7a32-4d87-b7f1-dfb86291c207
md"# 12 Continuous Latent Variables"

# ╔═╡ 244cb529-0ef9-4951-b4d3-22b9556eb977
md"# 13 Sequential Data"

# ╔═╡ db2381f9-3f6a-4217-bee6-4bc57d7b8f9e
md"# 14 Combining Models"

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
# ╟─7ddfd1a9-14ef-478b-ae6b-6e0677222639
# ╟─5858da92-2497-11ed-02bd-358df40aab94
# ╟─6e290f04-b4b9-468c-9be4-2fcebf928ef2
# ╟─8238c50b-de47-4545-b725-94bd46503d63
# ╟─eebbd0cf-4ac7-4664-bda5-948f8ac99106
# ╟─e74eb618-d30d-4d62-bfd6-70ba171fef48
# ╟─19ab052d-b322-4307-9854-55fd6b2f0f71
# ╟─44ed8c4a-aa7d-4610-9a16-7c917293e567
# ╟─b933201e-13bf-446f-8721-3dcea73dfdae
# ╟─2cfec58a-90c5-4101-82e8-33ccfbb617da
# ╟─f21c4a07-c251-4651-8992-57ed9c83dd55
# ╟─1831cc5d-56a5-4305-a874-7678b3611b5c
# ╟─d776f1b4-5bec-459b-b42b-4469505f8864
# ╟─d9b1145b-15f2-403c-8483-fa6dafca10c1
# ╟─5dd1a76e-cdf6-4be3-99db-b233241a76d1
# ╟─3179d4fa-fc6a-4e9a-8a91-4b08e013de3a
# ╟─896a07f3-7ce0-47cc-ae23-c3f6ed9c980f
# ╟─5b03d15b-a097-46cc-9dab-811fbc6e0d44
# ╟─6b124afe-dee8-41d0-b3bf-ff0a4edd5f42
# ╟─18f9c711-32ab-4953-a166-3e2f7f7ef5aa
# ╟─0e992c43-beb9-46af-ab03-5bc8edd53eaf
# ╟─32047173-3d0d-4bd9-b2f2-799397e6059e
# ╟─b040310c-41dd-494b-8a0f-7f3c136768a1
# ╟─698e68be-2e9a-474f-b9c9-a26760e7c4f4
# ╟─a0541733-9c31-4faa-bb80-fbb6459f5f67
# ╟─855c1ad2-4688-495e-82b8-1d0e3fb2fd6c
# ╟─a0aa4ed4-a7b0-437d-9729-8ba13da221b3
# ╟─32b36b7d-8993-4e88-ba4e-39dac03fe5d4
# ╟─b3bf6b33-ac3e-43c4-894f-7a6af6a52c55
# ╟─62d3c536-9c7c-4b1f-9883-8f19a133c60c
# ╟─fa537b9b-08c8-4745-98be-827fac7d84cc
# ╟─fadc4cc4-4bfa-42b7-b2b9-0090bb548492
# ╟─8b6436b4-871e-4cc2-adab-e2b96e5afee2
# ╟─d4be159e-b3d8-4219-bc98-32454e492653
# ╟─ce800802-cafe-47b2-8bc8-3b78f5209efa
# ╟─a4a42099-790c-4236-ae3f-c5f2a6089a7b
# ╟─e0176195-581a-459c-809e-106badf22698
# ╟─3d805fb2-4d0e-4665-a680-b7e155844b82
# ╟─0ca250eb-522c-481b-8133-43c6324cdad2
# ╟─ad30cd7e-f62b-497c-903b-e938eb15a39b
# ╟─1ada959f-fbaa-4aaa-81f2-ca1ea1a8ee99
# ╟─13fc8f22-6765-43c0-bf1a-d067a8a9a170
# ╟─ccab518f-4363-4515-ba91-d71bb383c592
# ╟─06b3cfbb-53a7-4554-90b1-286ebfeb8dc4
# ╟─feda4dfb-63d9-408d-80d8-8079c4783f07
# ╟─df8cec1f-150e-49a0-8874-112d378c65ae
# ╟─bd0c8210-df81-45e3-8a98-066632e36d03
# ╟─4541b521-7a32-4d87-b7f1-dfb86291c207
# ╟─244cb529-0ef9-4951-b4d3-22b9556eb977
# ╟─db2381f9-3f6a-4217-bee6-4bc57d7b8f9e
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
