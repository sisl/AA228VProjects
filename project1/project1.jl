### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 173388ab-207a-42a6-b364-b2c1cb335f6b
# ╠═╡ show_logs = false
begin
	using Test
	using Base64
	using PlutoUI
	using Pkg
	Pkg.develop(path=joinpath("..", "..", "StanfordAA228V.jl"))
	using StanfordAA228V
	using StanfordAA228V.Distributions
	using StanfordAA228V.Random
	using StanfordAA228V.Plots
	using StanfordAA228V.ForwardDiff
	using StanfordAA228V.Optim
	using StanfordAA228V.Parameters
	using StanfordAA228V.BSON
	using StanfordAA228V.GridInterpolations
	using LinearAlgebra

	default(fontfamily="Computer Modern", framestyle=:box) # LaTeX-style plotting

	md"> **Package management**: _Hidden_ (click the \"eye\" icon to reveal)."
end

# ╔═╡ 60f72d30-ab80-11ef-3c20-270dbcdf0cc4
md"""
# Project 1: Finding the most-likely failure

**Task**: Efficiently find likely failures using $n$ total function calls to the system `step` function.
- **Small system**: 1D Gaussian $\mathcal{N}(0,1)$. With $n=100$ `step` calls.
- **Medium system**: Swinging inverted pendulum. With $n=1{,}000$ `step` calls.
- **Large system**: Aircraft collision avoidance system (CAS). With $n=10{,}000$ `step` calls.

Your job is to write the following function that returns the failure trajectory `τ` (i.e., a `Vector` of $(s,a,o,x)$ tuples) with the highest likelihood you found:
```julia
most_likely_failure(sys, ψ; n)::Vector
```

If you encounter issues, [please ask us on Ed](https://edstem.org/us/courses/69226/discussion).
"""

# ╔═╡ fd8c851a-3a42-41c5-b0fd-a12085543c9b
md"""
# 1️⃣ **Small**: 1D Gaussian
The small system is a simple 1D Gaussian system.
- There are no dynamics (rollout depth $d=1$).
- There are no disturbances.
- The (initial and only) state $s$ is sampled from $\mathcal{N}(0,1)$.

> **Reminder: One rollout has a fixed length of $d=1$.**
"""

# ╔═╡ 17fa8557-9656-4347-9d44-213fd3b635a6
Markdown.parse("""
## Small system
The system is comprised of an `agent`, environment (`env`), and `sensor`.
""")

# ╔═╡ 22feee3d-4627-4358-9937-3c780b7e8bcb
sys_small = System(NoAgent(), SimpleGaussian(), IdealSensor());

# ╔═╡ 6f3e24de-094c-49dc-b892-6721b3cc54ed
SmallSystem::Type = typeof(sys_small) # Type used for multiple dispatch

# ╔═╡ 45f7c3a5-5763-43db-aba8-41ef8db39a53
md"""
## Small environment
The environment is a standard normal (Gaussian) distribution $\mathcal{N}(0, 1)$.
"""

# ╔═╡ 9c1daa96-76b2-4a6f-8d0e-f95d26168d2b
ps_small = Ps(sys_small.env)

# ╔═╡ ab4c6807-5b4e-4688-b794-159e26a1599b
ψ_small = LTLSpecification(@formula □(s->s > -2));

# ╔═╡ 370a15eb-df4b-493a-af77-00914b4616ea
Markdown.parse("""
## Small specification \$\\psi\$
The specification \$\\psi\$ (written `\\psi<TAB>` in code) indicates what the system should do:

\$\$\\psi(\\tau) = \\square(s > $(ψ_small.formula.ϕ.c))\$\$

i.e., "the state \$s\$ in the trajectory \$\\tau\$ should _always_ (\$\\square\$) be greater than \$$(ψ_small.formula.ϕ.c)\$, anything else is a failure."
""")

# ╔═╡ 166bd412-d433-4dc9-b874-7359108c0a8b
Markdown.parse("""
A failure is unlikely given that the probability of failure is:

\$\$P(s > $(ψ_small.formula.ϕ.c)) \\approx $(round(cdf(ps_small, ψ_small.formula.ϕ.c), sigdigits=4))\$\$
""")

# ╔═╡ 9132a200-f63b-444b-9830-b03cf075021b
md"""
## Baseline
The following function is a baseline random falsification algorithm that returns the trajectory that led to the most-likely failure.

> **Your algorithm should do better than the random baseline.**
"""

# ╔═╡ 99eb3a5f-c6d4-48b6-8e96-0adbd123b160
md"""
**TODO**: `d` and `m`
"""

# ╔═╡ c2ae204e-dbcc-453a-81f5-791ba4be39db
@tracked function most_likely_failure_baseline(sys, ψ; d=1, m=100)
	pτ = NominalTrajectoryDistribution(sys, d)         # Trajectory distribution
	τs = [rollout(sys, pτ; d) for _ in 1:m]            # Rollout with pτ, n*d steps
	τs_failures = filter(τ->isfailure(ψ, τ), τs)       # Filter to get failure trajs
	τ_most_likely = argmax(τ->logpdf(pτ, τ), τs_failures) # Most-likely failure traj
	return τ_most_likely
end

# ╔═╡ e73635cc-2b1e-4162-8760-b62184e70b6d
md"""
### Example usage of small baseline
Example usage with default rollout depth of `d=1` and `n=100` number of rollouts.
"""

# ╔═╡ 7fe03702-25e5-473a-a92b-3b77eb753bc3
begin
	Random.seed!(4)
	τ_baseline_small = most_likely_failure_baseline(sys_small, ψ_small)
	p_τ_small = NominalTrajectoryDistribution(sys_small)
	ℓ_τ_small = pdf(p_τ_small, τ_baseline_small)
	n_steps_small = step_counter.count
end;

# ╔═╡ 73da2a56-8991-4484-bcde-7d397214e552
Markdown.parse("""
### Baseline results (small)

\$\$\\begin{align}
\\ell_\\text{baseline} &= $(round(ℓ_τ_small, sigdigits=3))\\tag{most-likely failure log-likelihood} \\\\
n_\\text{steps} &= $(n_steps_small) \\tag{number of \\texttt{step} calls}
\\end{align}\$\$

Reminder that the number of `step` calls \$n\$ is equal to the number of rollouts \$m\$ for the small system. This is because the rollout depth is \$d=1\$.
""")

# ╔═╡ a6603deb-57fa-403e-a2e5-1195ae7c016c
md"""
Here we plot $100$ states showing which ones were _successes_ and which ones were _failures_.
"""

# ╔═╡ 92f20cc7-8bc0-4aea-8c70-b0f759748fbf
html"""
<h2>⟶ <b>Task (Small)</b>: Most-likely failure</h2>
<p>Please fill in the following <code>most_likely_failure</code> function.</p>
<ul>
	<li><b>Note</b>: You have a maximum of <code>n=100</code> total calls to <code>step</code>.</li>
</ul>
"""

# ╔═╡ f6589984-e24d-4aee-b7e7-db159ae7fea6
md"""
	most_likely_failure(sys::SimpleGaussian, ψ; d, m)::Vector

A function that takes in a system `sys` (1D Gaussian for the _small_ setting) and a specification `ψ` and **returns the trajectory that led to the most-likely failure**.

- `d` = rollout depth (leave at `d=1` for the `SmallSystem`)
- `m` = number of rollouts

**Note**: `ψ` is written as `\psi<TAB>`
"""

# ╔═╡ fc2d34da-258c-4460-a0a4-c70b072f91ca
@tracked function most_likely_failure(sys::SmallSystem, ψ; n=100)
	# TODO: WRITE YOUR CODE HERE
end

# ╔═╡ ec776b30-6a30-4643-a22c-e071a365d50b
md"""
## Hints
Expand the sections below for some helpful hints.
"""

# ╔═╡ dba42df0-3199-4c31-a735-b6b514703d50
md"""
## Common Errors
These are some common errors you may run into.
"""

# ╔═╡ 8c78529c-1e00-472c-bb76-d984b37235ab
md"""
# 2️⃣ **Medium**: Inverted Pendulum
The medium system is a swinging inverted pendulum.
- It uses a proportional controller to keep it upright.
- The state is comprised of the angle $\theta$ and angular velocity $\omega$: $s = [\theta, \omega]$
- Actions are left/right adjustments in the range $[-2, 2]$
- Disturbances $x$ are treated as addative noise: $x \sim \mathcal{N}(\mathbf{0}, 0.1^2I)$

> **One rollout has a fixed length of $d=41$.**

"""

# ╔═╡ daada216-11d4-4f8b-807c-d347130a3928
LocalResource(joinpath(@__DIR__, "..", "media", "inverted_pendulum.svg"))

# ╔═╡ d18c2105-c2af-4dda-8388-617aa816a567
Markdown.parse("""
## Medium system
An inverted pendulum comprised of a `ProportionalController` with an `AdditiveNoiseSensor`.
""")

# ╔═╡ 77637b5e-e3ce-4ecd-90fc-95611af18002
sys_medium = System(
	ProportionalController([-15.0, -8.0]),
	InvertedPendulum(),
	AdditiveNoiseSensor(MvNormal(zeros(2), 0.1^2*I))
);

# ╔═╡ c4c0328d-8cb3-41d5-9740-0197cbf760c2
MediumSystem::Type = typeof(sys_medium) # Type used for multiple dispatch

# ╔═╡ b1e9bd40-a401-4630-9a1f-d61b276e72f7
md"""
## Medium specification $\psi$
The inverted pendulum specification $\psi$ indicates what the system should do:

$$\psi(\tau) = \square\big(|\theta| < \pi/4\big)$$

i.e., "the absolute value of the pendulum angle $\theta$ (first element of the state $s$) in the trajectory $\tau$ should _always_ ($\square$) be less than $\pi/4$, anything else is a failure."
"""

# ╔═╡ fe272c1b-421c-49de-a513-80c7bcefdd9b
ψ_medium = LTLSpecification(@formula □(s -> abs(s[1]) < π / 4));

# ╔═╡ a16cf110-4afa-4792-9d3f-f13b24349886
md"""
## Medium example rollouts
Example rollouts of the pendulum system and their plot below.
"""

# ╔═╡ 8b82eb8d-f6fe-4b73-8617-8c75dd65b769
begin
	Random.seed!(4)
	pτ_medium_ex = NominalTrajectoryDistribution(sys_medium, 41)
	τs_rollout_medium = [rollout(sys_medium, pτ_medium_ex; d=41) for i in 1:1000]
end;

# ╔═╡ bdb27ba8-782c-467c-818d-f68c7790e845
md"""
## Baseline: Medium
Example usage with rollout depth of `d=41` and `m=1000` number of rollouts.
"""

# ╔═╡ 3d00dc65-4c48-4988-9bb9-4cd3af6b9c5b
begin
	Random.seed!(4)
	τ_base_medium = most_likely_failure_baseline(sys_medium, ψ_medium; d=41, m=1000)
	p_τ_medium = NominalTrajectoryDistribution(sys_medium, 41)
	ℓ_τ_medium = logpdf(p_τ_medium, τ_base_medium)
	n_steps_medium = step_counter.count
end;

# ╔═╡ 7ef66a50-6acc-474f-b406-7b27a7b18510
Markdown.parse("""
### Baseline results (medium)

\$\$\\begin{align}
\\ell_\\text{baseline} &= $(round(ℓ_τ_medium, sigdigits=3))\\tag{most-likely failure log-likelihood} \\\\
n_\\text{steps} &= $(n_steps_medium) \\tag{number of \\texttt{step} calls \$d\\times m\$}
\\end{align}\$\$
""")

# ╔═╡ 1da9695f-b7fc-46eb-9ef9-12160246018d
md"""
## ⟶ **Task (Medium)**: Most-likely failure
Please fill in the following `most_likely_failure` function.
- **Note**: You have a maximum of $n = d\times m = 1{,}000$ total calls to `step`.
    - For example $d=100$ and $m=10$
"""

# ╔═╡ 9657f5ff-f21c-43c5-838d-402a2a723d5e
md"""
	most_likely_failure(sys::SimpleGaussian, ψ; d, m)::Vector

A function that takes in a system `sys` (inverted pendulum for the _medium_ setting) and a specification `ψ` and **returns the trajectory that led to the most-likely failure**.

- `d` = rollout depth
- `m` = number of rollouts

**Note**: `ψ` is written as `\psi<TAB>`
"""

# ╔═╡ cb7b9b9f-59da-4851-ab13-c451c26117df
@tracked function most_likely_failure(sys::MediumSystem, ψ; n=1000)
	# TODO: WRITE YOUR CODE HERE
end

# ╔═╡ 4943ca08-157c-40e1-acfd-bd9326082f56
md"""
## Hints
Useful tips to watch out for.
"""

# ╔═╡ 7d054465-9f80-4dfb-9b5f-76c3977de7cd
Markdown.parse("""
## Large system
An aircraft collision avoidance system that uses an interpolated lookup-table policy.
""")

# ╔═╡ 1ec68a39-8de9-4fd3-be8a-26cf7706d1d6
begin
	grid, Q = load_cas_policy(joinpath(@__DIR__, "cas_policy.bson"))

	cas_agent = InterpAgent(grid, Q)
	cas_env = CollisionAvoidance(Ds=Normal(0, 1.5))
	cas_sensor = IdealSensor()
	sys_large = System(cas_agent, cas_env, cas_sensor)

	LargeSystem::Type = typeof(sys_large) # Type used for multiple dispatch
end

# ╔═╡ d23f0299-981c-43b9-88f3-fb6e07927498
md"""
## Large environment
The collision avoidance system has disturbances applied to the relative vertical rate variable $\dot{h}$ of the state (i.e., environment disturbances).

$$\dot{h} + x \quad \text{where} \quad x \sim \mathcal{N}(0, 1.5)$$
"""

# ╔═╡ 641b92a3-8ff2-4aed-8482-9fa686803b68
cas_env.Ds

# ╔═╡ be426908-3fee-4ecd-b054-2497ce9a2e50
md"""
## Large specification $\psi$
The collision avoidance system specification $\psi$ indicates what the system should do:

$$\psi(\tau) = \square_{[41]}\big(|h| > 50\big)$$

i.e., "the absolute valued relative altitude $h$ (first element of the state $s$) in the trajectory $\tau$ should _always_ ($\square$) be greater than $50$ meters at the end of the encounter ($t=41$), anything else is a failure."
"""

# ╔═╡ 258e14c4-9a2d-4515-9a8f-8cd96f31a6ff
ψ_large = LTLSpecification(@formula □(41:41, s->abs(s[1]) > 50));

# ╔═╡ 1a097a88-e4f0-4a8d-a5d6-2e3858ee417c
begin
	Random.seed!(4)
	p_τ_large_ex = NominalTrajectoryDistribution(sys_large, 41)
	τs_rollout_large = [rollout(sys_large, p_τ_large_ex; d=41) for i in 1:10000]
end;

# ╔═╡ a4e0000b-4b4a-4262-bf0a-85509c4ee47e
md"""
## Baseline: Large
"""

# ╔═╡ b5d02715-b7c9-4bf2-a284-42da40a70a68
begin
	Random.seed!(4)
	τ_base_large = most_likely_failure_baseline(sys_large, ψ_large; d=41, m=10000)
	p_τ_large = NominalTrajectoryDistribution(sys_large, 41)
	ℓ_τ_large = logpdf(p_τ_large, τ_base_large)
end;

# ╔═╡ 204feed7-cde8-40a8-b6b5-051a1c768fd9
Markdown.parse("""
### Baseline results (large)

\$\$\\begin{align}
\\ell_\\text{baseline} &= $(round(ℓ_τ_large, sigdigits=3))\\tag{most-likely failure log-likelihood} \\\\
n_\\text{steps} &= $(step_counter.count) \\tag{number of \\texttt{step} calls \$d\\times m\$}
\\end{align}\$\$
""")

# ╔═╡ 23fd490a-74d2-44b4-8a12-ea1460d95f85
md"""
## ⟶ **Task (Large)**: Most-likely failure
Please fill in the following `most_likely_failure` function.
- **Note**: You have a maximum of $n = d\times m = 1{,}025{,}000$ total calls to `step`.
    - For $d=41$ and $m=25{,}000$
> **_TODO_**.
"""

# ╔═╡ 3471a623-16af-481a-8f66-5bd1e7890188
@tracked function most_likely_failure(sys::LargeSystem, ψ; n=10000)
	# TODO: WRITE YOUR CODE HERE
end

# ╔═╡ 2827a6f3-47b6-4e6f-b6ae-63271715d1f3
Markdown.parse("""
# 📊 Tests
The tests below run your `num_failures` function to see if it works properly.

This will automatically run anytime the `num_failures` function is changed and saved (due to Pluto having dependent cells).
""")

# ╔═╡ 83884eb4-6718-455c-b731-342471325326
# ╠═╡ disabled = true
#=╠═╡
function run_project0_test(num_failures::Function; d=100, n=1000, seed=SEED)
	Random.seed!(seed) # For determinism
	return num_failures(sys, ψ; d, n)
end
  ╠═╡ =#

# ╔═╡ 4a91853f-9685-47f3-998a-8e0cfce688f8
Markdown.parse("""
## Running tests
Run two tests, controlling the RNG seed for deterministic outputs.
""")

# ╔═╡ 2ff6bb9c-5282-4ba1-b62e-a9fd0fe1969c
md"""
### Test 1: $n = 1000$
"""

# ╔═╡ b6f15d9c-33b8-40e3-be57-d91eda1c9753
#=╠═╡
begin
	test1_n = 1000
	test1_output = run_project0_test(num_failures; d=100, n=test1_n, seed=SEED)
end
  ╠═╡ =#

# ╔═╡ 522bb285-bc06-4c92-82ee-c0d0f68b184c
#=╠═╡
if isa(test1_output, Number)
	Markdown.parse("""
	The frequentist failure probability estimate for test 1 would be:
	
	\$\$\\hat{p}_{\\rm failure}^{({\\rm test}_1)} = \\frac{$(test1_output)}{$test1_n} =  $(test1_output/test1_n)\$\$
	""")
else
	md"*Update `num_failures` to get an estimated failure probability for test 1.*"
end
  ╠═╡ =#

# ╔═╡ 089581ec-8aff-4c56-9a65-26d394d5eec3
md"""
### Test 2: $n = 5000$
"""

# ╔═╡ 3314f402-10cc-434c-acbc-d38e59e4b846
#=╠═╡
begin
	test2_n = 5000
	test2_output = run_project0_test(num_failures; d=100, n=test2_n, seed=SEED)
end
  ╠═╡ =#

# ╔═╡ d72be566-6ad7-4817-8590-a504a699a4da
#=╠═╡
if isa(test2_output, Number)
	Markdown.parse("""
	The frequentist failure probability estimate for test 2 would be:
	
	\$\$\\hat{p}_{\\rm failure}^{({\\rm test}_2)} = \\frac{$(test2_output)}{$test2_n} =  $(test2_output/test2_n)\$\$
	""")
else
	md"*Update `num_failures` to get an estimated failure probability for test 2.*"
end
  ╠═╡ =#

# ╔═╡ 712e69bf-48e7-47e9-a14e-25cce64d4ae4
#=╠═╡
test2_n * 100
  ╠═╡ =#

# ╔═╡ cee165f0-049f-4ea3-8f19-04e66947a397
html"""
<h3>Check tests</h3>
<p>If the following test indicator is <span style='color:#759466'><b>green</b></span>, you can submit <code>project0.jl</code> (this file) to Gradescope.</p>
"""

# ╔═╡ 6302729f-b34a-4a18-921b-d194fe834208
#=╠═╡
begin
	# ⚠️ Note: PLEASE DO NOT MODIFY. Why are you in here anyhow :)?

	test1_passed::Bool = test1_output == 19
	test2_passed::Bool = test2_output == 110

	if test1_passed && test2_passed
		correct()
	elseif isnothing(test1_output) && isnothing(test1_output)
		almost()
	else
		keep_working()
	end
end
  ╠═╡ =#

# ╔═╡ 95e3d42f-b33f-4294-81c5-f34a300dc9b4
# This needs to be in the cell above.
html"""
<script>
let cell = currentScript.closest('pluto-cell')
let id = cell.getAttribute('id')
let cells_below = document.querySelectorAll(`pluto-cell[id='${id}'] ~ pluto-cell`)
let cell_below_ids = [cells_below[0]].map((el) => el.getAttribute('id'))
cell._internal_pluto_actions.set_and_run_multiple(cell_below_ids)
</script>
"""

# ╔═╡ ba6c082b-6e62-42fc-a85c-c8b7efc89b88
# ╠═╡ show_logs = false
begin
	########################################################
	# NOTE: DECODING THIS IS A VIOLATION OF THE HONOR CODE.
	########################################################
	ModuleTA = "UsingThisViolatesTheHonorCode_$(basename(tempname()))"
	try
		eval(Meta.parse("""
		module $ModuleTA
		$(String(base64decode("IyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMKIyBMT09LSU5HIEFUIFRISVMgSVMgQSBWSU9MQVRJT04gT0YgVEhFIEhPTk9SIENPREUKIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMKClRoaXNNb2R1bGUgPSBzcGxpdChzdHJpbmcoQF9fTU9EVUxFX18pLCAiLiIpW2VuZF0KCiMgTG9hZCBhbGwgY29kZSBhbmQgcGFja2FnZXMgZnJvbSBwYXJlbnQgbW9kdWxlClBhcmVudCA9IHBhcmVudG1vZHVsZShAX19NT0RVTEVfXykKCm1vZHVsZXMobTo6TW9kdWxlKSA9IGNjYWxsKDpqbF9tb2R1bGVfdXNpbmdzLCBBbnksIChBbnksKSwgbSkKCiMgTG9hZCBmdW5jdGlvbnMgYW5kIHZhcmlhYmxlcwpmb3IgbmFtZSBpbiBuYW1lcyhQYXJlbnQsIGltcG9ydGVkPXRydWUpCglpZiBuYW1lICE9IFN5bWJvbChUaGlzTW9kdWxlKSAmJiAhb2NjdXJzaW4oIiMiLCBzdHJpbmcobmFtZSkpICYmICFvY2N1cnNpbigiVXNpbmdUaGlzVmlvbGF0ZXNUaGVIb25vckNvZGUiLCBzdHJpbmcobmFtZSkpCgkJQGV2YWwgY29uc3QgJChuYW1lKSA9ICQoUGFyZW50KS4kKG5hbWUpCgllbmQKZW5kCgpleGNsdWRlcyA9IFsiUGx1dG9SdW5uZXIiLCAiSW50ZXJhY3RpdmVVdGlscyIsICJNYXJrZG93biIsICJDb3JlIiwgIkJhc2UiLCAiQmFzZS5NYWluSW5jbHVkZSJdCgojIExvYWQgcGFja2FnZXMKZm9yIG1vZCBpbiBtb2R1bGVzKFBhcmVudCkKCXN0cmluZyhtb2QpIGluIGV4Y2x1ZGVzICYmIGNvbnRpbnVlCgl0cnkKCQlAZXZhbCB1c2luZyAkKFN5bWJvbChtb2QpKQoJY2F0Y2ggZXJyCgkJaWYgZXJyIGlzYSBBcmd1bWVudEVycm9yCgkJCXRyeQoJCQkJQGV2YWwgdXNpbmcgU3RhbmZvcmRBQTIyOFYuJChTeW1ib2wobW9kKSkKCQkJY2F0Y2ggZXJyMgoJCQkJQHdhcm4gZXJyMgoJCQllbmQKCQllbHNlCgkJCUB3YXJuIGVycgoJCWVuZAoJZW5kCmVuZAoKIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIw0KIyBPcHRpbWl6YXRpb24tYmFzZWQgbW9zdC1saWtlbHkgZmFpbHVyZQ0KIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIw0KDQpmdW5jdGlvbiByb2J1c3RuZXNzX29iamVjdGl2ZShpbnB1dCwgc3lzLCDPiDsgc21vb3RobmVzcz0wLjApDQoJcywg8J2QsSA9IGV4dHJhY3Qoc3lzLmVudiwgaW5wdXQpDQoJz4QgPSByb2xsb3V0KHN5cywgcywg8J2QsSkNCgnwnZCsID0gW3N0ZXAucyBmb3Igc3RlcCBpbiDPhF0NCglyZXR1cm4gcm9idXN0bmVzcyjwnZCsLCDPiC5mb3JtdWxhLCB3PXNtb290aG5lc3MpDQplbmQNCg0KZnVuY3Rpb24gd2VpZ2h0ZWRfbGlrZWxpaG9vZF9vYmplY3RpdmUoaW5wdXQsIHN5cywgz4g7IHNtb290aG5lc3M9MS4wLCDOuz0xLjApDQoJcywg8J2QsSA9IGV4dHJhY3Qoc3lzLmVudiwgaW5wdXQpDQoJz4QgPSByb2xsb3V0KHN5cywgcywg8J2QsSkNCgnwnZCsID0gW3N0ZXAucyBmb3Igc3RlcCBpbiDPhF0NCglwID0gTm9taW5hbFRyYWplY3RvcnlEaXN0cmlidXRpb24oc3lzLCBsZW5ndGgo8J2QsSkpDQoJcmV0dXJuIHJvYnVzdG5lc3Mo8J2QrCwgz4guZm9ybXVsYSwgdz1zbW9vdGhuZXNzKSAtIM67ICogbG9ncGRmKHAsIM+EKQ0KZW5kDQoNCmZ1bmN0aW9uIGxpa2VsaWhvb2Rfb2JqZWN0aXZlKHgsIHN5cywgz4g7IHNtb290aG5lc3M9MS4wKQ0KCXMsIPCdkLEgPSBleHRyYWN0KHN5cy5lbnYsIHgpDQoJz4QgPSByb2xsb3V0KHN5cywgcywg8J2QsSkNCglpZiBpc2ZhaWx1cmUoz4gsIM+EKQ0KCQlwID0gTm9taW5hbFRyYWplY3RvcnlEaXN0cmlidXRpb24oc3lzLCBsZW5ndGgo8J2QsSkpDQoJCXJldHVybiAtbG9ncGRmKHAsIM+EKQ0KCWVsc2UNCgkJ8J2QrCA9IFtzdGVwLnMgZm9yIHN0ZXAgaW4gz4RdDQoJCXJldHVybiByb2J1c3RuZXNzKPCdkKwsIM+ILmZvcm11bGEsIHc9c21vb3RobmVzcykNCgllbmQNCmVuZA0KDQpzdHJ1Y3QgT3B0aW1pemF0aW9uQmFzZWRGYWxzaWZpY2F0aW9uDQoJb2JqZWN0aXZlICMgb2JqZWN0aXZlIGZ1bmN0aW9uDQoJb3B0aW1pemVyICMgb3B0aW1pemF0aW9uIGFsZ29yaXRobQ0KZW5kDQoNCmZ1bmN0aW9uIGZhbHNpZnkoYWxnOjpPcHRpbWl6YXRpb25CYXNlZEZhbHNpZmljYXRpb24sIHN5cywgz4gpDQoJZih4KSA9IGFsZy5vYmplY3RpdmUoeCwgc3lzLCDPiCkNCglyZXR1cm4gYWxnLm9wdGltaXplcihmLCBzeXMsIM+IKQ0KZW5kDQoNCkB0cmFja2VkIGZ1bmN0aW9uIGdyYWRpZW50X2Rlc2NlbnQoc3lzLCDPiDsgbiwgzrE9MWUtMykNCgkjIGYoeCkgPSB3ZWlnaHRlZF9saWtlbGlob29kX29iamVjdGl2ZSh4LCBzeXMsIM+IOyDOuz0wLjAxLCBzbW9vdGhuZXNzPTApDQoJIyBmKHgpID0gbGlrZWxpaG9vZF9vYmplY3RpdmUoeCwgc3lzLCDPiDsgc21vb3RobmVzcz0xKQ0KCWYoeCkgPSByb2J1c3RuZXNzX29iamVjdGl2ZSh4LCBzeXMsIM+IKQ0KCeKIh2YoeCkgPSBGb3J3YXJkRGlmZi5ncmFkaWVudChmLCB4KQ0KCXggPSBpbml0aWFsX2d1ZXNzKHN5cykNCglmb3IgaSBpbiAxOm4NCgkJeCAtPSDOsSriiIdmKHgpDQoJZW5kDQoJcmV0dXJuIHgNCmVuZA0KDQpmdW5jdGlvbiBydW5fZ2Qoc3lzLCDPiDsgbj0xMDAsIM6xPTFlLTMpDQoJUmFuZG9tLnNlZWQhKDQpDQoJeCA9IGdyYWRpZW50X2Rlc2NlbnQoc3lzLCDPiDsgbiwgzrEpDQoJQHNob3cgc3RlcF9jb3VudGVyLmNvdW50DQoJz4QgPSByb2xsb3V0KHN5cywgZXh0cmFjdChzeXMuZW52LCB4KS4uLikNCglyZXR1cm4gz4QNCmVuZA0KDQpAdHJhY2tlZCBmdW5jdGlvbiBtb3N0X2xpa2VseV9mYWlsdXJlX25lbGRlcl9tZWFkKHN5czo6U3lzdGVtLCDPiDsgbj0xMDAsIHZlcmJvc2U9ZmFsc2UpDQoJZnVuY3Rpb24gbmVsZGVyX21lYWQoZiwgc3lzLCDPiCkNCgkJeOKCgCA9IGluaXRpYWxfZ3Vlc3Moc3lzKQ0KCQlhbGcgPSBPcHRpbS5OZWxkZXJNZWFkKCkNCgkJb3B0aW9ucyA9IE9wdGltLk9wdGlvbnMoDQoJCQlzdG9yZV90cmFjZT10cnVlLA0KCQkJZXh0ZW5kZWRfdHJhY2U9dHJ1ZSwNCgkJCSMgZl9jYWxsc19saW1pdD1uLA0KCQkJIyBmX2NhbGxzX2xpbWl0PW4sDQoJCQkjIGl0ZXJhdGlvbnM9biwNCgkJKQ0KCQlyZXN1bHRzID0gb3B0aW1pemUoZiwgeOKCgCwgYWxnLCBvcHRpb25zKQ0KCQlpZiB2ZXJib3NlDQoJCQlAc2hvdyBzdGVwX2NvdW50ZXIuY291bnQNCgkJCWRpc3BsYXkocmVzdWx0cykNCgkJZW5kDQoJCc+EID0gcm9sbG91dChzeXMsIGV4dHJhY3Qoc3lzLmVudiwgT3B0aW0ubWluaW1pemVyKHJlc3VsdHMpKS4uLikNCgllbmQNCg0KCSMgb2JqZWN0aXZlKHgsIHN5cywgz4gpID0gd2VpZ2h0ZWRfbGlrZWxpaG9vZF9vYmplY3RpdmUoeCxzeXMsz4g7IHNtb290aG5lc3M9MSwgzrs9MC4wMDAxKQ0KCSMgb2JqZWN0aXZlKHgsIHN5cywgz4gpID0gcm9idXN0bmVzc19vYmplY3RpdmUoeCwgc3lzLCDPiDsgc21vb3RobmVzcz0wKQ0KCW9iamVjdGl2ZSh4LCBzeXMsIM+IKSA9IGxpa2VsaWhvb2Rfb2JqZWN0aXZlKHgsIHN5cywgz4g7IHNtb290aG5lc3M9MSkNCglhbGcgPSBPcHRpbWl6YXRpb25CYXNlZEZhbHNpZmljYXRpb24ob2JqZWN0aXZlLCBuZWxkZXJfbWVhZCkNCgnPhF9mYWlsdXJlID0gZmFsc2lmeShhbGcsIHN5cywgz4gpDQoJcmV0dXJuIM+EX2ZhaWx1cmUNCmVuZA0KDQpAdHJhY2tlZCBmdW5jdGlvbiBtb3N0X2xpa2VseV9mYWlsdXJlX2xiZmdzKHN5czo6U3lzdGVtLCDPiDsgdz0xLCBuPTEwMCwgzrs9MC4wMDAxLCB2ZXJib3NlPWZhbHNlKQ0KCWZ1bmN0aW9uIGxiZmdzKGYsIHN5cywgz4gpDQoJCXjigoAgPSBpbml0aWFsX2d1ZXNzKHN5cykNCgkJYWxnID0gT3B0aW0uTEJGR1MoKQ0KCQlvcHRpb25zID0gT3B0aW0uT3B0aW9ucygNCgkJCXN0b3JlX3RyYWNlPXRydWUsDQoJCQlleHRlbmRlZF90cmFjZT10cnVlLA0KCQkJZl9jYWxsc19saW1pdD1uLA0KCQkJIyBnX2NhbGxzX2xpbWl0PW7DtzIsDQoJCQkjIGl0ZXJhdGlvbnM9biwNCgkJKQ0KCQlyZXN1bHRzID0gb3B0aW1pemUoZiwgeOKCgCwgYWxnLCBvcHRpb25zOyBhdXRvZGlmZj06Zm9yd2FyZCkNCgkJaWYgdmVyYm9zZQ0KCQkJQHNob3cgc3RlcF9jb3VudGVyLmNvdW50DQoJCQlkaXNwbGF5KHJlc3VsdHMpDQoJCWVuZA0KCQnPhCA9IHJvbGxvdXQoc3lzLCBleHRyYWN0KHN5cy5lbnYsIE9wdGltLm1pbmltaXplcihyZXN1bHRzKSkuLi4pDQoJCXJldHVybiDPhA0KCWVuZA0KDQoJIyBvYmplY3RpdmUoeCwgc3lzLCDPiCkgPSB3ZWlnaHRlZF9saWtlbGlob29kX29iamVjdGl2ZSh4LCBzeXMsIM+IOyBzbW9vdGhuZXNzPXcsIM67KQ0KCSMgb2JqZWN0aXZlKHgsIHN5cywgz4gpID0gcm9idXN0bmVzc19vYmplY3RpdmUoeCwgc3lzLCDPiDsgc21vb3RobmVzcz13KQ0KCW9iamVjdGl2ZSh4LCBzeXMsIM+IKSA9IGxpa2VsaWhvb2Rfb2JqZWN0aXZlKHgsIHN5cywgz4g7IHNtb290aG5lc3M9dykNCglhbGcgPSBPcHRpbWl6YXRpb25CYXNlZEZhbHNpZmljYXRpb24ob2JqZWN0aXZlLCBsYmZncykNCgnPhF9mYWlsdXJlID0gZmFsc2lmeShhbGcsIHN5cywgz4gpDQoJcmV0dXJuIM+EX2ZhaWx1cmUNCmVuZA0KDQpAdHJhY2tlZCBmdW5jdGlvbiBtb3N0X2xpa2VseV9mYWlsdXJlX3NnZChzeXM6OlN5c3RlbSwgz4g7IG49MTAwLCDOuz0wLjAwMDEsIHc9MSwgdmVyYm9zZT1mYWxzZSkNCglmdW5jdGlvbiBzZ2QoZiwgc3lzLCDPiCkNCgkJeOKCgCA9IGluaXRpYWxfZ3Vlc3Moc3lzKQ0KCQkjIGFsZyA9IE9wdGltLkdyYWRpZW50RGVzY2VudChhbHBoYWd1ZXNzPTAuMSkNCgkJYWxnID0gT3B0aW0uR3JhZGllbnREZXNjZW50KGFscGhhZ3Vlc3M9MWUtNykNCgkJb3B0aW9ucyA9IE9wdGltLk9wdGlvbnMoDQoJCQlzdG9yZV90cmFjZT10cnVlLA0KCQkJZXh0ZW5kZWRfdHJhY2U9dHJ1ZSwNCgkJCWZfY2FsbHNfbGltaXQ9biwNCgkJCSMgZ19jYWxsc19saW1pdD1uw7cyLA0KCQkJIyBpdGVyYXRpb25zPW4sDQoJCSkNCgkJcmVzdWx0cyA9IG9wdGltaXplKGYsIHjigoAsIGFsZywgb3B0aW9uczsgYXV0b2RpZmY9OmZvcndhcmQpDQoJCWlmIHZlcmJvc2UNCgkJCUBzaG93IHN0ZXBfY291bnRlci5jb3VudA0KCQkJZGlzcGxheShyZXN1bHRzKQ0KCQllbmQNCgkJz4QgPSByb2xsb3V0KHN5cywgZXh0cmFjdChzeXMuZW52LCBPcHRpbS5taW5pbWl6ZXIocmVzdWx0cykpLi4uKQ0KCQlyZXR1cm4gz4QNCgllbmQNCg0KCSMgb2JqZWN0aXZlKHgsIHN5cywgz4gpID0gcm9idXN0bmVzc19vYmplY3RpdmUoeCwgc3lzLCDPiDsgc21vb3RobmVzcz13KQ0KCSMgb2JqZWN0aXZlKHgsIHN5cywgz4gpID0gd2VpZ2h0ZWRfbGlrZWxpaG9vZF9vYmplY3RpdmUoeCxzeXMsz4g7IHNtb290aG5lc3M9MSwgzrspDQoJb2JqZWN0aXZlKHgsIHN5cywgz4gpID0gbGlrZWxpaG9vZF9vYmplY3RpdmUoeCwgc3lzLCDPiDsgc21vb3RobmVzcz0xKQ0KCWFsZyA9IE9wdGltaXphdGlvbkJhc2VkRmFsc2lmaWNhdGlvbihvYmplY3RpdmUsIHNnZCkNCgnPhF9mYWlsdXJlID0gZmFsc2lmeShhbGcsIHN5cywgz4gpDQoJcmV0dXJuIM+EX2ZhaWx1cmUNCmVuZA0KDQoNCiMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMNCiMgRnV6emluZw0KIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIw0KDQpzdHJ1Y3QgR2F1c3NpYW5GdXp6aW5nRGlzdHJpYnV0aW9uIDw6IFRyYWplY3RvcnlEaXN0cmlidXRpb24NCiAgICDOvCAjIHByb3Bvc2FsOiBzdGF0ZSBtZWFuDQogICAgz4MgIyBwcm9wb3NhbDogc3RhdGUgc3RhbmRhcmQgZGV2aWF0aW9uDQplbmQNCg0KZnVuY3Rpb24gU3RhbmZvcmRBQTIyOFYuaW5pdGlhbF9zdGF0ZV9kaXN0cmlidXRpb24ocDo6R2F1c3NpYW5GdXp6aW5nRGlzdHJpYnV0aW9uKQ0KICAgIHJldHVybiBOb3JtYWwocC7OvCwgcC7PgykNCmVuZA0KDQojIFRPRE86IERlZmF1bHQuDQpmdW5jdGlvbiBTdGFuZm9yZEFBMjI4Vi5kaXN0dXJiYW5jZV9kaXN0cmlidXRpb24ocDo6R2F1c3NpYW5GdXp6aW5nRGlzdHJpYnV0aW9uLCB0KQ0KICAgIEQgPSBEaXN0dXJiYW5jZURpc3RyaWJ1dGlvbigobyktPkRldGVybWluaXN0aWMoKSwNCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgKHMsYSktPkRldGVybWluaXN0aWMoKSwNCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgKHMpLT5EZXRlcm1pbmlzdGljKCkpDQogICAgcmV0dXJuIEQNCmVuZA0KDQpTdGFuZm9yZEFBMjI4Vi5kZXB0aChwOjpHYXVzc2lhbkZ1enppbmdEaXN0cmlidXRpb24pID0gMQ0KDQpAdHJhY2tlZCBmdW5jdGlvbiBtb3N0X2xpa2VseV9mYWlsdXJlX2Z1enp5KHN5czo6U21hbGxTeXN0ZW0sIM+IOyBkPTEsIG09MTAwKQ0KCXDPhCA9IE5vbWluYWxUcmFqZWN0b3J5RGlzdHJpYnV0aW9uKHN5cywgZCkgICAgICAgICAjIFRyYWplY3RvcnkgZGlzdHJpYnV0aW9uDQoJcc+EID0gR2F1c3NpYW5GdXp6aW5nRGlzdHJpYnV0aW9uKDAsIDUpICAgICAgICAgICMgVHJhamVjdG9yeSBkaXN0cmlidXRpb24NCgkjIHHPhCA9IEdhdXNzaWFuRnV6emluZ0Rpc3RyaWJ1dGlvbijPiC5mb3JtdWxhLs+VLmMsIDAuMSkgICAgICAgICAgIyBUcmFqZWN0b3J5IGRpc3RyaWJ1dGlvbg0KCc+EcyA9IFtyb2xsb3V0KHN5cywgcc+EOyBkKSBmb3IgXyBpbiAxOm1dICAgICAgICAgICAgIyBSb2xsb3V0IHdpdGggcM+ELCBuKmQgc3RlcHMNCgnPhHNfZmFpbHVyZXMgPSBmaWx0ZXIoz4QtPmlzZmFpbHVyZSjPiCwgz4QpLCDPhHMpICAgICAgICMgRmlsdGVyIHRvIGdldCBmYWlsdXJlIHRyYWpzDQoJz4RfbW9zdF9saWtlbHkgPSBhcmdtYXgoz4QtPmxvZ3BkZihwz4QsIM+EKSwgz4RzX2ZhaWx1cmVzKSAjIE1vc3QtbGlrZWx5IGZhaWx1cmUgdHJhag0KCXJldHVybiDPhF9tb3N0X2xpa2VseQ0KZW5kDQoNCg0Kc3RydWN0IFBlbmR1bHVtRnV6emluZ0Rpc3RyaWJ1dGlvbiA8OiBUcmFqZWN0b3J5RGlzdHJpYnV0aW9uDQogICAgzqPigpIgIyBzZW5zb3IgZGlzdHVyYmFuY2UgY292YXJpYW5jZQ0KICAgIGQgIyBkZXB0aA0KZW5kDQoNCmZ1bmN0aW9uIFN0YW5mb3JkQUEyMjhWLmluaXRpYWxfc3RhdGVfZGlzdHJpYnV0aW9uKHA6OlBlbmR1bHVtRnV6emluZ0Rpc3RyaWJ1dGlvbikNCiAgICByZXR1cm4gUHJvZHVjdChbVW5pZm9ybSgtz4AgLyAxNiwgz4AgLyAxNiksIFVuaWZvcm0oLTEuLCAxLildKQ0KZW5kDQoNCmZ1bmN0aW9uIFN0YW5mb3JkQUEyMjhWLmRpc3R1cmJhbmNlX2Rpc3RyaWJ1dGlvbihwOjpQZW5kdWx1bUZ1enppbmdEaXN0cmlidXRpb24sIHQpDQogICAgRCA9IERpc3R1cmJhbmNlRGlzdHJpYnV0aW9uKChvKS0+RGV0ZXJtaW5pc3RpYygpLA0KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAocyxhKS0+RGV0ZXJtaW5pc3RpYygpLA0KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAocyktPk12Tm9ybWFsKHplcm9zKDIpLCBwLs6j4oKSKSkNCiAgICByZXR1cm4gRA0KZW5kDQoNClN0YW5mb3JkQUEyMjhWLmRlcHRoKHA6OlBlbmR1bHVtRnV6emluZ0Rpc3RyaWJ1dGlvbikgPSBwLmQNCg0KQHRyYWNrZWQgZnVuY3Rpb24gbW9zdF9saWtlbHlfZmFpbHVyZV9mdXp6eShzeXM6Ok1lZGl1bVN5c3RlbSwgz4g7IGQ9NDEsIG09MTAwKQ0KCXDPhCA9IE5vbWluYWxUcmFqZWN0b3J5RGlzdHJpYnV0aW9uKHN5cywgZCkgICAgICAgICAjIFRyYWplY3RvcnkgZGlzdHJpYnV0aW9uDQoJcc+EID0gUGVuZHVsdW1GdXp6aW5nRGlzdHJpYnV0aW9uKDAuMTVeMipJLCBkKSAgICAgICAgICMgVHJhamVjdG9yeSBkaXN0cmlidXRpb24NCgnPhHMgPSBbcm9sbG91dChzeXMsIHHPhDsgZCkgZm9yIF8gaW4gMTptXSAgICAgICAgICAgICMgUm9sbG91dCB3aXRoIHDPhCwgbipkIHN0ZXBzDQoJz4RzX2ZhaWx1cmVzID0gZmlsdGVyKM+ELT5pc2ZhaWx1cmUoz4gsIM+EKSwgz4RzKSAgICAgICAjIEZpbHRlciB0byBnZXQgZmFpbHVyZSB0cmFqcw0KCc+EX21vc3RfbGlrZWx5ID0gYXJnbWF4KM+ELT5sb2dwZGYocM+ELCDPhCksIM+Ec19mYWlsdXJlcykgIyBNb3N0LWxpa2VseSBmYWlsdXJlIHRyYWoNCglyZXR1cm4gz4RfbW9zdF9saWtlbHkNCmVuZA0KDQoNCnN0cnVjdCBDQVNGdXp6aW5nRGlzdHJpYnV0aW9uIDw6IFRyYWplY3RvcnlEaXN0cmlidXRpb24NCiAgICDPg+KCmyAjIGVudmlyb25tZW50IGRpc3R1cmJhbmNlIGNvdmFyaWFuY2UNCiAgICBkICAjIGRlcHRoDQplbmQNCg0KZnVuY3Rpb24gU3RhbmZvcmRBQTIyOFYuaW5pdGlhbF9zdGF0ZV9kaXN0cmlidXRpb24ocDo6Q0FTRnV6emluZ0Rpc3RyaWJ1dGlvbikNCiAgICByZXR1cm4gcHJvZHVjdF9kaXN0cmlidXRpb24oVW5pZm9ybSgtMTAwLCAxMDApLCBVbmlmb3JtKC0xMCwgMTApLCBEaXNjcmV0ZU5vblBhcmFtZXRyaWMoWzBdLCBbMS4wXSksIERpc2NyZXRlTm9uUGFyYW1ldHJpYyhbNDBdLCBbMS4wXSkpDQplbmQNCg0KZnVuY3Rpb24gU3RhbmZvcmRBQTIyOFYuZGlzdHVyYmFuY2VfZGlzdHJpYnV0aW9uKHA6OkNBU0Z1enppbmdEaXN0cmlidXRpb24sIHQpDQogICAgRCA9IERpc3R1cmJhbmNlRGlzdHJpYnV0aW9uKChvKS0+RGV0ZXJtaW5pc3RpYygpLA0KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAocyxhKS0+Tm9ybWFsKDAsIHAuz4PigpspLA0KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAocyktPkRldGVybWluaXN0aWMoKSkNCiAgICByZXR1cm4gRA0KZW5kDQoNClN0YW5mb3JkQUEyMjhWLmRlcHRoKHA6OkNBU0Z1enppbmdEaXN0cmlidXRpb24pID0gcC5k")))
		end"""))
		global UsingThisViolatesTheHonorCode = getfield(@__MODULE__, Symbol(ModuleTA))
	catch err
		@warn err
	end
	most_likely_failure_nelder_mead = 
		UsingThisViolatesTheHonorCode.most_likely_failure_nelder_mead

	most_likely_failure_lbfgs = 
		UsingThisViolatesTheHonorCode.most_likely_failure_lbfgs

	most_likely_failure_sgd = 
		UsingThisViolatesTheHonorCode.most_likely_failure_sgd

	most_likely_failure_fuzzy = 
		UsingThisViolatesTheHonorCode.most_likely_failure_fuzzy

	PendulumFuzzingDistribution = 
		UsingThisViolatesTheHonorCode.PendulumFuzzingDistribution

	CASFuzzingDistribution = 
		UsingThisViolatesTheHonorCode.CASFuzzingDistribution

	run_gd = UsingThisViolatesTheHonorCode.run_gd
end; md"""
# Backend
_Helper functions and project management._
"""

# ╔═╡ c151fc99-af4c-46ae-b55e-f50ba21f1f1c
begin
	start_code() = html"""
	<div class='container'><div class='line'></div><span class='text' style='color:#B1040E'><b><code>&lt;START CODE&gt;</code></b></span><div class='line'></div></div>
	<p> </p>
	<!-- START_CODE -->
	"""

	end_code() = html"""
	<!-- END CODE -->
	<p><div class='container'><div class='line'></div><span class='text' style='color:#B1040E'><b><code>&lt;END CODE&gt;</code></b></span><div class='line'></div></div></p>
	"""

	function hint(text; title="Hint")
		return Markdown.MD(Markdown.Admonition("hint", title, [text]))
	end

	function almost()
		text=md"""
		Please modify the `num_failures` function (currently returning `nothing`, which is the default).

		(Please only submit when this is **green**.)
		"""
		return Markdown.MD(Markdown.Admonition("warning", "Warning!", [text]))
	end

	function keep_working()
		text = md"""
		The answers are not quite right.

		(Please only submit when this is **green**.)
		"""
		return Markdown.MD(Markdown.Admonition("danger", "Keep working on it!", [text]))
	end

	function correct()
		text = md"""
		All tests have passed, you're done with Project 0!

		Please submit `project0.jl` (this file) to Gradescope.
		"""
		return Markdown.MD(Markdown.Admonition("correct", "Tests passed!", [text]))
	end

	function combine_html_md(contents::Vector; return_html=true)
		process(str) = str isa HTML ? str.content : html(str)
		return join(map(process, contents))
	end

	function html_expand(title, content::Markdown.MD)
		return HTML("<details><summary>$title</summary>$(html(content))</details>")
	end

	function html_expand(title, contents::Vector)
		html_code = combine_html_md(contents; return_html=false)
		return HTML("<details><summary>$title</summary>$html_code</details>")
	end

	html_space() = html"<br><br><br><br><br><br><br><br><br><br><br><br><br><br>"
	html_half_space() = html"<br><br><br><br><br><br><br>"
	html_quarter_space() = html"<br><br><br>"

	function set_aspect_ratio!()
		x_range = xlims()[2] - xlims()[1]
		y_range = ylims()[2] - ylims()[1]
		plot!(ratio=x_range/y_range)
	end

	rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])

	global SEED = sum(Int.(collect("AA228V Project 1"))) # Cheeky seed value :)

	DarkModeIndicator() = PlutoUI.HypertextLiteral.@htl("""
		<span>
		<script>
			const span = currentScript.parentElement
			span.value = window.matchMedia('(prefers-color-scheme: dark)').matches
		</script>
		</span>
	""")

	md"> **Helper functions and variables**."
end

# ╔═╡ fe044059-9102-4e7f-9888-d9f03eec69ff
html_expand("Expand for general Julia/Pluto tips.", md"""
1. You can create as many new cells anywhere as you like.
    - **Important**: Please do not modify/delete any existing cells.
2. To put multple lines of code in a single cell in Pluto, wrap with `begin` and `end`:
```julia
begin
	x = 10
	y = x^2
end
```
3. To suppress the Pluto output of a cell, add a semicolon `;` at the end.
```julia
x = 10;
```
or
```julia
begin
	x = 10
	y = x^2
end;
```
""")

# ╔═╡ a46702a3-4a8c-4749-bd00-52f8cce5b8ee
html_half_space()

# ╔═╡ e52ffc4f-947d-468e-9650-b6c67a57a62b
html_quarter_space()

# ╔═╡ a003beb6-6235-455c-943a-e381acd00c0e
start_code()

# ╔═╡ c494bb97-14ef-408c-9de1-ecabe221eea6
end_code()

# ╔═╡ e2418154-4471-406f-b900-97905f5d2f59
html_quarter_space()

# ╔═╡ 18754cc6-c089-4245-ad10-2848594e49b4
html_expand("Expand for useful interface functions.", [
	html"<h2hide>Useful interface functions</h2hide>",
	md"""
	The following functions are provided by `StanfordAA228V.jl` that you may use.
	""",
	html"<h3hide><code>logpdf</code></h3hide>",
	md"""
**`logpdf(p::TrajectoryDistribution, τ::Vector)::Float64`** — Evaluate the log probability density of the trajectory `τ` using the trajectory distribution `p`.
- Use `logpdf` instead of `pdf` for numerical stability.
""",
	html"<h3hide><code>rollout</code></h3hide>",
	md"""
**`rollout(sys::System; d)::Array`** — Run a single rollout of the system `sys` to a depth of `d`.
- `τ` is written as `\tau<TAB>` in code.
```julia
function rollout(sys::System; d=1)
    s = rand(Ps(sys.env))
    τ = []
    for t in 1:d
        o, a, s′ = step(sys, s) # For each rollout call, step is called d times.
        push!(τ, (; s, o, a))
        s = s′
    end
    return τ
end
```
""",
	html"<h3hide><code>isfailure</code></h3hide>",
	md"""
**`isfailure(ψ, τ)::Bool`** — Using the specification `ψ`, check if the trajector `τ` led to a failure.
- `ψ` is written as `\psi<TAB>` in code.
"""])

# ╔═╡ d566993e-587d-4aa3-995b-eb955dec5758
html_expand("Expand for baseline implementation using <code>DirectFalsification</code>.", [
	html"<h2hide>Using <code>DirectFalsification</code> algorithm</h2hide>",
	md"""
We could instead use the `DirectFalsification` algorithm for the small system where instead of using the `NominalTrajectoryDistribution`, we evaluate the pdf directly on the initial state distribution `ps_small`:
```julia
struct DirectFalsification
	d # depth
	m # number of samples
end

function falsify(alg::DirectFalsification, sys, ψ)
	d, m = alg.d, alg.m
	τs = [rollout(sys, d=d) for i in 1:m]
	return filter(τ->isfailure(ψ, τ), τs)
end

alg = DirectFalsification(1, 100)
τ_failures = falsify(alg, sys_small, ψ_small)
ℓτ = maximum(s->pdf(ps_small, s[1].s), τ_failures)
```
**Note**: We use the `NominalTrajectoryDistribution` to keep the algorithm general for the medium/large that _do_ have disturbances.
"""])

# ╔═╡ e888241c-b89f-4db4-ac35-6d826ec4c36c
html_expand("Expand if using optimization-based falsification.", [
	html"<h2hide>Robustness and gradients</h2hide>",
	md"""
Robustness can be a useful metric to find failures. If the robustness is $\le 0$, this indicates a failure.

- To take a gradient of _robustness_ w.r.t. a trajectory `τ`, you can use `ForwardDiff` like so:
```julia
function robustness_gradient(sys, ψ, τ)
	𝐬 = [step.s for step in τ]
	f(x) = robustness_objective(x, sys, ψ)
	return ForwardDiff.gradient(f, 𝐬)
end
```
- For the `robustness_objective` function of:
```julia
function robustness_objective(input, sys, ψ; smoothness=1.0)
	s, 𝐱 = extract(sys.env, input)
	τ = rollout(sys, s, 𝐱)
	𝐬 = [step.s for step in τ]
	return robustness(𝐬, ψ.formula, w=smoothness)
end
```
- You can then evaluate the robustness gradient of a single trajectory like so:
```julia
τ = rollout(sys_small)
robustness_gradient(sys_small, ψ_small, τ)
```
- **However**, your objective is not quite to minimize robustness.
    - **Hint**: You also want to _maximize likelihood_ (i.e., minimize negative likelihood).
""",
	html"<h2hide>Optimization-based falsification</h2hide>",
	md"""
- If you are using **Optim.jl**, the following options may be helpful (especially `f_calls_limit` for gradient free methods, `g_calls_limit` (typically n÷2) for gradient-based methods, and `iterations`): [https://julianlsolvers.github.io/Optim.jl/v0.9.3/user/config/](https://julianlsolvers.github.io/Optim.jl/v0.9.3/user/config/)
    - Optim also requires an initial guess `x0`, you can use the following for each environment (see Example 4.5 in the textbook):
```julia
x0 = initial_guess(sys::SmallSystem)  # SimpleGaussian
x0 = initial_guess(sys::MediumSystem) # InvertedPendulum
x0 = initial_guess(sys::LargeSystem)  # CollisionAvoidance

initial_guess(sys::SmallSystem) = [0.0]
initial_guess(sys::MediumSystem) = zeros(84)
initial_guess(sys::LargeSystem) = [rand(Normal(0,100)), zeros(42)...]
```
- To explain where these numbers came from:
    - `SmallSystem`: the initial guess is $0$ for the only search parameter: the initial state.
    - `MediumSystem`: the initial guess is $d \times |x| + |s_0| = 84$ for $d = 41$, $|x| = 2$ (disturbance on both $\theta$ and $\omega$), and $|s_0| = 2$ for both components of the initial state.
    - `LargeSystem`: the initial guess is $d \times |x| + |\{s_0^{(1)}, s_0^{(2)}\}| = 43$ for $d = 41$, $|x| = 1$ (disturbance is only on the environment), and $|\{s_0^{(1)}, s_0^{(2)}\}| = 2$ for searching only over the $h$ and $\dot{h}$ initial state variables, setting the initial $h$ to $h \sim \mathcal{N}(0, 100)$.
- Or you can write your own optimization algorithm :)
""",
	html"<h2hide>Details on the <code>extract</code> function</h2hide>",
	md"""
- The `extract` function is used to _extract_ the initial state `s` and the set of disturbances `𝐱` (written `\bfx<TAB>`) so that off-the-shelf optimization algorithms (e.g., from Optim.jl) can search over the required variables.
- The `SimpleGaussian` environment only searches over initial states and has no disturbances.
```julia
function extract(env::SimpleGaussian, input)
	s = input[1]             # Objective is simply over the initial state
	𝐱 = [Disturbance(0,0,0)] # No disturbances for the SimpleGaussian
	return s, 𝐱
end
```
- **Note**: We provide the `extract` algorithms for each of the environment types:
```julia
s, 𝐱 = extract(env::SimpleGaussian, input)
s, 𝐱 = extract(env::InvertedPendulum, input)
s, 𝐱 = extract(env::CollisionAvoidance, input)
```
"""])

# ╔═╡ c4fa9af9-1a79-43d7-9e8d-2854652a4ea2
html_expand("Stuck? Expand for hint on what to try.", md"""
$(hint(md"Try fuzzing! See _Example 4.3_ in the textbook. _Other techniques_: optimization or planning."))""")

# ╔═╡ a0a60728-4ee0-4fd0-bd65-c056956b9712
html_expand("Expand if you get an error <code>reducing over an empty collection</code>.", md"""
The following error may occur:
> **ArgumentError**: reducing over an empty collection is not allowed; consider supplying `init` to the reducer

This is usually because there were no failures found and you are trying to iterate over an empty set. Example: `τs_failures` may be equal to `[]`, resulting in the error:
```julia
τ_most_likely = argmax(τ->pdf(pτ, τ), τs_failures)
```

**Potential solution**: Try increasing `m` to sample more rollouts.
""")

# ╔═╡ b0a4461b-89d0-48ee-9bcf-b544b9f08154
html_expand("Expand if you're getting <code>NaN</code> likelihood errors.", md"""
Likelihoods or log-likelihoods equal to `NaN` may be a result of `log(pdf(p, τ))` due to numerical stability issues.

**Instead**, please use `logpdf(p, τ)` instead (better numerical stability).
""")

# ╔═╡ fda151a1-5069-44a8-baa1-d7903bc89797
html_space()

# ╔═╡ d95b0228-71b0-4cae-990e-4bab368c25d9
function plot_pendulum(sys::MediumSystem, ψ, τ=missing;
					   title="Inverted Pendulum", max_lines=100)
	plot(size=(680,350), grid=false)
	plot!(rectangle(2, 1, 0, π/4), opacity=0.5, color="#F5615C", label=false)
	plot!(rectangle(2, 1, 0, -π/4-1), opacity=0.5, color="#F5615C", label=false)
	xlabel!("Time (s)")
	ylabel!("𝜃 (rad)")
	title!(title)
	xlims!(0, 2)
	ylims!(-1.2, 1.2)
	set_aspect_ratio!()

	function plot_pendulum_traj!(τ; lw=2, α=1, color="#009E73")
		X = range(0, step=sys.env.dt, length=length(τ))
		plot!(X, [step.s[1] for step in τ]; lw, color, α, label=false)
	end

	if τ isa Vector{<:Vector}
		# Multiple trajectories
		τ_successes = filter(τᵢ->!isfailure(ψ, τᵢ), τ)
		τ_failures = filter(τᵢ->isfailure(ψ, τᵢ), τ)
		for (i,τᵢ) in enumerate(τ_successes)
			if i > max_lines
				break
			else
				plot_pendulum_traj!(τᵢ; lw=1, α=0.25, color="#009E73")
			end
		end

		for τᵢ in τ_failures
			plot_pendulum_traj!(τᵢ; lw=2, α=1, color="#F5615C")
		end
	elseif τ isa Vector
		# Single trajectory
		get_color(ψ, τ) = isfailure(ψ, τ) ? "#F5615C" : "#009E73"
		plot_pendulum_traj!(τ; lw=2, color=get_color(ψ, τ))
	end

	return plot!()
end

# ╔═╡ 44c8fbe0-21e7-482b-84a9-c3d32a4737dd
plot_pendulum(sys_medium, ψ_medium, τs_rollout_medium; max_lines=100)

# ╔═╡ e12b102e-785b-46e9-980c-e9f7943eda60
plot_pendulum(sys_medium, ψ_medium, τ_base_medium; title="Most-likely failure found")

# ╔═╡ bac5c489-553c-436f-b332-8a8e97126a51
html_quarter_space()

# ╔═╡ 0606d827-9c70-4a79-afa7-14fb6b806546
start_code()

# ╔═╡ 759534ca-b40b-4824-b7ec-3a5c06cbd23e
end_code()

# ╔═╡ 4d2675e1-947c-4cd5-a7e7-49ab6c604577
html_quarter_space()

# ╔═╡ 420e2a64-a96b-4e12-a846-06de7cf0bae1
html_expand("Expand if using optimization-based falsification.", md"""
Note that the number of function calls `f(x)` output by the Optim results when running `display(results)` may be different than the `step_counter`.

This is because Optim counts the number of objective function calls `f` and the objective function may run `rollout` (i.e., mulitple calls to `step` based on depth `d`) multiple times.

This is not applicable for the small problem, as the depth is $d=1$.
""")

# ╔═╡ 60ab8107-db65-4fb6-aeea-d4978aed77bd
html_space()

# ╔═╡ fd8e765e-6c38-47d2-a10f-c3f712607c77
function plot_cas(sys::LargeSystem, ψ, τ=missing; max_lines=100, title="")
	plot(size=(680,350), grid=false, xflip=true)
	xlims!(0, 40)
	ylims!(-400, 400)
	set_aspect_ratio!()
	xlabel!("\$t_\\mathrm{col}\$ (s)")
	ylabel!("\$h\$ (m)")
	title!(title)

	# Collision region
	plot!(rectangle(1, 100, 0, -50), opacity=0.5, color="#F5615C", label=false)

	function plot_cas_traj!(τ; lw=2, α=1, color="#009E73")
		X = reverse(range(0, 41, length=length(τ)))
		plot!(X, [step.s[1] for step in τ]; lw, color, α, label=false)
	end

	if τ isa Vector{<:Vector}
		# Multiple trajectories
		τ_successes = filter(τᵢ->!isfailure(ψ, τᵢ), τ)
		τ_failures = filter(τᵢ->isfailure(ψ, τᵢ), τ)
		for (i,τᵢ) in enumerate(τ_successes)
			if i > max_lines
				break
			else
				plot_cas_traj!(τᵢ; lw=1, α=0.25, color="#009E73")
			end
		end

		for τᵢ in τ_failures
			plot_cas_traj!(τᵢ; lw=2, α=1, color="#F5615C")
		end
	elseif τ isa Vector
		# Single trajectory
		get_color(ψ, τ) = isfailure(ψ, τ) ? "#F5615C" : "#009E73"
		plot_cas_traj!(τ; lw=2, color=get_color(ψ, τ))
	end

	return plot!()
end

# ╔═╡ aa0c4ffc-d7f0-484e-a1e2-7f6f92a3a53d
md"""
# 3️⃣ **Large**: Aircraft Collision Avoidance
The large system is an aircraft collision avoidance system (CAS).
- It uses an interpolated lookup-table policy.
- The state is comprised of the relative altitude (m) $h$, the relative vertical rate $\dot{h}$ (m/s), the previous action $a_\text{prev}$, and the time to closest point of approach $t_\text{col}$ (sec): $s = [h, \dot{h}, a_\text{prev}, t_\text{col}]$
- Actions are $a \in [-5, 0, 5]$ vertical rate changes.
- Disturbances $x$ are applied to $\dot{h}$ as sensor noise: $x \sim \mathcal{N}(0, 1.5)$

> **One rollout has a fixed length of $d=41$ (time from $40-0$ seconds to collision).**

$(plot_cas(sys_large, ψ_large, τ_base_large))
"""

# ╔═╡ 797cbe41-a5f3-4179-9143-9ef6e6888a4d
plot_cas(sys_large, ψ_large, τs_rollout_large)

# ╔═╡ 4ae85f59-4e94-48aa-8ccb-91311466c51f
plot_cas(sys_large, ψ_large, τ_base_large)

# ╔═╡ e3d6fdf1-3a9e-446b-8482-49d6f64b652e
html_quarter_space()

# ╔═╡ 18a70925-3c2a-4317-8bbc-c2a096ec56d0
start_code()

# ╔═╡ 4c5210d6-598f-4167-a6ee-93bceda7223b
end_code()

# ╔═╡ 74aeca7b-0658-427f-8c02-d093a0d725ee
html_space()

# ╔═╡ 5a1ed20d-788b-4655-bdd8-069545f48929
begin
	function extract(env::SimpleGaussian, input)
		s = input[1]             # Objective is simply over the initial state
		𝐱 = [Disturbance(0,0,0)] # No disturbances for the SimpleGaussian
		return s, 𝐱
	end

	function extract(env::InvertedPendulum, x)
		s = x[1:2]
		𝐱 = [Disturbance(0, 0, x[i:i+1]) for i in 3:2:length(x)]
		return s, 𝐱
	end

	function extract(env::CollisionAvoidance, x)
		s = [x[1], x[2], 0, 40] # [h, ḣ, a_prev, t_col]
		𝐱 = [Disturbance(0, x[i], 0) for i in 3:length(x)]
		return s, 𝐱
	end

	initial_guess(sys::SmallSystem) = [0.0]
	initial_guess(sys::MediumSystem) = zeros(84)
	initial_guess(sys::LargeSystem) = [rand(Normal(0,100)), zeros(42)...]

	md"> **Helper `extract` and `initial_guess` functions**."
end

# ╔═╡ a6931d1e-08ad-4592-a54c-fd76cdc51294
@bind dark_mode DarkModeIndicator()

# ╔═╡ 381240d5-32b2-4d68-9c7f-18aca1cff0a9
function plot_gaussian(sys, ψ, τ=missing; is_dark_mode=dark_mode, max_points=100)
	ps = Ps(sys.env)
	
	if is_dark_mode
		plot(
			bg="transparent",
			background_color_inside="black",
			bglegend="black",
			fg="white",
			gridalpha=0.5,
		)
	else
		plot()
	end

	# Create a range of x values
	_X = range(-4, 4, length=1000)
	_Y = pdf.(ps, _X)

	# Plot the Gaussian density
	plot!(_X, _Y,
	     xlim=(-4, 4),
	     ylim=(-0.001, 0.41),
	     linecolor=is_dark_mode ? "white" : "black",
		 fillcolor=is_dark_mode ? "darkgray" : "lightgray",
		 fill=true,
	     xlabel="state \$s\$",
	     ylabel="density \$p(s)\$",
	     size=(600, 300),
	     label=false)

	# Identify the indices where x <= -2
	idx = _X .<= ψ.formula.ϕ.c

	# Extract the x and y values for the region to fill
	x_fill = _X[idx]
	y_fill = _Y[idx]

	# Create the coordinates for the filled polygon
	# Start with the x and y values where x <= -2
	# Then add the same x values in reverse with y = 0 to close the polygon
	polygon_x = vcat(x_fill, reverse(x_fill))
	polygon_y = vcat(y_fill, zeros(length(y_fill)))

	# Add the filled area to the plot
	plot!(polygon_x, polygon_y,
	      fill=true,
	      fillcolor="crimson",
	      linecolor="transparent", # No border for the filled area
		  alpha=0.5,
	      label=false)

	# Draw failure threshold
	vline!([ψ.formula.ϕ.c], color="crimson", label="Failure threshold")

	if !ismissing(τ)
		count_plotted_succeses = 0
		count_plotted_failures = 0
		function plot_point!(τᵢ)
			if isfailure(ψ, τᵢ) && count_plotted_failures == 0
				label = "Failure state"
				count_plotted_failures += 1
			elseif !isfailure(ψ, τᵢ) && count_plotted_succeses == 0
				label = "Succes state"
				count_plotted_succeses += 1
			else
				label = false
			end
			color = isfailure(ψ, τᵢ) ? "black" : "#009E73"
			τₓ = τᵢ[1].s[1]
			scatter!([τₓ], [pdf(ps, τₓ)], color=color, msc="white", m=:circle, label=label)
		end

		if τ isa Vector{<:Vector}
			# Multiple rollouts
			for (i,τᵢ) in enumerate(τ)
				i > max_points && break
				plot_point!(τᵢ)
			end
		elseif τ isa Vector
			# Single rollout
			plot_point!(τ)
		end
	end

	return plot!()
end

# ╔═╡ e86d260f-c93d-4561-a9f1-44e4c7af827e
plot_gaussian(sys_small, ψ_small)

# ╔═╡ d4d057d7-cc9d-4949-9e3f-44a8aa67d725
plot_gaussian(sys_small, ψ_small, τ_baseline_small)

# ╔═╡ fe7f4a79-1a63-4272-a776-358a309c8550
begin
	Random.seed!(4)
	τs_baseline_small = [rollout(sys_small, p_τ_small) for i in 1:100]
	plot_gaussian(sys_small, ψ_small, τs_baseline_small)
end

# ╔═╡ ef084fea-bf4d-48d9-9c84-8cc1dd98f2d7
TableOfContents()

# ╔═╡ 97042a5e-9691-493f-802e-2262f2da4627
html"""
<style>
	h3 {
		border-bottom: 1px dotted var(--rule-color);
	}

	summary {
		font-weight: 500;
		font-style: italic;
	}

	.container {
      display: flex;
      align-items: center;
      width: 100%;
      margin: 1px 0;
    }

    .line {
      flex: 1;
      height: 2px;
      background-color: #B83A4B;
    }

    .text {
      margin: 0 5px;
      white-space: nowrap; /* Prevents text from wrapping */
    }

	h2hide {
		border-bottom: 2px dotted var(--rule-color);
		font-size: 1.8rem;
		font-weight: 700;
		margin-bottom: 0.5rem;
		margin-block-start: calc(2rem - var(--pluto-cell-spacing));
	    font-feature-settings: "lnum", "pnum";
	    color: var(--pluto-output-h-color);
	    font-family: Vollkorn, Palatino, Georgia, serif;
	    line-height: 1.25em;
	    margin-block-end: 0;
	    display: block;
	    margin-inline-start: 0px;
	    margin-inline-end: 0px;
	    unicode-bidi: isolate;
	}
	h3hide {
	    border-bottom: 1px dotted var(--rule-color);
		font-size: 1.6rem;
		font-weight: 600;
		color: var(--pluto-output-h-color);
	    font-feature-settings: "lnum", "pnum";
		font-family: Vollkorn, Palatino, Georgia, serif;
	    line-height: 1.25em;
		margin-block-start: 0;
	    margin-block-end: 0;
	    display: block;
	    margin-inline-start: 0px;
	    margin-inline-end: 0px;
	    unicode-bidi: isolate;
	}
</style>"""

# ╔═╡ Cell order:
# ╟─60f72d30-ab80-11ef-3c20-270dbcdf0cc4
# ╟─fe044059-9102-4e7f-9888-d9f03eec69ff
# ╟─a46702a3-4a8c-4749-bd00-52f8cce5b8ee
# ╟─fd8c851a-3a42-41c5-b0fd-a12085543c9b
# ╟─17fa8557-9656-4347-9d44-213fd3b635a6
# ╠═22feee3d-4627-4358-9937-3c780b7e8bcb
# ╠═6f3e24de-094c-49dc-b892-6721b3cc54ed
# ╟─45f7c3a5-5763-43db-aba8-41ef8db39a53
# ╠═9c1daa96-76b2-4a6f-8d0e-f95d26168d2b
# ╟─370a15eb-df4b-493a-af77-00914b4616ea
# ╠═ab4c6807-5b4e-4688-b794-159e26a1599b
# ╟─381240d5-32b2-4d68-9c7f-18aca1cff0a9
# ╠═e86d260f-c93d-4561-a9f1-44e4c7af827e
# ╟─166bd412-d433-4dc9-b874-7359108c0a8b
# ╟─9132a200-f63b-444b-9830-b03cf075021b
# ╟─99eb3a5f-c6d4-48b6-8e96-0adbd123b160
# ╠═c2ae204e-dbcc-453a-81f5-791ba4be39db
# ╟─e73635cc-2b1e-4162-8760-b62184e70b6d
# ╠═7fe03702-25e5-473a-a92b-3b77eb753bc3
# ╟─73da2a56-8991-4484-bcde-7d397214e552
# ╠═d4d057d7-cc9d-4949-9e3f-44a8aa67d725
# ╟─a6603deb-57fa-403e-a2e5-1195ae7c016c
# ╠═fe7f4a79-1a63-4272-a776-358a309c8550
# ╟─e52ffc4f-947d-468e-9650-b6c67a57a62b
# ╟─92f20cc7-8bc0-4aea-8c70-b0f759748fbf
# ╟─a003beb6-6235-455c-943a-e381acd00c0e
# ╟─f6589984-e24d-4aee-b7e7-db159ae7fea6
# ╠═fc2d34da-258c-4460-a0a4-c70b072f91ca
# ╟─c494bb97-14ef-408c-9de1-ecabe221eea6
# ╟─e2418154-4471-406f-b900-97905f5d2f59
# ╟─ec776b30-6a30-4643-a22c-e071a365d50b
# ╟─18754cc6-c089-4245-ad10-2848594e49b4
# ╟─d566993e-587d-4aa3-995b-eb955dec5758
# ╟─e888241c-b89f-4db4-ac35-6d826ec4c36c
# ╟─c4fa9af9-1a79-43d7-9e8d-2854652a4ea2
# ╟─dba42df0-3199-4c31-a735-b6b514703d50
# ╟─a0a60728-4ee0-4fd0-bd65-c056956b9712
# ╟─b0a4461b-89d0-48ee-9bcf-b544b9f08154
# ╟─fda151a1-5069-44a8-baa1-d7903bc89797
# ╟─8c78529c-1e00-472c-bb76-d984b37235ab
# ╟─daada216-11d4-4f8b-807c-d347130a3928
# ╟─d18c2105-c2af-4dda-8388-617aa816a567
# ╠═77637b5e-e3ce-4ecd-90fc-95611af18002
# ╠═c4c0328d-8cb3-41d5-9740-0197cbf760c2
# ╟─b1e9bd40-a401-4630-9a1f-d61b276e72f7
# ╠═fe272c1b-421c-49de-a513-80c7bcefdd9b
# ╟─a16cf110-4afa-4792-9d3f-f13b24349886
# ╠═8b82eb8d-f6fe-4b73-8617-8c75dd65b769
# ╟─d95b0228-71b0-4cae-990e-4bab368c25d9
# ╠═44c8fbe0-21e7-482b-84a9-c3d32a4737dd
# ╟─bdb27ba8-782c-467c-818d-f68c7790e845
# ╠═3d00dc65-4c48-4988-9bb9-4cd3af6b9c5b
# ╠═e12b102e-785b-46e9-980c-e9f7943eda60
# ╟─7ef66a50-6acc-474f-b406-7b27a7b18510
# ╟─bac5c489-553c-436f-b332-8a8e97126a51
# ╟─1da9695f-b7fc-46eb-9ef9-12160246018d
# ╟─0606d827-9c70-4a79-afa7-14fb6b806546
# ╟─9657f5ff-f21c-43c5-838d-402a2a723d5e
# ╠═cb7b9b9f-59da-4851-ab13-c451c26117df
# ╟─759534ca-b40b-4824-b7ec-3a5c06cbd23e
# ╟─4d2675e1-947c-4cd5-a7e7-49ab6c604577
# ╟─4943ca08-157c-40e1-acfd-bd9326082f56
# ╟─420e2a64-a96b-4e12-a846-06de7cf0bae1
# ╟─60ab8107-db65-4fb6-aeea-d4978aed77bd
# ╟─aa0c4ffc-d7f0-484e-a1e2-7f6f92a3a53d
# ╟─7d054465-9f80-4dfb-9b5f-76c3977de7cd
# ╠═1ec68a39-8de9-4fd3-be8a-26cf7706d1d6
# ╟─d23f0299-981c-43b9-88f3-fb6e07927498
# ╠═641b92a3-8ff2-4aed-8482-9fa686803b68
# ╟─be426908-3fee-4ecd-b054-2497ce9a2e50
# ╠═258e14c4-9a2d-4515-9a8f-8cd96f31a6ff
# ╠═1a097a88-e4f0-4a8d-a5d6-2e3858ee417c
# ╟─fd8e765e-6c38-47d2-a10f-c3f712607c77
# ╠═797cbe41-a5f3-4179-9143-9ef6e6888a4d
# ╟─a4e0000b-4b4a-4262-bf0a-85509c4ee47e
# ╠═b5d02715-b7c9-4bf2-a284-42da40a70a68
# ╠═4ae85f59-4e94-48aa-8ccb-91311466c51f
# ╟─204feed7-cde8-40a8-b6b5-051a1c768fd9
# ╟─e3d6fdf1-3a9e-446b-8482-49d6f64b652e
# ╟─23fd490a-74d2-44b4-8a12-ea1460d95f85
# ╟─18a70925-3c2a-4317-8bbc-c2a096ec56d0
# ╠═3471a623-16af-481a-8f66-5bd1e7890188
# ╟─4c5210d6-598f-4167-a6ee-93bceda7223b
# ╟─74aeca7b-0658-427f-8c02-d093a0d725ee
# ╟─2827a6f3-47b6-4e6f-b6ae-63271715d1f3
# ╠═83884eb4-6718-455c-b731-342471325326
# ╟─4a91853f-9685-47f3-998a-8e0cfce688f8
# ╟─2ff6bb9c-5282-4ba1-b62e-a9fd0fe1969c
# ╠═b6f15d9c-33b8-40e3-be57-d91eda1c9753
# ╟─522bb285-bc06-4c92-82ee-c0d0f68b184c
# ╟─089581ec-8aff-4c56-9a65-26d394d5eec3
# ╠═3314f402-10cc-434c-acbc-d38e59e4b846
# ╟─d72be566-6ad7-4817-8590-a504a699a4da
# ╠═712e69bf-48e7-47e9-a14e-25cce64d4ae4
# ╟─cee165f0-049f-4ea3-8f19-04e66947a397
# ╠═6302729f-b34a-4a18-921b-d194fe834208
# ╟─95e3d42f-b33f-4294-81c5-f34a300dc9b4
# ╟─ba6c082b-6e62-42fc-a85c-c8b7efc89b88
# ╟─173388ab-207a-42a6-b364-b2c1cb335f6b
# ╟─c151fc99-af4c-46ae-b55e-f50ba21f1f1c
# ╟─5a1ed20d-788b-4655-bdd8-069545f48929
# ╠═a6931d1e-08ad-4592-a54c-fd76cdc51294
# ╠═ef084fea-bf4d-48d9-9c84-8cc1dd98f2d7
# ╟─97042a5e-9691-493f-802e-2262f2da4627
