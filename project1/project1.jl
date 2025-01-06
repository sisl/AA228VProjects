### A Pluto.jl notebook ###
# v0.20.4

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

# ╔═╡ 14964632-98d8-4a2f-b2f6-e3f28b558803
# ╠═╡ show_logs = false
using StanfordAA228V

# ╔═╡ 173388ab-207a-42a6-b364-b2c1cb335f6b
# ╠═╡ show_logs = false
begin
	import MarkdownLiteral: @mdx

	using ProgressLogging
	using Test
	using Base64
	using PlutoUI
	using Distributions
	using Random
	using Plots
	using ReverseDiff
	using Optim
	using Parameters
	using BSON
	using GridInterpolations
	using LinearAlgebra

	default(fontfamily="Computer Modern", framestyle=:box) # LaTeX-style plotting

	md"> _Additional package management._"
end

# ╔═╡ 117d0059-ce1a-497e-8667-a0c2ef20c632
md"""
# Project 1: Finding the most-likely failure
_Please wait until the entire notebook is finished loading before proceeding (you may get temporary errors)._
"""

# ╔═╡ f8612f4e-8519-4ee3-ac09-80b7c126b238
highlight(md"""_See the three **"⟶ Task"** sections below for where to fill out the algorithms._""")

# ╔═╡ da5b4000-0bce-4fc2-be85-dada21264ca3
textbook_details([
	"Chapter 4. _Falsification through Optimization_",
	"Chapter 5. _Falsification through Planning_"])

# ╔═╡ 0456a732-2672-4108-a241-db9ae879a913


# ╔═╡ 6e8ab7c9-fb49-4d89-946d-c7d7588c199a
md"""
## Julia/Pluto tips
Useful tips you may be interested in regarding Julia and Pluto.
"""

# ╔═╡ fe044059-9102-4e7f-9888-d9f03eec69ff
html_expand("Expand for general Julia/Pluto tips.", [
	html"<h2hide>Julia packages</h2hide>",
	md"""
	Feel free to use as many external Julia packages as you like. Running `using PackageName` will automatically install it in this notebook.

	[List of Julia packages.](https://juliapackages.com/)""",
	html"<h2hide>Dependent cells</h2hide>",
	md"""
	Unlike Jupyter notebooks, Pluto has _dependent cells_. Meaning, if one cell uses the variable `x` and another cell defines `x`, then the first cell will _re-run_ when `x` is changed. This means there is no "hidden global state".

	See the [Pluto README](https://github.com/fonsp/Pluto.jl?tab=readme-ov-file) for details/examples.""",
	html"<h2hide>New cells</h2hide>",
	md"""
You can create as many new cells anywhere as you like. Click the `+` icon on the right or hit `CTRL+Enter` within an existing cell to create a new one below it.
- **Important**: Please do not modify/delete any existing cells.""",
	html"<h2hide>Running code</h2hide>",
	md"""
After editing a cell, you can run it several ways:
1. To run the current cell: `<SHIFT+ENTER>`
1. To run the current cell and create a new one below: `<CTRL+S>` or `<CMD+S>`
1. To run all cells with unsaved changes: `<CTRL+S>` or `<CMD+S>`
	""",
	html"<h2hide>Multiple lines of code in a cell</h2hide>",
	md"""
To put multple lines of code in a single cell in Pluto, wrap with `begin` and `end`:
```julia
begin
	x = 10
	y = x^2
end
```""",
	html"<h2hide>Locally scoped cells</h2hide>",
	md"""
Use Julia's `let` block to create locally scoped variables:
```julia
ℓ = let d = get_depth(sys)
	p = NominalTrajectoryDistribution(sys, d)
	τ = rollout(sys, d)
	logpdf(p, τ)
end
```
The last line of code in the `let` block will be returned and assigned to the globally scoped `ℓ` variable in this case.

This way, you can reuse variable names such as `τ` without affecting other cells that may also use that name in global scope.

You could also just define a new function:
```julia
function my_test(sys)
	d = get_depth(sys)
	p = NominalTrajectoryDistribution(sys, d)
	τ = rollout(sys, d)
	return logpdf(p, τ)
end

ℓ = my_test(sys)
```
	""",
	html"<h2hide>Suppress cell output</h2hide>",
	md"""
To suppress the Pluto output of a cell, add a semicolon `;` at the end.
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
""",
	html"<h2hide>Underscore as digit separator</h2hide>",
	md"""
You can use the underscore `_` as a convenient digit separator:
```julia
1000000 == 1_000_000 # true
100000 == 100_000 # true
10000 == 10_000 # true
1000 == 1_000 # true
```
[Link to Julia docs](https://docs.julialang.org/en/v1/manual/integers-and-floating-point-numbers/#:~:text=The%20underscore).
""",
	html"<h2hide>Unicode symbols</h2hide>",
	md"""
You can use Unicode—and even emojis 🙃—as variable and function names. Here are some common ones we use throughout this course:

| Unicode | Code |
|:-------:|:----:|
| `τ` | `\tab` |
| `ψ` | `\psi` |
| `ℓ` | `\ell` |
| `π` | `\pi` |
| `σ` | `\sigma` |
| `Σ` | `\Sigma` |
| `θ` | `\theta` |
| `ω` | `\omega` |
| `²` | `\^2` |
| `₂` | `\_2` |
| `🍕` | `\:pizza:` |

To enter them into cells, type the above "**Code**" and hit `<TAB><TAB>` (or `<TAB><ENTER>`). Feel free to use any Unicode/emojis to your hearts desire.

See the Julia docs for more examples: [https://docs.julialang.org/en/v1/manual/unicode-input/](https://docs.julialang.org/en/v1/manual/unicode-input/)
"""
])

# ╔═╡ a21612a1-1092-4892-9132-629833e7c867


# ╔═╡ ec776b30-6a30-4643-a22c-e071a365d50b
md"""
## Hints
Expand the sections below for some helpful hints.
"""

# ╔═╡ 18754cc6-c089-4245-ad10-2848594e49b4
html_expand("Expand for useful interface functions.", [
	html"<h2hide>Useful interface functions</h2hide>",
	md"""
	The following functions are provided by `StanfordAA228V.jl` that you may use.
	""",
	html"<h3hide><code>NominalTrajectoryDistribution</code></h3hide>",
	md"""
**`NominalTrajectoryDistribution(sys::System, d::Int)::TrajectoryDistribution`** — Returns the nominal trajectory distribution for the system `sys` over depth `d`.
- Use this to evaluate the likelihood of the trajectory (using `logpdf` below).
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

# ╔═╡ e888241c-b89f-4db4-ac35-6d826ec4c36c
html_expand("Expand if using optimization-based falsification.", [
	html"<h2hide>Robustness and gradients</h2hide>",
	md"""
Robustness can be a useful metric to find failures. If the robustness is $\le 0$, this indicates a failure.

- To take a gradient of _robustness_ w.r.t. a trajectory `τ`, you can use `ReverseDiff` like so (where we already load the `ReverseDiff` package for you):
```julia
function robustness_gradient(sys, ψ, τ)
	𝐬 = [step.s for step in τ]
	f(x) = robustness_objective(x, sys, ψ)
	return ReverseDiff.gradient(f, 𝐬)
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
initial_guess(sys::LargeSystem) = [rand(Normal(0,100)), 0 0, 40, zeros(39)...]
```
- To explain where these numbers came from:
    - `SmallSystem`: the initial guess is $0$ for the only search parameter: the initial state.
    - `MediumSystem`: the initial guess is $d \times |x| + |s_0| = 84$ for $d = 41$, $|x| = 2$ (disturbance on both $\theta$ and $\omega$), and $|s_0| = 2$ for both components of the initial state.
    - `LargeSystem`: the initial guess is $d \times |x| + |\{s_0^{(1)}, s_0^{(2)}\}| = 43$ for $d = 41$, $|x| = 1$ (disturbance is only on the environment), and $|\{s_0^{(1)}, s_0^{(2)}\}| = 2$ for searching only over the $h$ and $\dot{h}$ initial state variables, setting the initial $h$ to $h \sim \mathcal{N}(0, 100)$, the initial $t_\text{col}$ to $40$ and the other initial state variables and initial disturbances to zero.
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
""",
	html"<h2hide>Differing <code>step</code> calls using Optim.jl</h2hide>",
	md"""
Note that the number of function calls `f(x)` output by the Optim results when running `display(results)` may be different than the `stepcount()`.

This is because Optim counts the number of objective function calls `f` and the objective function may run `rollout` (i.e., mulitple calls to `step` based on depth `d`) multiple times.

This is not applicable for the small problem, as the depth is $d=1$."""
])

# ╔═╡ c4fa9af9-1a79-43d7-9e8d-2854652a4ea2
html_expand("Stuck? Expand for hints on what to try.", md"""
$(hint(md"Try fuzzing! See _Example 4.3_ in the textbook.

_Other techniques_: optimization or planning (or something entirely different!?)"))""")

# ╔═╡ 6bad6e8b-c021-41d2-afbb-bcd0242138dd


# ╔═╡ dba42df0-3199-4c31-a735-b6b514703d50
md"""
## Common errors
These are some common errors you may run into.
"""

# ╔═╡ a0a60728-4ee0-4fd0-bd65-c056956b9712
html_expand("Expand if you get an error <code>reducing over an empty collection</code>.", md"""
The following error may occur:
> **ArgumentError**: reducing over an empty collection is not allowed; consider supplying `init` to the reducer

This is usually because there were no failures found and you are trying to iterate over an empty set. Example: `τs_failures` may be equal to `[]`, resulting in the error:
```julia
τ_most_likely = argmax(τ->logpdf(pτ, τ), τs_failures)
```

**Potential solution**: Try increasing `m` to sample more rollouts.
""")

# ╔═╡ b0a4461b-89d0-48ee-9bcf-b544b9f08154
html_expand("Expand if you're getting <code>NaN</code> likelihood errors.", md"""
Likelihoods or log-likelihoods equal to `NaN` may be a result of `log(pdf(p, τ))` due to numerical stability issues.

**Instead**, please use `logpdf(p, τ)` instead (better numerical stability).
""")

# ╔═╡ 109c3d27-2c23-48a7-9fd7-be8a1f359e55
html_expand("Expand if you're using <code>Normal</code> and/or <code>MvNormal</code>.", md"""
The univariate normal (Gaussian) distribution in Julia takes in a mean $\mu$ and standard deviation $\sigma$:
```julia
Normal(μ, σ)
```

Where the **multivariate** normal distribution takes in a mean vector $\mathbf{\mu}$ and the _covariance_ matrix $\mathbf{\Sigma}$:
```julia
MvNormal(𝛍, 𝚺)
```

Meaning, if you want a 2d diagonal multivariate normal with mean zero and standard deviation of $\sigma = 0.1$, then you can do:
```julia
MvNormal(zeros(2), 0.1^2*I)
```
where "`I`" comes from the `LinearAlgebra` module (already loaded for you).
""")

# ╔═╡ bc2f62f5-1330-46cd-bb81-411baa483488
html_expand("Expand if you're using <code>initial_state_distribution</code>, <code>disturbance_distribution</code>, or <code>depth</code>.", md"""
The `StanfordAA228V` module defines several functions that you **might be adding new methods to (i.e., new type signatures)**.
- `initial_state_distribution`
- `disturbance_distribution`
- `depth`

Say you're implementing fuzzing and you define a new `TrajectoryDistribution` type and want to create your own version of `disturbance_distribution` for your new type:
```julia
struct NewTrajectoryDistribution <: TrajectoryDistribution
	some_parameter
end
```
Then you will need to use the Julia dot notation to add a method to `StanfordAA228V`:
```julia
function StanfordAA228V.disturbance_distribution(p::NewTrajectoryDistribution)
	# some code
end
```
This code will add a new method for `disturbance_distribution` with the input type of `NewTrajectoryDistribution`. Make sure to add the `StanfordAA228V.` to the function names that you create which are _also_ defined in `StanfordAA228V`:
- See the [`StanfordAA228V.jl`](https://github.com/sisl/StanfordAA228V.jl/blob/d2357ba8cdaf680b207a261495d785456981c66d/src/StanfordAA228V.jl#L39-L41) file.

This is common in Julia where you need to use the funciton name qualified with the module name. Read more in the ["Namespace Management" section of the Julia docs.](https://docs.julialang.org/en/v1/manual/modules/#namespace-management)
""")

# ╔═╡ a46702a3-4a8c-4749-bd00-52f8cce5b8ee
html_half_space()

# ╔═╡ 17fa8557-9656-4347-9d44-213fd3b635a6
Markdown.parse("""
## Small system
The system is comprised of an `agent`, environment (`env`), and `sensor`.
""")

# ╔═╡ 22feee3d-4627-4358-9937-3c780b7e8bcb
sys_small = System(NoAgent(), SimpleGaussian(), IdealSensor());

# ╔═╡ fd8c851a-3a42-41c5-b0fd-a12085543c9b
Markdown.MD(
	md"""
	# 1️⃣ **Small**: 1D Gaussian
	The small system is a simple 1D Gaussian system.
	- There are no dynamics (rollout depth $d=1$).
	- There are no disturbances.
	- The (initial and only) state $s$ is sampled from $\mathcal{N}(0,1)$.
	""",
	depth_highlight(sys_small)
)

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
A failure is highly unlikely given that the probability of failure is:

\$\$P(\\neg\\psi(\\tau)) = P(s < $(ψ_small.formula.ϕ.c)) \\approx $(round(cdf(ps_small, ψ_small.formula.ϕ.c), sigdigits=4))\$\$
where \$\\neg\\psi(\\tau)\$ indicates that the specification was violated and \$P\$ is the _cumulative distribution function_ (`cdf` in Julia).
""")

# ╔═╡ 9132a200-f63b-444b-9830-b03cf075021b
md"""
## Random baseline
The following function is a baseline random falsification algorithm that returns the trajectory that led to the most-likely failure.
"""

# ╔═╡ bb3b33e6-fd05-4631-b0bd-c71ef3dbee38
n_baseline_small = 100

# ╔═╡ cc11217f-e070-4d20-8ebe-18e7eb977487
highlight(md"""**Note**: You can access the number of `step` calls via `stepcount()`""")

# ╔═╡ a6603deb-57fa-403e-a2e5-1195ae7c016c
md"""
Here we plot $100$ states showing which ones were _successes_ and which ones were _failures_.

_(Note that since trajectories for the small problem have depth $1$, each trajectory holds a single state, so here you can think about each of the points below as individual trajectories)._
"""

# ╔═╡ e52ffc4f-947d-468e-9650-b6c67a57a62b
html_quarter_space()

# ╔═╡ 92f20cc7-8bc0-4aea-8c70-b0f759748fbf
Markdown.parse("""
## ⟶ **Task (Small)**: Most-likely failure
Please fill in the following `most_likely_failure` function.
""")

# ╔═╡ a003beb6-6235-455c-943a-e381acd00c0e
start_code()

# ╔═╡ c494bb97-14ef-408c-9de1-ecabe221eea6
end_code()

# ╔═╡ e2418154-4471-406f-b900-97905f5d2f59
html_quarter_space()

# ╔═╡ 1789c8b5-b314-4aba-ad44-555be9a85984
md"""
# 📊 Small Tests
We'll automatically test your `most_likely_failure(::SmallSystem, ψ)` function below.

**Note**: The next three tests are _only_ local validation tests.

_The **graded** tests to be submitted to Gradescope are located [below](#graded-test)._
"""

# ╔═╡ 535261e3-4cb3-4b0b-954d-7452b2a91b5d
md"""
## Different failure threshold
Let's test a different failure threshold.
"""

# ╔═╡ 052cc2e3-ca8a-4043-9a7d-7947a7f1fd0c
md"""
## Random failure threshold
In most cases, we don't know the _failure distribution_. If we did, we could just sample from it!

In this test, we make sure that your algorithm is robust to random failure thresholds.
"""

# ╔═╡ ce99f0cc-5fe8-42c2-af78-ac7211b6b699
@bind rerun_rand_small Button("Click to rerun random test.")

# ╔═╡ 7910c15c-a231-4a0f-a4ed-1fe0b52f62c7
@bind γ Slider(-3:0.1:3, default=0, show_value=true)

# ╔═╡ cbc3a060-b4ec-4572-914c-e07880dd3537
md"""
_You can also click the slider then use the arrow keys for finer control._
"""

# ╔═╡ fda151a1-5069-44a8-baa1-d7903bc89797
html_space()

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

# ╔═╡ 8c78529c-1e00-472c-bb76-d984b37235ab
Markdown.MD(
	md"""
	# 2️⃣ **Medium**: Inverted Pendulum
	The medium system is a swinging inverted pendulum.
	- It uses a proportional controller to keep it upright.
	- The state is comprised of the angle $\theta$ and angular velocity $\omega$ making $s = [\theta, \omega]$
	- Actions are left/right adjustments in the range $[-2, 2]$
	- Disturbances $x$ are treated as addative noise: $x \sim \mathcal{N}(\mathbf{0}, 0.1^2I)$
	""",
	depth_highlight(sys_medium)
)

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

# ╔═╡ f005da72-d7b5-4f01-8882-ed4e2bdcf4bd
n_baseline_medium = 41_000

# ╔═╡ d75f34d3-384c-486b-b648-61ef8fd52167
Markdown.parse("""
**Large likelihood values.** \$\\,\$ _It's perfectly normal for the likelihood to be extremely large, \$\\exp(\\ell) \\gg 1\$, this is because we're dealing with probablity **density** functions which will **integrate** to one. Don't be alarmed._

_This is particularly apparent when the distribution has **small variance**. Here's an example at \$x = 0\$:_
```julia
 pdf(Normal(0, 1e-15), 0) # $(round(pdf(Normal(0, 1e-15), 0), sigdigits=3))
```

*For the inverted pendulum, the `AdditiveNoiseSensor` is a multivariate Gaussian with mean zeros and diagonal standard deviation of \$\\sigma = \\mathit{0.1}\$. Say the disturbances were \$\\mathit{x_t = [\\!0,0]}\$ for all \$d = \\mathit{41}\$ steps, the trajectory likelihood would be:*

\$\$\\prod_{t=1}^{41} p(x_t) \\quad \\text{where} \\quad x_t \\sim \\mathcal{N}(\\mathbf{0}, 0.1^2I)\$\$

*For the disturbances \$\\mathbf{x} = \\big\\{[0,0], \\ldots, [0,0]\\big\\}\$ (the most-likely values of \$x_t\$), we get the following likelihood:*

```julia
 prod(pdf(MvNormal(zeros(2), 0.1^2*I), [0,0]) for t in 1:41) # $(round(prod(pdf(MvNormal(zeros(2), 0.1^2*I), [0,0]) for t in 1:41), sigdigits=3))
```

_Note that this ignores the initial state distribution._

_But we tend to work with **log likelihoods** to avoid numerical issues with these large magnitude numbers._

\$\$\\sum_{t=1}^{41} \\log p(x_t) \\quad \\text{where} \\quad x_t \\sim \\mathcal{N}(\\mathbf{0}, 0.1^2I)\$\$

```julia
 sum(logpdf(MvNormal(zeros(2), 0.1^2*I), [0,0]) for t in 1:41) # $(round(sum(logpdf(MvNormal(zeros(2), 0.1^2*I), [0,0]) for t in 1:41), digits=3))
```
""")

# ╔═╡ bac5c489-553c-436f-b332-8a8e97126a51
html_quarter_space()

# ╔═╡ 1da9695f-b7fc-46eb-9ef9-12160246018d
Markdown.parse("""
## ⟶ **Task (Medium)**: Most-likely failure
Please fill in the following `most_likely_failure` function.
""")

# ╔═╡ 0606d827-9c70-4a79-afa7-14fb6b806546
start_code()

# ╔═╡ 759534ca-b40b-4824-b7ec-3a5c06cbd23e
end_code()

# ╔═╡ 7987c20d-68e8-441b-bddc-3f0ae7c3591d
html_quarter_space()

# ╔═╡ da2d692a-8378-435e-bd6b-c0e65caef542
md"""
# 📊 Medium Test
We'll automatically test your `most_likely_failure(::MediumSystem, ψ)` function below.
"""

# ╔═╡ 60ab8107-db65-4fb6-aeea-d4978aed77bd
html_space()

# ╔═╡ 7d054465-9f80-4dfb-9b5f-76c3977de7cd
Markdown.parse("""
## Large system
An aircraft collision avoidance system that uses an interpolated lookup-table policy.
""")

# ╔═╡ 1ec68a39-8de9-4fd3-be8a-26cf7706d1d6
begin
	local grid, Q = load_cas_policy(joinpath(@__DIR__, "cas_policy.bson"))

	cas_agent = InterpAgent(grid, Q)
	cas_env = CollisionAvoidance(Ds=Normal(0, 1.5))
	cas_sensor = IdealSensor()
	sys_large = System(cas_agent, cas_env, cas_sensor)

	LargeSystem::Type = typeof(sys_large) # Type used for multiple dispatch
end

# ╔═╡ 9f739929-1cd3-4935-b229-ae3aeac7e131
begin
	global ThisProject = Project1
	max_steps(sys::SmallSystem)  = 20
	max_steps(sys::MediumSystem) = 1_000
	max_steps(sys::LargeSystem)  = 10_000
end;

# ╔═╡ 60f72d30-ab80-11ef-3c20-270dbcdf0cc4
begin
	function try_max_steps(sys)
		try
			return "\$n = $(format(max_steps(sys); latex=true))\$"
		catch

			return "**LOADING...**"
		end
	end

	Markdown.parse("""
	**Task**: Efficiently find likely failures using \$n\$ total function calls to the system `step` function.
	- **Small system**: 1D Gaussian \$\\mathcal{N}(0,1)\$. With $(try_max_steps(sys_small)) `step` calls.
	- **Medium system**: Swinging inverted pendulum. With $(try_max_steps(sys_medium)) `step` calls.
	- **Large system**: Aircraft collision avoidance system (CAS). With $(try_max_steps(sys_large)) `step` calls.
	
	Your job is to write the following function that returns the failure trajectory `τ` (i.e., a `Vector` of \$(s,a,o,x)\$ tuples) with the highest likelihood you found:
	```julia
	most_likely_failure(sys, ψ; n)::Vector{NamedTuple}
	```
	""")
end

# ╔═╡ d566993e-587d-4aa3-995b-eb955dec5758
html_expand("Expand for baseline implementation using <code>DirectFalsification</code>.", [
	html"<h2hide>Using <code>DirectFalsification</code> algorithm</h2hide>",
	Markdown.parse("""
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

alg = DirectFalsification(1, $(max_steps(sys_small)))
τ_failures = falsify(alg, sys_small, ψ_small)
ℓτ = maximum(s->pdf(ps_small, s[1].s), τ_failures)
```
**Note**: _But we want to use the `NominalTrajectoryDistribution` to keep the algorithm general for the medium/large problems that **do** have disturbances._
""")])

# ╔═╡ c2ae204e-dbcc-453a-81f5-791ba4be39db
@tracked function most_likely_failure_baseline(sys, ψ; n=max_steps(sys), full=false)
	d = get_depth(sys)
	m = n ÷ d                                          # Get num rollouts, \div for ÷
	pτ = NominalTrajectoryDistribution(sys, d)         # Trajectory distribution
	τs = [rollout(sys, pτ; d) for _ in 1:m]            # Rollout with pτ, m*d steps
	τs_failures = filter(τ->isfailure(ψ, τ), τs)       # Filter to get failure trajs.
	τ_most_likely = argmax(τ->logpdf(pτ, τ), τs_failures) # Most-likely failure traj
	return full ? (τ_most_likely, τs) : τ_most_likely     # Return MLF, or all trajs.
end

# ╔═╡ 254956d0-8f58-4e2b-b8a9-5dd10dd074a2
function run_baseline(sys::System, ψ; n, seed=4)
	Random.seed!(seed)
	τ, τs = most_likely_failure_baseline(sys, ψ; n, full=true)
	d = get_depth(sys)
	p = NominalTrajectoryDistribution(sys, d)
	ℓ = logpdf(p, τ)
	n = stepcount()
	return (τ=τ, τs=τs, ℓ=ℓ, n=n) # return these variables as a NamedTuple
end

# ╔═╡ 3385fcb3-8b93-4da8-ba75-77877cc77ce4
baseline_small_results = run_baseline(sys_small, ψ_small; n=n_baseline_small);

# ╔═╡ 73da2a56-8991-4484-bcde-7d397214e552
Markdown.parse("""
### Baseline results (small)

\$\$\\begin{align}
\\ell_\\text{baseline} &= $(round(baseline_small_results.ℓ, digits=3))\\tag{failure log-likelihood} \\\\
n_\\text{steps} &= $(baseline_small_results.n) \\tag{number of \\texttt{step} calls}
\\end{align}\$\$

Reminder that the number of `step` calls \$n\$ is equal to the number of rollouts \$m\$ for the small system. This is because the rollout depth is \$d=1\$.
""")

# ╔═╡ 77a6e704-33e8-4241-84f0-0e58c29c06ef
baseline_medium_results = run_baseline(sys_medium, ψ_medium; n=n_baseline_medium);

# ╔═╡ 7ef66a50-6acc-474f-b406-7b27a7b18510
Markdown.parse("""
\$\$\\begin{align}
\\ell_\\text{baseline} &= $(round(baseline_medium_results.ℓ; digits=3))\\tag{failure log-likelihood} \\\\
n_\\text{steps} &= $(format(baseline_medium_results.n; latex=true)) \\tag{number of \\texttt{step} calls \$d\\times m\$}
\\end{align}\$\$
""")

# ╔═╡ 42456abf-4930-4b01-afd1-fce3b4881e28
baseline_details(sys_small; n_baseline=n_baseline_small, descr="simple Gaussian", max_steps)

# ╔═╡ fc2d34da-258c-4460-a0a4-c70b072f91ca
@small function most_likely_failure(sys::SmallSystem, ψ; n=max_steps(sys))
	# TODO: WRITE YOUR CODE HERE
end

# ╔═╡ 307afd9c-6dac-4a6d-89d7-4d8cabfe3fe5
Markdown.MD(
	md"""
$(@bind rerun_small LargeCheckBox(text="⟵ Click to re-run the <code>SmallSystem</code> evaluation."))""",
	Markdown.parse("""
	↑ This will re-run **`most_likely_failure(::SmallSystem, ψ)`** and re-save **`$(get_filename(sys_small, ThisProject))`**

	_Uncheck this to load results from the file._
	""")
)

# ╔═╡ 772cf17e-0fdb-470e-9f12-9480af811edd
baseline_details(sys_medium; n_baseline=n_baseline_medium, descr="pendulum", max_steps)

# ╔═╡ cb7b9b9f-59da-4851-ab13-c451c26117df
@medium function most_likely_failure(sys::MediumSystem, ψ; n=max_steps(sys))
	# TODO: WRITE YOUR CODE HERE
end

# ╔═╡ 38f26afd-ffa5-48d6-90cc-e3ec189c2bf1
Markdown.MD(
	md"""
$(@bind rerun_medium LargeCheckBox(text="⟵ Click to re-run the <code>MediumSystem</code> evaluation."))""",
	Markdown.parse("""
	↑ This will re-run **`most_likely_failure(::MediumSystem, ψ)`** and re-save **`$(get_filename(sys_medium, ThisProject))`**

	_Uncheck this to load results from the file._
	""")
)

# ╔═╡ be8c37e8-45db-4198-b0b9-d287e73fb818
try
	submission_details(@bind(directory_trigger, OpenDirectory(@__DIR__)), ThisProject, [SmallSystem, MediumSystem, LargeSystem])
catch end

# ╔═╡ 0c520f93-49ce-45eb-899d-a31105d856c8
if directory_trigger
	@info "Opening local directory..."
	sleep(1)
end

# ╔═╡ aa0c4ffc-d7f0-484e-a1e2-7f6f92a3a53d
Markdown.MD(
	Markdown.parse("""
	# 3️⃣ **Large**: Aircraft Collision Avoidance
	The large system is an aircraft collision avoidance system (CAS).
	- It uses an interpolated lookup-table policy.
	- The state is comprised of the relative altitude (m) \$h\$, the relative vertical rate \$\\dot{h}\$ (m/s), the previous action \$a_\\text{prev}\$, and the time to closest point of approach \$t_\\text{col}\$ (sec): \$s = [h, \\dot{h}, a_\\text{prev}, t_\\text{col}]\$
	- Actions are \$a \\in [-5, 0, 5]\$ vertical rate changes.
	- Disturbances \$x\$ are applied to \$\\dot{h}\$ as environment noise: \$x \\sim \\mathcal{N}(0, 1.5)\$
	- Finite horizon (i.e., rollout depth) of \$d=$(get_depth(sys_large))\$ for \$t_\\text{col}\$ from \$40-0\$ seconds.
	"""),
	depth_highlight(sys_large)
)

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

# ╔═╡ 3328d818-391a-440a-8f1b-f2b7f3e00958
n_baseline_large = 410_000

# ╔═╡ 35434537-9b9c-4528-b58c-420d01813598
baseline_details(sys_large; n_baseline=n_baseline_large, descr="CAS", max_steps)

# ╔═╡ 06b14338-ea3b-45c8-bf6c-28b82db2ea70
baseline_large_results = run_baseline(sys_large, ψ_large; n=n_baseline_large);

# ╔═╡ 204feed7-cde8-40a8-b6b5-051a1c768fd9
Markdown.parse("""
\$\$\\begin{gather}
\\ell_\\text{baseline} = $(round(baseline_large_results.ℓ; digits=3))\\tag{failure log-likelihood} \\\\
n_\\text{steps} = $(format(baseline_large_results.n; latex=true)) \\tag{number of \\texttt{step} calls \$d\\times m\$}
\\end{gather}\$\$
""")

# ╔═╡ e3d6fdf1-3a9e-446b-8482-49d6f64b652e
html_quarter_space()

# ╔═╡ 23fd490a-74d2-44b4-8a12-ea1460d95f85
Markdown.parse("""
## ⟶ **Task (Large)**: Most-likely failure
Please fill in the following `most_likely_failure` function.
""")

# ╔═╡ 18a70925-3c2a-4317-8bbc-c2a096ec56d0
start_code()

# ╔═╡ 3471a623-16af-481a-8f66-5bd1e7890188
@large function most_likely_failure(sys::LargeSystem, ψ; n=max_steps(sys))
	# TODO: WRITE YOUR CODE HERE
end

# ╔═╡ 4c5210d6-598f-4167-a6ee-93bceda7223b
end_code()

# ╔═╡ 2ba2d3a2-3f6c-4d5f-8c45-8d00947f6e05
html_quarter_space()

# ╔═╡ ea2d7eb7-d576-415c-ac4c-fea7f90de637
md"""
# 📊 Large Test
We'll automatically test your `most_likely_failure(::LargeSystem, ψ)` function below.
"""

# ╔═╡ 7fe1c3d7-469c-47d9-9d46-e5b8b263edb9
Markdown.MD(
	md"""
$(@bind rerun_large LargeCheckBox(text="⟵ Click to re-run the <code>LargeSystem</code> evaluation."))""",
	Markdown.parse("""
	↑ This will re-run **`most_likely_failure(::LargeSystem, ψ)`** and re-save **`$(get_filename(sys_large, ThisProject))`**

	_Uncheck this to load results from the file._
	""")
)

# ╔═╡ 74aeca7b-0658-427f-8c02-d093a0d725ee
html_half_space()

# ╔═╡ 6d5c805b-330c-4b04-a51c-15e674352b1b
html_quarter_space()

# ╔═╡ 860ec509-3a86-4842-9471-6b1a0b8f366d
md"""
## Comparing likelihoods
Since the likelihoods across the three problems vary widely in range, we compute the log-likelihoods of your trajectories and take the difference of logs between the _most-likely trajectory_ (i.e., the mean trajectory $\tau_\text{mean}$). The most-likely trajectory is simply the trajectory rolled-out with the mean initial state and mean disturbances.

The score for problem $k$ is then computed as:

$$\begin{gather}
\text{score}_k = \log \left( \frac{p(\tau)}{p(\tau_\text{mean})} \right) = \log p(\tau) - \log p(\tau_\text{mean})
\end{gather}$$

_The maximum possible score is $0$ (but is impossible as the mean trajectories are not failures for these systems)._
"""

# ╔═╡ 54741d81-39e0-4a47-b84d-c41c8eb7611b
function score(sys::System, τ)
	ps = NominalTrajectoryDistribution(sys, get_depth(sys))
	τ_mean = mean_rollout(sys, ps)
	ℓ_mean = logpdf(ps, τ_mean)
	ℓ = logpdf(ps, τ)
	return ℓ - ℓ_mean
end

# ╔═╡ 6559cf16-a474-4533-a2c7-ccbc02480a76
md"""
Since the small system runs several tests, we take the average score over the tests:

$$\begin{gather}
\text{score}_\text{small} = \mathbb{E}_i \left[ \log\left( \frac{p(\tau_i)}{p(\tau_\text{mean})} \right) \right]
\end{gather}$$

To balance out the difficulty, we use weights $\mathbf{w} = [1,2,3]$ (normalized to sum to one):

$$\bar{w_i} = \frac{w_i}{\sum_j w_j}$$
"""

# ╔═╡ cfdba748-45d5-4eaa-97b3-fdc9fe7e4333
𝐰 = [1,2,3]

# ╔═╡ 6beda870-0cb0-40f5-9531-fa3e2f7bb020
md"""
The final score on the leaderboard is then a weighted sum:

$$\begin{gather}
\mathbf{s} = \big[\text{score}_\text{small},\, \text{score}_\text{medium},\, \text{score}_\text{large} \big] \\
\text{score} = \mathbf{w}^\top\mathbf{s}
\end{gather}$$
"""

# ╔═╡ 5c3d24f6-0106-444c-b7df-89bba8c01b37
function leaderboard_scores(systems::Vector{<:System}, τs; 𝐰=ones(length(τs)))
	score_small = mean(score(systems[1], τ) for τ in τs[1])
	score_medium = score(systems[2], τs[2])
	score_large = score(systems[3], τs[3])
	𝐬 = [score_small, score_medium, score_large]
	𝐰 = 𝐰 ./ sum(𝐰)
	return 𝐰'𝐬
end

# ╔═╡ 4edc5933-9457-4c7c-8456-a26974e0587e
html_half_space()

# ╔═╡ 50f1c7a3-9219-40ce-a060-77beb243b7af
begin
	function task_description(sys, details)
		return Markdown.parse(
		"""
			most_likely_failure(sys::$(system_name(sys)), ψ; n)::Vector{NamedTuple}
		
		A function that takes in a system `sys` ($details) and a specification `ψ` and **returns the trajectory that led to the most-likely failure**.
		
		- `n` = number of `step` calls allotted (\$n = $(format(max_steps(sys); latex=true))\$ for `$(system_name(sys))`)
		
		**Note**: `ψ` is written as `\\psi<TAB>`
		""")
	end

	md"> _Project 1 specific functions._"
end

# ╔═╡ f6589984-e24d-4aee-b7e7-db159ae7fea6
task_description(sys_small, "1D Gaussian for the _small_ setting")

# ╔═╡ 9657f5ff-f21c-43c5-838d-402a2a723d5e
task_description(sys_medium, "inverted pendulum for the _medium_ setting")

# ╔═╡ 45c79345-89da-498c-9a98-2ad55a0a6114
task_description(sys_large, "collision avoidance system for the _large_ setting")

# ╔═╡ 247f4c17-bee1-4315-aff9-017407ef9219
begin
	if !ismissing(directory_trigger) && directory_trigger
		try
			if Sys.iswindows()
				run(`explorer $(abspath(@__DIR__))`)
			elseif Sys.isapple()
				run(`open $(abspath(@__DIR__))`)
			elseif Sys.islinux()
				run(`xdg-open $(abspath(@__DIR__))`)
			end
		catch end
	end

	md"> _Helper for opening local directories._"
end

# ╔═╡ db7d4de5-9166-4e56-b5bc-1356e43286a9
begin
	function log_likelihood(sys::System, τ)
		ps = NominalTrajectoryDistribution(sys, get_depth(sys))
		ℓ = logpdf(ps, τ)
		return ℓ
	end

	rd(x::String) = x
	rd(x::Number) = round(x; sigdigits=6)

	md"> _Leaderboard helper functions._"
end

# ╔═╡ 5a1ed20d-788b-4655-bdd8-069545f48929
begin
	extract(sys::System, input) = extract(sys.env, input)

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
	initial_guess(sys::LargeSystem) =
		[rand(Normal(0,100)), 0, 0, 40, zeros(39)...]

	md"> *Helper `extract` and `initial_guess` functions.*"
end

# ╔═╡ 6c8b3077-876e-42fd-aa47-f3fa7c37f4dd
Markdown.MD(
	md"$(@bind dark_mode DarkModeIndicator())",
	md"> _Dark mode indicator._")

# ╔═╡ 6b17139e-6caf-4f07-a607-e403bf1ad794
begin
	version_mds = []
	try
		if !validate_version(StanfordAA228V)
			push!(version_mds, Markdown.MD(
			almost(md"""
			Your `StanfordAA228V` package is out-of-date. Please update it via the instructions below.

			**Then restart the notebook.**

			_(This warning may persist after restart, wait until the notebook finishes loading entirely)_"""),
			md"""$(LocalResource(joinpath(@__DIR__, "..", "media", dark_mode ? "update-package-dark.gif" : "update-package.gif")))"""
			))
		end
	catch end
	try
		if !validate_project_version(@__DIR__)
			push!(version_mds, almost(Markdown.parse("""
			Your `AA228Projects/$(basename(@__DIR__))` code is out-of-date. Please update it via the instructions below.
			1. Kill the notebook (i.e., close the `Pluto` terminal window).
			2. In a terminal, `cd` to your `AA228VProjects` directory:
			    - `cd $(abspath(joinpath(@__DIR__, "..")))`
			3. Pull the latest changes:
			    - `git pull`
			4. Reopen the Pluto notebook:
			    - Run `julia`.
			    - Launch Pluto in `julia`:
			```julia
			using Pluto
			Pluto.run()
			```
			_(This warning may persist after restart, wait until the notebook finishes loading entirely)_""")))
		end
	catch end
	if !isempty(version_mds)
		Markdown.MD(version_mds)
	end
end

# ╔═╡ daada216-11d4-4f8b-807c-d347130a3928
try
	if dark_mode
		LocalResource(joinpath(@__DIR__, "..", "media", "inverted_pendulum_dark.svg"))
	else
		LocalResource(joinpath(@__DIR__, "..", "media", "inverted_pendulum.svg"))
	end
catch end

# ╔═╡ 5aef09eb-ddc5-4b99-abf2-568621b871d5
# ╠═╡ show_logs = false
begin
	DarkModeHandler.setdarkmode!(dark_mode)

	import StanfordAA228V:
		Always, Predicate, FlippedPredicate,
		plot, plot!, plot_cdf, plot_pendulum # dark_mode triggers

	import StanfordAA228V.SignalTemporalLogic: ρ

	pkg_trigger = true
	md"> _AA228V/CS238V package management._"
end

# ╔═╡ 44c8fbe0-21e7-482b-84a9-c3d32a4737dd
plot(sys_medium, ψ_medium, baseline_medium_results.τs; max_lines=100)

# ╔═╡ e12b102e-785b-46e9-980c-e9f7943eda60
begin
	plot(
		begin
			plot(sys_medium, ψ_medium, baseline_medium_results.τs; max_lines=100)
			plot!(title="All baseline trajectories", titlefontsize=12)
		end,
		begin
			plot(sys_medium, ψ_medium, baseline_medium_results.τ)
			plot!(title="Most-likely failure found", titlefontsize=12)
		end,
		layout=(1,2),
		margin=5Plots.mm,
		topmargin=-5Plots.mm,
	)
end

# ╔═╡ 797cbe41-a5f3-4179-9143-9ef6e6888a4d
plot(sys_large, ψ_large, baseline_large_results.τs)

# ╔═╡ 4ae85f59-4e94-48aa-8ccb-91311466c51f
plot(sys_large, ψ_large, baseline_large_results.τ)

# ╔═╡ 95e3d42f-b33f-4294-81c5-f34a300dc9b4
# This needs to be in the cell above.
begin
	pkg_trigger
	html"""
	<script>
	let cell = currentScript.closest('pluto-cell')
	let id = cell.getAttribute('id')
	let cells_below = document.querySelectorAll(`pluto-cell[id='${id}'] ~ pluto-cell`)
	let cell_below_ids = [cells_below[0]].map((el) => el.getAttribute('id'))
	cell._internal_pluto_actions.set_and_run_multiple(cell_below_ids)
	</script>
	"""
end

# ╔═╡ ba6c082b-6e62-42fc-a85c-c8b7efc89b88
# ╠═╡ show_logs = false
begin
	pkg_trigger

	global UsingThisViolatesTheHonorCode = @load ThisProject.backend
	create_specification = UsingThisViolatesTheHonorCode.create_specification
	ψ2latex = UsingThisViolatesTheHonorCode.ψ2latex
	small_extras = UsingThisViolatesTheHonorCode.small_extras
	rerun = UsingThisViolatesTheHonorCode.rerun
	rerun_multiple = UsingThisViolatesTheHonorCode.rerun_multiple
	centered = UsingThisViolatesTheHonorCode.centered

	md"""
	# Backend
	_Helper functions and project management. Please do not edit._
	"""
end

# ╔═╡ e86d260f-c93d-4561-a9f1-44e4c7af827e
centered(plot(sys_small, ψ_small))

# ╔═╡ d4d057d7-cc9d-4949-9e3f-44a8aa67d725
centered(begin
	plot(sys_small, ψ_small, baseline_small_results.τ)
	title!("Baseline most-likely failure found")
end)

# ╔═╡ fe7f4a79-1a63-4272-a776-358a309c8550
centered(begin
	plot(sys_small, ψ_small, baseline_small_results.τs)
	title!("States from baseline")
end)

# ╔═╡ beaec161-ad89-4f83-9066-f420a1d04d39
rerun(sys_small, ψ_small;
	  save=false, f=most_likely_failure_small,
	  project=ThisProject, latextras=small_extras(sys_small, ψ_small))[3]

# ╔═╡ c524297f-2bf3-4dd2-b7b4-fc5ce9a81738
begin
	ψ_small_different = LTLSpecification(@formula □(s->s < 2))
	latextras_different = small_extras(sys_small, ψ_small_different)
	rerun(sys_small, ψ_small_different;
	      f=most_likely_failure_small, save=false,
		  project=ThisProject, latextras=latextras_different)[3]
end

# ╔═╡ 61173ec6-c7d6-44fa-8c47-5f7295dd49cf
begin
	rerun_rand_small # trigger
	ψ_small_rand = create_specification()
	latextras_rand = small_extras(sys_small, ψ_small_rand)
	rerun(sys_small, ψ_small_rand;
	      f=most_likely_failure_small, save=false,
		  project=ThisProject, latextras=latextras_rand)[3]
end

# ╔═╡ 57d321cd-2029-4e49-8b56-9c5c48721ac4
ψ_small_slider = create_specification(γ);

# ╔═╡ d647ac21-738b-43e7-bbbd-582b6294560e
Markdown.parse("""
## Slider to control threshold
If your `most_likely_failure` function for the small system is fast enough, you can explore how it performs when controlling the failure threshold \$γ\$ (we flip the comparison operator when \$γ > 0\$):

\$\$$(ψ2latex(sys_small, ψ_small_slider))\$\$
""")

# ╔═╡ b61b8f52-9bf3-4766-9208-3e40134a3573
try
	global τ_small_slider = most_likely_failure_small(sys_small, ψ_small_slider)
	global ℓ_small_slider = logpdf(NominalTrajectoryDistribution(sys_small), τ_small_slider)
	Markdown.parse("""
	\$\$\\begin{gather}
	\\ell_\\text{truth} = $(logpdf(Normal(), ψ_small_slider.formula.ϕ.c)) \\tag{true MLF log-likelihood} \\\\
	\\ell = $ℓ_small_slider \\tag{failure log-likelihood} \\\\
	n_\\text{steps} = $(stepcount()) \\tag{number of \\texttt{step} calls}
	\\end{gather}\$\$""")
catch end

# ╔═╡ 57c5a6f0-2527-4988-9bf0-140495ba9b7e
if !isnothing(τ_small_slider)
	centered(plot(sys_small, ψ_small_slider, τ_small_slider))
else
	almost(md"*Slider plot will show when issues are fixed with `most_likely_failure` above*")
end

# ╔═╡ d0a3770a-2c48-42db-9a71-6b7f695f22d8
begin
	τs_small, log_small, pass_small = rerun_multiple(sys_small;
		                                             f=most_likely_failure_small,
												     run=rerun_small, 
												     project=ThisProject)
	log_small
end

# ╔═╡ f286f3b2-3bac-4384-9b40-522e974a14ee
begin
	local pass
	try
		pass = pass_small
	catch
		pass = false
	end
	
	Markdown.MD(HTML("<h2 id='graded-test'>$(pass ? "✔️" : "✖️") Graded small test ($(pass ? "$(ThisProject.points_small)/$(ThisProject.points_small)" : "0/$(ThisProject.points_small)") points)</h2>"),
		md"""
	✳️ **If the following tests pass, then you're finished with the small problem.**
	
	We'll test multiple failure thresholds in the specification $\psi$. Make sure the above 'randon test' works well across different failure thresholds to ensure this will pass.""")
end

# ╔═╡ b417e370-efae-40e8-9247-5daf14fcc749
begin
	τ_medium, _, log_medium, pass_medium = rerun(sys_medium, ψ_medium;
											  f=most_likely_failure_medium,
											  run=rerun_medium, project=ThisProject)
	log_medium
end

# ╔═╡ 23999cd9-543b-47dc-a0b2-e133ba95891e
begin
	local pass
	try
		pass = pass_medium
	catch
		pass = false
	end
	Markdown.parse("""
	## $(pass ? "✔️" : "✖️") Graded medium test ($(pass ? "$(ThisProject.points_medium)/$(ThisProject.points_medium)" : "0/$(ThisProject.points_medium)") points)
	""")
end

# ╔═╡ f6eb6d1a-a9a0-4234-8699-269a92f666c0
begin
	τ_large, _, log_large, pass_large = rerun(sys_large, ψ_large;
										   f=most_likely_failure_large,
										   run=rerun_large, project=ThisProject)
	log_large
end

# ╔═╡ 7c473630-6555-4ada-85f3-0d40aefe6370
begin
	local pass
	try
		pass = pass_large
	catch
		pass = false
	end
	Markdown.parse("""
	## $(pass ? "✔️" : "✖️") Graded large test ($(pass ? "$(ThisProject.points_large)/$(ThisProject.points_large)" : "0/$(ThisProject.points_large)") points)
	""")
end

# ╔═╡ dbd088d1-f4c9-4e6a-b280-960b06da76e4
begin
	local passes = falses(3)
	try
		passes = [pass_small, pass_medium, pass_large]
	catch end
	Markdown.MD(Markdown.parse("# $(all(passes) ? "✅" : "❌") Final Check"),
	@mdx("""If the following test indicator is <span style='color:#759466'><b>green</b></span>, you can submit to Gradescope."""))
end

# ╔═╡ 1bb92755-65e3-457e-84cd-252eae5e4d7e
begin
	local passes = falses(3)

	try
		passes[1] = pass_small
	catch end

	try
		passes[2] = pass_medium
	catch end

	try
		passes[3] = pass_large
	catch end

	if all(passes)
		correct(Markdown.MD(md"""
	All tests have passed, **_you're done with Project 1!_**""",
	@mdx("""
	|  System  |  Passed?  |  Points  |
	| :------: | :-------: | :------: |
	| Small | $(passes[1] ? HTML("<span style='color:#759466'><b>Passed!</b></span>") : HTML("<span style='color:#B83A4B'><b>Failed.</b></span>")) | $(passes[1] ? "$(ThisProject.points_small)/$(ThisProject.points_small)" : "0/$(ThisProject.points_small)") |
	| Medium | $(passes[2] ? HTML("<span style='color:#759466'><b>Passed!</b></span>") : HTML("<span style='color:#B83A4B'><b>Failed.</b></span>")) | $(passes[2] ? "$(ThisProject.points_medium)/$(ThisProject.points_medium)" : "0/$(ThisProject.points_medium)") |
	| Large | $(passes[3] ? HTML("<span style='color:#759466'><b>Passed!</b></span>") : HTML("<span style='color:#B83A4B'><b>Failed.</b></span>")) | $(passes[3] ? "$(ThisProject.points_large)/$(ThisProject.points_large)" : "0/$(ThisProject.points_large)") |
	"""),
	md"""
	**📩 Please see the [Submission](#submission) section at the top of the page.**
	"""))
	else
		almost(Markdown.MD(md"**_Some tests have failed:_**", @mdx("""
	|  System  |  Passed?  | Points |
	| :------: | :-------: | :----: |
	| Small | $(passes[1] ? HTML("<span style='color:#759466'><b>Passed!</b></span>") : HTML("<span style='color:#B83A4B'><b>Failed.</b></span>")) | $(passes[1] ? "$(ThisProject.points_small)/$(ThisProject.points_small)" : "0/$(ThisProject.points_small)") |
	| Medium | $(passes[2] ? HTML("<span style='color:#759466'><b>Passed!</b></span>") : HTML("<span style='color:#B83A4B'><b>Failed.</b></span>")) | $(passes[2] ? "$(ThisProject.points_medium)/$(ThisProject.points_medium)" : "0/$(ThisProject.points_medium)") |
	| Large | $(passes[3] ? HTML("<span style='color:#759466'><b>Passed!</b></span>") : HTML("<span style='color:#B83A4B'><b>Failed.</b></span>")) | $(passes[3] ? "$(ThisProject.points_large)/$(ThisProject.points_large)" : "0/$(ThisProject.points_large)") |
	"""),
	md"""
	_Please fix the above failing tests before submission._
	
	_You may partially submit individual `.val` files to Gradescope, you have unlimited Gradescope submissions until the deadline. But please make sure to submit all **three** `.val` files once complete._"""))
	end
end

# ╔═╡ d9ab8278-eb76-4a36-aa0e-4ec74704f5e0
begin
	negative_infinity = -1e228

	function check_inf(score)
		if isinf(score)
			negative_infinity, negative_infinity
		else
			score, rd(score)
		end
	end

	global 𝔼_logpdf_small = 𝔼_logpdf_small_rd = "—"
	try
		global 𝔼_logpdf_small, 𝔼_logpdf_small_rd =
			check_inf(mean(log_likelihood(sys_small, τ) for τ in τs_small))
	catch end

	global logpdf_medium = logpdf_medium_rd = "—"
	try
		global logpdf_medium, logpdf_medium_rd =
			check_inf(log_likelihood(sys_medium, τ_medium))
	catch end

	global logpdf_large = logpdf_large_rd = "—"
	try
		global logpdf_large, logpdf_large_rd =
			check_inf(log_likelihood(sys_large, τ_large))
	catch end

	global user_score = user_score_rd = "—"
	try
		global user_score, user_score_rd = 
			check_inf(leaderboard_scores(
				[sys_small, sys_medium, sys_large],
				[τs_small, τ_medium, τ_large]; 𝐰=𝐰))
	catch end


	Markdown.parse("""
# Leaderboard
If the above tests pass, then you will receive full credit for your submission on Gradescope under the **`"Project 1 (.val files)"`** assignment.

_However_, we have a leaderboard so that students can participate in a friendly competition to find the most-likely failures for each problem.
	
## Leaderboard entry
Your leaderboard entry on Gradescope should look something like this:

| Rank | Submission Name | Score | 𝔼[log 𝑝(small)] | log 𝑝(medium) | log 𝑝(large) |
| :--: | :-------------: | :---: | :-----------: | :------------: | :-----------: |
| — | $(guess_username()) | $(user_score_rd) | $(𝔼_logpdf_small_rd) | $(logpdf_medium_rd) | $(logpdf_large_rd) |

_Note, if any of the scores are \$-\\infty\$, then they will show up as \$$(expnum(negative_infinity))\$ (a Gradescope quirk)._
""")
end

# ╔═╡ 5563f0da-7552-4879-a38a-ba1748d39d52
begin
	pendulum_gif_name = dark_mode ? "pendulum-dark.gif" : "pendulum.gif"

	if false
		pendulum_anim = @animate for t in 1:get_depth(sys_medium)
			θ_medium_failure = rad2deg(baseline_medium_results.τ[t].s[1])
			θ_medium_success = rad2deg(baseline_medium_results.τs[2][t].s[1])
			plot(
				plot_pendulum(θ_medium_failure; title="Failure"),
				plot_pendulum(θ_medium_success; title="Success"),
				layout=(1,2),
				size=(700,300),
				dpi=400,
			)
		end

		gif(pendulum_anim, joinpath(@__DIR__, "..", "media", pendulum_gif_name); fps=15, show_msg=false)
	end

	md"> _Inverted pendulum animated GIF._"
end

# ╔═╡ 4ea18122-b681-4de1-89e3-5fb7ce2f7a0b
try LocalResource(joinpath(@__DIR__, "..", "media", pendulum_gif_name)) catch end

# ╔═╡ 98cbe931-d362-4039-97ba-41e0049619a3
begin
	cas_gif_name = dark_mode ? "cas-dark.gif" : "cas.gif"
	cas_fps = 15
	cas_repeat = cas_fps # Repeat last frame x times

	if false
		cas_anim_T = 1:get_depth(sys_large)
		cas_anim_T = vcat(cas_anim_T, fill(cas_anim_T[end], cas_repeat))
		cas_anim = @animate for t in cas_anim_T
			plot(
				plot(sys_large, ψ_large, baseline_large_results.τ; t=t, title="CAS: Failure"),
				plot(sys_large, ψ_large, baseline_large_results.τs[1234]; t=t, title="CAS: Success"),
				layout=(1,2), dpi=300)
		end

		gif(cas_anim, joinpath(@__DIR__, "..", "media", cas_gif_name); fps=cas_fps, show_msg=false)

		function n_minimum_robustness(τs, ψ, n)
			ρs = [robustness([step.s for step in τ], ψ.formula) for τ in τs]
			perm = sortperm(ρs)
			return reverse(perm[1:n])
		end
	end

	md"> _Collision avoidance animated GIF._"
end

# ╔═╡ e189b31e-7e24-4c32-989f-3e600a44d4bc
try LocalResource(joinpath(@__DIR__, "..", "media", cas_gif_name)) catch end

# ╔═╡ 97042a5e-9691-493f-802e-2262f2da4627
Markdown.MD(notebook_style(), md"> _Notebook styling._")

# ╔═╡ 9865ed62-b4fd-4e49-9259-3e5997c589f3
Markdown.MD(button_style(rerun_rand_small), md"> _Button styling._")

# ╔═╡ ef084fea-bf4d-48d9-9c84-8cc1dd98f2d7
Markdown.MD(TableOfContents(), md"> _Table of contents._")

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BSON = "fbb218c0-5317-5bc6-957e-2ee96dd4b1f0"
Base64 = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
GridInterpolations = "bb4c363b-b914-514b-8517-4eb369bc008a"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MarkdownLiteral = "736d6165-7244-6769-4267-6b50796e6954"
Optim = "429524aa-4258-5aef-a3af-852621145aeb"
Parameters = "d96e819e-fc66-5662-9728-84c9c7592b0a"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
ProgressLogging = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
StanfordAA228V = "6f6e590e-f8c2-4a21-9268-94576b9fb3b1"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[compat]
BSON = "~0.3.9"
Distributions = "~0.25.115"
GridInterpolations = "~1.2.1"
MarkdownLiteral = "~0.1.1"
Optim = "~1.10.0"
Parameters = "~0.12.3"
Plots = "~1.40.9"
PlutoUI = "~0.7.60"
ProgressLogging = "~0.1.4"
ReverseDiff = "~1.15.3"
StanfordAA228V = "~0.1.22"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.2"
manifest_format = "2.0"
project_hash = "f20fac971195dcd2201decc798f06a70644f597d"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "50c3c56a52972d78e8be9fd135bfb91c9574c140"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.1.1"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "017fcb757f8e921fb44ee063a7aafe5f89b86dd1"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.18.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = "CUDSS"
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "c3b238aa28c1bebd4b5ea4988bebf27e9a01b72b"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "1.0.1"

    [deps.Atomix.extensions]
    AtomixCUDAExt = "CUDA"
    AtomixMetalExt = "Metal"
    AtomixoneAPIExt = "oneAPI"

    [deps.Atomix.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.BSON]]
git-tree-sha1 = "4c3e506685c527ac6a54ccc0c8c76fd6f91b42fb"
uuid = "fbb218c0-5317-5bc6-957e-2ee96dd4b1f0"
version = "0.3.9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "f1dff6729bc61f4d49e140da1af55dcd1ac97b2f"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.5.0"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "8873e196c2eb87962a2048b3b8e08946535864a1"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+4"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CRlibm]]
deps = ["CRlibm_jll"]
git-tree-sha1 = "32abd86e3c2025db5172aa182b982debed519834"
uuid = "96374032-68de-5a5b-8d9e-752f78720389"
version = "1.0.1"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "009060c9a6168704143100f36ab08f06c2af4642"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.2+1"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "SparseInverseSubset", "Statistics", "StructArrays", "SuiteSparse"]
git-tree-sha1 = "4312d7869590fab4a4f789e97bd82f0a04eaaa05"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.72.2"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "1713c74e00545bfe14605d2a2be1712de8fbcb58"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CodecBzip2]]
deps = ["Bzip2_jll", "TranscodingStreams"]
git-tree-sha1 = "84990fa864b7f2b4901901ca12736e45ee79068c"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.8.5"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "bce6804e5e6044c6daab27bb533d1295e4a2e759"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.6"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "c785dfb1b3bfddd1da557e861b919819b82bbe5b"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.27.1"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "64e15186f0aa277e174aa81798f7eb8598e0157e"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.0"

[[deps.CommonMark]]
deps = ["Crayons", "PrecompileTools"]
git-tree-sha1 = "3faae67b8899797592335832fccf4b3c80bb04fa"
uuid = "a80b9123-70ca-4bc0-993e-6e3bcb318db6"
version = "0.8.15"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "f36e5e8fdffcb5646ea5da81495a5a7566005127"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.4.3"

[[deps.Configurations]]
deps = ["ExproniconLite", "OrderedCollections", "TOML"]
git-tree-sha1 = "4358750bb58a3caefd5f37a4a0c5bfdbbf075252"
uuid = "5218b696-f38b-4ac9-8b61-a12ec717816d"
version = "0.17.6"

[[deps.ConstructionBase]]
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fc173b380865f70627d7dd1190dc2fce6cc105af"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.14.10+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "4b138e4643b577ccf355377c2bc70fa975af25de"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.115"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EnumX]]
git-tree-sha1 = "bdb1942cd4c45e3c678fd11569d5cccd80976237"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.4"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ErrorfreeArithmetic]]
git-tree-sha1 = "d6863c556f1142a061532e79f611aa46be201686"
uuid = "90fa49ef-747e-5e6f-a989-263ba693cf1a"
version = "0.5.2"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e51db81749b0777b2147fbe7b783ee79045b8e99"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.4+3"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.ExpressionExplorer]]
git-tree-sha1 = "7005f1493c18afb2fa3bdf06e02b16a9fde5d16d"
uuid = "21656369-7473-754a-2065-74616d696c43"
version = "1.1.0"

[[deps.ExproniconLite]]
git-tree-sha1 = "4c9ed87a6b3cd90acf24c556f2119533435ded38"
uuid = "55351af7-c7e9-48d6-89ff-24e801d99491"
version = "0.10.13"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "53ebe7511fa11d33bec688a9178fac4e49eeee00"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.2"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FastRounding]]
deps = ["ErrorfreeArithmetic", "LinearAlgebra"]
git-tree-sha1 = "6344aa18f654196be82e62816935225b3b9abe44"
uuid = "fa42c844-2597-5d31-933b-ebd51ab2693f"
version = "0.3.1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Setfield"]
git-tree-sha1 = "84e3a47db33be7248daa6274b287507dd6ff84e8"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.26.2"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffSparseArraysExt = "SparseArrays"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "21fac3c77d7b5a9fc03b0ec503aa1a6392c34d2b"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.15.0+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "a2df1b776752e3f344e5116c06d75a10436ab853"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.38"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "786e968a8d2fb167f2e4880baba62e0e26bd8e4e"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.3+1"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "846f7026a9decf3679419122b49f8a1fdb48d2d5"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.16+0"

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.FuzzyCompletions]]
deps = ["REPL"]
git-tree-sha1 = "be713866335f48cfb1285bff2d0cbb8304c1701c"
uuid = "fb4132e2-a121-4a70-b8a1-d5b831dcdcc2"
version = "0.5.5"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "fcb0584ff34e25155876418979d4c8971243bb89"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+2"

[[deps.GLPK]]
deps = ["GLPK_jll", "MathOptInterface"]
git-tree-sha1 = "1d706bd23e5d2d407bfd369499ee6f96afb0c3ad"
uuid = "60bf3e95-4087-53dc-ae20-288a0d20c6a6"
version = "1.2.1"

[[deps.GLPK_jll]]
deps = ["Artifacts", "GMP_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6aa6294ba949ccfc380463bf50ff988b46de5bc7"
uuid = "e8aa6df9-e6ca-548a-97ff-1f85fc5b8b98"
version = "5.0.1+1"

[[deps.GMP_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "781609d7-10c4-51f6-84f2-b8444358ff6d"
version = "6.3.0+0"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
git-tree-sha1 = "4ec797b1b2ee964de0db96f10cce05b81f23e108"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "11.1.0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "424c8f76017e39fdfcdbb5935a8e6742244959e8"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.10"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "b90934c8cb33920a8dc66736471dc3961b42ec9f"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.10+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "b0036b392358c80d2d2124746c2bf3d48d457938"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.82.4+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "01979f9b37367603e2848ea225918a3b3861b606"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+1"

[[deps.GridInterpolations]]
deps = ["LinearAlgebra", "Printf", "StaticArrays"]
git-tree-sha1 = "e64e58d732c7c1f32575e2b057c0fb0f7f52e244"
uuid = "bb4c363b-b914-514b-8517-4eb369bc008a"
version = "1.2.1"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "c67b33b085f6e2faf8bf79a61962e7339a81129c"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.15"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "55c53be97790242c29031e5cd45e8ac296dadda3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.0+0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "b1c2585431c382e3fe5805874bda6aea90a95de9"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.25"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "950c3717af761bc3ff906c2e8e52bd83390b6ec2"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.14"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IntervalArithmetic]]
deps = ["CRlibm", "EnumX", "FastRounding", "LinearAlgebra", "Markdown", "Random", "RecipesBase", "RoundingEmulator", "SetRounding", "StaticArrays"]
git-tree-sha1 = "f59e639916283c1d2e106d2b00910b50f4dab76c"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "0.21.2"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "71b48d857e86bf7a1838c4736545699974ce79a2"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.9"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "a007feb38b422fbdab534406aeca1b86823cb4d6"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eac1206917768cb54957c65a615460d87b455fc1"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.1+0"

[[deps.JuMP]]
deps = ["LinearAlgebra", "MacroTools", "MathOptInterface", "MutableArithmetics", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays"]
git-tree-sha1 = "866dd0bf0474f0d5527c2765c71889762ba90a27"
uuid = "4076af6c-e467-56ae-b986-b466b2749572"
version = "1.23.5"

    [deps.JuMP.extensions]
    JuMPDimensionalDataExt = "DimensionalData"

    [deps.JuMP.weakdeps]
    DimensionalData = "0703355e-b756-11e9-17c0-8b28908087d0"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "MacroTools", "PrecompileTools", "Requires", "StaticArrays", "UUIDs"]
git-tree-sha1 = "b9a838cd3028785ac23822cded5126b3da394d1a"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.31"

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"

    [deps.KernelAbstractions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Preferences", "Printf", "Unicode"]
git-tree-sha1 = "d422dfd9707bec6617335dc2ea3c5172a87d5908"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "9.1.3"

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

    [deps.LLVM.weakdeps]
    BFloat16s = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "05a8bd5a42309a9ec82f700876903abce1017dd3"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.34+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "78211fb6cbc872f77cad3fc0b6cf647d923f4929"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "854a9c268c43b77b0a27f22d7fab8d33cdb3a731"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.2+3"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "ce5f5621cac23a86011836badfedf664a612cee4"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.5"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LazilyInitializedFields]]
git-tree-sha1 = "0f2da712350b020bc3957f269c9caad516383ee0"
uuid = "0e77f7df-68c5-4e49-93ce-4cd80f5598bf"
version = "1.3.0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LazySets]]
deps = ["Distributed", "GLPK", "IntervalArithmetic", "JuMP", "LinearAlgebra", "Random", "ReachabilityBase", "RecipesBase", "Reexport", "Requires", "SharedArrays", "SparseArrays", "StaticArraysCore"]
git-tree-sha1 = "ae9b6a027c694b9e0bab91fc25d0b2808f1bf755"
uuid = "b4f0291d-fe17-52bc-9479-3d1a343d9043"
version = "3.0.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "27ecae93dd25ee0909666e6835051dd684cc035e"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+2"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll"]
git-tree-sha1 = "8be878062e0ffa2c3f67bb58a595375eda5de80b"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.11.0+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "ff3b4b9d35de638936a525ecd36e86a8bb919d11"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "df37206100d39f79b3376afb6b9cee4970041c61"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.51.1+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "61dfdba58e585066d8bce214c5a51eaa0539f269"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "84eef7acd508ee5b3e956a2ae51b05024181dee0"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.40.2+2"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "4ab7581296671007fc33f07a721631b8855f4b1d"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.1+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "edbf5309f9ddf1cab25afc344b1e8150b7c832f9"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.40.2+2"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "e4c3be53733db1051cc15ecf573b1042b3a712a1"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.3.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f02b56007b064fbfddb4c9cd60161b6dd0f40df3"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.1.0"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Malt]]
deps = ["Distributed", "Logging", "RelocatableFolders", "Serialization", "Sockets"]
git-tree-sha1 = "02a728ada9d6caae583a0f87c1dd3844f99ec3fd"
uuid = "36869731-bdee-424d-aa32-cab38c994e3b"
version = "1.1.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MarkdownLiteral]]
deps = ["CommonMark", "HypertextLiteral"]
git-tree-sha1 = "0d3fa2dd374934b62ee16a4721fe68c418b92899"
uuid = "736d6165-7244-6769-4267-6b50796e6954"
version = "0.1.1"

[[deps.MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "DataStructures", "ForwardDiff", "JSON", "LinearAlgebra", "MutableArithmetics", "NaNMath", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays", "SpecialFunctions", "Test", "Unicode"]
git-tree-sha1 = "e065ca5234f53fd6f920efaee4940627ad991fb4"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.34.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.MsgPack]]
deps = ["Serialization"]
git-tree-sha1 = "f5db02ae992c260e4826fe78c942954b48e1d9c2"
uuid = "99f44e22-a591-53d1-9472-aa23ef4bd671"
version = "1.2.1"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "a2710df6b0931f987530f59427441b21245d8f5e"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.6.0"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

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
version = "0.3.27+1"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "38cb508d080d21dc1128f7fb04f20387ed4c0af4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7493f61f55a6cce7325f197443aa80d32554ba10"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.15+3"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "ab7edad78cdef22099f43c54ef77ac63c2c9cc64"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.10.0"
weakdeps = ["MathOptInterface"]

    [deps.Optim.extensions]
    OptimMOIExt = "MathOptInterface"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6703a85cb3781bd5909d48730a67205f3f31a575"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.3+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "12f1439c4f986bb868acda6ea33ebc78e19b95ad"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.7.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ed6834e95bd326c52d5675b4181386dfbe885afb"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.55.5+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "35621f10a7531bc8fa58f74610b1bfb70a3cfc6b"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.43.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "3ca9a356cd2e113c420f2c13bea19f8d3fb1cb18"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.3"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "dae01f8c2e069a683d3a6e17bbae5070ab94786f"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.9"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Pluto]]
deps = ["Base64", "Configurations", "Dates", "Downloads", "ExpressionExplorer", "FileWatching", "FuzzyCompletions", "HTTP", "HypertextLiteral", "InteractiveUtils", "Logging", "LoggingExtras", "MIMEs", "Malt", "Markdown", "MsgPack", "Pkg", "PlutoDependencyExplorer", "PrecompileSignatures", "PrecompileTools", "REPL", "RegistryInstances", "RelocatableFolders", "Scratch", "Sockets", "TOML", "Tables", "URIs", "UUIDs"]
git-tree-sha1 = "b5509a2e4d4c189da505b780e3f447d1e38a0350"
uuid = "c3e4b0f8-55cb-11ea-2926-15256bba5781"
version = "0.20.4"

[[deps.PlutoDependencyExplorer]]
deps = ["ExpressionExplorer", "InteractiveUtils", "Markdown"]
git-tree-sha1 = "e0864c15334d2c4bac8137ce3359f1174565e719"
uuid = "72656b73-756c-7461-726b-72656b6b696b"
version = "1.2.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "eba4810d5e6a01f612b948c9fa94f905b49087b0"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.60"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.PrecompileSignatures]]
git-tree-sha1 = "18ef344185f25ee9d51d80e179f8dad33dc48eb1"
uuid = "91cefc8d-f054-46dc-8f8c-26e11d7c5411"
version = "3.0.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.PtrArrays]]
git-tree-sha1 = "77a42d78b6a92df47ab37e177b2deac405e1c88f"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.2.1"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "492601870742dcd38f233b23c3ec629628c1d724"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.7.1+1"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll"]
git-tree-sha1 = "e5dd466bf2569fe08c91a2cc29c1003f4797ac3b"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.7.1+2"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "1a180aeced866700d4bebc3120ea1451201f16bc"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.7.1+1"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "729927532d48cf79f49070341e1d918a65aba6b0"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.7.1+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "cda3b045cf9ef07a08ad46731f5a3165e56cf3da"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.1"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.ReachabilityBase]]
deps = ["ExprTools", "InteractiveUtils", "LinearAlgebra", "Random", "Requires", "SparseArrays"]
git-tree-sha1 = "d28da1989cc21fcf57611f928061de5e8f27dc5c"
uuid = "379f33d0-9447-4353-bd03-d664070e549f"
version = "0.3.3"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RegistryInstances]]
deps = ["LazilyInitializedFields", "Pkg", "TOML", "Tar"]
git-tree-sha1 = "ffd19052caf598b8653b99404058fce14828be51"
uuid = "2792f1a3-b283-48e8-9a74-f99dce5104f3"
version = "0.1.0"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.ReverseDiff]]
deps = ["ChainRulesCore", "DiffResults", "DiffRules", "ForwardDiff", "FunctionWrappers", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "Random", "SpecialFunctions", "StaticArrays", "Statistics"]
git-tree-sha1 = "cc6cd622481ea366bb9067859446a8b01d92b468"
uuid = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
version = "1.15.3"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "852bd0f55565a9e973fcfee83a84413270224dc4"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.8.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.SetRounding]]
git-tree-sha1 = "d7a25e439d07a17b7cdf97eecee504c50fedf5f6"
uuid = "3cc68bcd-71a2-5612-b932-767ffbe40ab0"
version = "0.2.1"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SignalTemporalLogic]]
deps = ["InteractiveUtils", "Markdown", "PlutoUI", "Zygote"]
git-tree-sha1 = "40e6c51e6d2e7571de6ad1d4dc7e7c94a50f21dc"
uuid = "a79a9ddd-d50e-4d85-a979-5d85760e62a0"
version = "1.0.0"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SparseInverseSubset]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "52962839426b75b3021296f7df242e40ecfc0852"
uuid = "dc90abb0-5640-4711-901d-7e5b23a2fada"
version = "0.1.2"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "64cca0c26b4f31ba18f13f6c12af7c85f478cfde"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "83e6cce8324d49dfaf9ef059227f91ed4441a8e5"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.2"

[[deps.StanfordAA228V]]
deps = ["AbstractPlutoDingetjes", "BSON", "Base64", "Distributions", "Downloads", "ForwardDiff", "GridInterpolations", "LazySets", "LinearAlgebra", "Markdown", "Optim", "Parameters", "Pkg", "Plots", "Pluto", "PlutoUI", "ProgressLogging", "Random", "SignalTemporalLogic", "Statistics", "TOML"]
git-tree-sha1 = "5376632ae8604432fd7c8ce7308edf70950c1da8"
uuid = "6f6e590e-f8c2-4a21-9268-94576b9fb3b1"
version = "0.1.22"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "47091a0340a675c738b1304b58161f3b0839d454"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.10"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "29321314c920c26684834965ec2ce0dacc9cf8e5"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.4"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "b423576adc27097764a90e163157bcfc9acf0f46"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.2"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "5a3a31c41e15a1e042d60f2f4942adccba05d3c9"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.0"
weakdeps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "SparseArrays", "StaticArrays"]

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

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
version = "1.11.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "7822b97e99a1672bfb1b49b668a6d46d58d8cbcb"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.9"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "c0667a8e676c53d390a09dc6870b3d8d6650e2bf"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.22.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "975c354fcd5f7e1ddcc1f1a23e6e091d99e99bc8"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.4"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "b13c4edda90890e5b04ba24e20a310fbe6f249ff"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.3.0"
weakdeps = ["LLVM"]

    [deps.UnsafeAtomics.extensions]
    UnsafeAtomicsLLVM = ["LLVM"]

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "85c7811eddec9e7f22615371c3cc81a504c508ee"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+2"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "5db3e9d307d32baba7067b13fc7b5aa6edd4a19a"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.36.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "a2fccc6559132927d4c5dc183e3e01048c6dcbd6"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.5+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "7d1671acbe47ac88e981868a078bd6b4e27c5191"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.42+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "beef98d5aad604d9e7d60b2ece5181f7888e2fd6"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.6.4+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "326b4fea307b0b39892b3e85fa451692eda8d46c"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.1+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "3796722887072218eabafb494a13c963209754ce"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.4+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "9dafcee1d24c4f024e7edc92603cedba72118283"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+3"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2b0e27d52ec9d8d483e2ca0b72b3cb1a8df5c27a"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+3"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "807c226eaf3651e7b2c468f687ac788291f9a89b"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.3+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "02054ee01980c90297412e4c809c8694d7323af3"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+3"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "d7155fea91a4123ef59f42c4afb5ab3b4ca95058"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.6+3"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "6fcc21d5aea1a0b7cce6cab3e62246abd1949b86"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.0+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "984b313b049c89739075b8e2a94407076de17449"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.2+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "a1a7eaf6c3b5b05cb903e35e8372049b107ac729"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.5+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "b6f664b7b2f6a39689d822a6300b14df4668f0f4"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.4+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a490c6212a0e90d2d55111ac956f7c4fa9c277a6"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.11+1"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fee57a273563e273f0f53275101cd41a8153517a"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "1a74296303b6524a0472a8cb12d3d87a78eb3612"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "dbc53e4cf7701c6c7047c51e17d6e64df55dca94"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+1"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "ab2221d309eda71020cdda67a973aa582aa85d69"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+1"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b9ead2d2bdb27330545eb14234a2e300da61232e"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "622cf78670d067c738667aaa96c553430b65e269"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+0"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArrays", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "Requires", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "0b3c944f5d2d8b466c5d20a84c229c17c528f49e"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.75"

    [deps.Zygote.extensions]
    ZygoteColorsExt = "Colors"
    ZygoteDistancesExt = "Distances"
    ZygoteTrackerExt = "Tracker"

    [deps.Zygote.weakdeps]
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
    Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "27798139afc0a2afa7b1824c206d5e87ea587a00"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.5"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6e50f145003024df4f5cb96c7fce79466741d601"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.56.3+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0ba42241cb6809f1a278d0bcb976e0483c3f1f2d"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+1"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "522c1df09d05a71785765d19c9524661234738e9"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.11.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e17c115d55c5fbb7e52ebedb427a0dca79d4484e"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.2+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a22cf860a7d27e4f3498a0fe0811a7957badb38"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.3+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "b7bfd3ab9d2c58c3829684142f5804e4c6499abc"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.45+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "490376214c4721cdaca654041f635213c6165cb3"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+2"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

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
git-tree-sha1 = "63406453ed9b33a0df95d570816d5366c92b7809"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+2"
"""

# ╔═╡ Cell order:
# ╟─6b17139e-6caf-4f07-a607-e403bf1ad794
# ╠═14964632-98d8-4a2f-b2f6-e3f28b558803
# ╟─117d0059-ce1a-497e-8667-a0c2ef20c632
# ╟─60f72d30-ab80-11ef-3c20-270dbcdf0cc4
# ╟─f8612f4e-8519-4ee3-ac09-80b7c126b238
# ╟─9f739929-1cd3-4935-b229-ae3aeac7e131
# ╟─be8c37e8-45db-4198-b0b9-d287e73fb818
# ╟─0c520f93-49ce-45eb-899d-a31105d856c8
# ╟─da5b4000-0bce-4fc2-be85-dada21264ca3
# ╟─0456a732-2672-4108-a241-db9ae879a913
# ╟─6e8ab7c9-fb49-4d89-946d-c7d7588c199a
# ╟─fe044059-9102-4e7f-9888-d9f03eec69ff
# ╟─a21612a1-1092-4892-9132-629833e7c867
# ╟─ec776b30-6a30-4643-a22c-e071a365d50b
# ╟─18754cc6-c089-4245-ad10-2848594e49b4
# ╟─d566993e-587d-4aa3-995b-eb955dec5758
# ╟─e888241c-b89f-4db4-ac35-6d826ec4c36c
# ╟─c4fa9af9-1a79-43d7-9e8d-2854652a4ea2
# ╟─6bad6e8b-c021-41d2-afbb-bcd0242138dd
# ╟─dba42df0-3199-4c31-a735-b6b514703d50
# ╟─a0a60728-4ee0-4fd0-bd65-c056956b9712
# ╟─b0a4461b-89d0-48ee-9bcf-b544b9f08154
# ╟─109c3d27-2c23-48a7-9fd7-be8a1f359e55
# ╟─bc2f62f5-1330-46cd-bb81-411baa483488
# ╟─a46702a3-4a8c-4749-bd00-52f8cce5b8ee
# ╟─fd8c851a-3a42-41c5-b0fd-a12085543c9b
# ╟─17fa8557-9656-4347-9d44-213fd3b635a6
# ╠═22feee3d-4627-4358-9937-3c780b7e8bcb
# ╠═6f3e24de-094c-49dc-b892-6721b3cc54ed
# ╟─45f7c3a5-5763-43db-aba8-41ef8db39a53
# ╠═9c1daa96-76b2-4a6f-8d0e-f95d26168d2b
# ╟─370a15eb-df4b-493a-af77-00914b4616ea
# ╠═ab4c6807-5b4e-4688-b794-159e26a1599b
# ╟─e86d260f-c93d-4561-a9f1-44e4c7af827e
# ╟─166bd412-d433-4dc9-b874-7359108c0a8b
# ╟─9132a200-f63b-444b-9830-b03cf075021b
# ╠═c2ae204e-dbcc-453a-81f5-791ba4be39db
# ╟─42456abf-4930-4b01-afd1-fce3b4881e28
# ╠═bb3b33e6-fd05-4631-b0bd-c71ef3dbee38
# ╠═254956d0-8f58-4e2b-b8a9-5dd10dd074a2
# ╟─cc11217f-e070-4d20-8ebe-18e7eb977487
# ╟─73da2a56-8991-4484-bcde-7d397214e552
# ╠═3385fcb3-8b93-4da8-ba75-77877cc77ce4
# ╟─d4d057d7-cc9d-4949-9e3f-44a8aa67d725
# ╟─a6603deb-57fa-403e-a2e5-1195ae7c016c
# ╟─fe7f4a79-1a63-4272-a776-358a309c8550
# ╟─e52ffc4f-947d-468e-9650-b6c67a57a62b
# ╟─92f20cc7-8bc0-4aea-8c70-b0f759748fbf
# ╟─a003beb6-6235-455c-943a-e381acd00c0e
# ╟─f6589984-e24d-4aee-b7e7-db159ae7fea6
# ╠═fc2d34da-258c-4460-a0a4-c70b072f91ca
# ╟─c494bb97-14ef-408c-9de1-ecabe221eea6
# ╟─e2418154-4471-406f-b900-97905f5d2f59
# ╟─1789c8b5-b314-4aba-ad44-555be9a85984
# ╟─beaec161-ad89-4f83-9066-f420a1d04d39
# ╟─535261e3-4cb3-4b0b-954d-7452b2a91b5d
# ╟─c524297f-2bf3-4dd2-b7b4-fc5ce9a81738
# ╟─052cc2e3-ca8a-4043-9a7d-7947a7f1fd0c
# ╟─ce99f0cc-5fe8-42c2-af78-ac7211b6b699
# ╟─61173ec6-c7d6-44fa-8c47-5f7295dd49cf
# ╟─d647ac21-738b-43e7-bbbd-582b6294560e
# ╠═57d321cd-2029-4e49-8b56-9c5c48721ac4
# ╟─b61b8f52-9bf3-4766-9208-3e40134a3573
# ╟─57c5a6f0-2527-4988-9bf0-140495ba9b7e
# ╠═7910c15c-a231-4a0f-a4ed-1fe0b52f62c7
# ╟─cbc3a060-b4ec-4572-914c-e07880dd3537
# ╟─f286f3b2-3bac-4384-9b40-522e974a14ee
# ╟─307afd9c-6dac-4a6d-89d7-4d8cabfe3fe5
# ╟─d0a3770a-2c48-42db-9a71-6b7f695f22d8
# ╟─fda151a1-5069-44a8-baa1-d7903bc89797
# ╟─8c78529c-1e00-472c-bb76-d984b37235ab
# ╟─daada216-11d4-4f8b-807c-d347130a3928
# ╟─d18c2105-c2af-4dda-8388-617aa816a567
# ╠═77637b5e-e3ce-4ecd-90fc-95611af18002
# ╠═c4c0328d-8cb3-41d5-9740-0197cbf760c2
# ╟─b1e9bd40-a401-4630-9a1f-d61b276e72f7
# ╠═fe272c1b-421c-49de-a513-80c7bcefdd9b
# ╟─4ea18122-b681-4de1-89e3-5fb7ce2f7a0b
# ╟─a16cf110-4afa-4792-9d3f-f13b24349886
# ╠═44c8fbe0-21e7-482b-84a9-c3d32a4737dd
# ╟─772cf17e-0fdb-470e-9f12-9480af811edd
# ╠═f005da72-d7b5-4f01-8882-ed4e2bdcf4bd
# ╠═77a6e704-33e8-4241-84f0-0e58c29c06ef
# ╟─7ef66a50-6acc-474f-b406-7b27a7b18510
# ╟─e12b102e-785b-46e9-980c-e9f7943eda60
# ╟─d75f34d3-384c-486b-b648-61ef8fd52167
# ╟─bac5c489-553c-436f-b332-8a8e97126a51
# ╟─1da9695f-b7fc-46eb-9ef9-12160246018d
# ╟─0606d827-9c70-4a79-afa7-14fb6b806546
# ╟─9657f5ff-f21c-43c5-838d-402a2a723d5e
# ╠═cb7b9b9f-59da-4851-ab13-c451c26117df
# ╟─759534ca-b40b-4824-b7ec-3a5c06cbd23e
# ╟─7987c20d-68e8-441b-bddc-3f0ae7c3591d
# ╟─da2d692a-8378-435e-bd6b-c0e65caef542
# ╟─23999cd9-543b-47dc-a0b2-e133ba95891e
# ╟─38f26afd-ffa5-48d6-90cc-e3ec189c2bf1
# ╟─b417e370-efae-40e8-9247-5daf14fcc749
# ╟─60ab8107-db65-4fb6-aeea-d4978aed77bd
# ╟─aa0c4ffc-d7f0-484e-a1e2-7f6f92a3a53d
# ╟─e189b31e-7e24-4c32-989f-3e600a44d4bc
# ╟─7d054465-9f80-4dfb-9b5f-76c3977de7cd
# ╠═1ec68a39-8de9-4fd3-be8a-26cf7706d1d6
# ╟─d23f0299-981c-43b9-88f3-fb6e07927498
# ╠═641b92a3-8ff2-4aed-8482-9fa686803b68
# ╟─be426908-3fee-4ecd-b054-2497ce9a2e50
# ╠═258e14c4-9a2d-4515-9a8f-8cd96f31a6ff
# ╠═797cbe41-a5f3-4179-9143-9ef6e6888a4d
# ╟─35434537-9b9c-4528-b58c-420d01813598
# ╠═3328d818-391a-440a-8f1b-f2b7f3e00958
# ╠═06b14338-ea3b-45c8-bf6c-28b82db2ea70
# ╟─204feed7-cde8-40a8-b6b5-051a1c768fd9
# ╠═4ae85f59-4e94-48aa-8ccb-91311466c51f
# ╟─e3d6fdf1-3a9e-446b-8482-49d6f64b652e
# ╟─23fd490a-74d2-44b4-8a12-ea1460d95f85
# ╟─18a70925-3c2a-4317-8bbc-c2a096ec56d0
# ╟─45c79345-89da-498c-9a98-2ad55a0a6114
# ╠═3471a623-16af-481a-8f66-5bd1e7890188
# ╟─4c5210d6-598f-4167-a6ee-93bceda7223b
# ╟─2ba2d3a2-3f6c-4d5f-8c45-8d00947f6e05
# ╟─ea2d7eb7-d576-415c-ac4c-fea7f90de637
# ╟─7c473630-6555-4ada-85f3-0d40aefe6370
# ╟─7fe1c3d7-469c-47d9-9d46-e5b8b263edb9
# ╟─f6eb6d1a-a9a0-4234-8699-269a92f666c0
# ╟─74aeca7b-0658-427f-8c02-d093a0d725ee
# ╟─dbd088d1-f4c9-4e6a-b280-960b06da76e4
# ╟─1bb92755-65e3-457e-84cd-252eae5e4d7e
# ╟─6d5c805b-330c-4b04-a51c-15e674352b1b
# ╟─d9ab8278-eb76-4a36-aa0e-4ec74704f5e0
# ╟─860ec509-3a86-4842-9471-6b1a0b8f366d
# ╠═54741d81-39e0-4a47-b84d-c41c8eb7611b
# ╟─6559cf16-a474-4533-a2c7-ccbc02480a76
# ╠═cfdba748-45d5-4eaa-97b3-fdc9fe7e4333
# ╟─6beda870-0cb0-40f5-9531-fa3e2f7bb020
# ╠═5c3d24f6-0106-444c-b7df-89bba8c01b37
# ╟─4edc5933-9457-4c7c-8456-a26974e0587e
# ╟─95e3d42f-b33f-4294-81c5-f34a300dc9b4
# ╟─ba6c082b-6e62-42fc-a85c-c8b7efc89b88
# ╟─5aef09eb-ddc5-4b99-abf2-568621b871d5
# ╟─173388ab-207a-42a6-b364-b2c1cb335f6b
# ╟─50f1c7a3-9219-40ce-a060-77beb243b7af
# ╟─5563f0da-7552-4879-a38a-ba1748d39d52
# ╟─98cbe931-d362-4039-97ba-41e0049619a3
# ╟─247f4c17-bee1-4315-aff9-017407ef9219
# ╟─db7d4de5-9166-4e56-b5bc-1356e43286a9
# ╟─5a1ed20d-788b-4655-bdd8-069545f48929
# ╟─6c8b3077-876e-42fd-aa47-f3fa7c37f4dd
# ╟─97042a5e-9691-493f-802e-2262f2da4627
# ╟─9865ed62-b4fd-4e49-9259-3e5997c589f3
# ╟─ef084fea-bf4d-48d9-9c84-8cc1dd98f2d7
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
