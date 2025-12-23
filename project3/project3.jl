### A Pluto.jl notebook ###
# v0.20.20

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
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
	using Downloads
	using TOML
	using Test
	using Base64
	using PlutoUI
	using Distributions
	using Random
	using Plots
	using ForwardDiff
	import ForwardDiff: gradient, hessian, jacobian
	using IntervalArithmetic
	using Parameters
	using BSON
	using GridInterpolations
	using LinearAlgebra
	using LazySets
	using JuMP
	using SCS
	using Flux

	default(fontfamily="Computer Modern", framestyle=:box) # LaTeX-style plotting

	md"> _Additional package management._"
end

# ╔═╡ 268f31d2-e485-43e1-a925-48928bb3e360
if haskey(ENV, "AA228V_CI_SSH_KEY")
    # Check if age is installed
    if !success(`age --version`)
        @warn "age binary not found or not executable. Please install age."
    else
        # Create temp file that auto-cleans on exit
        mktempdir() do tmpdir
            tempfile = joinpath(tmpdir, "decrypted.jl")
            
            # Decrypt .p0 file using age with SSH key
            run(pipeline(
                `age --decrypt -i $(ENV["AA228V_CI_SSH_KEY"]) .p3`,
                tempfile
            ))
            
            # Include the decrypted file
            include(tempfile)
        end
    end
end

# ╔═╡ 117d0059-ce1a-497e-8667-a0c2ef20c632
md"""
# Project 3: Estimate reachable sets
_Please wait until the entire notebook is finished loading before proceeding (you may get temporary errors)._
"""

# ╔═╡ d7643abe-4619-4859-b2e3-9e932fe53b2f
highlight(md"""_See the three **"⟶ Task"** sections below for where to fill out the algorithms._""")

# ╔═╡ 78181077-5548-459d-970d-1d8a9d63b72c


# ╔═╡ da5b4000-0bce-4fc2-be85-dada21264ca3
textbook_details([
	"Chapter 8. _Reachability for Linear Systems_",
	"Chapter 9. _Reachability for Nonlinear Systems_",
])

# ╔═╡ 0456a732-2672-4108-a241-db9ae879a913


# ╔═╡ 6e8ab7c9-fb49-4d89-946d-c7d7588c199a
md"""
## Julia/Pluto tips
Useful tips you may be interested in regarding Julia, Pluto, and plotting.

[![html](https://img.shields.io/badge/static%20html-Julia+Pluto-0072B2)](https://sisl.github.io/AA228VLectureNotebooks/media/html/julia_pluto_session.html)
[![html](https://img.shields.io/badge/static%20html-Julia%20Plotting-0072B2)](https://sisl.github.io/AA228VLectureNotebooks/media/html/julia_plotting.html)
"""

# ╔═╡ a21612a1-1092-4892-9132-629833e7c867


# ╔═╡ ec776b30-6a30-4643-a22c-e071a365d50b
md"""
## Hints and FAQs
Expand the sections below for some helpful hints or refer to the FAQs.

[![faqs](https://img.shields.io/badge/AA228V-FAQs-0072B2)](https://github.com/sisl/AA228V-FAQs)
"""

# ╔═╡ 5aafffc6-8e3b-4591-99f6-6ee3b8082786
html_expand("Check out the <code>LazySets.jl</code> docs.", md"""
The documentation for the `LazySets` package is an excellent resource:
- [Latest documentation](https://juliareach.github.io/LazySets.jl/dev/)
- [`overapproximate`](https://juliareach.github.io/LazySets.jl/dev/lib/approximations/overapproximate/): Specifically the method that as an optional `[ε]::Real` error tolerance.
- [`box_approximation`](https://juliareach.github.io/LazySets.jl/dev/lib/approximations/box_approximation/)
""")

# ╔═╡ bd8cfc1d-cf0e-42e2-8e46-c763f142e8f3
html_expand("Stuck on the nonlinear system (medium)? Expand for hints on what to try.", hint(Markdown.parse("""Check out the _natural inclusion_ techniques in section 9.2 of the textbook.

We do _not_ restrict the number of vertices of your sets. So you can try other approaches detailed in chapter 9 _Reachability for Nonlinear System_ in the textbook.""")))

# ╔═╡ 5d389282-466d-47d8-ae69-0794cf620f27
html_expand("Stuck on the nonlinear neural network system (large)? Expand for hints on what to try.", hint(Markdown.parse("""Check out the _natural inclusion_ techniques in section 9.2 of the textbook.

We do _not_ restrict the number of vertices of your sets.

But if you're not satisfied with the (overly approximated) sets from natural inclusion, then check out the section on using `NeuralVerification.jl` ([link](#neural-verification)).

Note that algorithms like _Taylor inclusion_ and _conservative linearization_ are **unsound** for the nonlinear neural network because the ReLU activation functions are not continuous and differentiable (see details in section 9.7 of the textbook).""")))

# ╔═╡ 6bad6e8b-c021-41d2-afbb-bcd0242138dd


# ╔═╡ 1167fdfb-b097-4d18-b982-b1786794f8cf
html_space()

# ╔═╡ 55521bf3-eb1d-4ea1-9c18-aa8ba4461bff
html_expand("Expand for <code>SmallSystem</code> code.", md"""
```julia
## Agent
struct ProportionalController <: Agent
    k
end

(c::ProportionalController)(s, a=missing) = c.k' * s

Πo(agent::ProportionalController) = agent.k'

## Environment
@with_kw struct MassSpringDamper <: Environment
    m = 1.0
    k = 10.0
    c = 2.0
    dt = 0.05
end

function (env::MassSpringDamper)(s, a)
    return Ts(env) * s + Ta(env) * a
end

Ts(env::MassSpringDamper) = [1 env.dt; -env.k*env.dt/env.m 1-env.c*env.dt/env.m]
Ta(env::MassSpringDamper) = [0 env.dt/env.m]'
Ps(env::MassSpringDamper) = Product([Uniform(-0.2, 0.2), Uniform(-1e-12, 1e-12)])
𝒮₁(env::MassSpringDamper) = Hyperrectangle(low=[-0.2, 0.0], high=[0.2, 0.0])

## Sensor
struct AdditiveNoiseSensor <: Sensor
    Do
end

(sensor::AdditiveNoiseSensor)(s) = sensor(s, rand(Do(sensor, s)))
(sensor::AdditiveNoiseSensor)(s, x) = s + x

Do(sensor::AdditiveNoiseSensor, s) = sensor.Do
Os(sensor::AdditiveNoiseSensor) = I
```
""")

# ╔═╡ 17fa8557-9656-4347-9d44-213fd3b635a6
Markdown.parse("""
## Small system
The system is comprised of an `agent`, environment (`env`), and `sensor`.
""")

# ╔═╡ e93d0515-c7d4-4af6-9041-fda738af5caa
begin
	local ϵ = 0.5

	sys_small = System(
		ProportionalController([0.0, -1.0]),
		MassSpringDamper(),
		AdditiveNoiseSensor(Product([Uniform(-ϵ, ϵ), Uniform(-ϵ, ϵ)]))
	)
end;

# ╔═╡ 6f3e24de-094c-49dc-b892-6721b3cc54ed
SmallSystem::Type = typeof(sys_small) # Type used for multiple dispatch

# ╔═╡ 592e4e77-8104-4464-8e10-ee2834c7c0ab
Markdown.parse("""
## Small specification \$\\psi\$
The specification \$\\psi\$ (written `\\psi<TAB>` in code) indicates what the system should do:

\$\$\\psi(\\tau) = \\square\\left( |p| < 0.3 \\right)\$\$

i.e., "the state position \$p\$ in the trajectory \$\\tau\$ should _always_ (\$\\square\$) be less than \$0.3\$, anything else is a failure."

We use the `AvoidSetSpecification` to define a _failure set_ where we want to ensure that the reachable set \$\\mathcal{R}_{1:d}\$ does not intersect with its complement:

\$\$\\mathcal{R}_{1:d} \\cap \\neg\\psi = \\varnothing\$\$
""")

# ╔═╡ ab4c6807-5b4e-4688-b794-159e26a1599b
ψ_small = AvoidSetSpecification(
	HalfSpace([1.0, 0.0], -0.3) ∪ HalfSpace([-1.0, 0.0], -0.3));

# ╔═╡ 402c0eaa-727f-4c54-89ec-64c3dfb8002c
fbaseline(sys,ψ,seeds) =
	aggregate_performance(estimate_probability_baseline, sys, ψ; seeds);

# ╔═╡ 92f20cc7-8bc0-4aea-8c70-b0f759748fbf
Markdown.parse("""
## ⟶ **Task (Small)**: Estimate the reachable set
Please fill in the following `estimate_reachable_sets` function.
""")

# ╔═╡ a003beb6-6235-455c-943a-e381acd00c0e
start_code()

# ╔═╡ c494bb97-14ef-408c-9de1-ecabe221eea6
end_code()

# ╔═╡ 8082ce45-6e93-4d98-8f90-79935deadec8


# ╔═╡ 38f3d8cf-21cf-4c77-bd45-a618b9b2e1cd
highlight(md"""
We recommend adding `@progress` to for loops in the `reachable` algorithms, e.g.:

```julia
@progress for d in 1:alg.h
	# ...
end
```

This way you get incremental feedback that the algorithms are making progress.
""")

# ╔═╡ e2418154-4471-406f-b900-97905f5d2f59
html_quarter_space()

# ╔═╡ 1789c8b5-b314-4aba-ad44-555be9a85984
md"""
# 📊 Small Test
We'll automatically test your `estimate_reachable_sets(::SmallSystem, ψ)` function below.
"""

# ╔═╡ 97ddd327-94b8-4c2e-a79d-f5304725c25b
md"""
## Passing test criteria
Find a sequence of sets over time $t$ to depth $d$ that:
1. Over approximates the optimal reachable set using max $4$ vertices per time step.
2. The sets must not intersect with the failure region.
3. The sets must not be _under approximations_ of the optimal reachable sets.
"""

# ╔═╡ fda151a1-5069-44a8-baa1-d7903bc89797
html_space()

# ╔═╡ 6ec8a963-726d-4738-9a82-7e0b26b90b16
html_expand("Expand for <code>MediumSystem</code> code.", md"""
```julia
## Agent
struct ProportionalController <: Agent
    k
end

(c::ProportionalController)(s, a=missing) = c.k' * s

Πo(agent::ProportionalController) = agent.k'

## Environment
@with_kw struct InvertedPendulum <: Environment
    m::Float64 = 1.0
    l::Float64 = 1.0
    g::Float64 = 10.0
    dt::Float64 = 0.05
    ω_max::Float64 = 8.0
    a_max::Float64 = 2.0
end

function (env::InvertedPendulum)(s, a, xs=missing)
	θ, ω = s[1], s[2]
	dt, g, m, l = env.dt, env.g, env.m, env.l
	ω = ω + (3g / (2 * l) * sin(θ) + 3 * a / (m * l^2)) * dt # No `clamp`
	θ = θ + ω * dt
	return [θ, ω]
end

# Initial state distribution
Ps(env::InvertedPendulum) = Product([Uniform(-π/16, π/16), Uniform(-1.0, 1.0)])

## Sensor
struct AdditiveNoiseSensor <: Sensor
    Do
end

(sensor::AdditiveNoiseSensor)(s) = sensor(s, rand(Do(sensor, s)))
(sensor::AdditiveNoiseSensor)(s, x) = s + x

Do(sensor::AdditiveNoiseSensor, s) = sensor.Do
Os(sensor::AdditiveNoiseSensor) = I
```
""")

# ╔═╡ 86f92b1d-87eb-40b0-ad0b-6b888c5fb9cc
md"""
## Medium system
First we define a convenient struct that holds the parameters of the medium system.

$(highlight(md"May be useful when defining functions such as `sets` and `intervals`.

Please do not change these values."))
"""

# ╔═╡ 0ab3fe4f-a13d-4d92-b3e4-5653f05dafe7
@with_kw struct PendulumParameters
	disturbance_mag = 0.01
	θmin = -π/16
	θmax = π/16
	ωmin = -1.0
	ωmax = 1.0
end

# ╔═╡ 613443b1-ac8b-4570-b4c7-d107d63a36cd
md"""
An inverted pendulum is comprised of a `ProportionalController` with an `AdditiveNoiseSensor`.
"""

# ╔═╡ 99ccbce1-ac0b-4744-9c26-cb951490f482
md"""
We redefine the transition function as follows and remove clamping that was in Projects 1 and 2.
"""

# ╔═╡ 022f242e-f839-4b6a-b6ff-6ad3b09470a4
function (env::InvertedPendulum)(s, a, xs=missing)
	θ, ω = s[1], s[2]
	dt, g, m, l = env.dt, env.g, env.m, env.l
	ω = ω + (3g / (2 * l) * sin(θ) + 3 * a / (m * l^2)) * dt # No `clamp`
	θ = θ + ω * dt
	return [θ, ω]
end

# ╔═╡ 77637b5e-e3ce-4ecd-90fc-95611af18002
begin
	medium_params = PendulumParameters()

	local disturbance_mag = medium_params.disturbance_mag
	local θmin, θmax = medium_params.θmin, medium_params.θmax
	local ωmin, ωmax = medium_params.ωmin, medium_params.ωmax

	local agent = ProportionalController([-15.0, -8.0])
	local env = InvertedPendulum()

	# Different than Projects 1 and 2 (to be bounded)
	local sensor = AdditiveNoiseSensor(Product([
		Uniform(-disturbance_mag, disturbance_mag),
		Uniform(-disturbance_mag, disturbance_mag)]))

	# Different than Projects 1 and 2 (to be bounded)
	StanfordAA228V.Ps(env::InvertedPendulum) =
		Product([Uniform(θmin, θmax), Uniform(ωmin, ωmax)])

	sys_medium = System(agent, env, sensor)
end;

# ╔═╡ c4c0328d-8cb3-41d5-9740-0197cbf760c2
MediumSystem::Type = typeof(sys_medium) # Type used for multiple dispatch

# ╔═╡ 8c78529c-1e00-472c-bb76-d984b37235ab
begin
	# Different than Projects 1 and 2
	StanfordAA228V.get_depth(sys::MediumSystem) = 21

	Markdown.MD(
		md"""
		# 2️⃣ **Medium**: Inverted Pendulum
		The medium system is a swinging inverted pendulum.
		- It uses a proportional controller to keep it upright.
		- The state is comprised of the angle $\theta$ and angular velocity $\omega$ making $s = [\theta, \omega]$
		- Actions are left/right adjustments in the range $[-2, 2]$
		- Disturbances $x$ are treated as additive noise: $x \sim \mathcal{N}(\mathbf{0}, 0.1^2I)$
		""",
		depth_highlight(sys_medium)
	)
end

# ╔═╡ fd8c851a-3a42-41c5-b0fd-a12085543c9b
Markdown.MD(
	Markdown.parse("""
	# 1️⃣ **Small**: Mass-spring-damper
	The small system is a simple _mass-spring-damper_. The mass-spring-damper system consists of a mass \$m\$ attached to a wall by a spring with spring constant \$k\$ and a damper with damping coefficient \$c\$.
	- The state \$s\$ is the position (relative to the resting point) \$p\$ and velocity \$v\$ of the mass, \$s = [p,v]\$
	- The action is the force \$\\beta\$ applies to the mass.
	- The rollout depth is \$d=$(get_depth(sys_small))\$.
	- Disturbances are a noisy measurement of the state with uniform noise with \$\\epsilon = 0.5\$:

	\$\$\\begin{align}
		x_p \\sim \\mathcal{U}(-\\epsilon, \\epsilon) \\\\
		x_v \\sim \\mathcal{U}(-\\epsilon, \\epsilon)
	\\end{align}\$\$
	
	_See Appendix A.4 for more system details._
	"""),
	depth_highlight(sys_small)
)

# ╔═╡ 6a7b4f2b-187a-40a3-842f-126d220d40ed
md""" $t =$ $(@bind t_small Slider(1:get_depth(sys_small), show_value=true))

**Change the time $t$ slider to see the reachable set per time step.**"""

# ╔═╡ b1e9bd40-a401-4630-9a1f-d61b276e72f7
md"""
## Medium specification $\psi$
The inverted pendulum specification $\psi$ indicates what the system should do:

$$\psi(\tau) = \square\big(|\theta| < \pi/4\big)$$

i.e., "the absolute value of the pendulum angle $\theta$ (first element of the state $s$) in the trajectory $\tau$ should _always_ ($\square$) be less than $\pi/4$, anything else is a failure."
"""

# ╔═╡ fe272c1b-421c-49de-a513-80c7bcefdd9b
ψ_medium = LTLSpecification(@formula □(s -> abs(s[1]) < π / 4));

# ╔═╡ bac5c489-553c-436f-b332-8a8e97126a51
html_quarter_space()

# ╔═╡ 1da9695f-b7fc-46eb-9ef9-12160246018d
Markdown.parse("""
## ⟶ **Task (Medium)**: Estimate the reachable set
Please fill in the following `estimate_reachable_sets` function.
""")

# ╔═╡ 0606d827-9c70-4a79-afa7-14fb6b806546
start_code()

# ╔═╡ cb7b9b9f-59da-4851-ab13-c451c26117df
@medium function estimate_reachable_sets(sys::MediumSystem, ψ)
	# TODO: WRITE YOUR CODE HERE
end

# ╔═╡ 759534ca-b40b-4824-b7ec-3a5c06cbd23e
end_code()

# ╔═╡ 97dbe1e4-8045-4213-866f-6921c733fbeb


# ╔═╡ b1cf81ad-e5cb-40d7-b365-abda3fc67a88
highlight(md"""
We recommend adding `@progress` to for loops in the `reachable` algorithms, e.g.:

```julia
@progress for d in 1:alg.h
	# ...
end
```

This way you get incremental feedback that the algorithms are making progress.
""")

# ╔═╡ a228c1ac-62cb-4c18-89b9-bda4c3b1c5bb
html_quarter_space()

# ╔═╡ da2d692a-8378-435e-bd6b-c0e65caef542
md"""
# 📊 Medium Test
We'll automatically test your `estimate_reachable_sets(::MediumSystem, ψ)` function below.
"""

# ╔═╡ 50199170-f120-48ee-879c-bbdc11618f1e
md""" $t =$ $(@bind t_medium Slider(1:get_depth(sys_medium), show_value=true))

**Change the time $t$ slider to see the reachable set per time step.**"""

# ╔═╡ 428575d2-66a9-4a19-9eb0-cdf9b55277f7
md"""
## Passing test criteria
Find a sequence of sets over time $t$ to depth $d$ that:
1. Over approximates the reachable set (no limit on max vertices).
2. The sets _may_ intersect with the failure region.
    - Although, for better leaderboard scores you may want to avoid this.
3. The sets must not be _under approximations_ of the approximately optimal reachable sets.
"""

# ╔═╡ 60ab8107-db65-4fb6-aeea-d4978aed77bd
html_space()

# ╔═╡ f8ea2983-c2d0-40ea-b949-9fc478ea45f8
Markdown.parse("""_The figure above shows several states_ \$s_t\$ _sampled from trajectories and a single trajectory shown as a line._""")

# ╔═╡ 5aae63b2-54ec-421b-84ec-0d4bc9c00c10
html_expand("Expand for <code>LargeSystem</code> code (surrogate).", md"""
```julia
## Agent
struct NoAgent <: Agent end
(c::NoAgent)(s, a=missing) = nothing
Distributions.pdf(c::NoAgent, s, x) = 1.0

## Environment
@with_kw struct ContinuumWorldSurrogate <: Environment
    cw::ContinuumWorld = ContinuumWorld()
    model::Chain
    disturbance_mag = 0.1
end

(env::ContinuumWorldSurrogate)(s, a) = env(s, a, rand(Ds(env, s, a)))
(env::ContinuumWorldSurrogate)(s, a, x) = env.model(s) + x # Call neural network

Ps(env::ContinuumWorldSurrogate) = Ps(env.cw)

Ds(env::ContinuumWorldSurrogate, s, a) =
	Product([
	    Uniform(-env.disturbance_mag, env.disturbance_mag),
	    Uniform(-env.disturbance_mag, env.disturbance_mag)])

## Sensor
struct IdealSensor <: Sensor end

(sensor::IdealSensor)(s) = s
(sensor::IdealSensor)(s, x) = sensor(s)

Distributions.pdf(sensor::IdealSensor, s, xₛ) = 1.0
```
""")

# ╔═╡ 790f1562-d6ff-4c44-b1ea-1b0ab1dced85
html_expand("Expand for <code>LargeSystem</code> code (original).", md"""
```julia
## Agent
struct InterpAgent <: Agent
    grid::RectangleGrid
    Q
end

(c::InterpAgent)(s) = argmax([interpolate(c.grid, q, s) for q in c.Q])
(c::InterpAgent)(s, x) = c(s)

Distributions.pdf(c::InterpAgent, o, xₐ) = 1.0

function load_cw_policy(filename::String)
    res = BSON.load(filename)
	grid = res[:grid]
	Q = res[:Q]
    return grid, Q
end

## Environment
@with_kw struct ContinuumWorld <: Environment
	size = [10, 10]                          # dimensions
	terminal_centers = [[4.5,4.5],[6.5,7.5]] # obstacle and goal centers
	terminal_radii = [0.5, 0.5]              # radius of obstacle and goal
	directions = [[0,1],[0,-1],[-1,0],[1,0]] # up, down, left, right
end

(env::ContinuumWorld)(s, a) = env(s, a, rand(Ds(env, s, a)))

function (env::ContinuumWorld)(s, a, x)
	dir = env.directions[a]
	return s .+ dir
end

Ps(env::ContinuumWorld) = Product([Uniform(0,1), Uniform(0,1)])
Ds(env::ContinuumWorld, s, a) = Deterministic()

## Sensor
struct IdealSensor <: Sensor end

(sensor::IdealSensor)(s) = s
(sensor::IdealSensor)(s, x) = sensor(s)

Distributions.pdf(sensor::IdealSensor, s, xₛ) = 1.0
```
""")

# ╔═╡ b608b8b2-5166-419b-ac4e-18e87c93ac01
md"""
## Large system
First we define a convenient struct that holds the parameters of the large system.

$(highlight(md"May be useful when defining functions such as `sets` and `intervals`.

Please do not change these values."))
"""

# ╔═╡ 05892ece-4c38-4c39-841a-f8b9333ba3ef
@with_kw struct ContinuumWorldParameters
	disturbance_mag = 0.1 # disturbance magnitude x = [𝒰(-mag, mag), 𝒰(-mag, mag)]
	xmin = 0 # x initial state x minimum
	xmax = 1 # x initial state x maximum
	ymin = 0 # y initial state y minimum
	ymax = 1 # y initial state y maximum
end

# ╔═╡ 7d054465-9f80-4dfb-9b5f-76c3977de7cd
md"""
The continuum world system uses the neural network surrogate to predict next states.
"""

# ╔═╡ 005966cb-763d-4346-8fdb-e336a9514e8c
md"""
### Neural network surrogate 🤖
We trained a 3 layer **ReLU-based** neural network to predict next states $\hat{s}'$ given the current state $s$.
"""

# ╔═╡ 94a5b84a-b942-4317-b255-f49cc13532c6
hint(md"""Since we trained a **ReLU-based** network, what algorithms are _not_ applicable (i.e., not **sound**)?

Think: _Is ReLU continuous and differentiable?_""", title="Hint: What can we apply?")

# ╔═╡ 6a38fe32-d862-4d7d-9978-290a152ae575
model = BSON.load("cw_model.bson")[:model]

# ╔═╡ 1ec68a39-8de9-4fd3-be8a-26cf7706d1d6
begin
	large_params = ContinuumWorldParameters()

	local agent = NoAgent()
	local env = ContinuumWorldSurrogate(
		cw=ContinuumWorld(),
		model=model,
		disturbance_mag=large_params.disturbance_mag,
	)
	local sensor = IdealSensor()

	sys_large = System(agent, env, sensor)
end;

# ╔═╡ aa0c4ffc-d7f0-484e-a1e2-7f6f92a3a53d
Markdown.MD(
Markdown.parse("""
# 3️⃣ **Large**: Continuum World
The large system is an agent in the continuum world (continuous grid world).

The objective is to get to the _green_ goal region and avoid the _red_ failure region.

### (Original) Continuum World System
To train our surrogate neural network, we define the "original" continuum world system that uses a lookup-based policy **which we cannot easily apply reachability to**.
- States are the \$[x,y]\$ positions in a \$10\\times10\$ continuous world.
    - With an initial state distribution uniformly in \$[0,1]\$ for both \$x\$ and \$y\$.
- Actions \$a\$ are the cardinal directions `up`, `down`, `left`, `right`.
    - It uses an interpolated lookup-table policy.
- The transition dynamics apply the action \$a\$ to the current state.
    - e.g., \$s' = s + a\$ where \$a \\in \\big\\{[0,1],\\, [0,-1],\\, [-1,0],\\, [1,0]\\big\\}\$
- We do not apply disturbances to the original environment.

### (Surrogate) Continuum World System
- Same state space and initial state distribution as the original system.
- The agent does not directly take actions, but instead will predict the next state \$\\hat{s}'\$ using the trained _**neural network surrogate**_.
- Disturbances \$x\$ are applied to the predicted next state \$\\hat{s}'\$ as "slip" noise. The agent slips in a random direction modeled by adding a random vector \$x = [x_x, x_y]\$ sampled from two independent uniform distributions:
\$\$
\\begin{align}
x_x &\\sim \\mathcal{U}(-$(large_params.disturbance_mag), $(large_params.disturbance_mag)) \\\\
x_y &\\sim \\mathcal{U}(-$(large_params.disturbance_mag), $(large_params.disturbance_mag))
\\end{align}\$\$
- Thus, the transition dynamics apply the disturbances to the predicted next state:
\$\$s' = \\hat{s}' + x\$\$

_Modified from the system details in Appendix A.7._"""),
    depth_highlight(sys_large)
)

# ╔═╡ 29279d51-1162-4fa8-bdd5-f0ff0e4ef968
LargeSystem::Type = typeof(sys_large) # Type used for multiple dispatch

# ╔═╡ 9f739929-1cd3-4935-b229-ae3aeac7e131
begin
	ThisProject = Project3
	max_vertices(sys::SmallSystem)  = 4
	max_vertices(sys::MediumSystem) = Inf
	max_vertices(sys::LargeSystem)  = Inf
end;

# ╔═╡ 60f72d30-ab80-11ef-3c20-270dbcdf0cc4
Markdown.MD(
	Markdown.parse("""
**Task**: Estimating the reachable sets of three systems.
- **Small system**: Mass-spring-damper (_**linear system**_).
    - A maximum of \$n = $(format(max_vertices(sys_small); latex=true))\$ vertices per time step is allotted for the small system.
- **Medium system**: Swinging inverted pendulum (_**nonlinear system**_).
    - No restriction on number of vertices.
- **Large system**: Continuum world (_**nonlinear neural network**_).
    - No restriction on number of vertices.
"""),
highlight(md"""Your job is to write the following function that returns the estimated reachable sets for each time step from $1$ to depth $d$ (use `get_depth(sys)` to get `d`)."""),
md"""
```julia
estimate_reachable_sets(sys, ψ)
```
##### Note on the return type:

The return type of `estimate_reachable_sets` should either be:
- a [`UnionSet`](https://juliareach.github.io/LazySets.jl/dev/lib/lazy_operations/UnionSet) or [`UnionSetArray`](https://juliareach.github.io/LazySets.jl/dev/lib/lazy_operations/UnionSet/#def_UnionSetArray) (from `LazySets.jl`)
    - **This is already the type that is output from the `reachable` algorithms in the textbook.**
- or a `Vector{<:LazySet}` (see [docs](https://juliareach.github.io/LazySets.jl/dev/lib/interfaces/LazySet/#LazySets.LazySet) for the `LazySet` subtypes)
   - If you want to run your own custom reachability code, we also accept a `Vector` of `LazySet`s for each time step.
""")

# ╔═╡ c4fa9af9-1a79-43d7-9e8d-2854652a4ea2
html_expand("Stuck on the linear system (small)? Expand for hints on what to try.", hint(Markdown.parse("""Check out the _set propagation_ techniques in section 8.2 of the textbook.

Being a linear system, we can solve the reachability problem _**exactly**_. But we restrict the maximum number of vertices per time step to be \$n = $(format(max_vertices(sys_small); latex=true))\$ to make it more interesting.""")))

# ╔═╡ fc2d34da-258c-4460-a0a4-c70b072f91ca
@small function estimate_reachable_sets(sys::SmallSystem, ψ; n=max_vertices(sys))
	# TODO: WRITE YOUR CODE HERE
end

# ╔═╡ 307afd9c-6dac-4a6d-89d7-4d8cabfe3fe5
Markdown.MD(
	md"""
  $(@bind rerun_small LargeCheckBox(text="⟵ Click to re-run the <code>SmallSystem</code> evaluation.", default=parse(Bool, get(ENV, "AA228V_RUN_EVALS", "false"))))""",
	Markdown.parse("""
	↑ This will re-run **`estimate_reachable_sets(::SmallSystem, ψ)`** and re-save **`$(get_filename(sys_small, ThisProject))`**

	_Uncheck this to load results from the file._
	""")
)

# ╔═╡ 38f26afd-ffa5-48d6-90cc-e3ec189c2bf1
Markdown.MD(
	md"""
  $(@bind rerun_medium LargeCheckBox(text="⟵ Click to re-run the <code>MediumSystem</code> evaluation.", default=parse(Bool, get(ENV, "AA228V_RUN_EVALS", "false"))))""",
	Markdown.parse("""
	↑ This will run **`estimate_reachable_sets(::MediumSystem, ψ)`** and save **`$(get_filename(sys_medium, ThisProject))`**

	_Uncheck this to load results from the file._
	""")
)

# ╔═╡ 4eeaa9ae-eac5-478a-aca5-82de3dda24f7
submission_details(@bind(directory_trigger, OpenDirectory(@__DIR__)), ThisProject,
	[SmallSystem, MediumSystem, LargeSystem])

# ╔═╡ 0c520f93-49ce-45eb-899d-a31105d856c8
if directory_trigger
	@info "Opening local directory..."
	sleep(1)
end

# ╔═╡ 2a81d0a0-a609-4d20-bddb-e5cbe65caf4b
md"""
### Training code

If you're interested in the `Flux.jl` training code, expand this section below.
"""

# ╔═╡ 3ae14115-f7d6-495b-a3fa-d13d0b0cdd54
html_expand("Expand for the <code>Flux</code> training code.", md"""
You can use `CUDA` if your machine supports CUDA-based GPUs.

```julia
using Flux
using Metal # Training on Apple silicon
```

We define the "original" system to collect data from:
```julia
local grid, Q = load_cw_policy(joinpath(@__DIR__, "cw_policy.bson"))

cw_agent_orig = InterpAgent(grid, Q)
cw_env_orig = ContinuumWorld()
cw_sensor_orig = IdealSensor()
sys_large_orig = System(cw_agent_orig, cw_env_orig, cw_sensor_orig)

# Type used for multiple dispatch
LargeSystemOriginal::Type = typeof(sys_large_orig)
```

For data collection, we collect $(s, s')$ pairs which consist of $s = [x,y]$ and $s' = [x', y']$ state values (using the original continuum world system).

```julia
function data_collection(sys::LargeSystemOriginal; n=100_000, seed=4)
	Random.seed!(seed)
	𝒟 = []
	d = get_depth(sys)
	for i in 1:n
		τ = rollout(sys; d)
		𝐬 = [deepcopy(step.s) for step in τ[1:end-1]]
		𝐬′ = [deepcopy(step.s) for step in τ[2:end]]
		push!(𝒟, collect(zip(𝐬, 𝐬′))...)
	end
	return 𝒟
end
```

We train a simple 3 (hidden) layer ReLU-based neural network to predict next states $\hat{s}'$ given the current state $s$. We minimize the mean-squared error (MSE) loss function using the Adam optimizer with a learning rate of $\alpha = 5{e-}4$ for $100$ epochs.

```julia
function train(sys::LargeSystemOriginal; epochs=100, lr=5e-4, seed=4)
	𝒟 = data_collection(sys; seed)
	𝐬 = first.(𝒟)
	𝐬′ = last.(𝒟)
	inputs::Matrix{Float32} = hcat(first.(𝐬), last.(𝐬))'
	targets::Matrix{Float32} = hcat(first.(𝐬′), last.(𝐬′))'
	loader = Flux.DataLoader((inputs, targets), batchsize=length(𝐬)÷50)

	Random.seed!(seed)

	model = Chain(
		Dense(2, 6, relu),
		Dense(6, 6, relu),
		Dense(6, 6, relu),
		Dense(6, 2) # next state s′
	)
	model = gpu(model)
	opt = Flux.setup(Flux.Adam(lr), model)

	@withprogress for epoch in 1:epochs
		for (s, s′) in loader
			s, s′ = gpu(s), gpu(s′)
			loss, grads = Flux.withgradient(model) do m
				ŝ′ = m(s)
				Flux.mse(ŝ′, s′)
			end
			Flux.update!(opt, model, grads[1])
			@logprogress "[$epoch]\n$loss" epoch/epochs
		end
	end

	model = cpu(model)
	model = fmap(f64, model)
	return model
end
```

Then we run `train` and save the model.

```julia
model = train(sys_large_orig)
BSON.@save "cw_model.bson" model
```
""")

# ╔═╡ be426908-3fee-4ecd-b054-2497ce9a2e50
md"""
## Large specification $\psi$
The continuum world specification $\psi$ indicates what the system should do:

$$\psi(\tau) = \lozenge(G) \wedge \square(\neg F)$$

where

$$\begin{gather}
G(s_t): \lVert s_t - \text{goal} \rVert_2 \le r_g \tag{$s_t$ is in the goal region} \\
F(s_t): \lVert s_t - \text{fail} \rVert_2 \le r_f \tag{$s_t$ is in the obstace region}
\end{gather}$$

where $r_g$ and $r_f$ are the goal region and obstacle region radii, respectively.

i.e., "the agent must _eventually_ ($\lozenge$) reach the goal and _always_ ($\square$) avoid the obstace region."
"""

# ╔═╡ 258e14c4-9a2d-4515-9a8f-8cd96f31a6ff
begin
	local goal = sys_large.env.cw.terminal_centers[2]
	local fail = sys_large.env.cw.terminal_centers[1]
	local rg = sys_large.env.cw.terminal_radii[2]
	local rf = sys_large.env.cw.terminal_radii[1]

	local G = @formula s->norm(s .- goal) ≤ rg
	local F = @formula s->norm(s .- fail) ≤ rf

	ψ_large = LTLSpecification(@formula ◊(G) ∧ □(¬F))
end;

# ╔═╡ e3d6fdf1-3a9e-446b-8482-49d6f64b652e
html_quarter_space()

# ╔═╡ 23fd490a-74d2-44b4-8a12-ea1460d95f85
Markdown.parse("""
## ⟶ **Task (Large)**: Estimate the reachable set
Please fill in the following `estimate_reachable_sets` function.
""")

# ╔═╡ 18a70925-3c2a-4317-8bbc-c2a096ec56d0
start_code()

# ╔═╡ 3471a623-16af-481a-8f66-5bd1e7890188
@large function estimate_reachable_sets(sys::LargeSystem, ψ)
	# TODO: WRITE YOUR CODE HERE
end

# ╔═╡ 4c5210d6-598f-4167-a6ee-93bceda7223b
end_code()

# ╔═╡ ef6bf2ba-748e-4f43-ad53-05d1936c2ba9


# ╔═╡ 0a496e93-5853-46bd-a3c5-6f466df90441
highlight(md"""
We recommend adding `@progress` to for loops in the `reachable` algorithms, e.g.:

```julia
@progress for d in 1:alg.h
	# ...
end
```

This way you get incremental feedback that the algorithms are making progress.
""")

# ╔═╡ ba82b4e4-413c-4b78-a777-85d03e3554f4
html_quarter_space()

# ╔═╡ ba780732-7aaa-4e09-9c40-304aa62e564b
html"""
<h3 id='neural-verification'>(Optional) Not satisfied with your solution?</h3>
<p>This <code>import</code> may be a spoiler 🙃</p>
"""

# ╔═╡ 8cc343cc-f5be-4ff4-802f-adcb6a77674a
import NeuralVerification: Ai2, forward_network, network

# ╔═╡ b4a0cef4-25f6-4b5d-a739-6f9583d6fb3d
hint(md"""
Although a simple application of _natural inclusion_ will pass the autograder, you may not be satisfied with this solution for the large problem.

**Expand the section below for some hints on what else to try.**""")

# ╔═╡ 16502371-5139-40de-be3e-0926f55ce405
html_expand("Expand for a hints on a better solution to the large problem.", [
	html"<h2hide><code>NeuralVerification.jl</code></h2hide>",
	md"""
	The `NeuralVerification.jl` package developed by our lab, SISL, has an extensive library of implemented state-of-the-art neural network verification algorithms.
	- [Link to the documentation.](https://sisl.github.io/NeuralVerification.jl/latest/)

	You can use this package to find a **sound** solution (unlike _Taylor inclusion_ and _conservative linearization_) and a better solution than _natural inclusion_.""",

	html"<h2hide>Implement this!</h2hide>",
	md"""
	Copy the following function `nvreach` to a new cell and implement the comments `1-4` if you want to gain experience with state-of-the-art neural network verification!

	```julia
	function nvreach(sys::LargeSystem, d)
		solver = Ai2()               # We recommend using the AI² solver (see [1])
		net = network(sys.env.model) # Convert Flux model → NeuralVerification network
		𝒮, 𝒳 = sets(sys, 1)          # Get initial state set 𝒮 and disturbance set 𝒳
		ℛs = LazySet[𝒮]              # Initialze reachable sets w/ initial state set 𝒮
	
		@progress for t in 2:d
			# TODO:
			# 1. Forward pass 𝒮 through the net: forward_network(solver, net, 𝒮)
			# 2. Then what? ... \oplus<TAB> may be useful...
			# 3. Tip: concretize afterwards.
			# 4. Push to ℛs
		end
	
		return UnionSetArray([ℛs...])
	end
	```

	Then call this function within your `estimate_reachable_sets(sys::LargeSystem, ψ)` like so:
	```julia
	d = get_depth(sys)
	return nvreach(sys, d)
	```

	**(Or you can put the internals of `nvreach` directly into `estimate_reachable_sets`)**

	If you're curious how to type certain variables used in the `nvreach` function, see the table below.

	| Variable | How to produce it |
	| :------- | :--------------------------------------------- |
	| `𝒮`      | `\scrS<TAB>`                                   |
	| `𝒮′`     | `\scrS<TAB>\prime<TAB>`                        |
	| `𝒮′ ⊕ 𝒳` | `\scrS<TAB>\prime<TAB> \oplus<TAB> \scrX<TAB>` |
	| `ℛs`     | `\scrR<TAB>s`                                  |
	""",

	html"<h2hide>AI² solver</h2hide>",
	md"""
	The `nvreach` function uses the AI² solver [^1], which is a state-of-the-art neural network verification algorithm that works on ReLU-based networks. For more technical details, please refer to the paper below.
	- [Link to AI² paper.](https://www.cs.utexas.edu/~swarat/pubs/ai2.pdf)
	- [Link to AI² `solve` functionality.](https://github.com/sisl/NeuralVerification.jl/blob/9802458c3055ecdae7ecdebd45f680f11af76c67/src/reachability/ai2.jl#L54)

	[^1] T. Gehr, M. Mirman, D. Drashsler-Cohen, P. Tsankov, S. Chaudhuri, and M. Vechev, **"AI²: Safety and Robustness Certification of Neural Networks with Abstract Interpretation,"** in *2018 IEEE Symposium on Security and Privacy (SP)*, 2018.
	""",
	try LocalResource(joinpath(@__DIR__, "..", "media", "ai2.png")) catch end,

	html"<h2hide>Wanna go further?</h2hide>",
	hint(md"Maybe a little 🄿 🄰 🅁 🅃 🄸 🅃 🄸 🄾 🄽 🄸 🄽 🄶?"),
])

# ╔═╡ 5c490a85-cff5-46bb-a6fa-330c81d4cd3b
html_quarter_space()

# ╔═╡ ea2d7eb7-d576-415c-ac4c-fea7f90de637
md"""
# 📊 Large Test
We'll automatically test your `estimate_reachable_sets(::LargeSystem, ψ)` function below.
"""

# ╔═╡ 7fe1c3d7-469c-47d9-9d46-e5b8b263edb9
Markdown.MD(
	md"""
  $(@bind rerun_large LargeCheckBox(text="⟵ Click to re-run the <code>LargeSystem</code> evaluation."), default=parse(Bool, get(ENV, "AA228V_RUN_EVALS", "false"))""",
	Markdown.parse("""
	↑ This will re-run **`estimate_reachable_sets(::LargeSystem, ψ)`** and re-save **`$(get_filename(sys_large, ThisProject))`**

	_Uncheck this to load results from the file._
	""")
)

# ╔═╡ 571c8a87-fd6d-490d-8a54-ad76f35f0af4
md""" $t =$ $(@bind t_large Slider(1:get_depth(sys_large), show_value=true))

**Change the time $t$ slider to see the reachable set per time step.**"""

# ╔═╡ a434dd23-c4d4-498c-9195-10a9c5813e35
@bind fun LargeCheckBox(text="⟵ Click here for some fun! 🌈 🦄")

# ╔═╡ b5b95a2e-1b81-4b4a-bf12-38041ff01059
md"""
## Passing test criteria
Find a sequence of sets over time $t$ to depth $d$ that:
1. Over approximates the reachable set (no limit on max vertices).
2. The sets _may_ intersect with the failure region.
    - Although, for better leaderboard scores you may want to avoid this.
3. The sets must not be _under approximations_ of the approximately optimal reachable sets.
"""

# ╔═╡ d1443daa-4cd7-4258-be76-c8a4be9778a6
if fun
	cmap = palette(["#5ebd3e","#ffb900","#f78200","#e23838","#973999","#009cdf"])
else
	cmap = palette(reverse([get(cgrad(:viridis), i) for i in range(0.2, 0.8, 5)]))
end;

# ╔═╡ 74aeca7b-0658-427f-8c02-d093a0d725ee
html_quarter_space()

# ╔═╡ 6d5c805b-330c-4b04-a51c-15e674352b1b
html_quarter_space()

# ╔═╡ cfdba748-45d5-4eaa-97b3-fdc9fe7e4333
𝐰 = [1,10,100]

# ╔═╡ 860ec509-3a86-4842-9471-6b1a0b8f366d
Markdown.parse("""
## Comparing reachable volumes
Since the reachable volumes across the three problems vary widely in range, we weight the errors using the weights \$\\mathbf{w} = [$(𝐰[1]),$(𝐰[2]),$(𝐰[3])]\$ (normalized to sum to one):

\$\$\\bar{w_i} = \\frac{w_i}{\\sum_j w_j}\$\$

""")

# ╔═╡ 6beda870-0cb0-40f5-9531-fa3e2f7bb020
md"""
The final score on the leaderboard is then a weighted sum:

$$\begin{gather}
\mathbf{s} = \big[\text{vol}_\text{small},\, \text{vol}_\text{medium},\, \text{vol}_\text{large} \big] \\
\text{score} = \mathbf{w}^\top\mathbf{s}
\end{gather}$$
"""

# ╔═╡ 5c3d24f6-0106-444c-b7df-89bba8c01b37
function leaderboard_scores(𝐬, 𝐰=ones(length(τs)))
	𝐰 = 𝐰 ./ sum(𝐰)
	return 𝐰'𝐬
end

# ╔═╡ 4edc5933-9457-4c7c-8456-a26974e0587e
html_half_space()

# ╔═╡ 20cb2d9b-ad2d-4d06-be09-03bd5396687a
begin
	function task_description(sys, details)
		if isinf(max_vertices(sys))
			max_vertices_statement = "**Note**: No restrictions on the number of vertices"
			n_arg = ""
			spec = "a specification"
		else
			max_vertices_statement = "**Note**: \$n = $(format(max_vertices(sys); latex=true))\$ maximum number of vertices per time step"
			n_arg = "; n"
			spec = "an avoid-set specification"
		end
		max_vertices_highlight = highlight(Markdown.parse("$max_vertices_statement for the `$(system_name(sys))`."))

		return Markdown.MD(
			Markdown.parse(
		"""
			estimate_reachable_sets(sys::$(system_name(sys)), ψ$n_arg)::LazySet
		
		A function that takes in a system `sys` ($details) and $spec `ψ` and **returns the overapproximated reachable set**.
		"""),
		max_vertices_highlight,
		md"""		
		**Note**: `ψ` is written as `\psi<TAB>`, `ℛ` is written as `\scrR<TAB>`
		""")
	end
end; md"> _Project 3 specific functions._"

# ╔═╡ d0a25025-9309-463f-a09a-9d7ea3df8143
task_description(sys_small, "mass-spring-damper for the _small_ setting")

# ╔═╡ f180bd3a-12da-4942-b2af-2df2f5887201
task_description(sys_medium, "inverted pendulum for the _medium_ setting")

# ╔═╡ 45c79345-89da-498c-9a98-2ad55a0a6114
task_description(sys_large, "continuum world system for the _large_ setting")

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

	function extract(env::MassSpringDamper, x)
		s = x[1:2]
		𝐱 = [Disturbance(0, 0, x[i:i+1]) for i in 3:2:length(x)]
		return s, 𝐱
	end

	function extract(env::InvertedPendulum, x)
		s = x[1:2]
		𝐱 = [Disturbance(0, 0, x[i:i+1]) for i in 3:2:length(x)]
		return s, 𝐱
	end

	function extract(env::ContinuumWorldSurrogate, x)
		s = x[1:2]
		𝐱 = [Disturbance(0, x[i:i+1], 0) for i in 3:2:length(x)]
		return s, 𝐱
	end

	md"> *Helper `extract` functions.*"
end

# ╔═╡ 70e5bfca-7172-4f21-b3e6-e31ac16c4add
begin
	function sets(sys::Any, d::Any)
		# Hey, you should define `sets` with the specific `sys` type, e.g.:
		error("""Please define `sets(sys, d)` with the specific `sys` type:
			sets(sys::MediumSystem, d)
			sets(sys::LargeSystem, d)
		""")
	end


	function intervals(sys::Any, d::Any)
		# Hey, you should define `intervals` with the specific `sys` type, e.g.:
		error("""Please define `intervals(sys, d)` with the specific `sys` type:
			intervals(sys::MediumSystem, d)
			intervals(sys::LargeSystem, d)
		""")
	end

	md"> *Generic `sets` and `intervals` functions.*"
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

# ╔═╡ 851943ba-d6f8-4e97-bbcd-dec5eb8b8b76
try let
	msd = dark_mode ? "mass-spring-damper-dark.svg" : "mass-spring-damper.svg"
	LocalResource(joinpath(@__DIR__, "..", "media", msd))
end catch end

# ╔═╡ daada216-11d4-4f8b-807c-d347130a3928
try	LocalResource(joinpath(@__DIR__, "..", "media", dark_mode ? "inverted_pendulum_dark.svg" : "inverted_pendulum.svg")) catch end

# ╔═╡ 7b3f2065-0521-4d89-8125-b9c09565d586
try LocalResource(joinpath(@__DIR__, "..", "media", dark_mode ? "cw-neural-network-dark.svg" : "cw-neural-network.svg")) catch end

# ╔═╡ 02fac8f9-b442-40d7-b3f3-415a10570e8e
begin
	DarkModeHandler.setdarkmode!(dark_mode)

	import StanfordAA228V:
		plot, plot!, plot_pendulum, plot_optimal, 
		plot_msd_time_axis, plot_msd_traj!,
		plot_pendulum_state, plotsamples!, plotoutsiders!, plotset!,
		plot_cw_full_reachability, plot_cw_reachability, plot_cw_trajectory!, ¬
		# import plotting for dark_mode triggers

	function separator(rulecolor=dark_mode ? "#ffffff26" : "#00000026")
		Markdown.parse("""# \${\\color{$rulecolor}\\rule{1000px}{3px}}\$""")
	end

	pkg_trigger = true
	md"> _AA228V/CS238V package management._"
end

# ╔═╡ 3e392ff6-6a32-49cc-9eab-82ae2d6e9154
separator()

# ╔═╡ c498c964-9c36-4859-9b43-c203f5aa3390
separator()

# ╔═╡ 87c29996-943e-43c8-9a85-9f90689a63ae
separator()

# ╔═╡ 6bffdf52-4c8d-464e-acad-ec21d528bdbd
separator()

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
	Div = UsingThisViolatesTheHonorCode.Div
	divcenter = UsingThisViolatesTheHonorCode.divcenter
	centered = UsingThisViolatesTheHonorCode.centered

	small_system_linear_reachability =
		UsingThisViolatesTheHonorCode.small_system_linear_reachability
	ℛ_small_optimal = small_system_linear_reachability(sys_small)
	𝐫_optimal = extract_set(ℛ_small_optimal)
	check_volume = UsingThisViolatesTheHonorCode.check_volume
	test = UsingThisViolatesTheHonorCode.test

	ℛmax_small = UsingThisViolatesTheHonorCode.ℛmax_small
	ℛmax_medium = UsingThisViolatesTheHonorCode.ℛmax_medium
	ℛmax_large = UsingThisViolatesTheHonorCode.ℛmax_large

	ℛ_small_optimal_over_time = convex_hull.(concretize.(fan_sets(ℛ_small_optimal)));

	md"""
	# Backend
	_Helper functions and project management. Please do not edit._
	"""
end

# ╔═╡ 0e7aa498-3009-4473-a9a7-687cc0835ee4
begin
	ℛ_small = test(sys_small, ψ_small, estimate_reachable_sets_small;
				   rerun=rerun_small, project=ThisProject);
	global τs_small = missing
	global issound_small = missing
	global outsiders_small = missing
	Random.seed!(0)
	local d = get_depth(sys_small)
	global τs_small = [rollout(sys_small, d=d+1) for i in 1:300]
	
	try
		global issound_small, outsiders_small =
			precompute_soundness_and_outsiders(sys_small, ℛ_small, τs_small)
	catch end
end;

# ╔═╡ 9cbf5fa3-664f-418e-a791-124276d9ae24
begin
	global small_timestamp = 0
	try
		ℛ_small
		global small_timestamp = time() # Trigger for re-saving
	catch end
end;

# ╔═╡ d0a3770a-2c48-42db-9a71-6b7f695f22d8
begin
	global pass_small = false
	global log_small = nothing
	global vol_small = Inf

	Δt_small = abs(time() - small_timestamp)
	global pass_small, log_small, vol_small =
		check_volume(sys_small, ψ_small, ℛ_small;
			ℛ_optimal=ℛ_small_optimal,
			ℛ_optimal_over_time=ℛ_small_optimal_over_time,
			t=t_small,
			reran=rerun_small,
			τs=τs_small,
			issound=issound_small,
			outsiders=outsiders_small,
			save=(Δt_small ≤ 1 && rerun_small),
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
	
	Markdown.MD(HTML("<h2 id='graded-test'>$(pass ? "✅" : "❌") Graded small test ($(pass ? "$(ThisProject.points_small)/$(ThisProject.points_small)" : "0/$(ThisProject.points_small)") points)</h2>"),
		md"""
	If the following test passes, then you're finished with the small problem.
	""")
end

# ╔═╡ b3a79f6d-6d36-4371-a346-607f819d0fe4
begin
	ℛ_medium = test(sys_medium, ψ_medium, estimate_reachable_sets_medium;
				    ℛmax=ℛmax_medium, rerun=rerun_medium, project=ThisProject);

	global τs_medium = missing
	global issound_medium = missing
	global outsiders_medium = missing

	Random.seed!(4)
	local d = get_depth(sys_medium)
	global τs_medium = [rollout(sys_medium, d=d) for i in 1:1000]

	try
		global issound_medium, outsiders_medium =
			precompute_soundness_and_outsiders(sys_medium, ℛ_medium, τs_medium)
	catch end
end;

# ╔═╡ f0b7fd4f-2a76-4329-93d8-91d789c3445c
begin
	global medium_timestamp = 0
	try
		ℛ_medium
		global medium_timestamp = time() # Trigger for re-saving
	catch end
end;

# ╔═╡ 7cb60134-6377-4522-8232-6765e2f1f725
begin
	global pass_medium = false
	global log_medium = nothing
	global vol_medium = Inf	
	Δt_medium = abs(time() - medium_timestamp)
	global pass_medium, log_medium, vol_medium =
		check_volume(sys_medium, ψ_medium, ℛ_medium;
			t=t_medium, reran=rerun_medium,
			save=(Δt_medium ≤ 1 && rerun_medium),
			ℛmax=ℛmax_medium,
			τs=τs_medium,
			issound=issound_medium,
			outsiders=outsiders_medium,
			project=ThisProject)
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
	## $(pass ? "✅" : "❌") Graded medium test ($(pass ? "$(ThisProject.points_medium)/$(ThisProject.points_medium)" : "0/$(ThisProject.points_medium)") points)
	""")
end

# ╔═╡ f6eb6d1a-a9a0-4234-8699-269a92f666c0
begin
	ℛ_large = test(sys_large, ψ_large, estimate_reachable_sets_large;
				    ℛmax=ℛmax_large, rerun=rerun_large, project=ThisProject);

	global τs_large = missing
	global issound_large = missing
	global outsiders_large = missing
	Random.seed!(4)
	local d = get_depth(sys_large)
	global τs_large = [rollout(sys_large, d=d) for i in 1:1000]

	try
		global issound_large, outsiders_large =
			precompute_soundness_and_outsiders(sys_large, ℛ_large, τs_large)
	catch end
end;

# ╔═╡ 685ead39-822e-4207-9832-940da6a13de8
begin
	global large_timestamp = 0
	try
		ℛ_large
		global large_timestamp = time() # Trigger for re-saving
	catch end
end;

# ╔═╡ 28ba58b2-7cae-4d41-8898-307ba09c5fda
begin
	global pass_large = false
	global log_large = nothing
	global vol_large = Inf	

	Δt_large = abs(time() - large_timestamp)
	global pass_large, log_large, vol_large =
		check_volume(sys_large, ψ_large, ℛ_large;
			t=t_large, reran=rerun_large,
			save=(Δt_large ≤ 1 && rerun_large),
			ℛmax=ℛmax_large,
			cmap=cmap,
			τs=τs_large,
			issound=issound_large,
			outsiders=outsiders_large,
			project=ThisProject)
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
	## $(pass ? "✅" : "❌") Graded large test ($(pass ? "$(ThisProject.points_large)/$(ThisProject.points_large)" : "0/$(ThisProject.points_large)") points)
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
if all([pass_small, pass_medium, pass_large])
	correct(Markdown.MD(md"""
All tests have passed, **_you're done with Project 1!_**""",
@mdx("""
|  System  |  Passed?  |  Points  |
| :------: | :-------: | :------: |
| Small | $(pass_small ? HTML("<span style='color:#759466'><b>Passed!</b></span>") : HTML("<span style='color:#B83A4B'><b>Failed.</b></span>")) | $(pass_small ? "$(Project2.points_small)/$(Project2.points_small)" : "0/$(Project2.points_small)") |
| Medium | $(pass_medium ? HTML("<span style='color:#759466'><b>Passed!</b></span>") : HTML("<span style='color:#B83A4B'><b>Failed.</b></span>")) | $(pass_medium ? "$(Project2.points_medium)/$(Project2.points_medium)" : "0/$(Project2.points_medium)") |
| Large | $(pass_large ? HTML("<span style='color:#759466'><b>Passed!</b></span>") : HTML("<span style='color:#B83A4B'><b>Failed.</b></span>")) | $(pass_large ? "$(Project2.points_large)/$(Project2.points_large)" : "0/$(Project2.points_large)") |
"""),
md"""
**📩 Please see the [Submission](#submission) section at the top of the page.**
"""))
else
	almost(Markdown.MD(md"**_Some tests have failed:_**", @mdx("""
|  System  |  Passed?  | Points |
| :------: | :-------: | :----: |
| Small | $(pass_small ? HTML("<span style='color:#759466'><b>Passed!</b></span>") : HTML("<span style='color:#B83A4B'><b>Failed.</b></span>")) | $(pass_small ? "$(Project2.points_small)/$(Project2.points_small)" : "0/$(Project2.points_small)") |
| Medium | $(pass_medium ? HTML("<span style='color:#759466'><b>Passed!</b></span>") : HTML("<span style='color:#B83A4B'><b>Failed.</b></span>")) | $(pass_medium ? "$(Project2.points_medium)/$(Project2.points_medium)" : "0/$(Project2.points_medium)") |
| Large | $(pass_large ? HTML("<span style='color:#759466'><b>Passed!</b></span>") : HTML("<span style='color:#B83A4B'><b>Failed.</b></span>")) | $(pass_large ? "$(Project2.points_large)/$(Project2.points_large)" : "0/$(Project2.points_large)") |
"""),
md"""
_Please fix the above failing tests before submission._

_You may partially submit individual `.val` files to Gradescope, you have unlimited Gradescope submissions until the deadline. But please make sure to submit all **three** `.val` files once complete._"""))
end

# ╔═╡ d9ab8278-eb76-4a36-aa0e-4ec74704f5e0
begin
	infinity = 1e228

	function check_inf(score)
		if isinf(score) || score == infinity
			infinity, infinity
		else
			score, rd(score)
		end
	end

	global volume_small = volume_small_rd = "—"
	try
		global volume_small, volume_small_rd = check_inf(vol_small)
	catch end

	global volume_medium = volume_medium_rd = "—"
	try
		global volume_medium, volume_medium_rd = check_inf(vol_medium)
	catch end

	global volume_large = volume_large_rd = "—"
	try
		global volume_large, volume_large_rd = check_inf(vol_large)
	catch end

	global user_score = user_score_rd = "—"
	try
		global user_score, user_score_rd = 
			check_inf(leaderboard_scores(
				[volume_small, volume_medium, volume_large], 𝐰))
	catch end

	Markdown.parse("""
# Leaderboard
If the above tests pass, then you will receive full credit for your submission on Gradescope under the **`"Project $(ThisProject.project_num) (.val files + .jl files)"`** assignment.

_However_, we have a leaderboard so that students can participate in a friendly competition to find the best estimated volume of the reachable set for each problem.
	
## Leaderboard entry
Your leaderboard entry on Gradescope should look something like this (smaller is better):

| Rank | Submission Name | Score | vol(small) | vol(medium) | vol(large) |
| :--: | :-------------: | :---: | :--------: | :---------: | :--------: |
| —    | $(guess_username()) | $user_score_rd | $volume_small_rd | $volume_medium_rd | $volume_large_rd |

_**Note**, we will show \$$infinity\$ in place of \$\\infty\$ when a test failed._ 
""")
end

# ╔═╡ 98cbe931-d362-4039-97ba-41e0049619a3
begin
	msd_gif_name = dark_mode ? "msd-dark.gif" : "msd.gif"

	if false
		local d = get_depth(sys_small)
		local msd_fps = 10
		local msd_repeat = msd_fps # Repeat last frame x times

		msd_anim_T = 1:d
		msd_anim_T = vcat(msd_anim_T, fill(msd_anim_T[end], msd_repeat))

		Random.seed!(0)
		local τs = [rollout(sys_small, d=d) for _ in 1:300]

		msd_anim = @withprogress name="gif" begin
			@animate for (i,t) in enumerate(msd_anim_T)
				plot_optimal(sys_small, ψ_small, ℛ_small_optimal, legend=false)
				plot!(𝐫_optimal[t];
					color=StanfordAA228V.PASTEL_SKY_BLUE,
					mark=false,
					lc=StanfordAA228V.PASTEL_SKY_BLUE,
					lw=2,
					linealpha=0.5,
					fillalpha=0.05)
				scatter!([τ[t].s[1] for τ in τs], [τ[t].s[2] for τ in τs];
					ms=2,
					color=StanfordAA228V.PASTEL_SKY_BLUE,
					msc=StanfordAA228V.PASTEL_SKY_BLUE,
					label=false)
				
				plt_set = plot!(title="\$t = $t\$", size=(700,350), dpi=400)
			
				plt_msd_time = plot_msd_time_axis(sys_small, ψ_small; flip=false)
				plot_msd_traj!(sys_small, ψ_small, τs[2][1:t]; lw=2, flip=false)
				plot!(title="\$t = $t \\ ($(rpad(round(0.05(t-1); digits=4), 4, '0')) \\ s)\$", size=(700,350), dpi=400)
				
				plot(plt_set, plt_msd_time, layout=(1,2))
				@logprogress i/length(msd_anim_T)
			end
		end

		gif(msd_anim, joinpath(@__DIR__, "..", "media", msd_gif_name); fps=msd_fps, show_msg=false)
	end

	md"> _Mass-spring-damper animated GIF._"
end

# ╔═╡ 86f9b1ae-d17f-4084-999a-28598ddc2846
try LocalResource(joinpath(@__DIR__, "..", "media", msd_gif_name)) catch end

# ╔═╡ 5563f0da-7552-4879-a38a-ba1748d39d52
begin
	pendulum_gif_name = dark_mode ? "pendulum-dark.gif" : "pendulum.gif"

	if false
		let
			results = run_baseline_mlf(sys_medium, ψ_medium; n=41_000)
			pendulum_anim = @animate for t in 1:get_depth(sys_medium)
				θ_medium_failure = rad2deg(results.τ[t].s[1])
				θ_medium_success = rad2deg(results.τs[2][t].s[1])
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
	end

	md"> _Inverted pendulum animated GIF._"
end

# ╔═╡ 4ea18122-b681-4de1-89e3-5fb7ce2f7a0b
try LocalResource(joinpath(@__DIR__, "..", "media", pendulum_gif_name)) catch end

# ╔═╡ 41c6a7e9-48d6-4b87-94a2-f11a6adca655
begin
	cw_gif_name = dark_mode ? "cw-dark.gif" : "cw.gif"

	if false
		let
			cw_time_multiplier = 10
			cw_fps = 3
			cw_repeat = cw_fps # Repeat last frame x times

			d = get_depth(sys_large)
			cw_anim_T = 1:d
			cw_anim_T = vcat(cw_anim_T, fill(cw_anim_T[end], cw_repeat))

			Random.seed!(4)
			τs = [rollout(sys_large; d=get_depth(sys_large)) for _ in 1:2000];

			traj_color = dark_mode ? :white : :black
			traj_stroke_color = dark_mode ? :black : :white
			τi = 7

			cw_anim = @withprogress name="gif" begin
				@animate for (i,t) in enumerate(cw_anim_T)
					plot(sys_large, ψ_large)

					for tt in 1:t
						c = get(cmap, ((tt-1) % length(cmap)) / (length(cmap) - 1))
						scatter!([τ[tt].s[1] for τ in τs], [τ[tt].s[2] for τ in τs];
							ms=1,
							msw=0.1,
							color=c,
							msc=:white,
							alpha=tt < t ? 0.025 : 1,
							label=false)
					end
			
					plot!(map(step->step.s[1], τs[τi][1:t]), map(step->step.s[2], τs[τi][1:t]); lw=4, color=traj_stroke_color, label=false)
				
					plot!(map(step->step.s[1], τs[τi][1:t]), map(step->step.s[2], τs[τi][1:t]); lw=2, color=traj_color, label=false)

					scatter!(map(step->step.s[1], τs[τi][1:1]), map(step->step.s[2], τs[τi][1:1]); mark=:square, color=traj_color, msc=traj_stroke_color, label=false)
					
					plot!(size=(650,400), xlims=(0, 10), ylims=(0, 10), dpi=400)

					@logprogress i/length(cw_anim_T)
				end
			end
	
			gif(cw_anim, joinpath(@__DIR__, "..", "media", cw_gif_name); fps=cw_fps, show_msg=false)
		end
	end

	md"> _Continuum world animated GIF._"
end

# ╔═╡ 505c504a-6ce6-474e-9220-95702f909c01
try LocalResource(joinpath(@__DIR__, "..", "media", cw_gif_name)) catch end

# ╔═╡ 97042a5e-9691-493f-802e-2262f2da4627
Markdown.MD(notebook_style(), md"> _Notebook styling._")

# ╔═╡ 9865ed62-b4fd-4e49-9259-3e5997c589f3
Markdown.MD(button_style([]), md"> _Button styling._")

# ╔═╡ ef084fea-bf4d-48d9-9c84-8cc1dd98f2d7
Markdown.MD(TableOfContents(), md"> _Table of contents._")

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BSON = "fbb218c0-5317-5bc6-957e-2ee96dd4b1f0"
Base64 = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Downloads = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
GridInterpolations = "bb4c363b-b914-514b-8517-4eb369bc008a"
IntervalArithmetic = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
JuMP = "4076af6c-e467-56ae-b986-b466b2749572"
LazySets = "b4f0291d-fe17-52bc-9479-3d1a343d9043"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MarkdownLiteral = "736d6165-7244-6769-4267-6b50796e6954"
NeuralVerification = "146f25fa-00e7-11e9-3ae5-fdbac6e12fa7"
Parameters = "d96e819e-fc66-5662-9728-84c9c7592b0a"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
ProgressLogging = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SCS = "c946c3f1-0d1f-5ce8-9dea-7daa1f7e2d13"
StanfordAA228V = "6f6e590e-f8c2-4a21-9268-94576b9fb3b1"
TOML = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[compat]
BSON = "~0.3.9"
Distributions = "~0.25.122"
Flux = "~0.16.5"
ForwardDiff = "~0.10.39"
GridInterpolations = "~1.3.0"
IntervalArithmetic = "~0.21.2"
JuMP = "~1.29.2"
LazySets = "~2.14.2"
MarkdownLiteral = "~0.1.2"
NeuralVerification = "~0.1.0"
Parameters = "~0.12.3"
Plots = "~1.41.1"
PlutoUI = "~0.7.73"
ProgressLogging = "~0.1.5"
SCS = "~2.4.0"
StanfordAA228V = "~0.1.27"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.1"
manifest_format = "2.0"
project_hash = "017fb8327b1613ff16418f87fd0128fcfd9a28f9"

[[deps.ADTypes]]
git-tree-sha1 = "27cecae79e5cc9935255f90c53bb831cc3c870d7"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.18.0"
weakdeps = ["ChainRulesCore", "ConstructionBase", "EnzymeCore"]

    [deps.ADTypes.extensions]
    ADTypesChainRulesCoreExt = "ChainRulesCore"
    ADTypesConstructionBaseExt = "ConstructionBase"
    ADTypesEnzymeCoreExt = "EnzymeCore"

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

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "3b86719127f50670efe356bc11073d84b4ed7a5d"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.42"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "7e35fca2bdfba44d797c53dfe63a51fabf39bfc0"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.4.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgCheck]]
git-tree-sha1 = "f9e9a66c9b7be1ad7372bbd9b062d9230c30c5ce"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.5.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "d81ae5489e13bc03567d4fbbb06c546a5e53c857"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.22.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = ["CUDSS", "CUDA"]
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceMetalExt = "Metal"
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
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "29bb0eb6f578a587a49da16564705968667f5fa8"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "1.1.2"

    [deps.Atomix.extensions]
    AtomixCUDAExt = "CUDA"
    AtomixMetalExt = "Metal"
    AtomixOpenCLExt = "OpenCL"
    AtomixoneAPIExt = "oneAPI"

    [deps.Atomix.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    OpenCL = "08131aa3-fb12-5dee-8b74-c09406e224a2"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.BSON]]
git-tree-sha1 = "4c3e506685c527ac6a54ccc0c8c76fd6f91b42fb"
uuid = "fbb218c0-5317-5bc6-957e-2ee96dd4b1f0"
version = "0.3.9"

[[deps.BangBang]]
deps = ["Accessors", "ConstructionBase", "InitialValues", "LinearAlgebra"]
git-tree-sha1 = "a49f9342fc60c2a2aaa4e0934f06755464fcf438"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.6"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTablesExt = "Tables"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BenchmarkTools]]
deps = ["Compat", "JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "7fecfb1123b8d0232218e2da0c213004ff15358d"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.6.3"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CDDLib]]
deps = ["LinearAlgebra", "MathOptInterface", "Polyhedra", "SparseArrays", "cddlib_jll"]
git-tree-sha1 = "f13ac9e3b20183b1142d478907fd3333914ad05d"
uuid = "3391f64e-dcde-5f30-b752-e11513730f60"
version = "0.10.1"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CRlibm]]
deps = ["CRlibm_jll"]
git-tree-sha1 = "66188d9d103b92b6cd705214242e27f5737a1e5e"
uuid = "96374032-68de-5a5b-8d9e-752f78720389"
version = "1.0.2"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fde3bf89aead2e723284a8ff9cdf5b551ed700e8"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.5+0"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "SparseInverseSubset", "Statistics", "StructArrays", "SuiteSparse"]
git-tree-sha1 = "3b704353e517a957323bd3ac70fa7b669b5f48d4"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.72.6"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "e4c6a16e77171a5f5e25e9646617ab1c276c5607"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.0"
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
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.CommonMark]]
deps = ["PrecompileTools"]
git-tree-sha1 = "351d6f4eaf273b753001b2de4dffb8279b100769"
uuid = "a80b9123-70ca-4bc0-993e-6e3bcb318db6"
version = "0.9.1"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "d9d26935a0bcffc87d2613ce14c527c99fc543fd"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.0"

[[deps.Configurations]]
deps = ["ExproniconLite", "OrderedCollections", "TOML"]
git-tree-sha1 = "4358750bb58a3caefd5f37a4a0c5bfdbbf075252"
uuid = "5218b696-f38b-4ac9-8b61-a12ec717816d"
version = "0.17.6"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "e357641bb3e0638d353c4b29ea0e40ea644066a6"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.3"

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
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

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

[[deps.DifferentiationInterface]]
deps = ["ADTypes", "LinearAlgebra"]
git-tree-sha1 = "6d5153dc500d644d4d672723aa27a614ee84ab3b"
uuid = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
version = "0.7.11"

    [deps.DifferentiationInterface.extensions]
    DifferentiationInterfaceChainRulesCoreExt = "ChainRulesCore"
    DifferentiationInterfaceDiffractorExt = "Diffractor"
    DifferentiationInterfaceEnzymeExt = ["EnzymeCore", "Enzyme"]
    DifferentiationInterfaceFastDifferentiationExt = "FastDifferentiation"
    DifferentiationInterfaceFiniteDiffExt = "FiniteDiff"
    DifferentiationInterfaceFiniteDifferencesExt = "FiniteDifferences"
    DifferentiationInterfaceForwardDiffExt = ["ForwardDiff", "DiffResults"]
    DifferentiationInterfaceGPUArraysCoreExt = "GPUArraysCore"
    DifferentiationInterfaceGTPSAExt = "GTPSA"
    DifferentiationInterfaceMooncakeExt = "Mooncake"
    DifferentiationInterfacePolyesterForwardDiffExt = ["PolyesterForwardDiff", "ForwardDiff", "DiffResults"]
    DifferentiationInterfaceReverseDiffExt = ["ReverseDiff", "DiffResults"]
    DifferentiationInterfaceSparseArraysExt = "SparseArrays"
    DifferentiationInterfaceSparseConnectivityTracerExt = "SparseConnectivityTracer"
    DifferentiationInterfaceSparseMatrixColoringsExt = "SparseMatrixColorings"
    DifferentiationInterfaceStaticArraysExt = "StaticArrays"
    DifferentiationInterfaceSymbolicsExt = "Symbolics"
    DifferentiationInterfaceTrackerExt = "Tracker"
    DifferentiationInterfaceZygoteExt = ["Zygote", "ForwardDiff"]

    [deps.DifferentiationInterface.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DiffResults = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
    Diffractor = "9f5e2b26-1114-432f-b630-d3fe2085c51c"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FastDifferentiation = "eb9bf01b-bf85-4b60-bf87-ee5de06c00be"
    FiniteDiff = "6a86dc24-6348-571c-b903-95158fe2bd41"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    GTPSA = "b27dd330-f138-47c5-815b-40db9dd9b6e8"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    PolyesterForwardDiff = "98d1487c-24ca-40b6-b7ab-df2af84e126b"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
    SparseMatrixColorings = "0a514795-09f3-496d-8182-132a7b665d35"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "c7e3a542b999843086e2f29dac96a618c105be1d"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.12"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "3bc002af51045ca3b47d2e1787d6ce02e68b943a"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.122"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EnumX]]
git-tree-sha1 = "bddad79635af6aec424f53ed8aad5d7555dc6f00"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.5"

[[deps.EnzymeCore]]
git-tree-sha1 = "c55ba9649c7dc0f6430a096e35391955d7fc6ecd"
uuid = "f151be2c-9106-41f4-ab19-57ee4f262869"
version = "0.8.16"
weakdeps = ["Adapt"]

    [deps.EnzymeCore.extensions]
    AdaptExt = "Adapt"

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
git-tree-sha1 = "27af30de8b5445644e8ffe3bcb0d72049c089cf1"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.7.3+0"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.ExpressionExplorer]]
git-tree-sha1 = "4a8c0a9eebf807ac42f0f6de758e60a20be25ffb"
uuid = "21656369-7473-754a-2065-74616d696c43"
version = "1.1.3"

[[deps.ExproniconLite]]
git-tree-sha1 = "c13f0b150373771b0fdc1713c97860f8df12e6c2"
uuid = "55351af7-c7e9-48d6-89ff-24e801d99491"
version = "0.10.14"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "95ecf07c2eea562b5adbd0696af6db62c0f52560"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.5"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "ccc81ba5e42497f4e76553a5545665eed577a663"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "8.0.0+0"

[[deps.FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "0a2e5873e9a5f54abb06418d57a8df689336a660"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.2"

[[deps.FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

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
git-tree-sha1 = "5bfcd42851cf2f1b303f51525a54dc5e98d408a3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.15.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Setfield"]
git-tree-sha1 = "9340ca07ca27093ff68418b7558ca37b05f8aeb1"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.29.0"

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

[[deps.Flux]]
deps = ["Adapt", "ChainRulesCore", "Compat", "EnzymeCore", "Functors", "LinearAlgebra", "MLCore", "MLDataDevices", "MLUtils", "MacroTools", "NNlib", "OneHotArrays", "Optimisers", "Preferences", "ProgressLogging", "Random", "Reexport", "Setfield", "SparseArrays", "SpecialFunctions", "Statistics", "Zygote"]
git-tree-sha1 = "d0751ca4c9762d9033534057274235dfef86aaf9"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.16.5"

    [deps.Flux.extensions]
    FluxAMDGPUExt = "AMDGPU"
    FluxCUDAExt = "CUDA"
    FluxCUDAcuDNNExt = ["CUDA", "cuDNN"]
    FluxEnzymeExt = "Enzyme"
    FluxMPIExt = "MPI"
    FluxMPINCCLExt = ["CUDA", "MPI", "NCCL"]

    [deps.Flux.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"
    NCCL = "3fe64909-d7a1-4096-9b7d-7a0f12cf0f6b"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "afb7c51ac63e40708a3071f80f5e84a752299d4f"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.39"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "2c5512e11c791d1baed2049c5652441b28fc6a31"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.Functors]]
deps = ["Compat", "ConstructionBase", "LinearAlgebra", "Random"]
git-tree-sha1 = "60a0339f28a233601cb74468032b5c302d5067de"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.5.2"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

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
version = "6.3.0+2"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "ScopedValues", "Serialization", "Statistics"]
git-tree-sha1 = "8ddb438e956891a63a5367d7fab61550fc720026"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "11.2.6"

    [deps.GPUArrays.extensions]
    JLD2Ext = "JLD2"

    [deps.GPUArrays.weakdeps]
    JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "f52c27dd921390146624f3aab95f4e8614ad6531"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.18"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b0406b866ea9fdbaf1148bc9c0b887e59f9af68"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.18+0"

[[deps.GenericLinearAlgebra]]
deps = ["LinearAlgebra", "Printf", "Random", "libblastrampoline_jll"]
git-tree-sha1 = "fc6236b7fbc40bc283e0dc5a84ead1169a6d7dda"
uuid = "14197337-ba66-59df-a3e3-ca00e7dcff7a"
version = "0.3.19"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "50c11ffab2a3d50192a228c313f05b5b5dc5acb2"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.86.0+0"

[[deps.GracefulPkg]]
deps = ["Compat", "Pkg", "TOML"]
git-tree-sha1 = "9721d7ae22a51439656a73de721d973dbb9d3bad"
uuid = "828d9ff0-206c-6161-646e-6576656f7244"
version = "2.4.2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.GridInterpolations]]
deps = ["LinearAlgebra", "Printf", "StaticArrays"]
git-tree-sha1 = "6916860ab8162db1fd34c8e8027243dc3b3f12ea"
uuid = "bb4c363b-b914-514b-8517-4eb369bc008a"
version = "1.3.0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "5e6fe50ae7f23d171f44e311c2960294aaa0beb5"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.19"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f923f9a774fcf3f5cb761bfa43aeadd689714813"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.1+0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "68c173f4f449de5b438ee67ed0c9c748dc31a2ec"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.28"

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
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "57e9ce6cf68d0abf5cb6b3b4abf9bedf05c939c0"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.15"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "88a101217d7cb38a7b481ccd50d21876e1d1b0e0"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.15.1"

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

    [deps.Interpolations.weakdeps]
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.IntervalArithmetic]]
deps = ["CRlibm", "EnumX", "FastRounding", "LinearAlgebra", "Markdown", "Random", "RecipesBase", "RoundingEmulator", "SetRounding", "StaticArrays"]
git-tree-sha1 = "f59e639916283c1d2e106d2b00910b50f4dab76c"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "0.21.2"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "PrecompileTools", "StructTypes", "UUIDs"]
git-tree-sha1 = "411eccfe8aba0814ffa0fdf4860913ed09c34975"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.14.3"

    [deps.JSON3.extensions]
    JSON3ArrowExt = ["ArrowTypes"]

    [deps.JSON3.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4255f0032eafd6451d707a51d5f0248b8a165e4d"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.3+0"

[[deps.JuMP]]
deps = ["LinearAlgebra", "MacroTools", "MathOptInterface", "MutableArithmetics", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays"]
git-tree-sha1 = "d6ece925e8798b6f078731ab04ce82c5433b0d64"
uuid = "4076af6c-e467-56ae-b986-b466b2749572"
version = "1.29.2"

    [deps.JuMP.extensions]
    JuMPDimensionalDataExt = "DimensionalData"

    [deps.JuMP.weakdeps]
    DimensionalData = "0703355e-b756-11e9-17c0-8b28908087d0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "MacroTools", "PrecompileTools", "Requires", "StaticArrays", "UUIDs"]
git-tree-sha1 = "b5a371fcd1d989d844a4354127365611ae1e305f"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.39"
weakdeps = ["EnzymeCore", "LinearAlgebra", "SparseArrays"]

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Preferences", "Printf", "Unicode"]
git-tree-sha1 = "ce8614210409eaa54ed5968f4b50aa96da7ae543"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "9.4.4"

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

    [deps.LLVM.weakdeps]
    BFloat16s = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "8e76807afb59ebb833e9b131ebf1a8c006510f33"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.38+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LRUCache]]
git-tree-sha1 = "5519b95a490ff5fe629c4a7aa3b3dfc9160498b3"
uuid = "8ac3fa9e-de4c-5943-b1dc-09c6b5f20637"
version = "1.6.2"
weakdeps = ["Serialization"]

    [deps.LRUCache.extensions]
    SerializationExt = ["Serialization"]

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "Ghostscript_jll", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "44f93c47f9cd6c7e431f2f2091fcba8f01cd7e8f"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.10"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

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
git-tree-sha1 = "c46bea04a8a94fb7fa4e3007506601fa34d6a711"
uuid = "b4f0291d-fe17-52bc-9479-3d1a343d9043"
version = "2.14.2"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.11.1+1"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3acf07f130a76f87c041cfb2ff7d7284ca67b072"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.41.2+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "f04133fe05eff1667d2054c53d59f9122383fe05"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.2+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2a7a12fc0a4e7fb773450d17975322aa77142106"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.41.2+0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "4adee99b7262ad2a1a4bbbc59d993d24e55ea96f"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.4.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

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
git-tree-sha1 = "f00544d95982ea270145636c181ceda21c4e2575"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.2.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MLCore]]
deps = ["DataAPI", "SimpleTraits", "Tables"]
git-tree-sha1 = "73907695f35bc7ffd9f11f6c4f2ee8c1302084be"
uuid = "c2834f40-e789-41da-a90e-33b280584a8c"
version = "1.0.0"

[[deps.MLDataDevices]]
deps = ["Adapt", "Functors", "Preferences", "Random", "SciMLPublic"]
git-tree-sha1 = "f5c0536d7bcd78610c5195398e37ce2ab8fd8789"
uuid = "7e8f7934-dd98-4c1a-8fe8-92b47a384d40"
version = "1.13.0"

    [deps.MLDataDevices.extensions]
    MLDataDevicesAMDGPUExt = "AMDGPU"
    MLDataDevicesCUDAExt = "CUDA"
    MLDataDevicesChainRulesCoreExt = "ChainRulesCore"
    MLDataDevicesChainRulesExt = "ChainRules"
    MLDataDevicesComponentArraysExt = "ComponentArrays"
    MLDataDevicesFillArraysExt = "FillArrays"
    MLDataDevicesGPUArraysExt = "GPUArrays"
    MLDataDevicesMLUtilsExt = "MLUtils"
    MLDataDevicesMetalExt = ["GPUArrays", "Metal"]
    MLDataDevicesOneHotArraysExt = "OneHotArrays"
    MLDataDevicesReactantExt = "Reactant"
    MLDataDevicesRecursiveArrayToolsExt = "RecursiveArrayTools"
    MLDataDevicesReverseDiffExt = "ReverseDiff"
    MLDataDevicesSparseArraysExt = "SparseArrays"
    MLDataDevicesTrackerExt = "Tracker"
    MLDataDevicesZygoteExt = "Zygote"
    MLDataDevicescuDNNExt = ["CUDA", "cuDNN"]
    MLDataDevicesoneAPIExt = ["GPUArrays", "oneAPI"]

    [deps.MLDataDevices.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ComponentArrays = "b0b7db55-cfe3-40fc-9ded-d10e2dbeff66"
    FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
    MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    OneHotArrays = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
    RecursiveArrayTools = "731186ca-8d62-57ce-b412-fbd966d074cd"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "Compat", "DataAPI", "DelimitedFiles", "FLoops", "MLCore", "NNlib", "Random", "ShowCases", "SimpleTraits", "Statistics", "StatsBase", "Tables", "Transducers"]
git-tree-sha1 = "a772d8d1987433538a5c226f79393324b55f7846"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.4.8"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Malt]]
deps = ["Distributed", "Logging", "RelocatableFolders", "Serialization", "Sockets"]
git-tree-sha1 = "21ab3240653e26c8d54c1389779e260a48174be3"
uuid = "36869731-bdee-424d-aa32-cab38c994e3b"
version = "1.3.0"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MarkdownLiteral]]
deps = ["CommonMark", "HypertextLiteral"]
git-tree-sha1 = "f7d73634acd573bf3489df1ee0d270a5d6d3a7a3"
uuid = "736d6165-7244-6769-4267-6b50796e6954"
version = "0.1.2"

[[deps.MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "DataStructures", "ForwardDiff", "JSON3", "LinearAlgebra", "MutableArithmetics", "NaNMath", "OrderedCollections", "PrecompileTools", "Printf", "SparseArrays", "SpecialFunctions", "Test"]
git-tree-sha1 = "a2cbab4256690aee457d136752c404e001f27768"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.46.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3cce3511ca2c6f87b19c34ffc623417ed2798cbd"
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.10+0"

[[deps.Measures]]
git-tree-sha1 = "b513cedd20d9c914783d8ad83d08120702bf2c77"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.3"

[[deps.MicroCollections]]
deps = ["Accessors", "BangBang", "InitialValues"]
git-tree-sha1 = "44d32db644e84c75dab479f1bc15ee76a1a3618f"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.2.0"

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
version = "2025.5.20"

[[deps.MsgPack]]
deps = ["Serialization"]
git-tree-sha1 = "f5db02ae992c260e4826fe78c942954b48e1d9c2"
uuid = "99f44e22-a591-53d1-9472-aa23ef4bd671"
version = "1.2.1"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "22df8573f8e7c593ac205455ca088989d0a2c7a0"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.6.7"

[[deps.NLSolversBase]]
deps = ["ADTypes", "DifferentiationInterface", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "25a6638571a902ecfb1ae2a18fc1575f86b1d4df"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.10.0"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Random", "ScopedValues", "Statistics"]
git-tree-sha1 = "eb6eb10b675236cee09a81da369f94f16d77dc2f"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.31"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"
    NNlibCUDACUDNNExt = ["CUDA", "cuDNN"]
    NNlibCUDAExt = "CUDA"
    NNlibEnzymeCoreExt = "EnzymeCore"
    NNlibFFTWExt = "FFTW"
    NNlibForwardDiffExt = "ForwardDiff"
    NNlibSpecialFunctionsExt = "SpecialFunctions"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.NeuralVerification]]
deps = ["CDDLib", "GLPK", "Interpolations", "JuMP", "LazySets", "LinearAlgebra", "Parameters", "PicoSAT", "Pkg", "Polyhedra", "Random", "Requires", "SCS"]
git-tree-sha1 = "9b4eb28fb0b28bcdae002b10db69bbe061fbb08f"
uuid = "146f25fa-00e7-11e9-3ae5-fdbac6e12fa7"
version = "0.1.0"

[[deps.OffsetArrays]]
git-tree-sha1 = "117432e406b5c023f665fa73dc26e79ec3630151"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.17.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OneHotArrays]]
deps = ["Adapt", "ChainRulesCore", "Compat", "GPUArraysCore", "LinearAlgebra", "NNlib"]
git-tree-sha1 = "bfe8e84c71972f77e775f75e6d8048ad3fdbe8bc"
uuid = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
version = "0.2.10"

[[deps.OpenBLAS32_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ece4587683695fe4c5f20e990da0ed7e83c351e7"
uuid = "656ef2d0-ae68-5445-9ca0-591084a874a2"
version = "0.3.29+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "NetworkOptions", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "386b47442468acfb1add94bf2d85365dea10cbab"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.6.0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.1+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Optim]]
deps = ["Compat", "EnumX", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "61942645c38dd2b5b78e2082c9b51ab315315d10"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.13.2"
weakdeps = ["MathOptInterface"]

    [deps.Optim.extensions]
    OptimMOIExt = "MathOptInterface"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "ConstructionBase", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "131dc319e7c58317e8c6d5170440f6bdaee0a959"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.4.6"

    [deps.Optimisers.extensions]
    OptimisersAdaptExt = ["Adapt"]
    OptimisersEnzymeCoreExt = "EnzymeCore"
    OptimisersReactantExt = "Reactant"

    [deps.Optimisers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c392fc5dd032381919e3b22dd32d6443760ce7ea"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.5.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.44.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "d922b4d80d1e12c658da7785e754f4796cc1d60d"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.36"
weakdeps = ["StatsBase"]

    [deps.PDMats.extensions]
    StatsBaseExt = "StatsBase"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0662b083e11420952f2e62e17eddae7fc07d5997"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.57.0+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.PicoSAT]]
deps = ["PicoSAT_jll", "Test"]
git-tree-sha1 = "4b24a9cb86ff2dd91cd6ca500780ce9088c0d07a"
uuid = "ff2beb65-d7cd-5ff1-a187-74671133a339"
version = "0.4.1"

[[deps.PicoSAT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "8801eab0536eabe7602f124f27dfbe8c067499d5"
uuid = "e78fa76d-a187-569f-aede-ad11521a2edf"
version = "965.0.0+0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "db76b1ecd5e9715f3d043cec13b2ec93ce015d53"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.44.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.0"
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
git-tree-sha1 = "26ca162858917496748aad52bb5d3be4d26a228a"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.4"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "12ce661880f8e309569074a61d3767e5756a199f"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.41.1"

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
deps = ["Base64", "Configurations", "Dates", "Downloads", "ExpressionExplorer", "FileWatching", "GracefulPkg", "HTTP", "HypertextLiteral", "InteractiveUtils", "LRUCache", "Logging", "LoggingExtras", "MIMEs", "Malt", "Markdown", "MsgPack", "Pkg", "PlutoDependencyExplorer", "PrecompileSignatures", "PrecompileTools", "REPL", "Random", "RegistryInstances", "RelocatableFolders", "SHA", "Scratch", "Sockets", "TOML", "Tables", "URIs", "UUIDs"]
git-tree-sha1 = "91424262b8490c197b4d4451afd1ab9e412399fc"
uuid = "c3e4b0f8-55cb-11ea-2926-15256bba5781"
version = "0.20.21"

[[deps.PlutoDependencyExplorer]]
deps = ["ExpressionExplorer", "InteractiveUtils", "Markdown"]
git-tree-sha1 = "c3e5073a977b1c58b2d55c1ec187c3737e64e6af"
uuid = "72656b73-756c-7461-726b-72656b6b696b"
version = "1.2.2"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "3faff84e6f97a7f18e0dd24373daa229fd358db5"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.73"

[[deps.Polyhedra]]
deps = ["GenericLinearAlgebra", "LinearAlgebra", "MathOptInterface", "MutableArithmetics", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "754bc39995daff07ed01d7ebdc8c9cf6681d241e"
uuid = "67491407-f73d-577b-9b50-8179a7c68029"
version = "0.8.1"

    [deps.Polyhedra.extensions]
    PolyhedraGeometryBasicsExt = "GeometryBasics"
    PolyhedraJuMPExt = "JuMP"
    PolyhedraRecipesBaseExt = "RecipesBase"

    [deps.Polyhedra.weakdeps]
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    JuMP = "4076af6c-e467-56ae-b986-b466b2749572"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"

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
git-tree-sha1 = "07a921781cab75691315adc645096ed5e370cb77"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.3"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "0f27480397253da18fe2c12a4ba4eb9eb208bf3d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.0"

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
deps = ["StyledStrings"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "d95ed0324b0799843ac6f7a6a85e65fe4e5173f0"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.5"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "34f7e5d2861083ec7596af8b8c092531facf2192"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.8.2+2"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll"]
git-tree-sha1 = "da7adf145cce0d44e892626e647f9dcbe9cb3e10"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.8.2+1"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "9eca9fc3fe515d619ce004c83c31ffd3f85c7ccf"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.8.2+1"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "8f528b0851b5b7025032818eb5abbeb8a736f853"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.8.2+2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9da16da70037ba9d701192e27befedefb91ec284"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.2"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.ReachabilityBase]]
deps = ["ExprTools", "InteractiveUtils", "LinearAlgebra", "Random", "Requires", "SparseArrays"]
git-tree-sha1 = "0a8aab328cb42e6a928e37c66d5081f6dd810159"
uuid = "379f33d0-9447-4353-bd03-d664070e549f"
version = "0.3.5"

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
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "5b3d50eb374cea306873b371d3f8d3915a018f0b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.9.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

[[deps.SCS]]
deps = ["LinearAlgebra", "MathOptInterface", "OpenBLAS32_jll", "PrecompileTools", "SCS_jll", "SparseArrays"]
git-tree-sha1 = "48ec3c39787bc7b278789b9af17c157ea8774dae"
uuid = "c946c3f1-0d1f-5ce8-9dea-7daa1f7e2d13"
version = "2.4.0"

    [deps.SCS.extensions]
    SCSSCS_GPU_jllExt = ["SCS_GPU_jll"]
    SCSSCS_MKL_jllExt = ["SCS_MKL_jll"]

    [deps.SCS.weakdeps]
    SCS_GPU_jll = "af6e375f-46ec-5fa0-b791-491b0dfa44a4"
    SCS_MKL_jll = "3f2553a9-4106-52be-b7dd-865123654657"

[[deps.SCS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl", "libblastrampoline_jll"]
git-tree-sha1 = "05d6e31efa3debae6618dabee35dbd53cf4539d8"
uuid = "f4f2fc5b-1d94-523c-97ea-2ab488bedf4b"
version = "300.200.900+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SciMLPublic]]
git-tree-sha1 = "ed647f161e8b3f2973f24979ec074e8d084f1bee"
uuid = "431bcebd-1456-4ced-9d72-93c2757fff0b"
version = "1.0.0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "c3b2323466378a2ba15bea4b2f73b081e022f473"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.5.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.SetRounding]]
git-tree-sha1 = "d7a25e439d07a17b7cdf97eecee504c50fedf5f6"
uuid = "3cc68bcd-71a2-5612-b932-767ffbe40ab0"
version = "0.2.1"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.ShowCases]]
git-tree-sha1 = "7f534ad62ab2bd48591bdeac81994ea8c445e4a5"
uuid = "605ecd9f-84a6-4c9e-81e2-4798472b76a3"
version = "0.1.0"

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

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "be8eeac05ec97d379347584fa9fe2f5f76795bcb"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.5"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SparseInverseSubset]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "52962839426b75b3021296f7df242e40ecfc0852"
uuid = "dc90abb0-5640-4711-901d-7e5b23a2fada"
version = "0.1.2"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "f2685b435df2613e25fc10ad8c26dddb8640f547"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.6.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "4f96c596b8c8258cc7d3b19797854d368f243ddc"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.4"

[[deps.StanfordAA228V]]
deps = ["AbstractPlutoDingetjes", "BSON", "Base64", "Distances", "Distributions", "Downloads", "ForwardDiff", "GridInterpolations", "Interpolations", "LazySets", "LinearAlgebra", "Markdown", "Optim", "Parameters", "Pkg", "Plots", "Pluto", "PlutoUI", "ProgressLogging", "Random", "SignalTemporalLogic", "Statistics", "TOML"]
git-tree-sha1 = "86ea9251637f86e14eaf1be169462aa710d4817f"
uuid = "6f6e590e-f8c2-4a21-9268-94576b9fb3b1"
version = "0.1.27"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "b8693004b385c842357406e3af647701fe783f98"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.15"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

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
git-tree-sha1 = "9d72a13a3f4dd3795a195ac5a44d7d6ff5f552ff"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.1"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "064b532283c97daae49e544bb9cb413c26511f8c"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.8"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "91f091a8716a6bb38417a6e6f274602a19aaa685"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.5.2"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "a2c37d815bf00575332b7bd0389f771cb7987214"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.2"
weakdeps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "SparseArrays", "StaticArrays"]

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "159331b30e94d7b11379037feeb9b690950cace8"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.11.0"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

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
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

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

[[deps.Transducers]]
deps = ["Accessors", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "SplittablesBase", "Tables"]
git-tree-sha1 = "4aa1fdf6c1da74661f6f5d3edfd96648321dade9"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.85"

    [deps.Transducers.extensions]
    TransducersAdaptExt = "Adapt"
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

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
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "96478df35bbc2f3e1e791bc7a3d0eeee559e60e9"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.24.0+0"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fee71455b0aaa3440dfdd54a9a36ccef829be7d4"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.1+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "b5899b25d17bf1889d25906fb9deed5da0c15b3b"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.12+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a4c0ee07ad36bf8bbce1c3bb52d21fb1e0b987fb"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.7+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "a376af5c7ae60d29825164db40787f15c80c7c54"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.3+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "a5bc75478d323358a90dc36766f3c99ba7feb024"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.6+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "aff463c82a773cb86061bce8d53a0d976854923e"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.5+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "e3150c7400c41e207012b41659591f083f3ef795"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.3+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "9750dc53819eba4e9a20be42349a6d3b86c7cdf8"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.6+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f4fc02e384b74418679983a97385644b67e1263b"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll"]
git-tree-sha1 = "68da27247e7d8d8dafd1fcf0c3654ad6506f5f97"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "44ec54b0e2acd408b0fb361e1e9244c60c9c3dd4"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "5b0263b6d080716a02544c55fdff2c8d7f9a16a0"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.10+0"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f233c83cad1fa0e70b7771e0e21b061a116f2763"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.2+0"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "801a858fc9fb90c11ffddee1801bb06a738bda9b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.7+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "00af7ebdc563c9217ecc67776d1bbf037dbcebf4"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.44.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArrays", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "Requires", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "8462a20f0fd85b4ef4a1b7310d33e7475d2bb14f"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.77"

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
git-tree-sha1 = "434b3de333c75fc446aa0d19fc394edafd07ab08"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.7"

[[deps.cddlib_jll]]
deps = ["Artifacts", "GMP_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "08f5df03703af917b9bfec47b9767eb943220d08"
uuid = "f07e07eb-5685-515a-97c8-3014f6152feb"
version = "0.94.14+0"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c3b0e6196d50eab0c5ed34021aaa0bb463489510"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.14+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "371cc681c00a3ccc3fbc5c0fb91f58ba9bec1ecf"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.13.1+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "125eedcb0a4a0bba65b657251ce1d27c8714e9d6"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.4+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56d643b57b188d30cccc25e331d416d3d358e557"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.13.4+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "91d05d7f4a9f67205bd6cf395e488009fe85b499"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.28.1+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "07b6a107d926093898e82b3b1db657ebe33134ec"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.50+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b4d631fd51f2e9cdd93724ae25b2efc198b059b1"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.5.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "fbf139bce07a534df0e699dbb5f5cc9346f95cc1"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.9.2+0"
"""

# ╔═╡ Cell order:
# ╟─6b17139e-6caf-4f07-a607-e403bf1ad794
# ╠═14964632-98d8-4a2f-b2f6-e3f28b558803
# ╠═268f31d2-e485-43e1-a925-48928bb3e360
# ╟─117d0059-ce1a-497e-8667-a0c2ef20c632
# ╟─60f72d30-ab80-11ef-3c20-270dbcdf0cc4
# ╟─d7643abe-4619-4859-b2e3-9e932fe53b2f
# ╟─9f739929-1cd3-4935-b229-ae3aeac7e131
# ╟─4eeaa9ae-eac5-478a-aca5-82de3dda24f7
# ╟─0c520f93-49ce-45eb-899d-a31105d856c8
# ╟─78181077-5548-459d-970d-1d8a9d63b72c
# ╟─da5b4000-0bce-4fc2-be85-dada21264ca3
# ╟─0456a732-2672-4108-a241-db9ae879a913
# ╟─6e8ab7c9-fb49-4d89-946d-c7d7588c199a
# ╟─a21612a1-1092-4892-9132-629833e7c867
# ╟─ec776b30-6a30-4643-a22c-e071a365d50b
# ╟─5aafffc6-8e3b-4591-99f6-6ee3b8082786
# ╟─c4fa9af9-1a79-43d7-9e8d-2854652a4ea2
# ╟─bd8cfc1d-cf0e-42e2-8e46-c763f142e8f3
# ╟─5d389282-466d-47d8-ae69-0794cf620f27
# ╟─6bad6e8b-c021-41d2-afbb-bcd0242138dd
# ╟─1167fdfb-b097-4d18-b982-b1786794f8cf
# ╟─3e392ff6-6a32-49cc-9eab-82ae2d6e9154
# ╟─fd8c851a-3a42-41c5-b0fd-a12085543c9b
# ╟─851943ba-d6f8-4e97-bbcd-dec5eb8b8b76
# ╟─55521bf3-eb1d-4ea1-9c18-aa8ba4461bff
# ╟─17fa8557-9656-4347-9d44-213fd3b635a6
# ╠═e93d0515-c7d4-4af6-9041-fda738af5caa
# ╠═6f3e24de-094c-49dc-b892-6721b3cc54ed
# ╟─592e4e77-8104-4464-8e10-ee2834c7c0ab
# ╠═ab4c6807-5b4e-4688-b794-159e26a1599b
# ╟─86f9b1ae-d17f-4084-999a-28598ddc2846
# ╟─402c0eaa-727f-4c54-89ec-64c3dfb8002c
# ╟─92f20cc7-8bc0-4aea-8c70-b0f759748fbf
# ╟─a003beb6-6235-455c-943a-e381acd00c0e
# ╟─d0a25025-9309-463f-a09a-9d7ea3df8143
# ╠═fc2d34da-258c-4460-a0a4-c70b072f91ca
# ╟─c494bb97-14ef-408c-9de1-ecabe221eea6
# ╟─8082ce45-6e93-4d98-8f90-79935deadec8
# ╟─38f3d8cf-21cf-4c77-bd45-a618b9b2e1cd
# ╟─e2418154-4471-406f-b900-97905f5d2f59
# ╟─1789c8b5-b314-4aba-ad44-555be9a85984
# ╟─f286f3b2-3bac-4384-9b40-522e974a14ee
# ╟─307afd9c-6dac-4a6d-89d7-4d8cabfe3fe5
# ╟─0e7aa498-3009-4473-a9a7-687cc0835ee4
# ╟─d0a3770a-2c48-42db-9a71-6b7f695f22d8
# ╟─6a7b4f2b-187a-40a3-842f-126d220d40ed
# ╟─97ddd327-94b8-4c2e-a79d-f5304725c25b
# ╟─9cbf5fa3-664f-418e-a791-124276d9ae24
# ╟─fda151a1-5069-44a8-baa1-d7903bc89797
# ╟─c498c964-9c36-4859-9b43-c203f5aa3390
# ╟─8c78529c-1e00-472c-bb76-d984b37235ab
# ╟─daada216-11d4-4f8b-807c-d347130a3928
# ╟─6ec8a963-726d-4738-9a82-7e0b26b90b16
# ╟─86f92b1d-87eb-40b0-ad0b-6b888c5fb9cc
# ╠═0ab3fe4f-a13d-4d92-b3e4-5653f05dafe7
# ╟─613443b1-ac8b-4570-b4c7-d107d63a36cd
# ╠═77637b5e-e3ce-4ecd-90fc-95611af18002
# ╠═c4c0328d-8cb3-41d5-9740-0197cbf760c2
# ╟─99ccbce1-ac0b-4744-9c26-cb951490f482
# ╠═022f242e-f839-4b6a-b6ff-6ad3b09470a4
# ╟─b1e9bd40-a401-4630-9a1f-d61b276e72f7
# ╠═fe272c1b-421c-49de-a513-80c7bcefdd9b
# ╟─4ea18122-b681-4de1-89e3-5fb7ce2f7a0b
# ╟─bac5c489-553c-436f-b332-8a8e97126a51
# ╟─1da9695f-b7fc-46eb-9ef9-12160246018d
# ╟─0606d827-9c70-4a79-afa7-14fb6b806546
# ╟─f180bd3a-12da-4942-b2af-2df2f5887201
# ╠═cb7b9b9f-59da-4851-ab13-c451c26117df
# ╟─759534ca-b40b-4824-b7ec-3a5c06cbd23e
# ╟─97dbe1e4-8045-4213-866f-6921c733fbeb
# ╟─b1cf81ad-e5cb-40d7-b365-abda3fc67a88
# ╟─a228c1ac-62cb-4c18-89b9-bda4c3b1c5bb
# ╟─da2d692a-8378-435e-bd6b-c0e65caef542
# ╟─23999cd9-543b-47dc-a0b2-e133ba95891e
# ╟─38f26afd-ffa5-48d6-90cc-e3ec189c2bf1
# ╟─b3a79f6d-6d36-4371-a346-607f819d0fe4
# ╟─7cb60134-6377-4522-8232-6765e2f1f725
# ╟─50199170-f120-48ee-879c-bbdc11618f1e
# ╟─428575d2-66a9-4a19-9eb0-cdf9b55277f7
# ╟─f0b7fd4f-2a76-4329-93d8-91d789c3445c
# ╟─60ab8107-db65-4fb6-aeea-d4978aed77bd
# ╟─87c29996-943e-43c8-9a85-9f90689a63ae
# ╟─aa0c4ffc-d7f0-484e-a1e2-7f6f92a3a53d
# ╟─505c504a-6ce6-474e-9220-95702f909c01
# ╟─f8ea2983-c2d0-40ea-b949-9fc478ea45f8
# ╟─5aae63b2-54ec-421b-84ec-0d4bc9c00c10
# ╟─790f1562-d6ff-4c44-b1ea-1b0ab1dced85
# ╟─b608b8b2-5166-419b-ac4e-18e87c93ac01
# ╠═05892ece-4c38-4c39-841a-f8b9333ba3ef
# ╟─7d054465-9f80-4dfb-9b5f-76c3977de7cd
# ╠═1ec68a39-8de9-4fd3-be8a-26cf7706d1d6
# ╠═29279d51-1162-4fa8-bdd5-f0ff0e4ef968
# ╟─005966cb-763d-4346-8fdb-e336a9514e8c
# ╟─94a5b84a-b942-4317-b255-f49cc13532c6
# ╠═6a38fe32-d862-4d7d-9978-290a152ae575
# ╟─7b3f2065-0521-4d89-8125-b9c09565d586
# ╟─2a81d0a0-a609-4d20-bddb-e5cbe65caf4b
# ╟─3ae14115-f7d6-495b-a3fa-d13d0b0cdd54
# ╟─be426908-3fee-4ecd-b054-2497ce9a2e50
# ╠═258e14c4-9a2d-4515-9a8f-8cd96f31a6ff
# ╟─e3d6fdf1-3a9e-446b-8482-49d6f64b652e
# ╟─23fd490a-74d2-44b4-8a12-ea1460d95f85
# ╟─18a70925-3c2a-4317-8bbc-c2a096ec56d0
# ╟─45c79345-89da-498c-9a98-2ad55a0a6114
# ╠═3471a623-16af-481a-8f66-5bd1e7890188
# ╟─4c5210d6-598f-4167-a6ee-93bceda7223b
# ╟─ef6bf2ba-748e-4f43-ad53-05d1936c2ba9
# ╟─0a496e93-5853-46bd-a3c5-6f466df90441
# ╟─ba82b4e4-413c-4b78-a777-85d03e3554f4
# ╟─ba780732-7aaa-4e09-9c40-304aa62e564b
# ╠═8cc343cc-f5be-4ff4-802f-adcb6a77674a
# ╟─b4a0cef4-25f6-4b5d-a739-6f9583d6fb3d
# ╟─16502371-5139-40de-be3e-0926f55ce405
# ╟─5c490a85-cff5-46bb-a6fa-330c81d4cd3b
# ╟─ea2d7eb7-d576-415c-ac4c-fea7f90de637
# ╟─7c473630-6555-4ada-85f3-0d40aefe6370
# ╟─7fe1c3d7-469c-47d9-9d46-e5b8b263edb9
# ╟─f6eb6d1a-a9a0-4234-8699-269a92f666c0
# ╟─28ba58b2-7cae-4d41-8898-307ba09c5fda
# ╟─571c8a87-fd6d-490d-8a54-ad76f35f0af4
# ╟─a434dd23-c4d4-498c-9195-10a9c5813e35
# ╟─b5b95a2e-1b81-4b4a-bf12-38041ff01059
# ╟─d1443daa-4cd7-4258-be76-c8a4be9778a6
# ╟─685ead39-822e-4207-9832-940da6a13de8
# ╟─74aeca7b-0658-427f-8c02-d093a0d725ee
# ╟─6bffdf52-4c8d-464e-acad-ec21d528bdbd
# ╟─dbd088d1-f4c9-4e6a-b280-960b06da76e4
# ╟─1bb92755-65e3-457e-84cd-252eae5e4d7e
# ╟─6d5c805b-330c-4b04-a51c-15e674352b1b
# ╟─d9ab8278-eb76-4a36-aa0e-4ec74704f5e0
# ╟─860ec509-3a86-4842-9471-6b1a0b8f366d
# ╠═cfdba748-45d5-4eaa-97b3-fdc9fe7e4333
# ╟─6beda870-0cb0-40f5-9531-fa3e2f7bb020
# ╠═5c3d24f6-0106-444c-b7df-89bba8c01b37
# ╟─4edc5933-9457-4c7c-8456-a26974e0587e
# ╟─95e3d42f-b33f-4294-81c5-f34a300dc9b4
# ╟─ba6c082b-6e62-42fc-a85c-c8b7efc89b88
# ╟─02fac8f9-b442-40d7-b3f3-415a10570e8e
# ╟─173388ab-207a-42a6-b364-b2c1cb335f6b
# ╟─20cb2d9b-ad2d-4d06-be09-03bd5396687a
# ╟─98cbe931-d362-4039-97ba-41e0049619a3
# ╟─5563f0da-7552-4879-a38a-ba1748d39d52
# ╟─41c6a7e9-48d6-4b87-94a2-f11a6adca655
# ╟─247f4c17-bee1-4315-aff9-017407ef9219
# ╟─db7d4de5-9166-4e56-b5bc-1356e43286a9
# ╟─5a1ed20d-788b-4655-bdd8-069545f48929
# ╟─70e5bfca-7172-4f21-b3e6-e31ac16c4add
# ╟─6c8b3077-876e-42fd-aa47-f3fa7c37f4dd
# ╟─97042a5e-9691-493f-802e-2262f2da4627
# ╟─9865ed62-b4fd-4e49-9259-3e5997c589f3
# ╟─ef084fea-bf4d-48d9-9c84-8cc1dd98f2d7
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
