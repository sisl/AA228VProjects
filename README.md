# Stanford AA228V/CS238V Programming Projects
[![website](https://img.shields.io/badge/website-Stanford-b31b1b.svg)](https://aa228v.stanford.edu/)
[![textbook](https://img.shields.io/badge/textbook-MIT%20Press-0072B2.svg)](https://algorithmsbook.com/validation/)
[![package](https://img.shields.io/badge/package-StanfordAA228V.jl-175E54.svg)](https://github.com/sisl/StanfordAA228V.jl)

Programming projects for Stanford's AA228V/CS238V _Validation of Safety-Critical Systems_.

<p align="center"> <img src="./media/coverart.svg"> </p>

Uses the [`StanfordAA228V.jl`](https://github.com/sisl/StanfordAA228V.jl) Julia package.

# Projects
- **[Project 0](./project0)** — _A light-weight introduction to falsification._
- **[Project 1](./project1)** — _Efficiently finding the most-likely failure._
- **[Project 2](./project2)** — _Efficiently estimating the failure probability._
- **[Project 3](./project3)** — _Estimating the volume of a reachable set._

# Installation
For additional installation help, [please post on Ed](https://edstem.org/us/courses/69226/discussion).

You can also follow [this video](https://youtu.be/dhziWVLqGSc).

## 1. Install `git`
- https://git-scm.com/book/en/v2/Getting-Started-Installing-Git

## 2. Install Julia
**Recommend Julia 1.11+**: https://julialang.org/downloads/

- Windows users:
    ```
    winget install julia -s msstore
    ```
- Linux/macOS users:
    ```
    curl -fsSL https://install.julialang.org | sh
    ```

This will give you the `julia` command in your terminal.

## 3. Install Packages
_**Note**: You only need to do this once._
1. _Clone this git repo:_
    1. Open a terminal and navigate to where you want the code to live.
    1. Run:
        ```
        git clone https://github.com/sisl/AA228VProjects
        ```
1. _Install [**Pluto**](https://plutojl.org/) and the SISL Julia package registry:_
    1. Open `julia` in a terminal.
    1. Run:
        ```julia
        using Pkg
        Pkg.add("Pluto")
        Registry.add(url="https://github.com/sisl/Registry.git")
        ```

## (Skip unless otherwise instructed) Update AA228VProjects
1. Open a terminal and navigate to your "AA228VProjects" directory.
1. Run: `git pull`
1. Open the specific Pluto notebook and follow the update instructions within.

# Julia Compiler Speed
Julia is a just-in-time (JIT) compiled language. Don't be alarmed if the first time you run the installed code it's slow. This is because Julia will precompile the required packages. The next time you run it will be _much_ faster.


# Teaching Team

_Winter 2025 teaching team_:
- Sydney Katz: [@smkatz12](https://github.com/smkatz12)
- Mykel Kochenderfer: [@mykelk](https://github.com/mykelk)
- Robert Moss: [@mossr](https://github.com/mossr)
- Harrison Delecki: [@hdelecki](https://github.com/hdelecki)
- Francois Chaubard: [@fchaubard](https://github.com/fchaubard)
- Sidharth Tadeparti: [@sidt36](https://github.com/sidt36)
