get_filename(sys::SmallSystem)  = "project$project_num-small.val"
get_filename(sys::MediumSystem) = "project$project_num-medium.val"
get_filename(sys::LargeSystem)  = "project$project_num-large.val"

env_name(sys::System) = typeof(sys).types[2].name.name

system_size(sys::SmallSystem) = "Small"
system_size(sys::MediumSystem) = "Medium"
system_size(sys::LargeSystem) = "Large"

system_name(sys::System) = "$(system_size(sys))System"


function submission_details(bind_directory_trigger; details=md"**TODO: FILL OUT**")
	Markdown.MD(HTML("<h2 id='submission'>Submission</h2>"),
	Markdown.parse("""
	You will submit **three** results files (`.val`) to the **`"Project $project_num (.val files)"`** Gradescope assignment and **include the PDF** in your write up in the **"`Project $project_num (write up)`"** Gradescope assignment (see below).

	The `.val` files will contain $details (and will be automatically saved for you).
	1. `$(get_filename(sys_small))`
	1. `$(get_filename(sys_medium))`
	1. `$(get_filename(sys_large))`
	The files will be saved in the same directory as this notebook:"""),
	md"""
	- $(bind_directory_trigger)
	    - ↑ Click to open directory.
	""",
	Markdown.parse("""
	**Note**: You don't have to submit the `project$project_num.jl` file.

	### Export to PDF
	After you're finished, please export this notebook to PDF.
	- Click the `[↑]` icon in the top right and click "PDF", then "Print to PDF".

	Include the **`project$project_num.pdf`** in your write-up:
	-  \$\\LaTeX\$ Overleaf template: [`$overleaf_link`]($overleaf_link)

	### Algorithm write up
	Include a PDF write up describing the algorithm(s) you used to solve the three problems. Include the notebook PDF and any additional details you'd like regarding what you tried and what worked best.

	**You'll submit this in a separate Gradescope assignment named `"Project $project_num (Write Up)"`.**

	**See the \$\\LaTeX\$ Overleaf template above.**

	If you encounter issues, [please ask us on Ed](https://edstem.org/us/courses/69226/discussion).
	"""))
end


function textbook_details(chapters=[])
    Markdown.parse("""
    ## Textbook
    [![textbook](https://img.shields.io/badge/textbook-MIT%20Press-0072B2.svg)](https://algorithmsbook.com/validation/files/val.pdf)
    [![coverart](https://raw.githubusercontent.com/sisl/AA228VProjects/refs/heads/main/media/coverart.svg)](https://algorithmsbook.com/validation/files/val.pdf)
    You may find the _Algorithms for Validation_ textbook helpful, specifically the following chapters:
    $(join(map(ch->"- $ch", chapters), "\n"))
    """)
end


function baseline_details(sys::System; n_baseline, descr=md"**TODO**")
	d = get_depth(sys)
	n = max_steps(sys)
	n_formatted = format(n; latex=true)
	n_baseline_formatted = format(n_baseline; latex=true)
	m_baseline_formatted = format(n_baseline ÷ d; latex=true)

	return Markdown.parse("""
	## Baseline: $(system_size(sys))
	The $descr baseline was run with a rollout depth of \$d=$d\$ and \$m=$m_baseline_formatted\$ number of rollouts, for a total number of steps \$n = m \\times d = $(n_baseline_formatted)\$.

	**Note**: Here we increase \$n\$ to \$$(n_baseline_formatted)\$ because the random baseline needs more steps to find failures.

	> **Your algorithm should find likely failures more efficiently than than the baseline.**
	>
	> _This doesn't mean your likelihood needs to be better, as the baseline is given more steps to run. This just means you have to find failures more efficiently._
	>
	> **You'll be given fewer steps of \$n = $n_formatted\$.**
	""")
end


function notebook_style()
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

        .checkbox-label {
            font-feature-settings: "lnum", "pnum";
            color: var(--pluto-output-h-color);
            font-family: Vollkorn, Palatino, Georgia, serif;
            font-size: 1.4rem;
            font-weight: 600;
            line-height: 1.25em;
            /* border-top: 2px dotted var(--rule-color); */
            /* border-bottom: 2px dotted var(--rule-color); */
            /* padding-top: 0.5rem; */
            padding-bottom: 0.5rem;
            /* margin-bottom: 0.5rem; */
            /* margin-block-start: calc(2rem - var(--pluto-cell-spacing)); */
            /* margin-block-end: calc(1.5rem - var(--pluto-cell-spacing)); */
            display: inline-block;
            margin-inline-start: 0px;
            margin-inline-end: 0px;
        }

        .checkbox-input {
            transform: scale(1.5);
        }

        .styled-button {
            background-color: var(--pluto-output-color);
            color: var(--pluto-output-bg-color);
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-family: Alegreya Sans, Trebuchet MS, sans-serif;
        }
    </style>

    <script>
    const buttons = document.querySelectorAll('input[type="button"]');
    buttons.forEach(button => button.classList.add('styled-button'));
    </script>"""
end

function button_style(args...)
    html"""
	<style>
	.styled-button {
			background-color: var(--pluto-output-color);
			color: var(--pluto-output-bg-color);
			border: none;
			padding: 10px 20px;
			border-radius: 5px;
			cursor: pointer;
			font-family: Alegreya Sans, Trebuchet MS, sans-serif;
		}
	</style>

	<script>
	const buttons = document.querySelectorAll('input[type="button"]');
	buttons.forEach(button => button.classList.add('styled-button'));
	</script>"""
end