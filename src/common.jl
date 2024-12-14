get_filename(sys::SmallSystem)  = "project$project_num-small.val"
get_filename(sys::MediumSystem) = "project$project_num-medium.val"
get_filename(sys::LargeSystem)  = "project$project_num-large.val"

env_name(sys::System) = typeof(sys).types[2].name.name

system_size(sys::SmallSystem) = "Small"
system_size(sys::MediumSystem) = "Medium"
system_size(sys::LargeSystem) = "Large"

system_name(sys::System) = "$(system_size(sys))System"


function submission_details(bind_directory_trigger, Project)
	Markdown.MD(HTML("<h2 id='submission'>Submission</h2>"),
	Markdown.parse("""
	You will submit **three** results files (`.val`) to the **`"Project $project_num (.val files)"`** Gradescope assignment and **include the PDF** in your write up in the **"`Project $project_num (write up)`"** Gradescope assignment (see below).
	"""),
	Markdown.parse("""
	1. **Gradescope assignment `"Project $project_num (.val files)"`** (total $(Project.points_small + Project.points_medium + Project.points_large) points):
	    1. `$(get_filename(sys_small))` ($(Project.points_small) points)
	    1. `$(get_filename(sys_medium))` ($(Project.points_medium) points)
	    1. `$(get_filename(sys_large))` ($(Project.points_large) points)
	1. **Gradescope assignment `"Project $project_num (write up)"`** (total $(Project.points_writeup_descr + Project.points_writeup_code) points):
	    - Description of algorithms ($(Project.points_writeup_descr) points)
	    - PDF of Pluto notebook ($(Project.points_writeup_code) points)


	_The_ `.val` _files will be automatically saved for you in the same directory as this notebook:_
	"""),
	md"""
	- $(bind_directory_trigger)
	    - ↑ Click to open directory.
	""",
	Markdown.parse("""
	**Note**: You do _not_ have to submit the `project$project_num.jl` file.

	### Algorithm write up
	Include a PDF write up describing the algorithms you used to solve the three problems. Include the notebook PDF. This should not be more than 1 to 2 pages (excluding the PDF of the notebook code).

	**You'll submit this in a separate Gradescope assignment named `"Project $project_num (Write Up)"`.**

	### Export to PDF
	After you're finished coding, please export this notebook to PDF.
	- Click the `[↑]` icon in the top right and click "PDF", then "Print to PDF".

	Include the **`project$project_num.pdf`** in your write-up:
	-  \$\\LaTeX\$ Overleaf template: [`$overleaf_link`]($overleaf_link)
	    - **Note**: You do _not_ have to use the template or \$\\LaTeX\$.

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

        .highlight {
            background-color: #f0f8ff;
            padding: 10px;
            margin: 30px;
            border: 2px solid #add8e6;
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