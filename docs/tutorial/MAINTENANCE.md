# Maintaining the Markdown manual

## Source ownership

All textbook prose lives in the original `atv-docs/tutorial/*.md` files.
Edit those files directly. Do not create a parallel set of authored `.tex`
chapters, an alternative prose tree, or a generated Markdown copy that will
overwrite these sources. The only LaTeX entry is `main.tex`, responsible for
layout and assembly; chapter titles are read from each Markdown H1.
The rendering implementation lives in `tex/markdown-reader.tex`, which
contains no separately authored chapter text.

Keep the existing `chNN-*.md` names stable even when reading order changes.
New subjects use descriptive, unnumbered filenames. Link by subject title
and relative path rather than hard-coded printed chapter numbers.

## Editorial scope and evidence

The current detailed proofreading pass covers the first six chapters in
the reading order: systems, mathematics, workflows, molecular dynamics,
ensembles, and many-body potentials. It corrects overgeneralized claims and
adds intermediate reasoning and model-specific exercises. Later chapters
still require their own detailed passes; navigation checks do not certify
their scientific or API content.

On 16 September 2026, an earlier pass inspected the then helium UKS example
(since merged into `examples/cpp/02_open_shell.cpp`). The current pass inspected
`frame.h` for the frame representation. These
were moving-branch source inspections, not pinned release validations or
executed engine tests. Primary papers were consulted for the EAM motivation,
angular models, the counterexample to universal close packing under pair
potentials, and stochastic velocity rescaling; the relevant chapter references
identify these sources. No numerical benchmark has been rerun here.

## Chapter contract

Introduce a physical question, learning goals, prerequisites, and theory
before the practice plan and code. Define symbols, state theorem assumptions,
and distinguish numerical error from model error. Practice starts from a
small typical physical system and explains inputs, observable outputs,
validation, and limitations. Finish with questions for reflection.

Preserve code and figure data unless a task explicitly calls for changing
them. Label incomplete fragments rather than silently calling them runnable.
Historical benchmark values need their original provenance and must not be
described as new measurements. Version-dependent support belongs in the
roadmap and method reference, not in a timeless physical explanation.

## Topic boundaries

- Basis sets precede HF; numerical SCF follows open-shell theory.
- Pair and many-body potentials belong with classical motion.
- AM1 belongs with molecular electronic models.
- Hessians, including the former properties chapter's analytic Hessian
  section, belong in `ch14-vibrations.md`.
- `ch20-hf-properties-environment.md` now owns charges and multipoles only.
- TDHF/TDA belong in `electronic-excitations.md`.
- Fields, continuum solvent, and QM/MM belong in `molecular-environment.md`.
- Scalar relativity accompanies effective core potentials in `ch04-basis-sets.md`.
- Driver composition is introduced early; fair model comparison is the final project.

## Reference ownership

- `ch00-architecture.md` owns the introductory distinction between energy,
  geometry, and motion. `../developer/cpu-workflow-reference.md` owns the
  detailed execution fragments and is included directly in the print appendix.
- `ch16-many-body-potentials.md` owns the physical models and validation
  exercises. `methods/md-manybody-potentials.md` owns implementation scope and validation.
- Keep interpretation beside historical numerical tables, but do not present
  a source listing or a retained result as a newly executed validation.
- State theorem assumptions beside the theorem. Distinguish a sufficient
  condition, a necessary condition, and a heuristic expectation.
- Use subject-title links rather than embedded tutorial numbers, and check
  relative links whenever a section moves to another directory.

## Build and verify

From the workspace root, use the existing installations:

```sh
python <manual check>
python <math-render check>
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The Python checks are optional validation, not a preprocessing or build
dependency. `main.tex` directly reads the original Markdown on every build.
It needs no generated chapter body, virtual environment, or network access.
The rendering check uses the installed `pdflatex` and Python `pypdf`; it
checks mathematical fonts, literal code, and source-line diagnostics for
unclosed delimiters. Its outputs go into `build/`.
Update the reading-order table in `index.md` whenever the assembly order
changes; the checker compares it with `main.tex`.

For an editorial revision, oracle scripts
compares code/figure fences and table rows across all Markdown sources
assembled by the current and baseline books, including appendices. This
allows intact blocks to move into their proper reference chapter. Select a
baseline that actually precedes the revision.

If table descriptions are intentionally corrected, use
oracle scripts.
This still requires identical code/figure fences, table-row counts, and
numeric-token sequences per row. It is not permission to change result data;
review the textual diff as well. This pass corrects the CPU frame's host-side
storage label and the definition of numerical tolerance, which is not limited
to round-off. A document build does not execute or validate C++ examples.

Keep generated files in the ignored `build/` directory when possible. Render
temporary PDF page previews into the approved temporary preview directory,
never among project figures. Remaining table and interactive-figure
limitations are documented in `preface.md`; improve the renderer without
creating another authoritative version of the text.

## Mathematical notation

Use `$...$` or `\(...\)` for inline mathematics, and `$$...$$` or
`\[...\]` for displayed equations. Delimiters may span source lines;
standard `amsmath` constructs such as `aligned` and `pmatrix` are supported.
Write literal currency as `\$`, and place program expressions in backticks
or fenced code blocks. Backtick and tilde fences, including quoted fences,
remain literal; `molplot` blocks retain the explicit omission notice.

The reader protects math and inline code before stripping Markdown and
HTML markup. Only mathematical fragments regain TeX syntax; ordinary
prose and code do not. Do not globally reactivate dollar signs, backslashes,
braces, or underscores in the text reader. An unclosed math or inline-code
delimiter is a build error identifying the source file and opening line.
