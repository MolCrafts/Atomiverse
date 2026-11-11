# To the Reader

Why does a molecule bind? How does a liquid transport matter? What changes
when a molecule meets its surroundings? Atomistic simulation offers a way
to investigate such questions by connecting a physical model with a
calculation. Learning to make that connection is the purpose of this book.

The intended reader is an undergraduate beginning molecular simulation.
We assume elementary algebra and a willingness to examine a small model
carefully, rather than prior experience with electronic-structure theory
or a large simulation package. Mathematical ideas are introduced where
they become useful and collected in a chapter to which you can return.

## Questions before commands

A calculation begins by choosing what to represent and what to leave out.
It continues by finding an adequate numerical solution and deciding what
the result can support. These are related but distinct responsibilities.
A well-converged answer to an unsuitable model is not the same as an
accurate description of nature.

For this reason, each chapter begins with a physical question, learning
goals, and prerequisites. Theory and its assumptions precede the practice
plan. Small, recognizable systems connect the equations to code, while
checks of symmetry, derivatives, convergence, and uncertainty connect the
output to a defensible interpretation.

## Working through the book

The five parts form a progression. Models and mathematics establish the
language of the calculation. Classical motion introduces trajectories and
statistical ensembles. Molecular electrons develops basis representations
and electronic methods. Periodic and large systems extends the spatial
and computational setting. Structure and measurement asks what geometries,
spectra, charges, and other observables can tell us. The final comparison
chapter brings these ideas together without treating different models as
interchangeable merely because they share a software interface.

You can follow the complete sequence or choose a route from the
[reading guide](index.md). For a first trajectory, begin with the foundations
and the argon example. For molecular electronic structure, proceed from
basis sets to Hartree–Fock theory and self-consistency. Detailed execution
internals are collected in a developer reference, reproduced as a print
appendix, so they need not interrupt a first encounter with the physics.

Try a derivation or a small check before making a calculation larger. Write
down what you expect and why. If the result differs, first inspect the
model, units, initial state, and numerical controls. An unexpected result
is something to investigate, not a judgment of your ability. A short,
reproducible example often teaches more than a long run whose assumptions
have become difficult to trace.

## Reading results responsibly

The manual distinguishes an analytic result for a stated model, a numerical
comparison, and a statistical estimate. Each needs different evidence.
An energy minimum does not by itself establish a stable phase; a quiet
temperature trace does not establish the correct ensemble; agreement with
one reference value does not validate every force or approximation.

Existing code, numerical results, and figure data are retained as reference
material. Some code fragments omit initialization or contain placeholders;
they are not standalone programs. This editorial work does not constitute
a new execution of the Atomiverse engine or its benchmarks. Use a complete
example matching a recorded source revision, and report the inputs, units,
uncertainty, and validation relevant to the conclusion you wish to draw.

## A shared source

The original Markdown files remain the authoritative chapters for both
reading and revision. The print edition assembles them through `main.tex`,
including their mathematical notation; it does not maintain a second set
of authored LaTeX chapters. Code remains literal. Tables retain a text-first
layout, interactive plots have an omission notice, and static image
references currently contribute their alternative text rather than a figure.

Installation belongs in the user guide, implementation details in the
developer guide, and version-dependent capabilities in the method reports
and roadmap. This separation helps the physical explanations remain useful
as the software evolves. Source evidence, build instructions, and the
chapter contract are recorded in the [maintenance guide](MAINTENANCE.md).
