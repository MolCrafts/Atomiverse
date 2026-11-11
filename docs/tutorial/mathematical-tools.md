# Mathematical Tools

Mathematics helps us express a physical question precisely: how does an
energy change, which motion restores equilibrium, and how uncertain is an
average? The symbols in this chapter are tools to learn gradually. Work
through the examples with a pencil; return to the relevant section when a
later calculation gives the symbols a new physical meaning.

**Learning goals.** Use derivatives and the chain rule; interpret matrices,
variational bounds, fixed points, and statistical uncertainty; state the
assumptions behind each conclusion.

**Before you begin.** Elementary algebra and coordinates are sufficient.
[Systems and Units](ch01-foundations.md) explains what the physical
quantities represent. You need not master every section before proceeding.

## Coordinates and vectors

A vector such as $\mathbf R=(x,y,z)$ collects three components. Its length
is $|\mathbf R|=\sqrt{x^2+y^2+z^2}$. The dot product
$\mathbf a\cdot\mathbf b=\sum_\alpha a_\alpha b_\alpha$ measures how much
one vector points along another; perpendicular vectors have zero dot product.
The index $\alpha$ here runs over Cartesian directions, not atoms.

For atoms $A$ and $B$, define $\mathbf R_{AB}=\mathbf R_A-\mathbf R_B$ and
$r_{AB}=|\mathbf R_{AB}|$. These are different objects: one has direction,
the other is a nonnegative distance. This distinction fixes the sign of a
pair force. In a periodic system, use the displacement convention appropriate
to the cell and interaction rather than subtracting wrapped coordinates
without an image correction.

## Derivatives and the chain rule

The derivative $U'(x)$ is the limiting ratio of an energy change to a small
displacement. Its units are energy per length. A partial derivative changes
one coordinate while holding the others fixed. The gradient collects these
derivatives: $\nabla U=(\partial U/\partial x,\partial U/\partial y,
\partial U/\partial z)$ for one three-dimensional coordinate.

If an energy depends on a distance which in turn depends on coordinates,
the chain rule connects the two changes. For $r_{AB}>0$,

$$\nabla_A r_{AB}=\frac{\mathbf R_{AB}}{r_{AB}},\qquad
\mathbf F_A=-u'(r_{AB})\frac{\mathbf R_{AB}}{r_{AB}}.$$

The force on $B$ is opposite for this isolated pair interaction. For example,
a harmonic bond has $u(r)=k(r-r_0)^2/2$, with stiffness $k>0$. Stretching it
beyond $r_0$ makes $u'(r)>0$, so the force on $A$ points toward $B$.
Checking this direction is often more useful than checking a decimal place.

The same rule applies to a local environment. If an embedding energy is
$F(\rho)$ and $\rho$ depends on a neighbor distance, then
$dF/dr=F'(\rho)\,d\rho/dr$. The two derivatives describe distinct responses;
omitting either one changes the force model.

## Taylor expansion and numerical slopes

Taylor's theorem relates nearby values of a smooth function. If the third
derivative is continuous and bounded in the interval of interest,

$$U(x+h)=U(x)+hU'(x)+\frac12h^2U''(x)+O(h^3).$$

The notation $O(h^3)$ means that the magnitude of the remainder is bounded
by a constant times $|h|^3$ for sufficiently small $h$. It is a statement
about how an error scales, not its exact value at a particular step size.
Subtracting the expansions at $x+h$ and $x-h$ gives

$$\frac{U(x+h)-U(x-h)}{2h}=U'(x)+O(h^2).$$

Thus a central difference estimates the slope. A force needs its negative.
Away from round-off limits, halving $h$ reduces this leading truncation error
by about a factor of four. If each computed energy has an absolute error of
order $\delta$, subtraction can instead produce derivative error of order
$\delta/h$. Making $h$ arbitrarily small is therefore not a reliable strategy.

For a harmonic bond the central difference is exact in exact arithmetic,
because the energy is quadratic. This makes it a useful first check, but
not a demonstration of the usual truncation-error trend. To observe that
trend, use a smooth nonquadratic energy, such as a Lennard–Jones pair away
from its singularity and cutoff. Compare several displacement sizes.

## Curvature and stationary points

The second derivative measures curvature. In many coordinates, write the
coordinates as a vector $\mathbf q$ and define the Hessian by
$H_{ij}=\partial^2 U/(\partial q_i\partial q_j)$. If the second derivatives
are continuous, the Hessian is symmetric. Near a stationary point
$\mathbf q_*$, where $\nabla U(\mathbf q_*)=0$,

$$U(\mathbf q_*+\mathbf s)
=U(\mathbf q_*)+\frac12\mathbf s^T H\mathbf s+o(|\mathbf s|^2).$$

Here $\mathbf s$ is a displacement and the superscript $T$ denotes
transpose. The little-$o$ remainder divided by $|\mathbf s|^2$ tends to zero
as the displacement tends to zero. A positive-definite Hessian means
$\mathbf s^T H\mathbf s>0$ for every nonzero displacement; it guarantees a
strict local minimum under these smoothness assumptions. A negative
curvature direction rules out a local minimum. Zero eigenvalues alone do
not settle the question: they may reflect symmetry or require higher-order
terms. Translation and rotation of an isolated molecule provide important
examples of symmetry directions.

## Matrices and eigenvectors

A matrix is a linear rule for transforming a vector. For an eigenvector,
the transformed vector is a scalar multiple of the original:
$A\mathbf v=\lambda\mathbf v$. The scalar $\lambda$ is its eigenvalue.
Normalization sets $\mathbf v^T\mathbf v=1$, so that a vector's length does
not obscure its direction.

**Spectral theorem.** Every finite real symmetric matrix has real
eigenvalues and an orthonormal basis of eigenvectors. For complex matrices,
the corresponding condition is Hermiticity, $A^\dagger=A$; the dagger means
transpose followed by complex conjugation. This theorem lets us separate
a quadratic energy into independent curvature directions.

For two equivalent sites, consider

$$A=\begin{pmatrix}a&b\\b&a\end{pmatrix}.$$

Multiplication gives eigenvectors $(1,1)^T/\sqrt2$ and
$(1,-1)^T/\sqrt2$, with eigenvalues $a+b$ and $a-b$. They describe same-sign
and opposite-sign amplitudes. In a two-orbital model these combinations
help explain bonding and antibonding states, but which lies lower depends
on the Hamiltonian, here on the sign of $b$.

Basis functions need not be perpendicular. Their inner products form the
overlap matrix $S$, and normalization becomes $\mathbf c^\dagger S\mathbf c=1$.
The orbital problem then has the form $F\mathbf c=\varepsilon S\mathbf c$.
If $S$ is positive definite, choose $X=S^{-1/2}$ and write
$\mathbf c=X\widetilde{\mathbf c}$. Multiplication by $X^\dagger$ gives

$$\widetilde F\widetilde{\mathbf c}=\varepsilon\widetilde{\mathbf c},
\qquad \widetilde F=X^\dagger F X,\qquad X^\dagger S X=I.$$

The inverse square root is constructed by diagonalizing $S$ and replacing
each positive eigenvalue $s$ by $s^{-1/2}$. A very small $s$ produces a large
factor: nearly redundant basis functions can amplify numerical error.
Removing such directions changes the numerical space and should be followed
by a check of the physical quantity of interest.

## The variational principle

In quantum mechanics a normalized state has total probability one.
The notation $\langle\Psi|\hat H|\Psi\rangle$ denotes the expectation of
the energy operator, or Hamiltonian, in that state. For a self-adjoint
Hamiltonian bounded below, an admissible normalized trial state of finite
energy cannot have an expectation below the lowest spectral energy $E_0$.

In a discrete eigenbasis, write $\Psi=\sum_n c_n\Psi_n$, with
$\sum_n|c_n|^2=1$. Orthogonality removes cross terms, giving

$$\langle\Psi|\hat H|\Psi\rangle
=\sum_n|c_n|^2E_n\ge E_0.$$

The general statement also covers continuous spectra through their spectral
resolution. The discrete example shows the reason for the bound: a weighted
average cannot lie below its smallest contributing energy.

Enlarging a nested trial space cannot increase the best variational energy,
provided the Hamiltonian is unchanged and each minimization finds its best
state. An iterative solver that stops at a different local stationary point
need not display that trend. Nor does this theorem supply an upper bound
for a dipole, a bond length, or an energy produced by an unrelated approximate
functional. Later chapters distinguish these different uses of approximation.

## Fixed points and iteration

A fixed point satisfies $x_*=g(x_*)$: applying the update rule leaves it
unchanged. Iteration forms $x_{n+1}=g(x_n)$. The index $n$ counts numerical
updates; it is not a physical time coordinate.

**Contraction theorem.** Suppose a map takes a nonempty complete domain
into itself and, for all points in it, satisfies
$|g(x)-g(y)|\le q|x-y|$ with $0\le q<1$. Then there is one fixed point in
that domain and iteration from any point there converges to it. A closed
subset of Euclidean space is complete: limits of its Cauchy sequences stay
in the set. The estimate

$$|x_n-x_*|\le\frac{q^n}{1-q}|x_1-x_0|$$

follows by summing the geometrically shrinking changes between iterates.
For example, $g(x)=a x+b$ with $|a|<1$ has fixed point $b/(1-a)$.
Negative $a$ produces alternating errors, even though they decrease.

For a continuously differentiable scalar map near a fixed point,
$|g'(x_*)|<1$ establishes local attraction. Electronic self-consistency is
a higher-dimensional nonlinear problem and does not automatically satisfy
a global contraction bound. Convergence controls must be tested; an
iteration budget is not a theorem about the result.

## Averages and uncertainty

A probability density $p(x)$ is nonnegative and integrates to one. An
integral is the continuous counterpart of a sum. The expected value of an
observable $A$ is $\langle A\rangle=\int A(x)p(x)\,dx$; its variance is
$\sigma_A^2=\langle(A-\langle A\rangle)^2\rangle$.

For $M$ independent samples from the same distribution, the sample mean
$\overline A=M^{-1}\sum_j A_j$ has variance $\sigma_A^2/M$ when that
variance is finite. To see why, expand the variance of the sum: independence
sets the covariance between different samples to zero, leaving $M$ equal
diagonal terms. The standard error is the square root of the mean's
variance, not the spread of individual measurements.

Trajectory samples generally have nonzero covariances. Saving twice as often
does not usually double the independent information. Equilibration concerns
whether the sampled distribution has settled; correlation concerns how much
new information the next observation adds. Neither is repaired simply by
printing more digits. [Statistical Ensembles](ch11-ensembles.md) develops
a block-based way to estimate uncertainty in such data.

## Practice plan

1. For $U(x)=k(x-x_0)^2/2$, calculate the force and curvature by hand.
   Choose $x\ne x_0$ and compare the analytic force with central differences
   at several $h$. At $x=x_0$, use an absolute rather than relative force error.
2. For two atoms, separate center-of-mass motion from their relative bond
   coordinate. Its reduced mass is $\mu=m_1m_2/(m_1+m_2)$, giving angular
   frequency $\omega=\sqrt{k/\mu}$ in coherent units. The period is
   $2\pi/\omega$, not $1/\omega$.
3. Verify both eigenvectors of the two-site matrix by multiplication. Change
   the sign of $b$ and determine which state is lower without a numerical solver.
4. Iterate $x_{n+1}=a x_n+b$ for one positive and one negative $a$ with
   $|a|<1$. Compare the error with $a^n(x_0-x_*)$. Then examine $|a|>1$.

These small models give exact reference answers. They are preparation for
checking an implementation, not a demand to solve every molecular problem
analytically. Take a question that remains unclear to the next chapter and
return when a trajectory or orbital calculation provides another example.

## Questions for reflection

Why can a smaller finite-difference displacement give a worse force?
Why does a stationary geometry need a curvature check? Which assumption
fails when a converged iterative result does not improve in a larger basis?
How does a spread of individual measurements differ from uncertainty in their mean?
