# PW closed-form stress: implementation audit and design

Initial read-only design: 2026-10-27. The current CPU implementation covers
Gamma and supported general-k LSDA/PBE-family RKS/populated UKS with fixed,
FD and Gaussian filling. Ten fixed, ten FD and twelve Gaussian actual-QE
primary references cover the matched Gamma/full-eight-k fixtures. Fixed
occupations constrain band indices per k and do not imply a global insulating
gap; the negative indirect-gap observation retains its separate failed gate.
Gamma fixed LSDA additionally admits Na>0/Nb=0 with a producer declaration,
canonical zero-row beta blocks and exactly zero beta densities. One matched
stretched-triplet QE reference passes all original gates, rotation and MPI2;
all nine strain lanes pass at two steps on the constant-beta-zero tangent.
Other nominally empty fixed/Gaussian channels retain earlier paths. A narrow
MP1/PBE/actual-Gamma/populated-UKS selector has been added; its
[CPU integrated archive](../../tools/diagnostics/pw-analytic-stress/results/2026-10-28-mp-gamma-pbe/cpu-integrated/archive-manifest.json)
passes independent tensor, full-SCF and matched-QE checks. Its 1487-source
wave passes all scoped mathematical gates and a repository-independent
relocated replay. Three original strict-parser helper failures and seven
successful same-log recollections remain explicit; the original pipeline
exit is still one. The separately qualified
[restricted MP1 PBE Gamma Si2 slice](../../tools/diagnostics/pw-analytic-stress/results/2026-10-28-mp-rks-gamma-pbe/cpu-integrated/archive-manifest.json)
retains degeneracy two, signed occupations and positive total density. Its
fresh main-source wave completes 88 groups/132 rank records, 281 units and
18 artifact identities. Independent full-SCF strains and a freshly linked
same-C/f oracle pass; relocated parser replay hides the original repository,
external molpy and SDK. Its full physical and bookkeeping scope is limited
to the archived 12-Ry/.02-Ry/six-band model. Other MP settings keep their
earlier path.
CUDA covers Gamma RKS fixed/FD and fixed nonempty UKS LSDA/PBE-family total
force/stress with certified final-band ensembles and request-aware device
scratch. The [integrated UKS archive](../../tools/diagnostics/pw-analytic-stress/results/2026-10-28-cuda-uks-depth/archive-index.json)
seals 162 native records, 135 request cases, 33 unit tests, three clean
memchecks and 18 private artifact identities, with independent review and
relocated parser replay while the original repository is hidden. Both
backends preserve the accepted reference SCF result and avoid strained band
solves in their supported analytic domains. Finite unsupported spin-XC
fields use the inherited numerical epilogues with explicit host transfers
and no extra SCF. Separate positive-spin device XC controls cover all nine
strains, two steps, rotation and nontrivial cross-gradients for five models;
neither these controls nor the public fixed-UKS scope qualify clipped/empty
analytic derivatives, UKS FD or multiple k points.
The [earlier fixed/FD/Gaussian and empty-spin receipt](../../tools/diagnostics/pw-analytic-stress/results/2026-10-28-empty-spin-mixing-depth/collection-portable/wave-validation.json)
records 1480 exact source identities, native binary scopes, matched actual-QE
energy/full-force/six-stress checks and independent prior mathematical evidence.
Private profiles and earlier failed gap/link/diagnostic controls retain their
own attribution. Further CUDA spin/filling/k coverage remains open. Derivations
below describe the wider intended model; they do not imply full implementation.
The equations below are derived for Atomiverse's present scalar norm-conserving
Hamiltonian and normalized complex PW coefficients. Their signs, volume factors
and operator split were cross-checked against the cited primary QE sources.

## 1. Scope, geometry and result convention

Use the reference direct cell H (columns are lattice vectors), fractional ionic
coordinates s_I, fractional k points, and **fixed integer Miller lists** for every
wavefunction sphere and the density sphere. Under F=I+epsilon:

- H'=F H, R'_I=F R_I, G'=F^(-T)G, q'=F^(-T)(G+k).
- Omega'=det(F) Omega; normalized density Fourier coefficients rho_G scale as
  det(F)^(-1) at fixed reduced density; normalized PW orbital coefficients C_G
  are held fixed for the explicit derivative.
- Atomic phases q.R and G.R are invariant. There is no independent atomic-phase
  strain term at fixed fractional coordinates.

Accumulate a full 3x3 **compressive-positive virial** W_ab=-dF_free/d epsilon_ab
in Ha. Publish sigma_ab[eV/Angstrom^3]=W_ab*HartreeInEv/frame.box.volume(), and
kStress=W*HartreeInEv. Voigt ordering remains xx,yy,zz,xy,xz,yz.
An analytic tensor component requires no factor two. The factor two belongs only
to an oracle deforming both epsilon_ab and epsilon_ba together.

This reduced-coordinate choice follows the invariant-basis metric formulation
of Hamann/Wu/Rabe/Vanderbilt, PRB 71,035117 (2005). It is a finite-reference-basis
derivative. Reselecting Miller modes at fixed physical ecut can jump; do not add
an ad hoc cutoff derivative, smooth the cutoff, or fit a Pulay correction to
make a crossing case pass. Preserve the existing cutoff-crossing negative
control and report cutoff convergence separately.

Primary reference:
https://docs.abinit.org/developers/strain_nonlocal/strainpert.pdf

## 2. Present implementation and missing pieces

- src/cpu/planewave/internal/pw_stress.cpp currently performs one reference SCF
  and twelve strained NSCF Davidson evaluations. It is strain FD, despite the
  legacy `analytic` symbol. Keep it as a numerical oracle; do not wrap it and
  relabel the result closed-form.
- pw_setup.cpp owns local radial transforms and diagonalizes each full real D_l
  block before rotating its radial projectors. KB beta_G already contains
  4*pi/sqrt(Omega), (-i)^l, real spherical harmonics and exp(-i q.R_I).
- pw_forces.cpp already contracts local and KB operators, but its ionic phase
  derivative is not a cell derivative. Stress additionally needs reciprocal
  radial and angular derivatives plus normalization derivatives.
- xc_builder.cpp evaluates the actual padded-grid discrete energy and its
  adjoint potential, including clipped spin densities. Stress must differentiate
  that same discrete energy, not a base-grid copy or a separately clipped field.
- ewald.cpp owns independent precision-controlled real and reciprocal sums,
  self and **nonzero ionic-background** terms. Its six-component virial is now
  exposed internally and consumed by the accepted first slice.
- PWWorkspace has only the currently bound orbital matrix (plus a few optional
  channel copies); there is no persistent accepted (spin,k,band) tensor/spectrum
  cache. Density alone cannot supply kinetic or nonlocal stress.

The PUBLIC NormConservingPotential provider already supplies sufficient data:
V(r), beta(r)=r*p(r), full D, exact radial prefixes, dr/di and mesh. It needs no
new cell/strain callback, provider Driver or numerical beta'(r) callback.
The radial/Cartesian derivatives and fixed-orbital T/H/local/full-D KB virials
are now **internal operators** derived from this data. A call-scoped
`FinalBandState` owns the complete local C/f/w ensemble and orbital density;
Qualified semilocal fixed/FD/Gaussian RKS/populated-UKS force/stress consumers
share it at the existing private `PW::evaluate_` boundary, including supported
general-k layouts. Other stress admissions retain the numerical path.

## 3. Orbital terms

Let w_k sum to one over the full k grid. f_skn is 2 for restricted fixed
occupation, 1 for an occupied collinear fixed spin orbital, or the actual
reference fractional occupation. Wavefunctions obey sum_G |C_sknG|^2=1.
Atomiverse enumerates full complex G spheres: do not copy QE's gamma-compressed
factor two on top of these sums or on top of f=2.

### Kinetic

T=1/2 sum_sknG w_k f_skn |C_sknG|^2 q^2,

W^T_ab = sum_sknG w_k f_skn |C_sknG|^2 q_a q_b.

This gives 2T on the trace for an isotropic scale derivative. QE's kfac=1 NC
path has precisely this q_a q_b contraction; its e2=2 is the Ry convention.

Primary reference: https://raw.githubusercontent.com/QEF/q-e/qe-7.5/PW/src/stres_knl.f90

### Nonlocal norm-conserving KB

Write beta_Imi,G=e^(-iq.R_I) b_mi(q), where
b_mi(q)=(4*pi/sqrt(Omega))(-i)^l f_i,l(q)Y_lm(qhat).
For a given orbital c_Imi=sum_G e^(+iq.R_I) b_mi(q)^* C_G, and
E_NL=sum_skn,Im,ij w_k f_skn c_i^* D_ij c_j.

Define eta_i,ab=sum_G q_a e^(+iq.R_I) [partial b_i(q)^*/partial q_b] C_G,
where the partial derivative holds Omega fixed and differentiates **only the
radial/angular part**, not the atomic phase. Then

d c_i/d epsilon_ab = -1/2 delta_ab c_i - eta_i,ab,

W^NL_ab = delta_ab E_NL + 2 sum_skn,Im,ij w_k f_skn Re[c_i^* D_ij eta_j,ab].

Either use the full D block or the existing strain-independent diagonalization
D=U diag(d) U^T. Rotate f'_l with the **same U** used for f_l. Never differentiate
only radial factors, omit the angular derivative, discard D off-diagonals,
redifferentiate LAPACK eigenvectors, or add a q.R phase derivative that should
cancel under affine ionic motion.

For canonical beta_rad=r*p(r):

f_l(q) = integral r beta_rad(r) j_l(qr) dr,
f'_l(q) = integral r^2 beta_rad(r) j'_l(qr) dr.

Differentiate the existing Simpson sum on each exact radial prefix. Use stable
small-x Bessel series and Cartesian **solid harmonic products** to avoid
singular Y_lm gradients at q=0:

f_l(q)Y_lm(qhat) = [f_l(q)/q^l] H_lm(q).

j_l(x)=x^l/(2l+1)!! * [1-x^2/(2(2l+3))+...]. The q_a times derivative term
vanishes at q=0; the l=1 Cartesian derivative itself has a finite nonzero limit.
An isotropic-only test cannot catch a missing angular derivative: require all
shears and a rigid-rotation test with nonlocal p/d channels.

QE explicitly separates normalization, Bessel and spherical-harmonic pieces.
Primary reference: https://raw.githubusercontent.com/QEF/q-e/qe-7.5/PW/src/stres_us.f90

## 4. Density-dependent terms

Use full density-sphere sums. rho_G has density units Bohr^-3 and
rho_0=N_e/Omega. S_I(G)=sum_I exp(-i G.R_I) is invariant under the strain.

### Local potential

Let v(g)=U(g)/Omega be the single-atom form factor returned by
nc_vloc_to_gspace and V_loc,G=S_I(G)v(g).
E_loc=Omega sum_G Re[rho_G^* V_loc,G]. Then

W^loc_ab = delta_ab E_loc
          + Omega sum_G Re[rho_G^* S_I(G)] v'(g) G_a G_b/g.

For the current erf(r)-screened radial convention and g>0,

U(g)=4*pi [ integral A(r)j0(gr)dr - Z exp(-g^2/4)/g^2 ],
A(r)=r[r V(r)+Z erf(r)],

U'(g)=4*pi [ -integral r A(r)j1(gr)dr
             + Z exp(-g^2/4)(1/(2g)+2/g^3) ].

At G=0 use the existing finite alpha-Z value
U(0)=4*pi integral r[r V(r)+Z]dr; its radial-derivative contraction is zero,
but its volume-normalization term delta_ab E_loc,G0 is essential. Preserve the
Hartree/Ewald G=0 and jellium convention exactly. Derive from the same radial
samples for builtin GTH, UPF and external NC; an independent analytic GTH
transform is a useful test oracle, not a second production operator path.

Primary reference: https://raw.githubusercontent.com/QEF/q-e/qe-7.5/PW/src/stres_loc.f90

### Hartree

E_H,G=2*pi*Omega*|rho_G|^2/G^2 for G!=0,

W^H_ab=sum_G!=0 E_H,G [delta_ab-2 G_a G_b/G^2].

The G=0 term is absent in both energy and derivative. The density is already
MPI-complete: a replicated Hartree sum must not be multiplied by rank count.

Primary reference: https://raw.githubusercontent.com/QEF/q-e/qe-7.5/PW/src/stres_har.f90

### XC on the exact existing padded grid

Let e(rho_alpha,rho_beta,sigma_aa,sigma_ab,sigma_bb) be energy **per volume**.
At each existing XC quadrature point let raw density n_s, raw spectral gradient
g_s, local partial u_s=d e/d n_s (including the current clip chain rule), and
flux p_s=d e/d g_s be precisely those used by `evaluate_point`.
For example p_alpha=2 e_sigma_aa g_alpha + e_sigma_ab g_beta.

With dV=Omega/N_pad,

W^xc_ab=dV sum_points {
  delta_ab [sum_s n_s u_s + sum_s g_s.p_s - e]
  + sum_s g_s,a p_s,b
}.

Equivalently its diagonal pressure part is integral(sum_s n_s v_xc,s)-E_xc,
with the separate gradient tensor added. Prefer direct point energy/partial/
flux reduction so it differentiates the exact quadrature without introducing
a second divergence/projection or a factor-of-two ambiguity in sigma_ab.

RKS can be evaluated using n_alpha=n_beta=n/2 and g_alpha=g_beta=g/2; do not
apply another spin degeneracy. UKS uses both actual channel densities and
negative cross-spin sigma without alteration. Current clipping applies to n_s,
not the spectral gradient; a clipped channel retains its defined GGA flux.
Retain that chain rule. At hard density/near-vacuum branch crossings the
functional is not differentiable; do not silently smooth only the potential.
The known LSDA-MP signed-density convergence failure stays an explicit failure.

Keep the same density modes, FFT shape and current LDA/GGA quadrature padding.
For a fixed discrete grid in fractional coordinates, its strain derivative is
exactly the formula above; no extra grid-point interpolation derivative exists.

Primary references:
https://raw.githubusercontent.com/QEF/q-e/qe-7.5/PW/src/stress.f90
https://raw.githubusercontent.com/QEF/q-e/qe-7.5/PW/src/stres_gradcorr.f90

## 5. Ewald with the existing independent precision domains

At reference splitting parameter alpha, ionic charges Z_I, Q=sum_I Z_I,
S_Z(G)=sum_I Z_I exp(-iG.R_I):

W^real_ab = (1/2) sum_IJL' Z_I Z_J R_a R_b
 [ erfc(alpha R)/R^3 + (2 alpha/sqrt(pi))exp(-alpha^2 R^2)/R^2 ],

E^rec_G=(2*pi/Omega)exp(-G^2/(4alpha^2))|S_Z(G)|^2/G^2,

W^rec_ab = sum_G!=0 E^rec_G
 [delta_ab-(2/G^2+1/(2alpha^2))G_a G_b],

W^self_ab=0 at fixed alpha,
E_bg=-pi Q^2/(2 alpha^2 Omega), W^bg_ab=delta_ab E_bg.

Reuse the actual Ewald real-space image and reciprocal precision domains,
including its unique i<=j/fac convention; do not substitute the electronic
G cutoff or count pairs twice. **Q here is the ionic sum**, not the net charge
of ions plus electrons. It is nonzero even in a neutral valence calculation,
so discarding the existing background term gives wrong pressure.

The exact complete Ewald sum is independent of alpha. The code chooses alpha
from cell lengths and truncates at a stringent 1e-14-Ha tail domain. The tensor
above holds alpha fixed at the reference. Verify alpha-independence and strain
FD at that precision; if claiming the literal finite truncated algorithm's
exact derivative, include (dE/dalpha)(dalpha/depsilon) as well or bound the
residual explicitly. Image-domain changes near a cutoff must remain below the
declared precision bound, not be hidden by a stress tolerance change.

Keep the existing setup's Bohr conversion in this work. Assemble W in energy
units then divide by the **actual Frame volume** at publication, which also
avoids mixing the legacy setup conversion with a different volume conversion.

Primary reference: https://raw.githubusercontent.com/QEF/q-e/qe-7.5/PW/src/stres_ewa.f90

## 6. Occupations, entropy and accepted orbital data

At fixed electron count and fixed physical smearing width Delta [Ha],
F=E_KS-Delta*S. On the stationary filling branch,

dF_occ=sum_skn w_k [epsilon_skn-Delta*dS/df_skn] df_skn
      =mu sum_skn w_k df_skn=mu dN=0.

Therefore use reference f_skn in the explicit kinetic/nonlocal contractions.
There is **no extra standalone isotropic TS/Omega stress term** and no need to
finite-difference mu or entropy. FD, Gaussian and MP require their matching
existing generalized entropy/occupation definitions. MP quasi-occupations may
be signed: never clamp them or skip negative bands; use abs(f) only for tiny
contribution screening. The width is fixed in Ry on the method and converted
once to Ha, not scaled with the cell. Finite-temperature UKS uses the shared
mu over both spin spectra; multiplicity is an initial split, not a fixed
magnetization constraint on this path.

Use an accepted final orbital state. At present the SCF mixes density after
measuring energy, and the workspace lacks all-k accepted orbitals. A minimal
correct epilogue is a **single unstrained, frozen-reference-density band sweep**:
rebuild final V_H/V_xc/V_eff, solve each local k/spin once at the existing band
residual gate, collect spectra and per-band kinetic/nonlocal 6-tensors, find
the same global mu, then weight by reference occupations. Per-band tensors
need O(12*N_band*N_k*N_spin) scalars, so stress need not cache every C_G or run
another diagonalization after mu is known. Force and stress epilogues should
share this band sweep rather than independently refreshing the same orbitals.
This is NSCF finalization, not a second SCF: never remix density or change the
public reference-SCF verdict/iterations/energy/occupation metadata.

On MPI, distribute k/spin orbital contributions as the solver already does,
collect the complete spectrum before shared-mu filling, reduce only local
orbital virials once, and add replicated density/XC/Ewald virials once.

For an unconverged reference, preserve `last_result().converged=false` and the
reference metadata. A tensor evaluated at that state does not inherit the
stationary Hellmann-Feynman/energy-derivative qualification. Existing truncated
solve diagnostics must not be made falsely converged or silently discarded.

Finite-temperature strain stationarity is discussed in the metric paper above;
its present application here is an inference from the implemented matching
occupation/entropy functional, not a claim that MP response projectors work.

## 7. Smallest complete implementation sequence

1. Add INTERNAL local form-factor value+radial derivative and KB value+radial/
   solid-harmonic Cartesian derivative routines, preserving exact prefixes,
   D rotations, phases and normalization. Add a six-component Ewald virial
   reduction in the existing energy loops. Add XC point-strain reduction in
   the same padded-grid loop as energy/potential.
2. Add an internal no-SCF tensor assembler over the accepted PWProblem and
   reference orbital sweep. First opt-in vertical case: asymmetric skew-cell
   Si2, GTH-PBE-q4, Gamma RKS fixed occupations, LSDA, with nonzero local and
   nonlocal terms and all six physical components. H-only fixtures cannot
   by themselves qualify nonlocal stress.
3. Extend the same assembler to existing k meshes, UKS and all three smear
   kinds, and generic admitted GGA through point partials/fluxes. Use internal
   opt-in diagnostics until all currently supported public stress paths meet
   their gates. Then replace the production strain-FD call once; retain FD as
   an oracle, without a new permanent user-facing solver/configuration fork.
4. Port the accepted tensor kernels to CUDA using the same physical formulas.
   Do not delay density XC primitive validation on GPU availability, and do
   not qualify a device path from CPU-only compilation.

No new Driver, public StressMethod, strain-provider callback or duplicated SCF
loop is necessary. Operator derivatives are backend numerical kernels over the
same immutable model and method-owned execution state.

## 8. Validation gates and failure preservation

Keep all existing actual-QE H3 LSDA/PBE x FD/Gaussian/MP1 reference energies
(1e-5 Ha) and all six stress components (2e-3 eV/Angstrom^3), their explicit
nontrivial magnetization/entropy/shear checks, the two rotation cases, original
simple-deformation gates, and the max_iter-1 result-metadata checks. Preserve
the failed LSDA-MP compressed sample and cutoff-reselection negative control.
Do not compare only hydrostatic trace or project antisymmetric/force residuals
away before checking them.

New primitive gates should be fixed before native measurements:
- Local U'(g), rotated f'_l and complete Cartesian beta derivatives against
  independent central derivatives of the original value operator, at q=0,
  small q and generic directions; include l=0/1/2, signed/full off-diagonal D,
  exact unequal radial prefixes and a rotated representation of the same D.
- Every virial contribution against central strain derivatives of its exact
  frozen-C/f/reduced-density energy, keeping Miller labels/FFT shape fixed.
  Use a step sweep to demonstrate quadratic FD error then roundoff; start
  with component gates of 1e-8 Ha absolute plus 1e-7 relative for these staged
  bounded-energy fixtures, without reusing derivative code in the oracle.
- Ewald pressure/shear and alpha-invariance on skew/multi-ion cells, including
  nonzero ionic Q and explicit background, at its controlled tail accuracy.
- XC LDA/GGA with high-frequency modes, clipped-channel gradients and negative
  sigma_ab, on the actual padded grid and its exact discrete energy.
- Total stress versus simple-component full-SCF derivatives on geometries
  whose reference Miller lists are verified not to change. Retain the current
  2e-3-eV/Angstrom^3 public gates, and separately record stricter observed errors
  and h-sweep convergence; do not replace fixed-basis tests by recut basis FD.
- New actual-QE skew Si2 or a matched NC-UPF input with nonlocal p/d channels:
  energy, all six stress entries and full forces. Existing H3 pins establish
  spin/thermal/XC depth, not a nonlocal-stress external pin.
- 1-vs-2-rank k-mesh comparison at existing 1e-6-Ha and 2e-5-eV/Angstrom^3
  gates; check no replicated-term/rank multiplication and every tensor lane.

Termwise native measurements and an actual nonlocal-stress external reference
are required before claiming this design implemented or qualified. The source
cache under `qe-7.5/` has exact URLs and SHA256 hashes, including recorded 404s;
it is research provenance only, never a substitute for executed native data.

## Integration state audit (2026-10-28)

The current four CPU PW SCF branches form their iteration energy before
mixing, then retain the post-mix density. At finite residual the public
reference energy, chemical potential and entropy are therefore not exactly
the same state as the final workspace density. The workspace wavefunctions
also hold only the last visited k point and spin channel. Neither object is
an all-channel accepted orbital cache.

A derivative consumer must keep the public reference result and its retained
density distinct from an auxiliary final-band sweep. A single unstrained
sweep can gather every channel's coefficients, signed occupations and k
weights. Its orbital density must be identified separately. Component
energies and virials need one consistent density; combining orbital kinetic
and nonlocal terms with unrelated local/Hartree/XC densities is not a closed
functional derivative. The initial Gamma restricted fixed-occupation slice
quantified this difference under strict SCF convergence before changing
the public stress path. Later slices preserve the same separation. This audit
does not alter the existing SCF acceptance
order or claim that a final-band sweep is another SCF calculation.

## Padded-grid energy reduction accuracy (2026-10-28)

A small-strain Si2 PBEsol check exposed accumulation error in the existing
serial double-precision XC energy sum. Every point's density, sigma arguments
and returned energy density matched the independent reconstruction bit for
bit. The production result matched a fused multiply-add accumulation; its
maximum difference from a long-double reduction was 1.47e-13 Ha. Dividing two
such energies by a 3e-6 strain step caused one XC component to exceed its
original 1e-8 Ha plus 1e-7 relative gate. The failed production-observable run
remains failed; higher-precision mathematical evidence did not overwrite it.

Both RKS and spin loops now compensate their energy sum in working precision.
The point evaluator, potential, six-component XC virial and density residual
rule are unchanged. Actual isolated-library comparisons preserve every active
potential/density/force/W payload bit for bit and pass the original component
gates. Uniform G0-only densities provide a separate exact quadrature oracle:
E=Omega*e_xc. The same ten cases and 8*epsilon*max(1,abs(E)) fixture bound fail
on the old library and pass on x86 and ARM. This bound belongs to those
well-conditioned constant-density fixtures, not arbitrary cancellation or
external functionals.

Compensation must retain its arithmetic dependencies; reassociation can erase
the correction, as discussed in [Kahan's compiler analysis](https://people.eecs.berkeley.edu/~wkahan/SPECbnch.pdf).
[Ogita, Rump and Oishi](https://www.tuhh.de/ti3/paper/rump/OgRuOi05.pdf)
give the broader context of compensated summation and dot products; the present
small change does not claim their stronger algorithms' universal guarantees.
The [scoped acceptance](../../tools/diagnostics/pw-analytic-stress/results/2026-10-28-accuracy-stream/source-review/wave-validation.json)
keeps exact-integration, original tiny-strain failures and actual public/QE
checks separately attributable to their recorded libraries.

## Finite-temperature extension contract (2026-10-28)

The candidate Gamma extension keeps the orbital-functional tensor and the
accepted SCF free energy separate. For F=E-T*S at fixed temperature and
electron number, stationary shared-mu Fermi occupations cancel the
occupation-response terms: the remaining mu*dN is zero. The current FD
implementation uses that shared chemical potential and allows magnetization
to change. If a separate magnetization constraint is imposed, its multiplier
times dM also vanishes at fixed M. The tensor can
therefore use the explicit strain derivative at fixed auxiliary occupations.
This argument requires a stationary ensemble; a final-band sweep at finite
reference-density residual is only an approximation to it. Record the actual
residual, orbital/reference density mismatch, entropy and auxiliary free energy
in diagnostics rather than replacing the accepted result.

This interpretation is consistent with the explicit occupation-weighted
kinetic/nonlocal assembly in
[QE 7.5 stres_knl](https://raw.githubusercontent.com/QEF/q-e/qe-7.5/PW/src/stres_knl.f90)
and the component sum in
[QE 7.5 stress](https://raw.githubusercontent.com/QEF/q-e/qe-7.5/PW/src/stress.f90).
It is a derivation checked against reference source, not numerical acceptance
of the extension. The existing executed QE 7.6 pins remain the external oracle;
CPU FD multiple-k dispatch now has its separate matched eight-k qualification.
CPU Gaussian and CUDA Gamma RKS FD now have their own current qualification.
MP, further CUDA filling/spin/k tensors and exact empty-spin boundaries retain
separate qualification work.

### Gaussian stationarity contract

Gaussian broadening uses a generalized entropy, rather than the physical
Fermi–Dirac entropy. Write its width in Hartree as tau, let
`x=(epsilon-mu)/tau`, and include the spin degeneracy g in each occupation:

\[
f(x)=\frac{g}{2}\operatorname{erfc}(x),\qquad
s(x)=\frac{g}{2\sqrt{\pi}}e^{-x^2}.
\]

Then `s'(x)=x*f'(x)`. At a stationary spectrum, the occupation variation of
`E-tau*sum(w*s)` is therefore `sum(w*(epsilon-tau*x)*df)=mu*dNe`.
It vanishes at fixed electron number. The explicit strain tensor can use the
same final C/f state without adding a separate entropy pressure. The entropy
is a sum over levels; it has no explicit volume factor at fixed occupations.
Gaussian width is a numerical broadening parameter here, not a claim of
physical electronic temperature.

The Gaussian branches of
[QE wgauss](https://raw.githubusercontent.com/QEF/q-e/qe-7.5/Modules/wgauss.f90)
and [QE w1gauss](https://raw.githubusercontent.com/QEF/q-e/qe-7.5/Modules/w1gauss.f90)
provide the reference sign and degeneracy conventions. This derivation does
not qualify an implementation: matched grids, actual reference outputs,
nontrivial occupations, reference-state preservation and auxiliary residuals
still need independent checks. The MP signed-density convergence boundary
does not inherit a Gaussian pass.
