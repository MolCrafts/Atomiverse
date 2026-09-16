# GPW Γ-real ≡ k={Γ}: operator identity and how it is guarded

At k = Γ the Bloch problem reduces exactly to the real Γ problem. In ATV this
is a **numerical identity, not a physics approximation**: at the same input
density both paths must produce the same S, the same H_KS = ∂E/∂P, and the
same energy terms to machine precision. Guarded by
`GpwGammaKIdentity.FixedDensityOperatorsMatch`
(`tests/cpu/gpw/test_cpu_gpw_gamma_k_identity.cpp`, fixed-density
`max_iter == 0` evaluation, no SCF), plus the SCF-level gates
`GpwForces.H2ExplicitGammaUksMatchesGamma` and
`GpwDftU.KpointGammaLimitMatchesRealGammaUks`. None of these gates may be
widened.

## The CompactBasisGrid invariant

`build_compact_basis_grid` stores **one entry per (grid point, function)**:
the Γ Bloch sum χ_u(r_g) = Σ_T φ_u(r_g − T). When the unwrapped evaluation
window is wider than the box (Gaussian tail radius ≥ L/2 — diffuse bases or
small cells), several window cells map to one grid point; those per-image
values are **coalesced at build time** (they land adjacent after the stable
counting sort, so the merge is a linear pass and thread-count-independent).

Downstream contractions (`collocate_density*` triangle, `integrate_potential*`
upper-triangle + symmetrize, the force cross loops) all assume this invariant.
Per-image duplicate entries are NOT a valid CompactBasisGrid: the KS-integrate
triangle puts the cross-image product φ_u^{T1}·φ_u^{T2} on the **diagonal**
element (u,u), where "accumulate upper once, then symmetrize" counts it once
instead of twice — H_KS stops being the exact ∂E/∂P of the collocated energy
(≈5e-7 on H2/12 Bohr) while every *energy* stays exactly right, and the Γ SCF
converges to a fixed point ~1e-4 (max element of P) off the k={Γ} one. The
collocate triangle handles duplicates correctly (all ia<ib pairs ×2), which is
what made the two directions inconsistent rather than merely inexact.

The k path never had the problem: `build_kpoint_basis_grid` accumulates images
densely into χ^k before any product is formed.

## Debugging inference to keep

The energy is **stationary** in P; forces are **first order** in P:

  δP ~ 1e-5  ⇒  δE ~ δP² ~ 1e-10  and  δF ~ δP ~ 1e-5

so "the energies of two paths agree to 1e-10, therefore they sample the same
functional" is **not a valid inference** — matching energies with diverging
forces is precisely the signature of an operator (∂E/∂P) inconsistency.
The decisive probe is a **fixed-density evaluation** (`max_iter == 0` + density
guess) diffing H_KS between paths: it isolates the operator from the SCF in one
KS build. Parameter sweeps (cutoff, tol, eps) cannot close this class of gap —
each path converges to its own fixed point.
