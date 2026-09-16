/// @file cpu/planewave/internal/gth_upf.h
/// @brief Analytic GTH pseudopotential → `UPFData` radial tables.
///
/// The plane-wave path consumes norm-conserving pseudopotentials through
/// `UPFData` (radial mesh + local potential + Kleinman-Bylander projectors).
/// Goedecker-Teter-Hutter potentials are fully analytic, so they can be
/// evaluated onto that same mesh instead of being read from a file. Doing it
/// this way — rather than adding a second G-space route — means the existing
/// `upf_vloc_to_gspace`, `setup_kb_projectors`, force and DFPT code paths
/// serve GTH unchanged.
///
/// Provenance: S. Goedecker, M. Teter, J. Hutter, *Phys. Rev. B* **54**, 1703
/// (1996), Eqs. 1 and 3; C. Hartwigsen, S. Goedecker, J. Hutter, *Phys. Rev.
/// B* **58**, 3641 (1998). Parameters come from the in-tree published tables
/// (`cpu/gpw/basis_data/gpw_tables.h`), never from a parsed string.
#pragma once

#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/planewave/internal/upf_reader.h"

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

/// @brief Evaluate an analytic GTH potential onto UPF-shaped radial tables.
///
/// Unit conventions follow UPF, because the consumers do: `v_loc` is in
/// **Rydberg**, `d_ij` is in **Rydberg**, and each projector's `beta` holds
/// \f$r\,p_i^l(r)\f$ (not \f$p_i^l\f$ itself). `rho_atom` is deliberately
/// left empty — GTH carries no atomic pseudo-density, and the SCF falls back
/// to its uniform starting guess, which affects only the first iteration.
///
/// @param pot     Published GTH parameterization (element, zion, rloc, cloc,
///                nonlocal channels).
/// @param r_max   Outer radius of the log mesh [Bohr]. Must exceed the range
///                where \f$r V_{loc}(r) + Z\,\mathrm{erf}(r)\f$ has decayed,
///                which the screened transform in `upf_vloc_to_gspace`
///                assumes; the default is ample for every tabulated entry.
/// @param n_r     Number of log-spaced mesh points.
/// @return `UPFData` equivalent to a norm-conserving UPF for this potential.
UPFData upf_from_gth(const gpw::GthPotential& pot, AccReal r_max = AccReal(12),
                     int n_r = 2000);

/// @brief Closed-form GTH local potential form factor \f$V_{loc}(G)\f$
///        [Hartree], per atom, already divided by the cell volume.
///
/// Goedecker et al. (1996) Eq. 5. This is an **independent** evaluation of the
/// same quantity `upf_vloc_to_gspace(upf_from_gth(pot), …)` produces by radial
/// quadrature; the unit test compares the two, so a mistake in the radial
/// synthesis or an inadequate mesh cannot pass silently.
///
/// @param pot   Published GTH parameterization.
/// @param g     Reciprocal-lattice vector magnitude \f$|G|\f$ [1/Bohr].
/// @param omega Cell volume [Bohr³].
/// @return \f$V_{loc}(G)\f$ [Hartree]; at `g == 0` the divergent
///         \f$-4\pi Z/(\Omega G^2)\f$ piece is omitted, matching the
///         jellium-cancelled convention `upf_vloc_to_gspace` returns.
AccReal gth_vloc_gspace(const gpw::GthPotential& pot, AccReal g, AccReal omega);

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
