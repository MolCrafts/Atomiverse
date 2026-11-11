/// Pure fixed-Miller PW operator energies and cell virials; no SCF or solver state.
#pragma once

#include "atomiverse/cpu/planewave/internal/g_vector.h"
#include "atomiverse/cpu/planewave/internal/nc_potential.h"
#include <array>
#include <complex>
#include <span>

namespace ATV::cpu::pw::internal {

struct FrozenOperatorVirial {
    AccReal energy_ha = 0;
    // W_ab = -dE/d epsilon_ab, Hartree, row-major and never symmetrized.
    // Inspect antisymmetry before forming a physical symmetric stress tensor.
    std::array<AccReal, 9> virial_ha{};
};

struct FrozenBandView {
    // Band-major coefficients on the full complex per-k G list; no Gamma factor.
    std::span<const std::complex<AccReal>> coefficients;
    // Signed f, including any spin factor, excluding the explicit k weight.
    std::span<const AccReal> occupations;
    AccReal k_weight = 1;
};

// Deformation holds integer Miller modes, FFT shape, C and f fixed:
// q'=F^-T q, R'=F R, Omega'=det(F)Omega, rho_G'=rho_G/det(F).
// Inputs are immutable. Positions and volume below are in Bohr and Bohr^3.
FrozenOperatorVirial kinetic_frozen_virial(const GVectorList& wave_vectors,
                                           FrozenBandView orbitals);

// The density list is Gamma-centred (kvec=0). The g^2<1e-16 exclusion follows
// the existing Poisson operator, including its omission of the uniform mode.
FrozenOperatorVirial
hartree_frozen_virial(const GVectorList& density_vectors,
                      std::span<const std::complex<AccReal>> density,
                      AccReal omega);

// All positions in one call belong to this admitted NC species; sum calls for
// multiple species. The local G=0 finite part contributes its volume derivative.
FrozenOperatorVirial local_potential_frozen_virial(
    const NcPotentialData& data, const GVectorList& density_vectors,
    std::span<const Vec3> atoms, std::span<const std::complex<AccReal>> density,
    AccReal omega);

// Includes radial, angular and normalization derivatives and the complete D in
// its original projector representation. Affine atomic phase derivatives cancel.
// No small-D, small-projector or positive-occupation screening is performed.
// These are literal NcPotentialData energy derivatives. The caller must use
// this same operator for its orbitals; arbitrary exponent rescaling still
// requires representable floating-point intermediate values.
FrozenOperatorVirial nonlocal_frozen_virial(const NcPotentialData& data,
                                            const GVectorList& wave_vectors,
                                            std::span<const Vec3> atoms,
                                            FrozenBandView orbitals,
                                            AccReal omega);

} // namespace ATV::cpu::pw::internal
