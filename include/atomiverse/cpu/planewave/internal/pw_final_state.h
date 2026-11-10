#pragma once
#include <optional>

#include "atomiverse/cpu/planewave/internal/pw_setup.h"
#include "atomiverse/cpu/planewave/pw_solver.h"

namespace ATV::cpu::pw::internal {

struct FinalBandBlock {
    int spin = 0;
    int ik = 0;
    AccReal k_weight = 0;
    // Owned band-major, full-complex coefficient rows. Fixed empty-spin
    // blocks have zero rows; their density and every band contribution vanish.
    xt::xarray<std::complex<AccReal>> coefficients;
    std::vector<AccReal> epsilon_ha;
    std::vector<AccReal>
        occupations; // signed f, spin factor included, no k weight
};

struct FixedSpinOccupations {
    int n_alpha = 0;
    int n_beta = 0;
};

struct FinalBandState {
    PWResult
        reference_result; // original SCF metadata; never overwritten by NSCF
    std::vector<std::complex<AccReal>> reference_density_G;
    std::vector<std::vector<std::complex<AccReal>>> reference_spin_density_G;
    std::vector<FinalBandBlock> local_blocks;
    std::vector<std::complex<AccReal>> orbital_density_G;
    std::vector<std::vector<std::complex<AccReal>>> orbital_spin_density_G;
    AccReal auxiliary_mu_ha = 0;
    AccReal auxiliary_entropy = 0; // sum w*s, not -width*S
    AccReal auxiliary_electron_count = 0;
    AccReal auxiliary_magnetization = 0;
    AccReal density_mismatch_l2 = 0;
    AccReal max_local_eigen_residual_ha = 0;
    int local_band_solves = 0;
    // Call-scoped producer declaration; independent of reference verdict/markers.
    std::optional<FixedSpinOccupations> fixed_spin_occupations;
};

// Exactly one unstrained band sweep of H[reference density]. No SCF,
// mixing, radial admission or mutation of problem/result. The optional internal
// band count may add zero-filled fixed-occupation gap bands; default follows
// the existing force path. Geometry tables remain in the caller's setup.
// Each orbital must satisfy the solver's existing method.tol*.1 Hpsi residual
// target and |norm_squared-1|<=1e-8. The 40-expansion budget is unchanged.
// Local auxiliary-work failures are synchronized before spectrum/density
// collectives; no incomplete state is returned. Reference SCF convergence is
// independent and its original verdict is preserved.
FinalBandState extract_final_bands(
    PWGridView grid, const PWIons& ions, const PWKMesh& mesh,
    PWDensityView density, const xt::xarray<std::complex<AccReal>>& trial,
    const PW& method, const PWResult& reference, int final_band_count = 0);

} // namespace ATV::cpu::pw::internal
