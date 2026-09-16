/// @file cpu/scf/dftd3.cpp
/// @brief Molecular D3BJ — Frame (Å) → PBE-D3(BJ) energy [Hartree] and forces
///        [Hartree/Bohr].
///
/// Both entry points funnel through ONE call to `gpw::dftd3_pbe_bj` with the
/// same members, so the energy this class reports and the force it reports are
/// two fields of one evaluation of one functional. That is the invariant the
/// mean-field gradient depends on: `e_total_ += e_disp_` and the dispersion
/// gradient leg must never be able to describe different surfaces. The D3 math
/// itself (C6 interpolation, BJ damping, the dCN chain) lives once, in
/// `cpu/gpw/internal/dftd3.cpp`; there is no molecular copy of it.

#include "atomiverse/cpu/scf/dftd3.h"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/gpw/internal/dftd3.h"
#include "atomiverse/cpu/gpw/internal/dftd3_params.h"

namespace ATV {
namespace cpu {
namespace scf {

namespace {

/// @brief Frame (Å, Z) → the isolated PBE-D3(BJ) evaluation [a.u.].
///
/// Shared by `D3BJ::energy` and `D3BJ::forces` so neither can drift into a
/// different geometry marshalling, element check, or parameter set.
/// @param frame System geometry (Å, Z).
/// @param s6,s8,a1,a2 BJ scales (a2 [Bohr]).
/// @param rcut_bohr Pair cutoff [Bohr].
/// @return Energy [Hartree] + forces [Hartree/Bohr]; empty for an empty frame.
gpw::D3Result evaluate_d3bj(const Frame& frame, double s6, double s8, double a1,
                            double a2, double rcut_bohr) {
    const int n = static_cast<int>(frame.nlocal());
    gpw::D3Result empty;
    if (n <= 0)
        return empty;

    const auto type = frame.span(kAtom_type);
    const auto x = frame.span(kAtom_x);
    const auto y = frame.span(kAtom_y);
    const auto z = frame.span(kAtom_z);

    std::vector<int> Z(static_cast<std::size_t>(n));
    std::vector<std::array<double, 3>> pos(static_cast<std::size_t>(n));
    const double a2b = units::kAngstromInBohr;
    for (int i = 0; i < n; ++i) {
        const int zi = static_cast<int>(type[i]);
        if (!gpw::D3Params::supports_element(zi))
            throw std::runtime_error(
                "D3BJ: element Z=" + std::to_string(zi)
                + " is not in the in-tree D3 table (no fallback)");
        Z[static_cast<std::size_t>(i)] = zi;
        pos[static_cast<std::size_t>(i)] = {x[i] * a2b, y[i] * a2b, z[i] * a2b};
    }

    const double cell[3] = {0.0, 0.0, 0.0};
    const gpw::D3Params params{};
    return gpw::dftd3_pbe_bj(Z, pos, cell, /*periodic=*/false, params,
                             rcut_bohr, s6, s8, a1, a2);
}

} // namespace

double D3BJ::energy(const Frame& frame) const {
    return evaluate_d3bj(frame, s6, s8, a1, a2, rcut_bohr).energy;
}

std::vector<std::array<double, 3>> D3BJ::forces(const Frame& frame) const {
    return evaluate_d3bj(frame, s6, s8, a1, a2, rcut_bohr).forces;
}

} // namespace scf
} // namespace cpu
} // namespace ATV
