/// @file cpu/vibration/mass_weighting.cpp
/// @brief cpu::vib::MassWeighting — implementation of the mass-weighting
///        transform on the assembled numerical Hessian.

#include "atomiverse/cpu/vibration/mass_weighting.h"

#include <cmath>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/vibration/atomic_masses.h"
#include "atomiverse/cpu/vibration/nist_constants.h"

namespace ATV {
namespace cpu {
namespace vib {

void MassWeighting::launch(const cpu::LaunchContext& ctx) {
    const int n = ws_->n_atoms;
    const int dim = 3 * n;
    if (ctx.step < 2 * dim)
        return; // not the finalize pass yet.

    // Fetch IUPAC-2013 atomic masses from each atom's atomic number.
    // mass_amu() throws on an unsupported element (no fallback).
    // Hoisted once before the loop -- the 13.7x rule.
    const int* __restrict sp_type = ctx.frame.span(kAtom_type);
    for (int i = 0; i < n; ++i)
        ws_->mass_amu[i] = vib::mass_amu(sp_type[i]);

    // Per-atom inverse-sqrt mass (PySCF's `mass**-.5`).
    std::vector<double> inv_sqrt_m(n);
    for (int i = 0; i < n; ++i)
        inv_sqrt_m[i] = 1.0 / std::sqrt(ws_->mass_amu[i]);

    // eV/Angstrom^2 -> Hartree/Bohr^2, then divide by sqrt(m_i m_j).
    // Index c maps to atom c/3, axis c%3, so the per-coordinate mass is
    // inv_sqrt_m[c/3]. The result lands in `hessian_mw`; the raw link-1
    // `hessian` (eV/Angstrom^2) is left untouched so HessianResult keeps
    // reporting it in the link-1 units.
    constexpr double unit_conv = nist::EV_PER_ANG2_TO_HARTREE_PER_BOHR2;
    for (int ic = 0; ic < dim; ++ic) {
        const double wi = inv_sqrt_m[ic / 3];
        for (int jc = 0; jc < dim; ++jc) {
            const double wj = inv_sqrt_m[jc / 3];
            ws_->hessian_mw(ic, jc) =
                ws_->hessian(ic, jc) * unit_conv * wi * wj;
        }
    }

    ws_->mass_weighted = true;
}

} // namespace vib
} // namespace cpu
} // namespace ATV
