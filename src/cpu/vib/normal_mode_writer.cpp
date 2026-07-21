/// @file src/cpu/vib/normal_mode_writer.cpp
/// @brief cpu::vib::NormalModeWriter — per-mode XYZ-animation writer.

#include "atomiverse/cpu/vib/normal_mode_writer.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/vib/atomic_masses.h"
#include "atomiverse/cpu/vib/nist_constants.h"

namespace ATV {
namespace cpu {
namespace vib {

void NormalModeWriter::launch(const cpu::LaunchContext& ctx) {
    if (ws_ == nullptr)
        return; // test-only role-check ctor.
    const int n = ws_->n_atoms;
    const int dim = 3 * n;
    if (ctx.step < 2 * dim)
        return; // not the finalize pass yet.

    // Atom symbols from each atom's atomic number (no-fallback table).
    // Hoisted once before the loop -- the 13.7x rule.
    const int* __restrict sp_type = ctx.frame.span(kAtom_type);
    std::vector<std::string> symbol(n);
    for (int i = 0; i < n; ++i)
        symbol[i] = element_symbol(sp_type[i]);

    // Per-atom inverse-sqrt mass: the workspace eigenvectors are
    // mass-weighted, so the Cartesian displacement is dx_i = q_i /
    // sqrt(m_i) (the standard PySCF `norm_mode` form).
    std::vector<double> inv_sqrt_m(n);
    for (int i = 0; i < n; ++i)
        inv_sqrt_m[i] = 1.0 / std::sqrt(ws_->mass_amu[i]);

    const int n_modes = static_cast<int>(ws_->frequencies.size());
    for (int k = 0; k < n_modes; ++k) {
        // Imaginary modes (negative wavenumber) are not animated.
        if (ws_->frequencies(k) <= 0.0)
            continue;

        // Cartesian displacement vector for mode k, normalized so the
        // largest single-atom displacement has unit magnitude — the
        // sweep amplitude then sets the absolute peak excursion.
        std::vector<double> disp(dim);
        double max_norm = 0.0;
        for (int i = 0; i < n; ++i) {
            double comp[3];
            for (int d = 0; d < 3; ++d) {
                comp[d] = ws_->normal_modes(3 * i + d, k) * inv_sqrt_m[i];
                disp[3 * i + d] = comp[d];
            }
            const double nrm = std::sqrt(comp[0] * comp[0] + comp[1] * comp[1]
                                         + comp[2] * comp[2]);
            if (nrm > max_norm)
                max_norm = nrm;
        }
        const double scale = (max_norm > 0.0) ? amplitude_ / max_norm : 0.0;

        std::ofstream os(mode_path(k));
        if (!os)
            continue;
        os << std::setprecision(10);

        // Sinusoidal sweep: one full period of n_frames_ frames.
        for (int f = 0; f < n_frames_; ++f) {
            const double phase = 2.0 * nist::PI * static_cast<double>(f)
                                 / static_cast<double>(n_frames_);
            const double s = scale * std::sin(phase);

            os << n << '\n';
            os << "mode " << k << " freq " << ws_->frequencies(k)
               << " cm-1 frame " << f << '\n';
            for (int i = 0; i < n; ++i) {
                os << symbol[i] << '\t'
                   << ws_->r0[3 * i + 0] + s * disp[3 * i + 0] << '\t'
                   << ws_->r0[3 * i + 1] + s * disp[3 * i + 1] << '\t'
                   << ws_->r0[3 * i + 2] + s * disp[3 * i + 2] << '\n';
            }
        }
    }
}

} // namespace vib
} // namespace cpu
} // namespace ATV
