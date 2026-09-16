/// @file cpu/vibration/phonon_diagonalizer.cpp
/// @brief Full \(3N\) mass-weighted dynamical matrix → wavenumbers [cm⁻¹].

#include "atomiverse/cpu/vibration/phonon_diagonalizer.h"

#include <cmath>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/cpu/vibration/nist_constants.h"

namespace ATV {
namespace cpu {
namespace vib {

void PhononDiagonalizer::launch(const cpu::LaunchContext& ctx) {
    const int n = ws_->n_atoms;
    const int dim = 3 * n;
    if (ctx.step < 2 * dim)
        return;
    if (!ws_->mass_weighted)
        return;

    if (n < 1) {
        ws_->frequencies = xt::zeros<double>({0});
        ws_->normal_modes = xt::zeros<double>({0, 0});
        ws_->n_imaginary = 0;
        ws_->is_linear = false;
        return;
    }

    xt::xtensor<double, 2> h = xt::zeros<double>({dim, dim});
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j)
            h(i, j) = ws_->hessian_mw(i, j);

    auto [force_const_au, mode] = xt::linalg::eigh(h);
    const int n_modes = dim;
    const double au2cm = nist::au2wavenumber();
    ws_->frequencies = xt::zeros<double>({n_modes});
    int n_imag = 0;
    for (int k = 0; k < n_modes; ++k) {
        const double lambda = force_const_au(k);
        if (lambda < 0.0) {
            ws_->frequencies(k) = -std::sqrt(-lambda) * au2cm;
            ++n_imag;
        } else {
            ws_->frequencies(k) = std::sqrt(lambda) * au2cm;
        }
    }
    ws_->normal_modes = mode;
    ws_->n_imaginary = n_imag;
    ws_->is_linear = false; // crystal — rotor classification unused
}

} // namespace vib
} // namespace cpu
} // namespace ATV
