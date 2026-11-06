/// @file cpu/scf/internal/xc_response_kernel.cpp
/// @brief Closed-shell adiabatic f^xc kernel (see internal/xc_response_kernel.h).

#include "atomiverse/cpu/scf/internal/xc_response_kernel.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

namespace {
/// Central-difference step on the unit transition density [a.u.]; 1e-4
/// reproduces PySCF kernel elements to ~1e-9 Ha on H2/STO-3G.
constexpr double kStep = 1e-4;
} // namespace

XcResponseKernel::XcResponseKernel(const SlaterDeterminant& det,
                                   const IntegralView& pack, xc::XCType xc,
                                   const MolecularGrid& grid, Channel ch)
    : functional_(xc), grid_(grid), ch_(ch) {
    if (!det.restricted())
        throw std::invalid_argument(
            "XcResponseKernel: closed-shell restricted determinant required "
            "(no fallback)");
    if (pack.shells.empty())
        throw std::invalid_argument(
            "XcResponseKernel: integral view has no shells (no fallback)");
    using Spin = SlaterDeterminant::Spin;
    const RealMatrix& C = det.coefficients(Spin::alpha);
    const RealVector& f = det.occupations(Spin::alpha);
    const int n_mo = static_cast<int>(C.shape()[1]);
    std::vector<int> occ, virt;
    for (int p = 0; p < n_mo; ++p)
        (f(p) > 0.0 ? occ : virt).push_back(p);
    no_ = static_cast<int>(occ.size());
    nv_ = static_cast<int>(virt.size());

    const RealMatrix C_eng =
        pack.spherical() ? RealMatrix(xt::linalg::dot(pack.c2s, C)) : C;
    const std::size_t n = C_eng.shape()[0];
    Co_ = xt::zeros<double>({n, static_cast<std::size_t>(no_)});
    Cv_ = xt::zeros<double>({n, static_cast<std::size_t>(nv_)});
    for (std::size_t mu = 0; mu < n; ++mu) {
        for (int i = 0; i < no_; ++i)
            Co_(mu, static_cast<std::size_t>(i)) =
                C_eng(mu, static_cast<std::size_t>(occ[i]));
        for (int a = 0; a < nv_; ++a)
            Cv_(mu, static_cast<std::size_t>(a)) =
                C_eng(mu, static_cast<std::size_t>(virt[a]));
    }
    P0_ = xt::linalg::dot(Co_, xt::transpose(Co_));
    basis_.build(grid_, pack.shells, static_cast<int>(n),
                 (static_cast<unsigned>(functional_.ingredients()) & 2u) != 0);
}

RealVector XcResponseKernel::apply(const RealVector& v) const {
    const std::size_t nov = static_cast<std::size_t>(n_ov());
    if (v.size() != nov)
        throw std::invalid_argument("XcResponseKernel::apply: v.size() = "
                                    + std::to_string(v.size())
                                    + " != n_ov = " + std::to_string(nov));
    RealVector out = xt::zeros<double>({nov});
    double norm = 0.0;
    for (double x : v)
        norm += x * x;
    norm = std::sqrt(norm);
    if (!(norm > 0.0))
        return out;

    // Transition density D = Co X Cvᵀ (X(i,a) = v(a·no + i)), hermitized.
    RealMatrix X = xt::zeros<double>(
        {static_cast<std::size_t>(no_), static_cast<std::size_t>(nv_)});
    for (int a = 0; a < nv_; ++a)
        for (int i = 0; i < no_; ++i)
            X(i, a) = v(static_cast<std::size_t>(a * no_ + i)) / norm;
    const RealMatrix D = xt::linalg::dot(
        Co_, RealMatrix(xt::linalg::dot(X, xt::transpose(Cv_))));
    const RealMatrix Ds = 0.5 * (D + xt::transpose(D));

    // δP_β = +δP_α (singlet) or −δP_α (triplet).
    const double sb = (ch_ == Channel::singlet) ? 1.0 : -1.0;
    const RealMatrix Pa_p = P0_ + kStep * Ds, Pa_m = P0_ - kStep * Ds;
    const RealMatrix Pb_p = P0_ + sb * kStep * Ds, Pb_m = P0_ - sb * kStep * Ds;
    const XCGridResult xp =
        evaluate_xc_batched(basis_, grid_, Pa_p, Pb_p, functional_);
    const XCGridResult xm =
        evaluate_xc_batched(basis_, grid_, Pa_m, Pb_m, functional_);
    const RealMatrix dV = (xp.Vxc_alpha - xm.Vxc_alpha) * (norm / (2 * kStep));

    // (K v)_jb = Co_jᵀ dV Cv_b.
    const RealMatrix Kmo = xt::linalg::dot(
        xt::transpose(Co_), RealMatrix(xt::linalg::dot(dV, Cv_)));
    for (int b = 0; b < nv_; ++b)
        for (int j = 0; j < no_; ++j)
            out(static_cast<std::size_t>(b * no_ + j)) = Kmo(j, b);
    return out;
}

RealMatrix XcResponseKernel::dense() const {
    const std::size_t nov = static_cast<std::size_t>(n_ov());
    RealMatrix K = xt::zeros<double>({nov, nov});
    for (std::size_t q = 0; q < nov; ++q) {
        RealVector e = xt::zeros<double>({nov});
        e(q) = 1.0;
        const RealVector col = apply(e);
        for (std::size_t p = 0; p < nov; ++p)
            K(p, q) = col(p);
    }
    return 0.5 * (K + xt::transpose(K));
}

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
