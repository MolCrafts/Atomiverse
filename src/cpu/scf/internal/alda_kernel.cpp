/// @file cpu/scf/internal/alda_kernel.cpp
/// @brief Full-grid MO-product ALDA kernel (see internal/alda_kernel.h).

#include "atomiverse/cpu/scf/internal/alda_kernel.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>

#include "atomiverse/cpu/scf/basis_eval.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

RealMatrix alda_kernel_matrix(const SlaterDeterminant& det,
                              const internal::IntegralView& pack,
                              const MolecularGrid& grid,
                              ExcitationChannel /*channel*/) {
    // Pure Dirac exchange: f_αβ = 0 ⇒ singlet (f_αα+f_αβ) = triplet (f_αα−f_αβ).
    if (!det.restricted())
        throw std::invalid_argument(
            "alda_kernel_matrix: closed-shell restricted determinant required "
            "(no fallback)");
    if (pack.shells.empty())
        throw std::invalid_argument(
            "alda_kernel_matrix: internal::IntegralWorkspace has no shells (no fallback)");

    using Spin = SlaterDeterminant::Spin;
    const RealMatrix& C = det.coefficients(Spin::alpha);
    const RealVector& f = det.occupations(Spin::alpha);
    const int n_mo = static_cast<int>(C.shape()[1]);
    const int n_ao_work = static_cast<int>(C.shape()[0]);
    std::vector<int> occ, virt;
    for (int p = 0; p < n_mo; ++p)
        (f(p) > 0.0 ? occ : virt).push_back(p);
    const int no = static_cast<int>(occ.size());
    const int nv = static_cast<int>(virt.size());
    const int nov = no * nv;
    if (nov == 0)
        return xt::zeros<double>({std::size_t(0), std::size_t(0)});

    const int n_cart = static_cast<int>(pack.shells.size());
    const bool sph = pack.spherical();
    if (!sph && n_cart != n_ao_work)
        throw std::invalid_argument(
            "alda_kernel_matrix: AO dimension mismatch (C rows="
            + std::to_string(n_ao_work) + ", shells=" + std::to_string(n_cart)
            + ")");

    RealMatrix P = det.density_total();
    RealMatrix P_cart;
    RealMatrix C_cart;
    if (sph) {
        const auto& c2s = pack.c2s;
        P_cart = xt::linalg::dot(
            c2s, RealMatrix(xt::linalg::dot(P, xt::transpose(c2s))));
        C_cart = xt::linalg::dot(c2s, C);
    } else {
        P_cart = P;
        C_cart = C;
    }

    const int G = static_cast<int>(grid.points.size());
    RealMatrix K = xt::zeros<double>(
        {static_cast<std::size_t>(nov), static_cast<std::size_t>(nov)});

    std::vector<double> xi(static_cast<std::size_t>(nov));
    std::vector<double> phi_mo(static_cast<std::size_t>(n_mo));
    std::vector<double> chi(static_cast<std::size_t>(n_cart));

    for (int g = 0; g < G; ++g) {
        const auto& pt = grid.points[static_cast<std::size_t>(g)];
        const double w = pt.weight;
        if (!(std::fabs(w) > 0.0))
            continue;

        for (int mu = 0; mu < n_cart; ++mu)
            chi[static_cast<std::size_t>(mu)] = eval_basis(
                pack.shells[static_cast<std::size_t>(mu)], pt.x, pt.y, pt.z);

        double rho = 0.0;
        for (int mu = 0; mu < n_cart; ++mu) {
            const double cmu = chi[static_cast<std::size_t>(mu)];
            if (!(std::fabs(cmu) > 1e-20))
                continue;
            double row = 0.0;
            for (int nu = 0; nu < n_cart; ++nu)
                row += P_cart(mu, nu) * chi[static_cast<std::size_t>(nu)];
            rho += cmu * row;
        }
        if (rho < 0.0)
            rho = 0.0;
        const double fx = alda_fx_dirac(rho);
        if (!(std::fabs(fx) > 0.0))
            continue;
        const double wfx = w * fx;

        for (int p = 0; p < n_mo; ++p) {
            double v = 0.0;
            for (int mu = 0; mu < n_cart; ++mu)
                v += chi[static_cast<std::size_t>(mu)]
                     * C_cart(mu, static_cast<std::size_t>(p));
            phi_mo[static_cast<std::size_t>(p)] = v;
        }
        for (int a = 0; a < nv; ++a) {
            const int ia = virt[static_cast<std::size_t>(a)];
            const double pa = phi_mo[static_cast<std::size_t>(ia)];
            for (int i = 0; i < no; ++i) {
                const int ii = occ[static_cast<std::size_t>(i)];
                xi[static_cast<std::size_t>(a * no + i)] =
                    phi_mo[static_cast<std::size_t>(ii)] * pa;
            }
        }
        for (int p = 0; p < nov; ++p) {
            const double xp = xi[static_cast<std::size_t>(p)];
            if (!(std::fabs(xp) > 0.0))
                continue;
            const double c = wfx * xp;
            for (int q = p; q < nov; ++q) {
                const double v = c * xi[static_cast<std::size_t>(q)];
                K(p, q) += v;
                if (q != p)
                    K(q, p) += v;
            }
        }
    }
    return K;
}

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
