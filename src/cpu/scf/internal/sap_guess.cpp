/// @file cpu/scf/internal/sap_guess.cpp
/// @brief SapGuess — V_SAP quadrature on the molecular grid + F = T + V_SAP.

#include "atomiverse/cpu/scf/internal/sap_guess.h"

#include <stdexcept>

#include <xtensor-blas/xlinalg.hpp>

#include "atomiverse/cpu/scf/basis_eval.h"
#include "atomiverse/cpu/scf/internal/sap_data.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

SapGuess::SapGuess(const std::vector<native::Shell>& shells,
                   const RealMatrix& T, const std::vector<native::Atom>& atoms,
                   const MolecularGrid& grid,
                   const xt::xtensor<double, 2>& c2s) {
    const int n_cart = static_cast<int>(shells.size());
    const int n_work = static_cast<int>(T.shape()[0]);
    const bool spherical = (c2s.size() > 0);
    if (spherical
        && (static_cast<int>(c2s.shape()[0]) != n_cart
            || static_cast<int>(c2s.shape()[1]) != n_work))
        throw std::invalid_argument(
            "SapGuess: c2s dimensions do not bridge the Cartesian shell "
            "list to the working dimension of T");
    if (!spherical && n_work != n_cart)
        throw std::invalid_argument(
            "SapGuess: T dimension does not match the Cartesian AO count "
            "(and no c2s transform was supplied)");

    // Table rows for every atom up front — throws on unsupported Z.
    std::vector<SapAtomicPotential> pots;
    pots.reserve(atoms.size());
    for (const auto& at : atoms)
        pots.emplace_back(at.Z);

    // Quadrature: V_μν = Σ_g w_g φ_μ(r_g) V_SAP(r_g) φ_ν(r_g), Cartesian AOs.
    RealMatrix V_cart = xt::zeros<double>({static_cast<std::size_t>(n_cart),
                                           static_cast<std::size_t>(n_cart)});
    std::vector<double> phi(static_cast<std::size_t>(n_cart));
    std::vector<int> nz;
    nz.reserve(static_cast<std::size_t>(n_cart));
    for (const auto& gp : grid.points) {
        double v = 0.0;
        for (std::size_t a = 0; a < atoms.size(); ++a) {
            const double dx = gp.x - atoms[a].pos[0];
            const double dy = gp.y - atoms[a].pos[1];
            const double dz = gp.z - atoms[a].pos[2];
            v += pots[a].value(
                std::sqrt(dx * dx + dy * dy + dz * dz));
        }
        const double wv = gp.weight * v;
        nz.clear();
        for (int i = 0; i < n_cart; ++i) {
            const double p =
                eval_basis(shells[static_cast<std::size_t>(i)], gp.x, gp.y,
                           gp.z);
            phi[static_cast<std::size_t>(i)] = p;
            if (p != 0.0)
                nz.push_back(i);
        }
        for (std::size_t ii = 0; ii < nz.size(); ++ii) {
            const int i = nz[ii];
            const double wvp = wv * phi[static_cast<std::size_t>(i)];
            for (std::size_t jj = ii; jj < nz.size(); ++jj) {
                const int j = nz[jj];
                V_cart(i, j) += wvp * phi[static_cast<std::size_t>(j)];
            }
        }
    }
    for (int i = 0; i < n_cart; ++i)
        for (int j = 0; j < i; ++j)
            V_cart(i, j) = V_cart(j, i);

    V = spherical ? RealMatrix(xt::linalg::dot(
            xt::transpose(c2s), xt::linalg::dot(V_cart, c2s)))
                  : std::move(V_cart);
    F = T + V;
}

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
