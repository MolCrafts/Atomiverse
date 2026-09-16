/// @file cpu/scf/internal/hueckel_guess.cpp
/// @brief HueckelGuess — extended-Hückel guess Fock assembly (VSIP diagonal
///        + Wolfsberg-Helmholz off-diagonal over the overlap matrix).

#include "atomiverse/cpu/scf/internal/hueckel_guess.h"

#include <cmath>

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

HueckelGuess::HueckelGuess(const std::vector<native::Shell>& shells,
                           const RealMatrix& S,
                           const std::vector<native::Atom>& atoms) {
    // Per-working-AO VSIP diagonal. Shell blocks are consecutive Cartesian
    // entries with cart = 0..n_cart-1 (the emission order of the basis
    // resolver); the working AO count per block is n_cart (Cartesian
    // convention) or 2l+1 (spherical), selected by the dimension of S.
    const int n_work = static_cast<int>(S.shape()[0]);
    int n_cart_total = 0;
    int n_sph_total = 0;
    for (const auto& sh : shells) {
        if (sh.cart == 0) {
            n_cart_total += (sh.l + 1) * (sh.l + 2) / 2;
            n_sph_total += 2 * sh.l + 1;
        }
    }
    if (static_cast<int>(shells.size()) != n_cart_total)
        throw std::invalid_argument(
            "HueckelGuess: shell list is not grouped into contiguous "
            "Cartesian shell blocks");
    const bool spherical = (n_work == n_sph_total && n_work != n_cart_total);
    if (n_work != n_cart_total && n_work != n_sph_total)
        throw std::invalid_argument(
            "HueckelGuess: S dimension matches neither the Cartesian nor "
            "the spherical AO count of the shell list");

    std::vector<double> diag;
    diag.reserve(static_cast<std::size_t>(n_work));
    for (const auto& sh : shells) {
        if (sh.cart != 0)
            continue; // one VSIP per shell block, replicated per component
        // Element of this shell: the atom whose position coincides with the
        // shell center (centers are copied from atom positions in Bohr).
        int Z = 0;
        bool found = false;
        for (const auto& at : atoms) {
            const double dx = sh.center[0] - at.pos[0];
            const double dy = sh.center[1] - at.pos[1];
            const double dz = sh.center[2] - at.pos[2];
            if (dx * dx + dy * dy + dz * dz < 1e-16) {
                Z = at.Z;
                found = true;
                break;
            }
        }
        if (!found)
            throw std::invalid_argument(
                "HueckelGuess: shell center coincides with no atom");
        const double h = vsip(Z, sh.l); // throws on unsupported (Z, l)
        const int n_comp =
            spherical ? (2 * sh.l + 1) : (sh.l + 1) * (sh.l + 2) / 2;
        for (int c = 0; c < n_comp; ++c)
            diag.push_back(h);
    }

    // F_μμ = VSIP;  F_μν = ½ K (H_μμ + H_νν) S_μν  for μ ≠ ν.
    F = xt::zeros<double>(
        {static_cast<std::size_t>(n_work), static_cast<std::size_t>(n_work)});
    for (int i = 0; i < n_work; ++i) {
        F(i, i) = diag[static_cast<std::size_t>(i)];
        for (int j = i + 1; j < n_work; ++j) {
            const double f = 0.5 * kWolfsbergHelmholz
                             * (diag[static_cast<std::size_t>(i)]
                                + diag[static_cast<std::size_t>(j)])
                             * S(i, j);
            F(i, j) = f;
            F(j, i) = f;
        }
    }
}

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
