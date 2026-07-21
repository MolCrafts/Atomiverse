/// @file cpu/observer/mulliken.cpp
/// @brief Mulliken population analysis — pure C++/xtensor implementation.

#include "atomiverse/cpu/observer/mulliken.h"

#include "atomiverse/cpu/scf/internal/integrals.h"

#include <stdexcept>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xtensor.hpp>

namespace ATV {
namespace cpu {

namespace {

/// @brief Derive AO index -> atom index by matching shell center to atom pos.
///
/// Both native::Shell.center and native::Atom.pos are populated from the
/// same Frame positions converted to Bohr, so bit-exact equality holds
/// within a single IntegralPack. The observer derives the mapping itself
/// rather than caching it on IntegralPack: hot-path data structures carry
/// only fundamental physical quantities.
std::vector<int> derive_basis_to_atom(const scf::IntegralPack& pack) {
    const int n_basis = pack.n_basis;
    const int natm = static_cast<int>(pack.atoms.size());

    std::vector<int> b2a(static_cast<std::size_t>(n_basis), -1);
    for (int mu = 0; mu < n_basis; ++mu) {
        const double* c = pack.shells[static_cast<std::size_t>(mu)].center;
        for (int A = 0; A < natm; ++A) {
            const double* p = pack.atoms[static_cast<std::size_t>(A)].pos;
            if (c[0] == p[0] && c[1] == p[1] && c[2] == p[2]) {
                b2a[static_cast<std::size_t>(mu)] = A;
                break;
            }
        }
        if (b2a[static_cast<std::size_t>(mu)] < 0) {
            throw std::runtime_error("cpu::mulliken: shell center at AO "
                                     + std::to_string(mu)
                                     + " matches no atom in IntegralPack");
        }
    }
    return b2a;
}

/// @brief Scatter the diagonal of (D S) per atom using the AO->atom map.
std::vector<double>
accumulate_diag_ds_per_atom(const xt::xtensor<double, 2>& D,
                            const xt::xtensor<double, 2>& S,
                            const std::vector<int>& basis_to_atom, int natm) {
    const xt::xtensor<double, 2> DS = xt::linalg::dot(D, S);

    std::vector<double> out(static_cast<std::size_t>(natm), 0.0);
    const int n_basis = static_cast<int>(basis_to_atom.size());
    for (int mu = 0; mu < n_basis; ++mu) {
        const int A = basis_to_atom[static_cast<std::size_t>(mu)];
        out[static_cast<std::size_t>(A)] += DS(mu, mu);
    }
    return out;
}

} // namespace

std::vector<double> mulliken_charge(const scf::IntegralPack& pack,
                                    const xt::xtensor<double, 2>& P) {
    const int natm = static_cast<int>(pack.atoms.size());
    const auto b2a = derive_basis_to_atom(pack);

    auto population = accumulate_diag_ds_per_atom(P, pack.S, b2a, natm);

    std::vector<double> q(static_cast<std::size_t>(natm));
    for (int A = 0; A < natm; ++A) {
        const int Z = pack.atoms[static_cast<std::size_t>(A)].Z;
        q[static_cast<std::size_t>(A)] =
            static_cast<double>(Z) - population[static_cast<std::size_t>(A)];
    }
    return q;
}

std::vector<double> mulliken_spin(const scf::IntegralPack& pack,
                                  const xt::xtensor<double, 2>& P_alpha,
                                  const xt::xtensor<double, 2>& P_beta) {
    const int natm = static_cast<int>(pack.atoms.size());
    const auto b2a = derive_basis_to_atom(pack);

    const xt::xtensor<double, 2> P_diff = P_alpha - P_beta;
    return accumulate_diag_ds_per_atom(P_diff, pack.S, b2a, natm);
}

} // namespace cpu
} // namespace ATV
