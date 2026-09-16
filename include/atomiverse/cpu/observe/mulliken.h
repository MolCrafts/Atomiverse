/// @file cpu/observe/mulliken.h
/// @brief Mulliken population analysis — pure observer (compute only, no I/O).
///
/// Computes Mulliken gross populations by projecting a density matrix onto
/// the atom-centred basis. Callers inside the library pass a non-owning
/// `IntegralView` of method-owned fields; user code uses RHF/observers,
/// never constructs freestanding integral bags.
///
/// Pure C++/xtensor implementation; no FFI.
#pragma once

#include <vector>

#include <xtensor/containers/xtensor.hpp>

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {
struct IntegralView;
} // namespace internal
} // namespace scf

/// @brief Mulliken net charges q_A = Z_A - Tr_A(P S).
/// @param ints Non-owning view of method integral fields (S, atoms, shells).
/// @param P Density matrix [n_basis, n_basis].
/// @return Per-atom Mulliken net charges [e].
std::vector<double>
mulliken_charge(const cpu::scf::internal::IntegralView& ints,
                const xt::xtensor<double, 2>& P);

/// @brief Mulliken spin densities sigma_A = Tr_A((P^α − P^β) S).
/// @param ints Non-owning view of method integral fields.
/// @param P_alpha Alpha density [n_basis, n_basis].
/// @param P_beta Beta density [n_basis, n_basis].
/// @return Per-atom Mulliken spin densities [e].
std::vector<double>
mulliken_spin(const cpu::scf::internal::IntegralView& ints,
              const xt::xtensor<double, 2>& P_alpha,
              const xt::xtensor<double, 2>& P_beta);

} // namespace cpu
} // namespace ATV
