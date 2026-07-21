/// @file cpu/observer/mulliken.h
/// @brief Mulliken population analysis — pure observer (compute only, no I/O).
///
/// Computes Mulliken gross populations by projecting a density matrix onto
/// the atom-centred basis:
///
///   N_A     = sum_{mu in A} (P S)_{mu, mu}          (gross population on atom A)
///   q_A     = Z_A - N_A                              (Mulliken net charge)
///   sigma_A = sum_{mu in A} ((P^alpha - P^beta) S)_{mu, mu}   (Mulliken spin density)
///
/// Functions are stateless and side-effect free: given an IntegralPack and
/// density matrices, they return per-atom vectors. The AO -> atom mapping
/// is derived from shell centers inside the observer; IntegralPack itself
/// only carries fundamental physical quantities (atoms, shells, S, T, V, ...)
/// and never precomputed indices.
///
/// Pure C++/xtensor implementation; no FFI.
///
/// @see cpu::scf::IntegralPack for the S matrix and shell/atom tables
#pragma once

#include <vector>

#include <xtensor/containers/xtensor.hpp>

namespace ATV {
namespace cpu {
namespace scf {
struct IntegralPack; // defined in cpu/scf/internal/integrals.h
} // namespace scf

/// @brief Mulliken net charges q_A = Z_A - Tr_A(P S).
///
/// The returned vector is indexed by atom order (matches pack.atoms).
///
/// @param pack Integral pack carrying atoms, shells, and the overlap S.
/// @param P    Density matrix [n_basis, n_basis]. For RHF use total density
///             P_total = 2 * P_occ; for UHF / UKS use P_alpha + P_beta.
/// @return Per-atom Mulliken net charges [e], size == pack.atoms.size().
std::vector<double> mulliken_charge(const cpu::scf::IntegralPack& pack,
                                    const xt::xtensor<double, 2>& P);

/// @brief Mulliken spin densities sigma_A = Tr_A((P^alpha - P^beta) S).
///
/// @param pack    Integral pack carrying atoms, shells, and the overlap S.
/// @param P_alpha Alpha spin density matrix [n_basis, n_basis].
/// @param P_beta  Beta  spin density matrix [n_basis, n_basis].
/// @return Per-atom Mulliken spin densities [e], size == pack.atoms.size().
std::vector<double> mulliken_spin(const cpu::scf::IntegralPack& pack,
                                  const xt::xtensor<double, 2>& P_alpha,
                                  const xt::xtensor<double, 2>& P_beta);

} // namespace cpu
} // namespace ATV
