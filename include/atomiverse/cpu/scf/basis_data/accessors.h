/// @file cpu/scf/basis_data/accessors.h
/// @brief Lookup helpers into the static basis tables (no duplicated numbers).
///
/// Production data lives in the generated headers (`sto3g.h`, …). Call sites
/// that need a fixed contraction (CUDA device path, unit fixtures) **must**
/// read those tables through this header — never re-type exponents/coeffs.
#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/cpu/scf/basis_data/sto3g.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace basis_data {

/// @brief True when every STO-3G shell for `Z` is s-type (l = 0), ≥1 shell.
///
/// Allows multi-shell pure-s atoms (Wave 3). H/He today have one 1s shell;
/// elements with any p/d shell return false.
/// @param Z Atomic number (1 = H).
[[nodiscard]] inline bool is_s_only_sto3g(int Z) {
    const auto& table = sto3g();
    if (Z < 1 || static_cast<std::size_t>(Z) >= table.size())
        return false;
    const auto& shells = table[static_cast<std::size_t>(Z)].shells;
    if (shells.empty())
        return false;
    for (const auto& sh : shells) {
        if (sh.l != 0 || sh.alpha.empty()
            || sh.alpha.size() != sh.coeff.size())
            return false;
    }
    return true;
}

/// @brief Alias of `is_s_only_sto3g` (historical name from single-shell wave).
[[nodiscard]] inline bool is_pure_s_sto3g(int Z) { return is_s_only_sto3g(Z); }

/// @brief All s-only STO-3G shells for `Z` (table order).
/// @throws std::invalid_argument when `Z` is not s-only STO-3G.
[[nodiscard]] inline const std::vector<ContractedShell>&
s_only_sto3g_shells(int Z) {
    if (!is_s_only_sto3g(Z)) {
        throw std::invalid_argument(
            "basis_data::s_only_sto3g_shells: Z=" + std::to_string(Z)
            + " is not s-only STO-3G (has p/d or is unsupported)");
    }
    return sto3g()[static_cast<std::size_t>(Z)].shells;
}

/// @brief First s shell of an s-only STO-3G element (H/He 1s).
/// @throws std::invalid_argument when `Z` is not s-only STO-3G.
[[nodiscard]] inline const ContractedShell& pure_s_sto3g_shell(int Z) {
    return s_only_sto3g_shells(Z)[0];
}

/// @brief Common primitive count for all s-only shells of `Z` (STO-3G = 3).
/// @throws std::invalid_argument if shell prim counts differ (device path needs
///         uniform n_prim today).
[[nodiscard]] inline int s_only_sto3g_n_prim(int Z) {
    const auto& shells = s_only_sto3g_shells(Z);
    const int np = static_cast<int>(shells[0].alpha.size());
    for (const auto& sh : shells) {
        if (static_cast<int>(sh.alpha.size()) != np) {
            throw std::invalid_argument(
                "basis_data::s_only_sto3g_n_prim: mixed n_prim for Z="
                + std::to_string(Z));
        }
    }
    return np;
}

/// @brief True when every STO-3G shell for `Z` has l ≤ 1 (s or p), ≥1 shell.
///
/// Wave 4 device path: H…O in the generated table (no d). Unsupported Z or
/// empty table → false.
/// @param Z Atomic number (1 = H).
[[nodiscard]] inline bool is_sp_sto3g(int Z) {
    const auto& table = sto3g();
    if (Z < 1 || static_cast<std::size_t>(Z) >= table.size())
        return false;
    const auto& shells = table[static_cast<std::size_t>(Z)].shells;
    if (shells.empty())
        return false;
    for (const auto& sh : shells) {
        if (sh.l < 0 || sh.l > 1 || sh.alpha.empty()
            || sh.alpha.size() != sh.coeff.size())
            return false;
    }
    return true;
}

/// @brief STO-3G shells for an s/p element (table order).
/// @throws std::invalid_argument when `Z` is not s/p STO-3G.
[[nodiscard]] inline const std::vector<ContractedShell>&
sp_sto3g_shells(int Z) {
    if (!is_sp_sto3g(Z)) {
        throw std::invalid_argument(
            "basis_data::sp_sto3g_shells: Z=" + std::to_string(Z)
            + " is not s/p STO-3G (has d or is unsupported)");
    }
    return sto3g()[static_cast<std::size_t>(Z)].shells;
}

/// @brief Common primitive count for all s/p STO-3G shells of `Z`.
/// @throws std::invalid_argument if shell prim counts differ.
[[nodiscard]] inline int sp_sto3g_n_prim(int Z) {
    const auto& shells = sp_sto3g_shells(Z);
    const int np = static_cast<int>(shells[0].alpha.size());
    for (const auto& sh : shells) {
        if (static_cast<int>(sh.alpha.size()) != np) {
            throw std::invalid_argument(
                "basis_data::sp_sto3g_n_prim: mixed n_prim for Z="
                + std::to_string(Z));
        }
    }
    return np;
}

/// @brief Cartesian AO count for angular momentum l (s→1, p→3).
[[nodiscard]] inline constexpr int n_cart_l(int l) {
    return (l + 1) * (l + 2) / 2;
}

} // namespace basis_data
} // namespace scf
} // namespace cpu
} // namespace ATV
