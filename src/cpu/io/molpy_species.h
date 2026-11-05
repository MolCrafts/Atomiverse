/// @file src/cpu/io/molpy_species.h
/// @brief The one species conversion at the molpy CXX boundary: the engine's
/// `kAtom_type` (atomic number Z, `int`) to molpy' `atomic_number` column
/// (`u64`, 0..=118, 0 = wildcard `*`).
///
/// molpy validates the upper bound and names the offending row; a negative
/// `int` has no `u64` spelling and would wrap to a huge number, so it is
/// refused here before it crosses. Internal to `src/cpu/` (same status as
/// `molpy_meta.h`) — not a public header.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ATV::cpu::internal {

/// @brief Convert a whole species column into a reusable `u64` scratch.
/// @param z engine species values (`kAtom_type`). @param n row count.
/// @param out reused scratch, resized to `n`.
/// @param who public operation name used as the message prefix.
/// @throws std::invalid_argument on a negative value.
inline void molpy_atomic_numbers(const int* z, std::size_t n,
                                 std::vector<std::uint64_t>& out,
                                 const char* who) {
    out.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (z[i] < 0)
            throw std::invalid_argument(
                std::string(who) + ": atom " + std::to_string(i)
                + " has atomic number " + std::to_string(z[i])
                + "; kAtom_type carries Z (0..118, 0 = wildcard)");
        out[i] = static_cast<std::uint64_t>(z[i]);
    }
}

} // namespace ATV::cpu::internal
