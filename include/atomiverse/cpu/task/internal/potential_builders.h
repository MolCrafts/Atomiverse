/// @file cpu/task/internal/potential_builders.h
/// @brief Typed real-space params -> the components they mirror.
///
/// Only ONE style of the widened catalogue needs a builder: every other pair
/// or bonded mirror hands its fields straight to a constructor, in declaration
/// order, and a braced construction at the call site is the whole story there.
/// `PairBornParams` is different because the wire cannot carry a vector of
/// STRUCTS — its variable-length primitive is a length-prefixed run of one
/// scalar type — so the Fumi-Tosi table (the per-type-pair
/// Born-Mayer-Huggins parameterization of a molten salt: one row of
/// A/ρ/σ/C/D per interacting element pair) crosses as PARALLEL VECTORS and has
/// to be transposed back into `cpu::BornTypeCoeff` rows.
///
/// It is a NAMED function rather than a loop repeated at each call site — the
/// tree's default is to keep a one-call-site block inline — because it has
/// THREE call sites: the launcher replay's explicit binding and its
/// categorical setter (`src/cpu/task/task_materializer.cpp`), plus the shared
/// Python evaluator setter (`python/src/bindings/evaluator_setters.h`, which
/// serves both `MD` and `Minimize`). A transposition that reached only some of
/// them would put different coefficients on different paths while every one of
/// them still ran.
///
/// `internal`, not public API: this is a params-mirror-to-component factory,
/// which the public C++ surface does not have — a user constructs
/// `cpu::PairBorn` from the coefficient rows directly.
#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/cpu/potentials/pair_born.h"
#include "atomiverse/cpu/task/task_spec.h"

namespace ATV::cpu::task::internal {

/// @brief Transpose the parallel-vector Born mirror into coefficient rows.
///
/// Row `k` of the returned table is
/// `{type_i[k], type_j[k], A[k], rho[k], sigma[k], C[k], D[k]}` — the field
/// order of `cpu::BornTypeCoeff`, which is also the field order of the mirror,
/// which is also the byte order the codec writes.
///
/// The seven vectors must all be the same length: they are one table seen
/// column-wise, so a short one is a row whose coefficients were never supplied,
/// and reading past it would silently key a Born-Mayer wall on whatever
/// followed in memory. Refused here, at the transposition, rather than at
/// `cpu::PairBorn`'s own validation, which sees only well-formed rows.
///
/// Type keys are frame `kAtom_type` values as-is — atomic numbers in this tree
/// — and are not renumbered.
///
/// @param p Typed Born mirror (per-row A [eV], ρ [Å], σ [Å], C [eV·Å⁶],
///          D [eV·Å⁸] and the two type keys).
/// @return One `BornTypeCoeff` per row, in the mirror's row order.
/// @throws std::invalid_argument When the seven vectors differ in length.
inline std::vector<cpu::BornTypeCoeff> born_table(const PairBornParams& p) {
    const std::size_t rows = p.type_i.size();
    if (p.type_j.size() != rows || p.A.size() != rows || p.rho.size() != rows
        || p.sigma.size() != rows || p.C.size() != rows || p.D.size() != rows)
        throw std::invalid_argument(
            "PairBorn: the type_i/type_j/A/rho/sigma/C/D vectors are one "
            "coefficient table seen column-wise and must all have the same "
            "length; got type_i="
            + std::to_string(rows)
            + ", type_j=" + std::to_string(p.type_j.size())
            + ", A=" + std::to_string(p.A.size())
            + ", rho=" + std::to_string(p.rho.size())
            + ", sigma=" + std::to_string(p.sigma.size()) + ", C="
            + std::to_string(p.C.size()) + ", D=" + std::to_string(p.D.size()));
    std::vector<cpu::BornTypeCoeff> table;
    table.reserve(rows);
    for (std::size_t k = 0; k < rows; ++k)
        table.push_back(cpu::BornTypeCoeff{p.type_i[k], p.type_j[k], p.A[k],
                                           p.rho[k], p.sigma[k], p.C[k],
                                           p.D[k]});
    return table;
}

} // namespace ATV::cpu::task::internal
