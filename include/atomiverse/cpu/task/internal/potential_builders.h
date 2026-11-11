/// @file cpu/task/internal/potential_builders.h
/// @brief Typed real-space params -> the components they mirror.
///
/// `PairLJCut`, `PairCoulCut` and `PairLJCutCoulLong` have builders too
/// (below): several setters construct them and must agree on the table form,
/// the special-bond weights, the energy shift, the tail and the Coulomb
/// constant. Of the widened catalogue only ONE style needs a builder: every
/// other pair or bonded mirror hands its fields straight to a constructor, in declaration
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
#include <utility>
#include <vector>

#include "atomiverse/cpu/potentials/pair_born.h"
#include "atomiverse/cpu/potentials/pair_coul_cut.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/cpu/potentials/pair_lj_cut_coul_long.h"
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
/// Type keys are frame `kAtom_type` values as-is (the force-field type index,
/// molpy `atoms.type_id`) and are not renumbered.
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

/// @brief The `cpu::PairLJCut` a `PairLJCutParams` mirror describes.
///
/// One factory for every path that constructs the style (launcher replay,
/// the Python MD / Minimize / MC / vibration setters): one switch between the
/// single-type and per-type constructors (rule in `PairLJCutParams::typed`),
/// so the table form, the special-bond weights, the energy shift and the tail
/// reach all of them alike and no path can install a different Hamiltonian.
///
/// @param p Typed mirror (ε [eV], σ [Å], cutoff [Å], 1-2/1-3/1-4 weights,
///          LJ rows, r⁻⁴ rows, tail and shift flags).
/// @return The configured component.
/// @throws std::invalid_argument on an ambiguous mirror or bad rows.
inline cpu::PairLJCut make_pair_lj_cut(const PairLJCutParams& p) {
    if (!p.typed())
        return cpu::PairLJCut{p.epsilon, p.sigma, p.cutoff}
            .set_special_bonds(p.s12, p.s13, p.s14)
            .set_energy_shift(p.energy_shift)
            .set_tail_correction(p.tail);
    cpu::PairLJCut pair{p.coeffs, p.cutoff};
    if (!p.r4.empty())
        pair.set_r4_attraction(p.r4);
    return std::move(pair)
        .set_special_bonds(p.s12, p.s13, p.s14)
        .set_energy_shift(p.energy_shift)
        .set_tail_correction(p.tail);
}

/// @brief Build `cpu::PairCoulCut` from its mirror (cutoff, special-bond
///        weights, Coulomb constant) — same single-factory reason as above.
/// @param p Typed mirror.
/// @return The configured component.
inline cpu::PairCoulCut make_pair_coul_cut(const PairCoulCutParams& p) {
    auto pair =
        cpu::PairCoulCut{p.cutoff}.set_special_bonds(p.s12, p.s13, p.s14);
    if (p.coulomb > 0.0)
        pair.set_coulomb_constant(p.coulomb);
    return pair;
}

/// @brief The `cpu::PairLJCutCoulLong` a `PairLJCutCoulLongParams` mirror
///        describes (same call sites as `make_pair_lj_cut`).
///
/// @param p Typed mirror (LJ rows, α [1/Å], cutoff [Å], six special
///          weights, shift flag, r⁻⁴ rows).
/// @return The configured component.
/// @throws std::invalid_argument on bad rows or a non-positive α / cutoff.
inline cpu::PairLJCutCoulLong
make_pair_lj_cut_coul_long(const PairLJCutCoulLongParams& p) {
    cpu::PairLJCutCoulLong pair{p.coeffs, p.alpha, p.cutoff};
    pair.set_special_bonds(p.lj12, p.lj13, p.lj14, p.c12, p.c13, p.c14)
        .set_energy_shift(p.energy_shift)
        .set_tail_correction(p.tail);
    if (!p.r4.empty())
        pair.set_r4_attraction(p.r4);
    return pair;
}

} // namespace ATV::cpu::task::internal
