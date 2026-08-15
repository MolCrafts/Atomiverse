/// @file python/src/bindings/frame_guards.h
/// @brief Shared pre-run predicates: what a driver handle's registrations
///        demand of the molrs Frame handed to `run()`.
///
/// Two questions every driver facade asks before it starts a run — "did the
/// user register an electrostatic model?" and "does the frame carry the column
/// that model reads?" — and both answers must be identical whichever loop
/// asks. `MdHandle` and `MinimizeHandle` share the nine force-source setters
/// (`evaluator_setters.h`), so they inherit the same failure mode: a Coulomb
/// evaluator on a frame with no `charge` column. The predicates live here so
/// the two guards cannot drift; the refusal message itself stays on each
/// handle, in that loop's vocabulary.
#pragma once

#include <string_view>
#include <type_traits>
#include <variant>

#include "atomiverse/cpu/task/task_spec.h"
#include "atomiverse/molrs.hpp"

/// @brief Does `block` of a molrs Frame carry the column `key`?
///
/// The molrs boundary is the only place where a column's *absence* is still
/// visible: `cpu::from_molrs` allocates every optional per-atom column and
/// leaves an absent one zero-filled, so an evaluator downstream cannot tell
/// "no charges were given" from "every charge is 0 e".
///
/// @param frame Borrowed molrs Frame handle.
/// @param block Block name, e.g. `"atoms"`.
/// @param key   Column name, e.g. `"charge"`.
/// @return True when that block carries that column.
inline bool has_molrs_column(const molrs::FrameRef& frame, const char* block,
                             std::string_view key) {
    for (const ::rust::String& raw : molrs::frame_block_columns(frame, block))
        if (std::string_view{raw.data(), raw.size()} == key)
            return true;
    return false;
}

/// @brief Is `P` the mirror of an electrostatic evaluator?
///
/// Every Coulomb style reads per-atom charge: the truncated pair style, the
/// real-space Ewald half and the reciprocal PME half alike. The PME mirror is
/// always compiled (the wire layer does not need `ATV_HAS_KSPACE`), so no
/// build gate belongs here.
///
/// @tparam P One `*Params` mirror type.
template <typename P>
inline constexpr bool is_coulomb_params_v =
    std::is_same_v<P, ATV::cpu::task::PairCoulCutParams>
    || std::is_same_v<P, ATV::cpu::task::CoulLongParams>
    || std::is_same_v<P, ATV::cpu::task::KSpacePmeParams>;

/// @brief Is this mirror entry an electrostatic evaluator?
///
/// @tparam Entry A recorded registration — `MDComponentEntry` or
///               `MinimizeComponentEntry`; anything with a `params` variant.
/// @param entry One recorded registration.
/// @return True for `PairCoulCut` / `CoulLong` / `KSpacePME` entries.
template <typename Entry>
bool is_coulomb_entry(const Entry& entry) {
    return std::visit(
        [](const auto& p) {
            return is_coulomb_params_v<std::decay_t<decltype(p)>>;
        },
        entry.params);
}
