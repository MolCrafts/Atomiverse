/// @file python/src/bindings/xc_names.h
/// @brief The one exchange-correlation functional name table the bindings
///        expose to Python, both directions.
///
/// An *exchange-correlation functional* is the recipe a density-functional
/// method uses to supply everything its non-interacting reference system
/// leaves out; it is named by a string on the Python surface and by
/// `cpu::xc::XCType` in C++. Two bindings now need that mapping — the
/// Gaussian SCF methods (`RKS(xc=...)`, `UKS`) and the plane-wave method
/// (`PW(xc=...)`) — so the table lives here rather than being spelled twice
/// and drifting.
///
/// Both directions come off ONE table, which is the point: a name that
/// `xc_from_name` accepts is exactly the name `xc_name` gives back, so the
/// round-trip a Python property promises (`PW(..., xc="LSDA")` then
/// `pw.xc == "LSDA"`) cannot be broken by editing one list and forgetting the
/// other. Every binding takes `xc=` as a constructor argument and reads it
/// back through a read-only property, so both directions are used.
///
/// No-fallback rule: an unsupported name raises `std::invalid_argument`
/// (Python `ValueError`) listing every supported one. Silently substituting a
/// functional would report a converged energy for a model the caller never
/// asked for.
///
/// This header is included only from translation units compiled with a
/// density-functional domain built in (`ATV_HAS_HF` / `ATV_HAS_PW`) — the
/// enum it maps to belongs to those domains.
#pragma once

#include "atomiverse/cpu/xc/xc_functional.h"

#include <stdexcept>
#include <string>
#include <utility>

/// @brief Supported functional names and the enumerators they resolve to.
///
/// Order is the table's published order and is what the diagnostics list.
inline constexpr std::pair<const char*, ATV::cpu::xc::XCType> kXCNames[] = {
    {"LSDA", ATV::cpu::xc::XCType::LSDA},
    {"PBE", ATV::cpu::xc::XCType::PBE},
    {"PBEsol", ATV::cpu::xc::XCType::PBEsol},
    {"RevPBE", ATV::cpu::xc::XCType::RevPBE},
    {"RPBE", ATV::cpu::xc::XCType::RPBE},
    {"BLYP", ATV::cpu::xc::XCType::BLYP},
    {"OptB88VdW", ATV::cpu::xc::XCType::OptB88VdW},
    {"B3LYP", ATV::cpu::xc::XCType::B3LYP},
    {"PBE0", ATV::cpu::xc::XCType::PBE0},
    {"B2PLYP", ATV::cpu::xc::XCType::B2PLYP},
    {"CAMB3LYP", ATV::cpu::xc::XCType::CAMB3LYP},
    {"PADE", ATV::cpu::xc::XCType::PADE},
};

/// @brief Functional name → `XCType`.
/// @param name Exchange-correlation functional name, e.g. "PBE" or "B3LYP".
/// @return The matching enumerator.
/// @throws std::invalid_argument (Python `ValueError`) on an unsupported
///         name, listing every supported functional — no fallback.
inline ATV::cpu::xc::XCType xc_from_name(const std::string& name) {
    std::string supported;
    for (const auto& [key, value] : kXCNames) {
        if (name == key)
            return value;
        if (!supported.empty())
            supported += ", ";
        supported += key;
    }
    throw std::invalid_argument("unknown exchange-correlation functional '"
                                + name + "'; supported: " + supported);
}

/// @brief `XCType` → the name `xc_from_name` accepts for it.
/// @param xc A functional enumerator.
/// @return Its published name.
/// @throws std::invalid_argument when the enumerator is absent from the
///         table — that is a binding left behind by a new functional, not a
///         user error, and silently returning "unknown" would make the
///         property round-trip lie.
inline std::string xc_name(ATV::cpu::xc::XCType xc) {
    for (const auto& [key, value] : kXCNames)
        if (xc == value)
            return key;
    throw std::invalid_argument(
        "exchange-correlation functional enumerator "
        + std::to_string(static_cast<int>(xc))
        + " has no name in the Python binding table (xc_names.h)");
}
