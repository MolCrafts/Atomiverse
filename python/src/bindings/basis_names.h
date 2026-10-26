/// @file python/src/bindings/basis_names.h
/// @brief Shared CPU/CUDA Gaussian-basis names at the Python boundary.
#pragma once

#include "atomiverse/cpu/scf/basis_set.h"

#include <stdexcept>
#include <string>
#include <utility>

/// @brief Basis-set name → `BasisSet` (names verbatim from `basis_set_name`).
/// @param name Basis-set name, e.g. "STO-3G" or "def2-SVP".
/// @return The matching enumerator.
/// @throws std::invalid_argument (Python `ValueError`) on an unsupported
///         name, listing every supported name — no fallback basis.
inline ATV::cpu::scf::BasisSet basis_from_name(const std::string& name) {
    using ATV::cpu::scf::BasisSet;
    static constexpr std::pair<const char*, BasisSet> kTable[] = {
        {"STO-3G", BasisSet::STO3G},
        {"6-31G", BasisSet::Pople631G},
        {"6-31G*", BasisSet::Pople631Gs},
        {"6-31+G*", BasisSet::Pople631pGs},
        {"def2-SVP", BasisSet::Def2SVP},
        {"def2-TZVP", BasisSet::Def2TZVP},
        {"aug-cc-pVDZ", BasisSet::AugCcPVDZ},
        {"cc-pVDZ", BasisSet::CcPVDZ},
    };
    std::string supported;
    for (const auto& [key, value] : kTable) {
        if (name == key)
            return value;
        if (!supported.empty())
            supported += ", ";
        supported += key;
    }
    throw std::invalid_argument("unknown basis set '" + name
                                + "'; supported: " + supported);
}
