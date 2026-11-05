/// @file core/thermo_style.cpp
/// @brief `thermo_style custom` keyword parsing and formatting.

#include "atomiverse/core/thermo_style.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include <spdlog/fmt/fmt.h>

#include "atomiverse/core/units.h"

namespace ATV {

namespace {

struct Entry {
    const char* keyword;
    ThermoKeyword column;
    const char* header; // LAMMPS thermo column name
};

constexpr Entry kEntries[] = {
    {"step", ThermoKeyword::Step, "Step"},
    {"temp", ThermoKeyword::Temp, "Temp"},
    {"pe", ThermoKeyword::Pe, "PotEng"},
    {"ke", ThermoKeyword::Ke, "KinEng"},
    {"etotal", ThermoKeyword::Etotal, "TotEng"},
    {"econserve", ThermoKeyword::Econserve, "Econserve"},
    {"press", ThermoKeyword::Press, "Press"},
    {"pxx", ThermoKeyword::Pxx, "Pxx"},
    {"pyy", ThermoKeyword::Pyy, "Pyy"},
    {"pzz", ThermoKeyword::Pzz, "Pzz"},
    {"pxy", ThermoKeyword::Pxy, "Pxy"},
    {"pxz", ThermoKeyword::Pxz, "Pxz"},
    {"pyz", ThermoKeyword::Pyz, "Pyz"},
    {"vol", ThermoKeyword::Vol, "Volume"},
};

const Entry& entry(ThermoKeyword column) {
    for (const auto& e : kEntries)
        if (e.column == column)
            return e;
    throw std::logic_error("ThermoStyle: unmapped keyword");
}

// Voigt slot of a pressure-tensor column, -1 for the isotropic pressure.
int voigt(ThermoKeyword column) {
    switch (column) {
    case ThermoKeyword::Pxx:
        return 0;
    case ThermoKeyword::Pyy:
        return 1;
    case ThermoKeyword::Pzz:
        return 2;
    case ThermoKeyword::Pxy:
        return 3;
    case ThermoKeyword::Pxz:
        return 4;
    case ThermoKeyword::Pyz:
        return 5;
    default:
        return -1;
    }
}

bool is_pressure(ThermoKeyword column) {
    return column == ThermoKeyword::Press || voigt(column) >= 0;
}

} // namespace

ThermoStyle::ThermoStyle()
    : ThermoStyle(
          std::vector<std::string>{"step", "temp", "pe", "ke", "etotal"}) {}

ThermoStyle::ThermoStyle(const std::vector<std::string>& keywords) {
    if (keywords.empty())
        throw std::invalid_argument("thermo_style: no keywords");
    for (const auto& keyword : keywords) {
        const auto* found =
            std::find_if(std::begin(kEntries), std::end(kEntries),
                         [&](const Entry& e) { return keyword == e.keyword; });
        if (found == std::end(kEntries)) {
            std::string known;
            for (const auto& e : kEntries)
                known += std::string(known.empty() ? "" : " ") + e.keyword;
            throw std::invalid_argument("thermo_style: unknown keyword '"
                                        + keyword + "' (supported: " + known
                                        + ")");
        }
        if (std::find(columns_.begin(), columns_.end(), found->column)
            != columns_.end())
            throw std::invalid_argument("thermo_style: repeated keyword '"
                                        + keyword + "'");
        columns_.push_back(found->column);
    }
}

bool ThermoStyle::needs_energy() const {
    return std::any_of(columns_.begin(), columns_.end(), [](ThermoKeyword c) {
        return c == ThermoKeyword::Pe || c == ThermoKeyword::Etotal
               || c == ThermoKeyword::Econserve;
    });
}

bool ThermoStyle::needs_virial() const {
    return std::any_of(columns_.begin(), columns_.end(), is_pressure);
}

std::string ThermoStyle::header() const {
    std::string out;
    for (auto column : columns_)
        out += column == ThermoKeyword::Step
                   ? fmt::format("{:>10s}", entry(column).header)
                   : fmt::format(" {:>15s}", entry(column).header);
    return out;
}

std::string ThermoStyle::line(const ThermoSample& s) const {
    std::string out;
    // Pressure tensor P = (W + K) / V, reported in bar (LAMMPS metal units).
    auto pressure = [&](int a) {
        return s.volume > 0 ? (s.virial[a] + s.kinetic[a]) / s.volume
                                  / units::bar_to_ev_per_a3
                            : 0.0;
    };
    for (auto column : columns_) {
        double value = 0;
        switch (column) {
        case ThermoKeyword::Step:
            out += fmt::format("{:10d}", static_cast<long long>(s.step));
            continue;
        case ThermoKeyword::Temp:
            value = s.ndof > 0
                        ? 2.0 * s.ke
                              / (static_cast<double>(s.ndof) * units::md_boltz)
                        : 0.0;
            break;
        case ThermoKeyword::Pe:
            value = s.pe;
            break;
        case ThermoKeyword::Ke:
            value = s.ke;
            break;
        case ThermoKeyword::Etotal:
            value = s.pe + s.ke;
            break;
        case ThermoKeyword::Econserve:
            value = s.conserved;
            break;
        case ThermoKeyword::Press:
            value = (pressure(0) + pressure(1) + pressure(2)) / 3.0;
            break;
        case ThermoKeyword::Vol:
            value = s.volume;
            break;
        default:
            value = pressure(voigt(column));
            break;
        }
        out += fmt::format(" {:15.8g}", value);
    }
    return out;
}

} // namespace ATV
