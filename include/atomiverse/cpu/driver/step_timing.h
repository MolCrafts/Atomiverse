/// @file cpu/driver/step_timing.h
/// @brief MD timing categories + component-name classifier + CSV schema.
///
/// Categories match LAMMPS `Loop time` / section breakdown:
///   Pair | Bond | Kspace | Neigh | Comm | Output | Modify | Other
/// Accumulation lives in `ATV::TimerRegistry`; this header is pure metadata.
#pragma once

#include <array>
#include <string>
#include <string_view>

namespace ATV {
namespace cpu {
namespace md {

/// @brief Per-component MD timing taxonomy (LAMMPS-aligned sections).
struct StepTiming {
    enum Category {
        Pair = 0,
        Bond,
        Kspace,
        Neigh,
        Comm,
        Output,
        Modify,
        Other,
        kNumCategories
    };

    /// Display names — identical to LAMMPS log section labels (Bond not Bonded).
    static constexpr std::array<std::string_view, kNumCategories> kNames{
        "Pair", "Bond", "Kspace", "Neigh", "Comm", "Output", "Modify", "Other"};

    /// @brief Map a component `name()` to a timing category.
    ///
    /// Order matters: more specific prefixes before generic substrings
    /// (e.g. `coul/long` before bare `coul` is not needed if we check kspace
    /// first; `charmm_dihedral` contains neither "bond" alone as a section).
    static Category classify(std::string_view n) {
        auto has = [&](std::string_view s) {
            return n.find(s) != std::string_view::npos;
        };
        // Reciprocal mesh / Ewald first (before generic "pair").
        if (has("pppm") || has("pme") || has("ewald") || has("kspace"))
            return Kspace;
        // Real-space coulomb and LJ (including fused lj/cut/coul/long).
        if (has("lj") || has("morse") || has("coul") || has("pair") || has("sw")
            || has("eam"))
            return Pair;
        if (has("bond") || has("angle") || has("dihedral") || has("improper"))
            return Bond;
        if (has("neighbor") || has("cell") || has("neigh"))
            return Neigh;
        if (has("halo") || has("migrat") || has("rebalance") || has("comm")
            || has("reduce"))
            return Comm;
        if (has("thermo") || has("dump") || has("xyz") || has("recorder")
            || has("trajectory") || has("output"))
            return Output;
        if (has("nose") || has("langevin") || has("berendsen") || has("verlet")
            || has("nve") || has("nvt") || has("npt") || has("barostat")
            || has("thermostat") || has("rattle") || has("shake")
            || has("deform") || has("momentum") || has("fix_")
            || has("init_velocity") || has("kinetic"))
            return Modify;
        return Other;
    }

    /// @brief CSV column schema for the machine-readable per-stage breakdown.
    static std::string csv_header() {
        return "stage,ranks,steps,total_s,nlocal_min,nlocal_max,nlocal_avg,"
               "Pair_s,Bond_s,Kspace_s,Neigh_s,Comm_s,Output_s,Modify_s,"
               "Other_s,"
               "Pair_pct,Bond_pct,Kspace_pct,Neigh_pct,Comm_pct,Output_pct,"
               "Modify_pct,Other_pct";
    }
};

} // namespace md
} // namespace cpu
} // namespace ATV
