/// @file cpu/recorder/thermo.cpp
/// @brief CPU thermodynamic output: Thermo component + thermo_print helper.

#include "atomiverse/cpu/record/thermo.h"
#include "atomiverse/core/log.h"
#include "atomiverse/core/units.h"

#include <spdlog/fmt/fmt.h>

namespace ATV {
namespace cpu {

// ── Thermo component ───────────────────────────────────────────────

Thermo::Thermo(int freq) : freq_(freq) {}

void Thermo::launch(const cpu::LaunchContext& ctx) {
    if (ctx.runtime.mpi_rank != 0)
        return;

    auto& f = ctx.frame;
    // Ndof has ONE owner (Driver::publish_dof_): 3N - 3 - n_constraints. This
    // used to be hardcoded as 3N - 3 here, which is simply wrong the moment
    // anything is constrained — a rigid-water box would report T ~33% low.
    int n_dof = ctx.runtime.md.n_dof_global;

    double temperature = 0;
    if (n_dof > 0)
        temperature = double(2) * f.meta.get(kKinetic)
                      / (static_cast<double>(n_dof) * units::kB_ev_per_k);

    double total_energy = f.meta.get(kEnergy) + f.meta.get(kKinetic);

    // `Conserved` is the integrator's conserved quantity (extended
    // Hamiltonian for Nosé-Hoover NVT/NPT); `0` when no integrator
    // populates `frame.meta.get(kConserved)`. A flat column confirms the
    // integrator is symplectic — for NVT/NPT, `TotalE` itself is *not*
    // conserved (energy flows to the thermostat / barostat).
    if (ctx.step == 0) {
        ATV_INFO("  Step          PE              KE           Temp         "
                 "TotalE         Conserved");
        ATV_INFO("  ----  --------------- --------------- --------------- "
                 "--------------- ---------------");
    }

    ATV_INFO("  {:4d}  {:15.8e} {:15.8e} {:15.8e} {:15.8e} {:15.8e}", ctx.step,
             static_cast<double>(f.meta.get(kEnergy)),
             static_cast<double>(f.meta.get(kKinetic)),
             static_cast<double>(temperature),
             static_cast<double>(total_energy),
             static_cast<double>(f.meta.get(kConserved)));
}

// ── thermo_print free function ─────────────────────────────────────

void thermo_print(int step, double pe, double ke, int n_dof,
                  bool& header_printed, std::ofstream& file) {
    // Temperature: T = 2*KE / (Ndof * kB). Ndof is supplied by the caller — it is
    // owned by Driver::publish_dof_ and accounts for constraints.
    const int ndof = n_dof;
    const double temp =
        (ndof > 0) ? 2.0 * ke / (static_cast<double>(ndof) * units::kB_ev_per_k)
                   : 0.0;
    const double total = pe + ke;

    if (!header_printed) {
        ATV_INFO("{:>8s} {:>12s} {:>14s} {:>14s} {:>14s}", "Step", "Temp",
                 "PotEng", "KinEng", "TotEng");
        header_printed = true;
    }
    ATV_INFO("{:8d} {:12.4f} {:14.6f} {:14.6f} {:14.6f}", step,
             static_cast<double>(temp), static_cast<double>(pe),
             static_cast<double>(ke), static_cast<double>(total));

    if (file.is_open()) {
        file << fmt::format("{} {:.6f} {:.6f} {:.6f} {:.6f}\n", step,
                            static_cast<double>(temp), static_cast<double>(pe),
                            static_cast<double>(ke),
                            static_cast<double>(total));
    }
}

} // namespace cpu
} // namespace ATV
