/// @file cpu/scf/orbital_info_observer.h
/// @brief OrbitalInfoObserver -- per-step HOMO/LUMO/gap summary from the
///        method's SlaterDeterminant.
///
/// Reads orbital energies from the latest `SlaterDeterminant` published by a
/// sibling mean-field method via `MeanFieldOutputs`. Writes a TSV log and
/// optionally mirrors to stdout.
///
/// Usage:
/// @code
///   cpu::scf::RHF scf{};
///   auto handle = scf.result_handle();                       // share state
///   opt.on(minimizer::Phase::force).add(std::move(scf));
///   opt.on(minimizer::Phase::end_of_step)
///      .add(cpu::scf::OrbitalInfoObserver{handle, "orb.tsv"});
/// @endcode
///
/// Quantities logged (always written in eV):
///   HOMO        = highest-occupied orbital energy (alpha channel)
///   LUMO        = lowest-unoccupied orbital energy (alpha channel)
///   gap         = LUMO - HOMO
///   ip_koopmans = -HOMO (Koopmans ionization potential; Koopmans,
///                 Physica 1, 104 (1934), DOI 10.1016/S0031-8914(34)90011-2)
#pragma once

#include <fstream>
#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/scf/mean_field_method.h"

namespace ATV {
namespace cpu {
namespace scf {

/// @brief Orbital-energy observer component.
///
/// No default bindings: user chooses `end_of_step` of whichever driver hosts
/// the method (MD, Minimize, or Evaluate).
class OrbitalInfoObserver : public cpu::Component {
  public:
    /// @brief Construct.
    /// @param handle        Shared view of the method's latest state.
    /// @param filename      Output TSV path. Empty string disables file output.
    /// @param also_stdout   If true, mirror each row to stdout.
    OrbitalInfoObserver(MeanFieldOutputs handle, std::string filename = "",
                        bool also_stdout = false);

    std::string name() const override { return "orbital_info_observer"; }

    void build(const cpu::BuildContext& ctx) override;
    void launch(const cpu::LaunchContext& ctx) override;
    void destroy(const cpu::DestroyContext& ctx) override;

    /// @brief Number of rows written (excluding header).
    int n_rows() const { return n_rows_; }

    /// @brief HOMO energy [eV] from the last launch (undefined before first launch).
    double homo_eV() const { return homo_eV_; }

    /// @brief LUMO energy [eV] from the last launch.
    double lumo_eV() const { return lumo_eV_; }

    /// @brief HOMO-LUMO gap [eV] from the last launch.
    double gap_eV() const { return lumo_eV_ - homo_eV_; }

    /// @brief Koopmans ionization potential IP = −ε_HOMO [eV] from the last
    ///        launch (frozen-orbital estimate of the vertical IP).
    double ip_koopmans_eV() const { return -homo_eV_; }

  private:
    MeanFieldOutputs handle_;
    std::string filename_;
    bool also_stdout_;
    std::ofstream os_;
    int n_rows_ = 0;
    double homo_eV_ = 0;
    double lumo_eV_ = 0;
};

} // namespace scf
} // namespace cpu
} // namespace ATV
