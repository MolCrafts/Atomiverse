/// @file cpu/scf/orbital_info_observer.cpp
/// @brief OrbitalInfoObserver -- HOMO/LUMO/gap TSV writer.

#include "atomiverse/cpu/scf/orbital_info_observer.h"

#include <cstdio>
#include <stdexcept>

#include "atomiverse/core/units.h"

namespace ATV {
namespace cpu {
namespace scf {

OrbitalInfoObserver::OrbitalInfoObserver(MeanFieldOutputs handle,
                                         std::string filename, bool also_stdout)
    : handle_(std::move(handle)), filename_(std::move(filename)),
      also_stdout_(also_stdout) {
    if (!handle_.result || !handle_.ready)
        throw std::invalid_argument(
            "OrbitalInfoObserver: null MeanFieldOutputs (did you forget "
            "method.result_handle()?)");
}

void OrbitalInfoObserver::build(const cpu::BuildContext& ctx) {
    if (ctx.runtime.mpi_rank != 0)
        return;
    if (!filename_.empty()) {
        os_.open(filename_, std::ios::out | std::ios::trunc);
        os_ << "step\thomo_eV\tlumo_eV\tgap_eV\tip_koopmans\n";
    }
    if (also_stdout_)
        std::printf("%6s  %12s  %12s  %12s  %12s\n", "step", "HOMO[eV]",
                    "LUMO[eV]", "gap[eV]", "IP_K[eV]");
}

void OrbitalInfoObserver::launch(const cpu::LaunchContext& ctx) {
    if (ctx.runtime.mpi_rank != 0)
        return;
    if (!*handle_.ready)
        return; // no SCF completed yet this step

    const SlaterDeterminant& det = *handle_.result;

    // homo()/lumo() are occupation-defined [Hartree]; lumo() returns homo()
    // when the channel has no virtual orbital (minimal basis on H-like
    // systems), matching the historical observer semantics.
    homo_eV_ = det.homo(SlaterDeterminant::Spin::alpha) * units::kHartreeInEv;
    lumo_eV_ = det.lumo(SlaterDeterminant::Spin::alpha) * units::kHartreeInEv;
    double gap = lumo_eV_ - homo_eV_;

    if (os_.is_open()) {
        os_ << ctx.step << '\t' << homo_eV_ << '\t' << lumo_eV_ << '\t' << gap
            << '\t' << ip_koopmans_eV() << '\n';
        os_.flush();
    }
    if (also_stdout_)
        std::printf("%6d  %12.6f  %12.6f  %12.6f  %12.6f\n", ctx.step,
                    homo_eV_, lumo_eV_, gap, ip_koopmans_eV());
    ++n_rows_;
}

void OrbitalInfoObserver::destroy(const cpu::DestroyContext& /*ctx*/) {
    if (os_.is_open())
        os_.close();
}

} // namespace scf
} // namespace cpu
} // namespace ATV
