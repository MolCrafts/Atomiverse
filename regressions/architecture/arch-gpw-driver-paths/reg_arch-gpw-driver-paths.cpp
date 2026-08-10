/// @file regressions/architecture/arch-gpw-driver-paths/reg_arch-gpw-driver-paths.cpp
/// @brief Architecture gate: multi-path GPW DIIS is a Driver node (gpw::Phase).
///
/// Public headers only. Confirms `GpwDft : cpu::Driver<gpw::Phase>` and that
/// path-local stage Ops bind the five SCF slots used by Γ RKS/UKS and multi-k
/// DIIS. Full energy pins live in twin regs (`gpw-uks-o2-gamma`,
/// `gpw-kpoints-si`) — this case is shape-only (no multi-iter SCF).

#include <cstdio>
#include <type_traits>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/driver/driver.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/stages/diagonalize_stage.h"
#include "atomiverse/cpu/gpw/stages/density_stage.h"
#include "atomiverse/cpu/gpw/stages/gpw_converged.h"
#include "atomiverse/cpu/gpw/stages/ks_build_stage.h"
#include "atomiverse/cpu/gpw/stages/mix_stage.h"

int main() {
    using ATV::cpu::gpw::DiagonalizeStage;
    using ATV::cpu::gpw::DensityStage;
    using ATV::cpu::gpw::GpwConverged;
    using ATV::cpu::gpw::GpwDft;
    using ATV::cpu::gpw::KsBuildStage;
    using ATV::cpu::gpw::MixStage;

    static_assert(std::is_base_of_v<ATV::cpu::Driver<ATV::gpw::Phase>, GpwDft>);
    if (ATV::slot_count<ATV::gpw::Phase>() != 5) {
        std::fprintf(stderr, "FAIL: gpw::Phase slot_count != 5\n");
        return 1;
    }
    if (ATV::gpw::kind_of(ATV::gpw::Phase{}) != ATV::PhaseKind::SCF) {
        std::fprintf(stderr, "FAIL: gpw::Phase kind is not SCF\n");
        return 1;
    }

    bool skip = false;
    auto nop = []() {};
    KsBuildStage ks(nop);
    MixStage mix(nop, skip);
    DiagonalizeStage diag(nop, skip);
    DensityStage dens(nop, skip);
    GpwConverged crit(skip);

    auto phase = [](const auto& c) {
        auto b = c.default_bindings();
        if (b.size() != 1 || b[0].kind != ATV::PhaseKind::SCF)
            return -1;
        return b[0].phase;
    };
    if (phase(ks) != ATV::slot_of(ATV::gpw::Phase::ks_build)
        || phase(mix) != ATV::slot_of(ATV::gpw::Phase::mix)
        || phase(diag) != ATV::slot_of(ATV::gpw::Phase::diagonalize)
        || phase(dens) != ATV::slot_of(ATV::gpw::Phase::density)
        || phase(crit) != ATV::slot_of(ATV::gpw::Phase::check)) {
        std::fprintf(stderr, "FAIL: stage default_bindings phase mismatch\n");
        return 1;
    }

    GpwDft method;
    method.n_occ = 1;
    method.n_alpha = 7;
    method.n_beta = 5;
    // UKS config is recognized without a full SCF.
    if (!method.is_uks()) {
        std::fprintf(stderr, "FAIL: n_alpha/n_beta should select UKS\n");
        return 1;
    }

    std::printf("PASS arch-gpw-driver-paths: Driver + multi-path stage shape\n");
    return 0;
}
