/// @file reg_arch-pw-driver-paths.cpp
/// @brief Architecture gate: multi-path PW SCF is a Driver node (pw::Phase).
///
/// Public headers only. Confirms `PW : cpu::Driver<pw::Phase>` and that
/// path-local stage StepFns bind the five SCF slots used by restricted
/// fixed-occ, collinear fixed-occ, restricted metallic smear, and
/// collinear + metallic smear. Full energy pins live in twin regs
/// (`pw-smearing-al`, `pw-spin-polarized-o2`, …) — this case is shape-only.

#include <cstdio>
#include <type_traits>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/driver/driver.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/planewave/stages/density_stage.h"
#include "atomiverse/cpu/planewave/stages/diagonalize_stage.h"
#include "atomiverse/cpu/planewave/stages/mix_stage.h"
#include "atomiverse/cpu/planewave/stages/pw_converged.h"
#include "atomiverse/cpu/planewave/stages/veff_stage.h"

int main() {
    using ATV::cpu::pw::DensityStage;
    using ATV::cpu::pw::DiagonalizeStage;
    using ATV::cpu::pw::MixStage;
    using ATV::cpu::pw::PW;
    using ATV::cpu::pw::PwConverged;
    using ATV::cpu::pw::Smearing;
    using ATV::cpu::pw::VeffStage;

    static_assert(std::is_base_of_v<ATV::cpu::Driver<ATV::pw::Phase>, PW>);
    if (ATV::slot_count<ATV::pw::Phase>() != 5) {
        std::fprintf(stderr, "FAIL: pw::Phase slot_count != 5\n");
        return 1;
    }
    if (ATV::pw::kind_of(ATV::pw::Phase{}) != ATV::PhaseKind::SCF) {
        std::fprintf(stderr, "FAIL: pw::Phase kind is not SCF\n");
        return 1;
    }

    auto nop = []() {};
    bool conv = false;
    VeffStage veff(nop);
    DiagonalizeStage diag(nop);
    DensityStage dens(nop);
    MixStage mix(nop);
    PwConverged crit(conv);

    auto phase = [](const auto& c) {
        auto b = c.default_bindings();
        if (b.size() != 1 || b[0].kind != ATV::PhaseKind::SCF)
            return -1;
        return b[0].phase;
    };
    if (phase(veff) != ATV::slot_of(ATV::pw::Phase::veff)
        || phase(diag) != ATV::slot_of(ATV::pw::Phase::diagonalize)
        || phase(dens) != ATV::slot_of(ATV::pw::Phase::density)
        || phase(mix) != ATV::slot_of(ATV::pw::Phase::mix)
        || phase(crit) != ATV::slot_of(ATV::pw::Phase::check)) {
        std::fprintf(stderr, "FAIL: stage default_bindings phase mismatch\n");
        return 1;
    }

    // Multi-path public surface (no *Config).
    PW method;
    method.spin_polarized = true;
    method.multiplicity = 3;
    method.smearing = Smearing::fermi_dirac;
    method.degauss = 0.02;
    method.extra_bands = 4;
    if (!method.spin_polarized || method.multiplicity != 3
        || method.smearing != Smearing::fermi_dirac) {
        std::fprintf(stderr, "FAIL: multi-path members not sticky\n");
        return 1;
    }

    std::printf(
        "PASS: arch-pw-driver-paths  PW : Driver<pw::Phase>  "
        "multi-path stages (spin+smear members)  (harness shape)\n");
    return 0;
}
