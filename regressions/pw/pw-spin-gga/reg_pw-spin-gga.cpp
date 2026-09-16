/// @file regressions/pw/pw-spin-gga/reg_pw-spin-gga.cpp
/// @brief END-TO-END: true collinear spin-GGA PBE on O₂ triplet (e_tot, mag,
///        GGA vs LSDA activation). Public API only. See REFERENCE.md.
///
/// Requires ATV_BUILD_PW. No live CP2K/QE.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/scf/xc_functional.h"

using namespace ATV;
using namespace ATV::cpu::pw;

namespace {

cpu::Frame o2_frame() {
    cpu::Frame frame(2);
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_type)[i] = 8;
        frame.span(kAtom_mass)[i] = 15.999;
        frame.span(kAtom_x)[i] = 0.0;
        frame.span(kAtom_y)[i] = 0.0;
    }
    frame.span(kAtom_z)[0] = 0.0;
    frame.span(kAtom_z)[1] = 1.2075;
    frame.box = cpu::Box::cubic(8.0);
    return frame;
}

} // namespace

int main() {
    // Offline pins (REFERENCE.md). Magnetization is the literature hard gate.
    constexpr double want_mag = 2.0;
    constexpr double tol_mag = 1e-9;
    // Total energy [Ha] at the exact settings below with true collinear
    // spin-GGA PBE (spectral ∇ρ_σ). Offline method pin 2026-07-26.
    constexpr double want_e_ha = -28.7585210635;
    constexpr double tol_e_ha = 1.0e-3;
    // GGA activation: |E_PBE − E_LSDA| floor [Ha] on the same spin O₂ setup.
    constexpr double de_floor_ha = 1.0e-4;

    // Compiled-in absolute path (ATV_REG_DATA_DIR): resolves identically from
    // the build tree, the source tree, or a bare ./reg_<case>.
    const char* upf = ATV_REG_DATA_DIR "/pw/O_ONCV_PBE-1.2.upf";
    if (std::FILE* f = std::fopen(upf, "r")) {
        std::fclose(f);
    } else {
        std::printf("FAIL: cannot open %s\n", upf);
        return 1;
    }

    auto frame = o2_frame();

    PW pbe;
    pbe.upf_path = upf;
    pbe.ecut_ry = 12.0;
    pbe.k_mesh = 1;
    pbe.tol = 5e-4;
    pbe.max_iter = 60;
    pbe.mix_beta = 0.3;
    pbe.pulay_depth = 5;
    pbe.spin_polarized = true;
    pbe.multiplicity = 3;
    pbe.xc = cpu::scf::XCType::PBE;

    pbe.build(frame);
    const auto& r_pbe = pbe.run(frame);

    std::printf("PBE spin-GGA: E=%.10f Ha (%.10f Ry)  mag=%.6f  conv=%d iters=%d\n",
                r_pbe.energy_ha, r_pbe.energy, r_pbe.magnetization,
                (int)r_pbe.converged, r_pbe.iterations);

    bool ok = true;
    if (!std::isfinite(r_pbe.energy_ha) || !std::isfinite(r_pbe.energy)) {
        std::printf("FAIL: non-finite PBE energy\n");
        ok = false;
    }
    if (!r_pbe.converged) {
        std::printf("FAIL: PBE SCF not converged\n");
        ok = false;
    }
    if (std::fabs(r_pbe.magnetization - want_mag) > tol_mag) {
        std::printf("FAIL: mag got=%.12f want=%.12f tol=%g\n",
                    r_pbe.magnetization, want_mag, tol_mag);
        ok = false;
    }

    // LSDA spin twin for GGA activation (same geometry / occupations).
    auto frame_lsda = o2_frame();
    PW lsda;
    lsda.upf_path = upf;
    lsda.ecut_ry = 12.0;
    lsda.k_mesh = 1;
    lsda.tol = 5e-4;
    lsda.max_iter = 60;
    lsda.mix_beta = 0.3;
    lsda.pulay_depth = 5;
    lsda.spin_polarized = true;
    lsda.multiplicity = 3;
    lsda.xc = cpu::scf::XCType::LSDA;

    lsda.build(frame_lsda);
    const auto& r_lsda = lsda.run(frame_lsda);
    std::printf("LSDA spin:    E=%.10f Ha  mag=%.6f  conv=%d iters=%d\n",
                r_lsda.energy_ha, r_lsda.magnetization, (int)r_lsda.converged,
                r_lsda.iterations);

    if (!std::isfinite(r_lsda.energy_ha) || !r_lsda.converged) {
        std::printf("FAIL: LSDA spin twin failed\n");
        ok = false;
    }
    const double de =
        std::fabs(r_pbe.energy_ha - r_lsda.energy_ha);
    std::printf("|E_PBE - E_LSDA| = %.6e Ha (floor %.1e)\n", de, de_floor_ha);
    if (de <= de_floor_ha) {
        std::printf("FAIL: GGA activation |ΔE| too small — spin path stuck "
                    "in σ=0 LDA limit?\n");
        ok = false;
    }

    if (std::fabs(r_pbe.energy_ha - want_e_ha) > tol_e_ha) {
        std::printf("FAIL: E_ha got=%.12f want=%.12f tol=%g\n", r_pbe.energy_ha,
                    want_e_ha, tol_e_ha);
        ok = false;
    }

    if (!ok)
        return 1;
    std::printf("PASS: pw-spin-gga (mag + true spin-GGA energy + LSDA gap)\n");
    return 0;
}
