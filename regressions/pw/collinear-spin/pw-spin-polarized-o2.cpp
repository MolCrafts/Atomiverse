/// @file regressions/pw/pw-spin-polarized-o2/reg_pw-spin-polarized-o2.cpp
/// @brief END-TO-END: PW collinear spin-polarized O2 triplet (e_tot, mag).
///
/// See REFERENCE.md. Public API only. Requires ATV_BUILD_PW.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/xc/xc_functional.h"
#include "harness/registry.h"

namespace atv_reg_pw_pw_spin_polarized_o2 {

using namespace ATV;
using namespace ATV::cpu::pw;

int run() {
    const auto start = std::chrono::steady_clock::now();
    // ── Offline pins (REFERENCE.md) ────────────────────────────────────
    // Magnetization: O2 ³Σg⁻, multiplicity 3 → fixed occupations M = 2
    // (Herzberg molecular spectrum; standard computational triplet test).
    constexpr double want_mag = 2.0;
    constexpr double tol_mag = 1e-9;
    // Actual QE 7.5 energy at the matching NC-UPF, 12/48 Ry cutoffs,
    // 36^3 density grid and triplet occupations. See REFERENCE.md.
    constexpr double want_e_ha = -28.7326768250;
    constexpr double tol_e_ha = 1.0e-3; // Ha — modest ecut / SCF tol

    // Compiled-in absolute path (ATV_REG_DATA_DIR): resolves identically from
    // the build tree, the source tree, or a bare ./reg_<case>.
    const char* upf = ATV_REG_DATA_DIR "/pw/O_ONCV_PBE-1.2.upf";
    if (std::FILE* f = std::fopen(upf, "r")) {
        std::fclose(f);
    } else {
        std::printf("FAIL: cannot open %s\n", upf);
        return 1;
    }

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

    PW pw{std::make_shared<UpfPseudopotential>(upf), 12.0, 1,
          cpu::xc::XCType::PBE};
    pw.tol = 1e-7;
    pw.max_iter = 200;
    pw.mix_beta = 0.3;
    pw.pulay_depth = 5;
    pw.spin_polarized = true;
    pw.multiplicity = 3;
    // Default xc=PBE → true collinear spin-GGA (pw-spin-gga product path).

    pw.build(frame);
    const auto& r = pw.run(frame);

    std::printf("E=%.10f Ha (%.10f Ry)  mag=%.6f  conv=%d iters=%d\n",
                r.energy_ha, r.energy, r.magnetization, (int)r.converged,
                r.iterations);

    bool ok = true;
    if (!std::isfinite(r.energy_ha) || !std::isfinite(r.energy)) {
        std::printf("FAIL: non-finite energy\n");
        ok = false;
    }
    if (!r.converged) {
        std::printf("FAIL: SCF not converged\n");
        ok = false;
    }
    if (std::fabs(r.magnetization - want_mag) > tol_mag) {
        std::printf("FAIL: mag got=%.12f want=%.12f tol=%g\n", r.magnetization,
                    want_mag, tol_mag);
        ok = false;
    }
    if (std::fabs(r.energy_ha - want_e_ha) > tol_e_ha) {
        std::printf("FAIL: E_ha got=%.12f want=%.12f tol=%g\n", r.energy_ha,
                    want_e_ha, tol_e_ha);
        ok = false;
    }

    if (!ok)
        return 1;
    std::printf("PASS: pw-spin-polarized-o2 (mag + energy)\n");
    const double wall =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    std::printf("whole section got=%.6f want<=60 seconds\n", wall);
    if (wall > 60.0)
        return 1;
    return 0;
}

} // namespace atv_reg_pw_pw_spin_polarized_o2

ATV_REGRESSION("pw/collinear-spin", "pw-spin-polarized-o2",
               atv_reg_pw_pw_spin_polarized_o2::run);
