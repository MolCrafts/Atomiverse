/// @file regressions/pw/pw-spin-polarized-o2/reg_pw-spin-polarized-o2.cpp
/// @brief END-TO-END: PW collinear spin-polarized O2 triplet (e_tot, mag).
///
/// See REFERENCE.md. Public API only. Requires ATV_BUILD_PW.

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

namespace {} // namespace

int main() {
    // ── Offline pins (REFERENCE.md) ────────────────────────────────────
    // Magnetization: O2 ³Σg⁻, multiplicity 3 → fixed occupations M = 2
    // (Herzberg molecular spectrum; standard computational triplet test).
    constexpr double want_mag = 2.0;
    constexpr double tol_mag = 1e-9;
    // Total energy [Ha] at the exact settings below (SG15 ONCV O, ecut=12 Ry,
    // Γ, L=8 Å, R=1.2075 Å, multiplicity 3, default xc=PBE = true collinear
    // spin-GGA). See REFERENCE.md for oracle note.
    // Re-pinned 2026-07-26 (`pw-spin-gga`): XCBuilder::step_spin uses spectral
    // ∇ρ_σ (true spin-GGA); previous −28.6267 Ha was PBE@σ=0 LSDA-limit.
    constexpr double want_e_ha = -28.7585210635;
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

    PW pw;
    pw.upf_path = upf;
    pw.ecut_ry = 12.0;
    pw.k_mesh = 1;
    pw.tol = 5e-4;
    pw.max_iter = 60;
    pw.mix_beta = 0.3;
    pw.pulay_depth = 5;
    pw.spin_polarized = true;
    pw.multiplicity = 3;
    // Default xc=PBE → true collinear spin-GGA (pw-spin-gga product path).
    pw.xc = cpu::scf::XCType::PBE;

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
    return 0;
}
