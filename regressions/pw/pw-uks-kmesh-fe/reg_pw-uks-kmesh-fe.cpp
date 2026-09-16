/// @file regressions/pw/pw-uks-kmesh-fe/reg_pw-uks-kmesh-fe.cpp
/// @brief END-TO-END: k-point collinear UKS plane-wave (O2 multi-k).
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

namespace {

cpu::Frame o2_frame() {
    cpu::Frame f(2);
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_type)[i] = 8;
        f.span(kAtom_mass)[i] = 15.999;
        f.span(kAtom_x)[i] = 0.0;
        f.span(kAtom_y)[i] = 0.0;
    }
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_z)[1] = 1.2075; // Å
    f.box = cpu::Box::cubic(5.5);
    return f;
}

/// @brief One multi-k collinear UKS solve at the fixed case settings.
PWResult solve(const char* upf, cpu::scf::XCType xc) {
    auto frame = o2_frame();
    PW pw;
    pw.upf_path = upf;
    pw.ecut_ry = 5.0;
    pw.k_mesh = 2; // multi-k collinear UKS (this slug)
    pw.tol = 2e-3;
    pw.max_iter = 20;
    pw.mix_beta = 0.5;
    pw.pulay_depth = 3;
    pw.xc = xc;
    pw.spin_polarized = true;
    pw.multiplicity = 3;
    pw.build(frame);
    return pw.run(frame);
}

} // namespace

int main() {
    // ── Gates (REFERENCE.md) ───────────────────────────────────────────
    // 1. Magnetization: O2 ³Σg⁻, multiplicity 3 → fixed occupations M = 2.
    //    Literature identity (Herzberg); exact, XC-independent.
    constexpr double want_mag = 2.0;
    constexpr double tol_mag = 1e-9;
    // 2. The run must genuinely be multi-k: |{k}| = k_mesh³.
    constexpr int want_nk = 8;
    // 3. XC activation floor: `xc = PBE` must reach the collinear solver as a
    //    true spin-GGA, i.e. differ from the local (LSDA) answer by more than
    //    a solver wobble. Before 2026-07-26 `XCBuilder::step_spin` evaluated
    //    every functional at σ = 0, so a PBE request silently ran the LDA
    //    limit — with NO gate that could see it. This floor is that gate.
    //    It is an activation threshold, not a frozen value.
    constexpr double xc_gap_floor_ha = 1.0e-3;
    //
    // There is deliberately NO absolute total-energy pin: O2 at E_cut = 5 Ry
    // in a 5.5 Å box has no citable external value, and the number this case
    // used to carry was an ATV self-freeze recorded on the σ = 0 path (see
    // REFERENCE.md, "Void pin").

    // Compiled-in absolute path (ATV_REG_DATA_DIR): resolves identically from
    // the build tree, the source tree, or a bare ./reg_<case>.
    const char* upf = ATV_REG_DATA_DIR "/pw/O_ONCV_PBE-1.2.upf";
    if (std::FILE* f = std::fopen(upf, "r")) {
        std::fclose(f);
    } else {
        std::printf("FAIL: cannot open %s\n", upf);
        return 1;
    }

    const PWResult gga = solve(upf, cpu::scf::XCType::PBE);
    const PWResult lda = solve(upf, cpu::scf::XCType::LSDA);

    std::printf("PBE spin-GGA: E=%.10f Ha  mag=%.6f nk=%d conv=%d it=%d\n",
                gga.energy_ha, gga.magnetization, gga.n_kpts,
                (int)gga.converged, gga.iterations);
    std::printf("LSDA        : E=%.10f Ha  mag=%.6f nk=%d conv=%d it=%d\n",
                lda.energy_ha, lda.magnetization, lda.n_kpts,
                (int)lda.converged, lda.iterations);

    bool ok = true;
    for (const auto* r : {&gga, &lda}) {
        if (!std::isfinite(r->energy_ha) || !std::isfinite(r->energy)) {
            std::printf("FAIL: non-finite energy\n");
            ok = false;
        }
        if (!r->converged) {
            std::printf("FAIL: SCF not converged\n");
            ok = false;
        }
        if (r->n_kpts != want_nk) {
            std::printf("FAIL: n_kpts=%d want=%d (case must be multi-k)\n",
                        r->n_kpts, want_nk);
            ok = false;
        }
        if (std::fabs(r->magnetization - want_mag) > tol_mag) {
            std::printf("FAIL: mag got=%.12f want=%.12f tol=%g\n",
                        r->magnetization, want_mag, tol_mag);
            ok = false;
        }
    }

    const double xc_gap = std::fabs(gga.energy_ha - lda.energy_ha);
    std::printf("|E_PBE - E_LSDA| = %.6e Ha (floor %.1e)\n", xc_gap,
                xc_gap_floor_ha);
    if (!(xc_gap > xc_gap_floor_ha)) {
        std::printf("FAIL: spin-GGA inactive — PBE collapsed onto the local "
                    "(σ=0) answer\n");
        ok = false;
    }

    if (!ok)
        return 1;
    std::printf("PASS: pw-uks-kmesh-fe (multi-k UKS: mag identity, k-mesh, "
                "spin-GGA activation)\n");
    return 0;
}
