/// @file regressions/pw/pw-analytic-stress-si/reg_pw-analytic-stress-si.cpp
/// @brief END-TO-END: PW Nielsen–Martin analytic stress on diamond Si (NC-UPF).
///
/// Dual-gate vs offline strain-FD pin (REFERENCE.md). Public API only.
/// Requires ATV_BUILD_PW and ATV_PW_UPF_PATH.

#include <cmath>
#include <cstdio>
#include <string>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/scf/xc_functional.h"

#ifndef ATV_PW_UPF_PATH
#error "ATV_PW_UPF_PATH must be set to tests/data/pw/Si_lda_nc.upf"
#endif

using namespace ATV;
using namespace ATV::cpu::pw;

int main() {
    constexpr double kA = 5.43; // Å
    constexpr double kEcutRy = 15.0;
    constexpr int kMesh = 2;
    constexpr double kSymTol = 0.025; // eV/Å³ (soft: incomplete ecut/k)
    constexpr double kDualTol = 0.05; // eV/Å³ max |an − FD pin|

    // Offline multi-SCF strain-FD oracle (h=2e-3), same twin, measured
    // 2026-07-24 Release OMP=16 — see REFERENCE.md. Not a self-golden of
    // analytic output: re-measurable via pw_stress_fd when desired.
    // Mechanical / CP2K sign: σ = −(1/Ω)∂E/∂ε. Compressed a=5.43 vs a0≈5.47
    // → positive (compressive) hydrostatic stress. Offline multi-SCF FD pin
    // magnitude matches pre-product energy-derivative pin with a sign flip.
    constexpr double kFdPin[6] = {
        0.05755, 0.05755, 0.05755, // xx yy zz (isotropic within noise)
        0.0,     0.0,     0.0      // off-diag vanish by crystal symmetry
    };
    // Diagonal pin uses the mean of the three FD diagonals from the offline
    // run (~+0.0575 mechanical); off-diag pin is exact symmetry zero.

    cpu::Frame frame(2);
    frame.span(kAtom_type)[0] = 14;
    frame.span(kAtom_type)[1] = 14;
    frame.span(kAtom_mass)[0] = 28.0855;
    frame.span(kAtom_mass)[1] = 28.0855;
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.0;
    frame.span(kAtom_x)[1] = kA / 4.0;
    frame.span(kAtom_y)[1] = kA / 4.0;
    frame.span(kAtom_z)[1] = kA / 4.0;
    const double h = kA / 2.0;
    Mat3 cell;
    cell(0, 0) = 0;
    cell(0, 1) = h;
    cell(0, 2) = h;
    cell(1, 0) = h;
    cell(1, 1) = 0;
    cell(1, 2) = h;
    cell(2, 0) = h;
    cell(2, 1) = h;
    cell(2, 2) = 0;
    frame.box.set_cell(cell);

    PW method;
    method.upf_path = ATV_PW_UPF_PATH;
    method.ecut_ry = kEcutRy;
    method.k_mesh = kMesh;
    method.tol = 1e-6;
    method.max_iter = 50;
    method.xc = cpu::scf::XCType::LSDA;

    const auto san = pw_stress(frame, method);
    for (int v = 0; v < 6; ++v) {
        if (!std::isfinite(san[static_cast<std::size_t>(v)])) {
            std::printf("FAIL: non-finite analytic stress[%d]=%g\n", v,
                        san[static_cast<std::size_t>(v)]);
            return 1;
        }
    }

    // Cubic symmetry of diamond primitive cell stress.
    if (std::abs(san[3]) > kSymTol || std::abs(san[4]) > kSymTol
        || std::abs(san[5]) > kSymTol) {
        std::printf("FAIL: off-diag stress xy=%.6g xz=%.6g yz=%.6g tol=%.3g\n",
                    san[3], san[4], san[5], kSymTol);
        return 1;
    }
    if (std::abs(san[0] - san[1]) > kSymTol
        || std::abs(san[1] - san[2]) > kSymTol) {
        std::printf("FAIL: diag not equal xx=%.6g yy=%.6g zz=%.6g tol=%.3g\n",
                    san[0], san[1], san[2], kSymTol);
        return 1;
    }

    // Dual-gate vs offline FD pin (not live 12× SCF).
    double max_abs = 0.0;
    for (int v = 0; v < 6; ++v) {
        const double d = std::abs(san[static_cast<std::size_t>(v)] - kFdPin[v]);
        max_abs = std::max(max_abs, d);
    }
    if (max_abs > kDualTol) {
        std::printf("FAIL: dual-gate max|an-FDpin|=%.6g > tol=%.3g eV/A3\n"
                    "  an: %.6g %.6g %.6g | %.6g %.6g %.6g\n"
                    "  pin xx/yy/zz=%.5g off=0 (see REFERENCE.md)\n",
                    max_abs, kDualTol, san[0], san[1], san[2], san[3], san[4],
                    san[5], kFdPin[0]);
        return 1;
    }

    std::printf("PASS: PW analytic stress Si a=%.2f A ecut=%.0f k=%d  "
                "σ=%.6g %.6g %.6g | %.3g %.3g %.3g eV/A3  max|an-FDpin|=%.4g\n",
                kA, kEcutRy, kMesh, san[0], san[1], san[2], san[3], san[4],
                san[5], max_abs);
    return 0;
}
