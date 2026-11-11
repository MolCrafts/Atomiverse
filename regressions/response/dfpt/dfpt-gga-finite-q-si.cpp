/// @file regressions/response/dfpt-gga-finite-q-si/reg_dfpt-gga-finite-q-si.cpp
/// @brief END-TO-END: finite-q DFPT phonons at X for diamond Si with the
///        pure GGA (PBE) exchange-correlation kernel (PW method).
///
/// The two parent products pin each axis alone (`dfpt-finite-q-si`:
/// commensurate X supercell path, LSDA kernel; `dfpt-gga-si`: Γ with the
/// true PBE kernel). This case gates the JOINT path: q = X = (1/2,0,0) →
/// 2×1×1 supercell Γ-DFPT + IFC Fourier where K_xc is the central FD of the
/// same GGA `XCBuilder` operator the ground-state SCF converged.
///
/// Gates: existing external frequency windows, finite/converged response,
/// asymmetry and runtime q-pattern curvature supplement. No native frozen
/// constants are oracles.
///
/// Public API only. Requires ATV_BUILD_PW. Exit non-zero on failure.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/dfpt_phonon.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "finite_q_checks.h"
#include "harness/registry.h"

namespace atv_reg_response_dfpt_gga_finite_q_si {

using namespace ATV;
using namespace ATV::cpu::pw;

int run() {
    const auto section_start = std::chrono::steady_clock::now();
    // ── Pinned acceptance numbers (see REFERENCE.md) ──
    // CP2K 2026.1 supercell FD twin (X branch): TA 133.7 (×2), LA 373.8,
    // LO 417.9, TO 462.0 (×2) cm⁻¹. ATV minimal twin (10 Ry NC-UPF, k=1
    // supercell) is biased the same way as the LSDA parent:
    // |ω| ≈ 108, 108, 239, 451, 669, 669 cm⁻¹ with a soft TA pair.
    constexpr double kLo3Min = 50.0;
    constexpr double kLo3Max = 400.0;
    constexpr double kHi3Min = 300.0;
    constexpr double kHi3Max = 850.0;
    constexpr int kMaxImag = 2;
    constexpr double kAsymMaxEvA2 = 1e-4;
    // Same-q runtime consistency supplements; external frequency gates
    // remain the independent oracle. No frozen native force constants.
    constexpr double kC00TolEvA2 = 0.30;

    constexpr double kA = 5.43; // Å
    constexpr double kEcutRy = 10.0;
    constexpr int kMesh = 1;

    cpu::Frame frame(2);
    frame.span(kAtom_atomic_number)[0] = 14;
    frame.span(kAtom_atomic_number)[1] = 14;
    frame.span(kAtom_mass)[0] = 28.0855;
    frame.span(kAtom_mass)[1] = 28.0855;
    frame.span(kAtom_x)[1] = kA / 4.0;
    frame.span(kAtom_y)[1] = kA / 4.0;
    frame.span(kAtom_z)[1] = kA / 4.0;
    const double half = kA / 2.0;
    Mat3 cell;
    cell(0, 0) = 0;
    cell(0, 1) = half;
    cell(0, 2) = half;
    cell(1, 0) = half;
    cell(1, 1) = 0;
    cell(1, 2) = half;
    cell(2, 0) = half;
    cell(2, 1) = half;
    cell(2, 2) = 0;
    frame.box.set_cell(cell);

    PW method{{std::make_shared<UpfPseudopotential>(ATV_REG_DATA_DIR
                                                    "/pw/Si_lda_nc.upf")},
              kEcutRy,
              kMesh,
              cpu::xc::XCType::PBE};
    method.tol = 1e-8;
    method.max_iter = 200;

    DfptPhonon dfpt;
    dfpt.q_frac = {0.5, 0.0, 0.0}; // X → commensurate 2×1×1 supercell
    dfpt.response_tol = 1e-4;
    dfpt.response_max_iter = 120;
    dfpt.cg_tol = 1e-8;
    dfpt.mix_beta = 0.25;
    const DfptPhononResult r = dfpt.compute(method, frame);

    if (!r.scf_converged || !r.response_converged) {
        std::printf("FAIL: convergence scf=%d response=%d\n",
                    r.scf_converged ? 1 : 0, r.response_converged ? 1 : 0);
        return 1;
    }
    if (r.frequencies.size() != 6) {
        std::printf("FAIL: expected 6 modes, got %zu\n",
                    static_cast<std::size_t>(r.frequencies.size()));
        return 1;
    }

    int n_imag = 0;
    std::vector<double> abs_w(6);
    for (int k = 0; k < 6; ++k) {
        if (!std::isfinite(r.frequencies(k))) {
            std::printf("FAIL: non-finite frequency[%d]\n", k);
            return 1;
        }
        if (r.frequencies(k) < 0.0)
            ++n_imag;
        abs_w[static_cast<std::size_t>(k)] = std::abs(r.frequencies(k));
    }
    if (n_imag > kMaxImag) {
        std::printf("FAIL: too many imaginary modes (%d > %d)\n", n_imag,
                    kMaxImag);
        return 1;
    }
    std::sort(abs_w.begin(), abs_w.end());

    for (int k = 0; k < 3; ++k) {
        if (abs_w[static_cast<std::size_t>(k)] < kLo3Min
            || abs_w[static_cast<std::size_t>(k)] > kLo3Max) {
            std::printf("FAIL: low |omega|[%d]=%.2f outside [%.0f, %.0f]\n", k,
                        abs_w[static_cast<std::size_t>(k)], kLo3Min, kLo3Max);
            return 1;
        }
    }
    for (int k = 3; k < 6; ++k) {
        if (abs_w[static_cast<std::size_t>(k)] < kHi3Min
            || abs_w[static_cast<std::size_t>(k)] > kHi3Max) {
            std::printf("FAIL: high |omega|[%d]=%.2f outside [%.0f, %.0f]\n", k,
                        abs_w[static_cast<std::size_t>(k)], kHi3Min, kHi3Max);
            return 1;
        }
    }

    if (r.max_asymmetry > kAsymMaxEvA2) {
        std::printf("FAIL: force-constant asymmetry %.3e eV/A^2 > %.1e\n",
                    r.max_asymmetry, kAsymMaxEvA2);
        return 1;
    }

    const double c00 = r.force_constants(0, 0);
    const double curvature =
        atv_dfpt_finite_q_checks::x_pattern_curvature(method, frame);
    if (!std::isfinite(c00) || std::abs(c00 - curvature) > kC00TolEvA2) {
        std::printf(
            "FAIL: Re C00(X)=%.8f q-pattern curvature=%.8f tol=%.2f eV/Ang^2\n",
            c00, curvature, kC00TolEvA2);
        return 1;
    }
    const double wall = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - section_start)
                            .count();
    std::printf("whole section got=%.6f want<=60 seconds\n", wall);
    if (wall > 60.0)
        return 1;
    std::printf(
        "PASS: dfpt-gga-finite-q-si external spectrum + Re C00(X) runtime "
        "consistency C00=%.8f curvature=%.8f tol=%.2f asym=%.3e\n",
        c00, curvature, kC00TolEvA2, r.max_asymmetry);
    return 0;
}

} // namespace atv_reg_response_dfpt_gga_finite_q_si

ATV_REGRESSION("response/dfpt", "dfpt-gga-finite-q-si",
               atv_reg_response_dfpt_gga_finite_q_si::run);
