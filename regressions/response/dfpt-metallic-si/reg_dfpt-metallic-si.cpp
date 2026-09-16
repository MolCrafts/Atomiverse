/// @file regressions/response/dfpt-metallic-si/reg_dfpt-metallic-si.cpp
/// @brief END-TO-END: Γ-point DFPT phonons for diamond Si with Fermi–Dirac
///        metallic occupation smearing (PW LSDA).
///
/// Gates (all pins in REFERENCE.md):
///   1. acoustic modes ≈ 0 with NO acoustic-sum-rule imposition,
///   2. optical triplet inside a literature-style LDA window,
///   3. near-exact optical degeneracy and force-constant symmetry,
///   4. diagonal force constant vs the OFFLINE central second difference
///      of the smeared free energy at identical settings,
///   5. row-sum (translational invariance) of the DFPT matrix.
///
/// Public API only. Requires ATV_BUILD_PW. Exit non-zero on failure.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/dfpt_phonon.h"
#include "atomiverse/cpu/planewave/pw_solver.h"

#ifndef ATV_PW_UPF_PATH
#error "ATV_PW_UPF_PATH must be set to tests/data/pw/Si_lda_nc.upf"
#endif

using namespace ATV;
using namespace ATV::cpu::pw;

int main() {
    // ── Pinned acceptance numbers (see REFERENCE.md) ──
    // Literature anchor: DFPT-LDA Si optical Γ ~ 517 cm⁻¹ (Giannozzi et al.).
    // Minimal twin: ecut = 10 Ry, 2×2×2 MP, a = 5.43 Å, Fermi–Dirac smear.
    constexpr double kOpticalLoCm = 500.0;
    constexpr double kOpticalHiCm = 650.0;
    constexpr double kOpticalSpreadCm = 10.0;
    constexpr double kAcousticMaxCm = 30.0;
    // Offline free-energy FD pin: C[(0x),(0x)] = (F(+h) − 2F0 + F(−h))/h²,
    // h = 0.03 Å (REFERENCE.md). Matches fixed-occ twin at these settings
    // (gapped fill, near-integer FD occupations).
    constexpr double kC00FdPinEvA2 = 17.2393;
    constexpr double kC00TolEvA2 = 0.30;
    constexpr double kAsrMaxEvA2 = 0.05;
    constexpr double kAsymMaxEvA2 = 1e-5;

    constexpr double kA = 5.43; // Å
    constexpr double kEcutRy = 10.0;
    constexpr int kMesh = 2;
    constexpr double kDegaussRy = 0.02;

    cpu::Frame frame(2);
    frame.span(kAtom_type)[0] = 14;
    frame.span(kAtom_type)[1] = 14;
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

    PW method;
    method.upf_path = ATV_PW_UPF_PATH;
    method.ecut_ry = kEcutRy;
    method.k_mesh = kMesh;
    method.xc = cpu::scf::XCType::LSDA;
    method.smearing = Smearing::fermi_dirac;
    method.degauss = kDegaussRy;
    // Gapped Si twin: extra empty states are optional for FD fill (near
    // integer occupations). Keep 0 so the Sternheimer manifold matches
    // fixed-occ valence; true multi-band metals need denser k/ecut.
    method.extra_bands = 0;
    method.tol = 1e-8;
    method.max_iter = 200;

    DfptPhonon dfpt;
    dfpt.response_tol = 1e-5;
    dfpt.cg_tol = 1e-9;
    dfpt.mix_beta = 0.7;
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
    for (int k = 0; k < 6; ++k)
        if (!std::isfinite(r.frequencies(k))) {
            std::printf("FAIL: non-finite frequency[%d]\n", k);
            return 1;
        }

    for (int k = 0; k < 3; ++k) {
        if (std::abs(r.frequencies(k)) > kAcousticMaxCm) {
            std::printf("FAIL: acoustic |omega[%d]|=%.2f cm^-1 > %.1f\n", k,
                        std::abs(r.frequencies(k)), kAcousticMaxCm);
            return 1;
        }
    }
    for (int k = 3; k < 6; ++k) {
        if (r.frequencies(k) < kOpticalLoCm
            || r.frequencies(k) > kOpticalHiCm) {
            std::printf(
                "FAIL: optical omega[%d]=%.2f cm^-1 outside [%.0f, %.0f]\n", k,
                r.frequencies(k), kOpticalLoCm, kOpticalHiCm);
            return 1;
        }
    }
    const double spread = r.frequencies(5) - r.frequencies(3);
    if (spread > kOpticalSpreadCm) {
        std::printf("FAIL: optical triplet spread %.2f cm^-1 > %.1f\n", spread,
                    kOpticalSpreadCm);
        return 1;
    }
    if (r.max_asymmetry > kAsymMaxEvA2) {
        std::printf("FAIL: force-constant asymmetry %.3e eV/A^2 > %.1e\n",
                    r.max_asymmetry, kAsymMaxEvA2);
        return 1;
    }
    const double c00 = r.force_constants(0, 0);
    if (std::abs(c00 - kC00FdPinEvA2) > kC00TolEvA2) {
        std::printf("FAIL: C[(0x),(0x)] got=%.4f want=%.4f +/- %.2f eV/A^2 "
                    "(offline free-energy FD pin)\n",
                    c00, kC00FdPinEvA2, kC00TolEvA2);
        return 1;
    }
    double asr_max = 0;
    for (int i = 0; i < 6; ++i)
        for (int b = 0; b < 3; ++b) {
            const double s =
                r.force_constants(i, b) + r.force_constants(i, 3 + b);
            asr_max = std::max(asr_max, std::abs(s));
        }
    if (asr_max > kAsrMaxEvA2) {
        std::printf("FAIL: DFPT row-sum (ASR) max %.4f eV/A^2 > %.3f\n",
                    asr_max, kAsrMaxEvA2);
        return 1;
    }

    std::printf(
        "PASS: DFPT Si Gamma FD-smear LSDA  acoustic "
        "(%.2f, %.2f, %.2f) cm^-1  optical (%.2f, %.2f, %.2f) cm^-1 in "
        "[%.0f, %.0f]  spread %.2f  C00 %.4f vs FD pin %.4f (tol %.2f)  "
        "ASRmax %.4f  asym %.2e\n",
        r.frequencies(0), r.frequencies(1), r.frequencies(2), r.frequencies(3),
        r.frequencies(4), r.frequencies(5), kOpticalLoCm, kOpticalHiCm, spread,
        c00, kC00FdPinEvA2, kC00TolEvA2, asr_max, r.max_asymmetry);
    return 0;
}
