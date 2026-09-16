/// @file regressions/response/dfpt-finite-q-smearing/reg_dfpt-finite-q-smearing.cpp
/// @brief END-TO-END: finite-q DFPT phonons at X for diamond Si with
///        Fermi–Dirac metallic occupation smearing (PW LSDA).
///
/// Commensurate path: q = X = (1/2,0,0) → 2×1×1 supercell Γ-DFPT (smear-
/// aware) + IFC Fourier. Gapped Si + FD keeps near-integer occupations so
/// spectra track the fixed-occ twin (`dfpt-finite-q-si`) within windows.
///
/// Public API only. Requires ATV_BUILD_PW. Exit non-zero on failure.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

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
    // Sibling fixed-occ: Giannozzi et al., PRB 43, 7231 (1991) LDA Si at X.
    // Minimal twin (10 Ry, k=1 on 2×1×1 SC, FD 0.02 Ry) — same windows as
    // reg_dfpt-finite-q-si (gapped fill ⇒ F ≈ E).
    constexpr double kLo3Min = 50.0;
    constexpr double kLo3Max = 400.0;
    constexpr double kHi3Min = 300.0;
    constexpr double kHi3Max = 850.0;
    constexpr double kAsymMaxEvA2 = 1e-3;
    constexpr int kMaxImag = 2;

    constexpr double kA = 5.43;
    constexpr double kEcutRy = 10.0;
    constexpr int kMesh = 1;
    constexpr double kDegaussRy = 0.02;

    cpu::Frame frame(2);
    frame.span(kAtom_type)[0] = 14;
    frame.span(kAtom_type)[1] = 14;
    frame.span(kAtom_mass)[0] = 28.0855;
    frame.span(kAtom_mass)[1] = 28.0855;
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
    method.xc = cpu::scf::XCType::LSDA;
    method.smearing = Smearing::fermi_dirac;
    method.degauss = kDegaussRy;
    // Gapped Si twin: extra empty states optional for FD fill.
    method.extra_bands = 0;
    method.tol = 1e-8;
    method.max_iter = 200;

    DfptPhonon dfpt;
    dfpt.q_frac = {0.5, 0.0, 0.0};
    dfpt.response_tol = 1e-4;
    dfpt.response_max_iter = 120;
    dfpt.cg_tol = 1e-8;
    dfpt.mix_beta = 0.25;
    const DfptPhononResult r = dfpt.compute(method, frame);

    if (!r.scf_converged) {
        std::printf("FAIL: ground-state SCF not converged\n");
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
            std::printf(
                "FAIL: low |omega|[%d]=%.2f outside [%.0f, %.0f]\n", k,
                abs_w[static_cast<std::size_t>(k)], kLo3Min, kLo3Max);
            return 1;
        }
    }
    for (int k = 3; k < 6; ++k) {
        if (abs_w[static_cast<std::size_t>(k)] < kHi3Min
            || abs_w[static_cast<std::size_t>(k)] > kHi3Max) {
            std::printf(
                "FAIL: high |omega|[%d]=%.2f outside [%.0f, %.0f]\n", k,
                abs_w[static_cast<std::size_t>(k)], kHi3Min, kHi3Max);
            return 1;
        }
    }

    if (r.max_asymmetry > kAsymMaxEvA2) {
        std::printf("FAIL: asymmetry %.3e eV/A^2 > %.1e\n", r.max_asymmetry,
                    kAsymMaxEvA2);
        return 1;
    }

    std::printf(
        "PASS: DFPT Si X FD-smear  omega (%.2f, %.2f, %.2f | %.2f, %.2f, "
        "%.2f) cm^-1  |ω|sorted (%.0f,%.0f,%.0f|%.0f,%.0f,%.0f)  n_imag=%d  "
        "asym %.2e\n",
        r.frequencies(0), r.frequencies(1), r.frequencies(2), r.frequencies(3),
        r.frequencies(4), r.frequencies(5), abs_w[0], abs_w[1], abs_w[2],
        abs_w[3], abs_w[4], abs_w[5], n_imag, r.max_asymmetry);
    return 0;
}
