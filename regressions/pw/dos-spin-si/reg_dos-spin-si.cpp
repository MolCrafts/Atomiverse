/// @file regressions/pw/dos-spin-si/reg_dos-spin-si.cpp
/// @brief END-TO-END: collinear spin-polarized PW DOS on diamond Si.
///
/// See REFERENCE.md. Public API only. Requires ATV_BUILD_PW.
/// Exit non-zero on failure with got / want / tol.

#include <cmath>
#include <cstdio>
#include <string>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/bands_dos.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/scf/xc_functional.h"

using namespace ATV;
using namespace ATV::cpu::pw;

namespace {

cpu::Frame make_si_diamond(double a_ang) {
    cpu::Frame frame(2);
    frame.span(kAtom_type)[0] = 14;
    frame.span(kAtom_type)[1] = 14;
    frame.span(kAtom_mass)[0] = 28.0855;
    frame.span(kAtom_mass)[1] = 28.0855;
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.0;
    frame.span(kAtom_x)[1] = a_ang / 4.0;
    frame.span(kAtom_y)[1] = a_ang / 4.0;
    frame.span(kAtom_z)[1] = a_ang / 4.0;
    const double h = a_ang / 2.0;
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
    return frame;
}

} // namespace

int main() {
    // Twin settings — REFERENCE.md (same geometry as bands-spin-si /
    // bands-dos-si; collinear M=0 + denser-mesh Gaussian DOS).
    constexpr double kA = 5.43;
    constexpr double kEcutRy = 15.0;
    constexpr int kMeshScf = 2;
    constexpr int kNBands = 8;
    constexpr double kNelec = 8.0;
    constexpr double kNelecTol = 1.0;
    constexpr double kMagTol = 1e-6;

    // Compiled-in absolute path (ATV_REG_DATA_DIR): resolves identically from
    // the build tree, the source tree, or a bare ./reg_<case>.
    const char* upf = ATV_REG_DATA_DIR "/pw/Si_lda_nc.upf";
    if (std::FILE* f = std::fopen(upf, "r")) {
        std::fclose(f);
    } else {
        std::printf("FAIL: cannot open %s\n", upf);
        return 1;
    }

    auto frame = make_si_diamond(kA);

    PW method;
    method.upf_path = upf;
    method.ecut_ry = kEcutRy;
    method.k_mesh = kMeshScf;
    method.tol = 1e-6;
    method.max_iter = 80;
    method.mix_beta = 0.5;
    method.xc = cpu::scf::XCType::LSDA;
    method.spin_polarized = true;
    method.multiplicity = 1;

    Dos dos;
    dos.k_mesh = 4;
    dos.n_bands = kNBands;
    dos.sigma_ry = 0.05;
    dos.n_bins = 200;

    const DosResult dr = dos.compute(method, frame);

    if (!dr.scf_converged) {
        std::printf("FAIL: spin DOS SCF did not converge scf_energy_ry=%g\n",
                    dr.scf_energy_ry);
        return 1;
    }
    if (dr.nspin != 2) {
        std::printf("FAIL: nspin got=%d want=2\n", dr.nspin);
        return 1;
    }
    if (!(std::abs(dr.magnetization) <= kMagTol)) {
        std::printf("FAIL: magnetization got=%.3e want |M|<=%.1e\n",
                    dr.magnetization, kMagTol);
        return 1;
    }
    if (dr.dos.empty() || dr.energy_ry.size() != dr.dos.size()) {
        std::printf("FAIL: empty or mismatched total DOS grid\n");
        return 1;
    }
    if (dr.dos_alpha.size() != dr.dos.size()
        || dr.dos_beta.size() != dr.dos.size()) {
        std::printf(
            "FAIL: per-spin DOS size alpha=%zu beta=%zu total=%zu want equal\n",
            dr.dos_alpha.size(), dr.dos_beta.size(), dr.dos.size());
        return 1;
    }
    if (!(std::isfinite(dr.n_elec_to_vbm)
          && std::abs(dr.n_elec_to_vbm - kNelec) <= kNelecTol)) {
        std::printf(
            "FAIL: n_elec_to_vbm got=%.4f want=%.1f ± %.1f\n",
            dr.n_elec_to_vbm, kNelec, kNelecTol);
        return 1;
    }
    if (!(dr.gap_ry > 0.0)) {
        std::printf("FAIL: expected open gap on Si twin gap_ry=%g\n",
                    dr.gap_ry);
        return 1;
    }

    // Total D ≈ Dα + Dβ on the shared energy grid (trap consistency).
    const double n_a =
        integrate_dos_to(dr.energy_ry, dr.dos_alpha, dr.e_vbm_ry);
    const double n_b =
        integrate_dos_to(dr.energy_ry, dr.dos_beta, dr.e_vbm_ry);
    if (!(std::abs((n_a + n_b) - dr.n_elec_to_vbm) <= 0.05)) {
        std::printf(
            "FAIL: n_alpha+n_beta=%.4f vs n_total=%.4f (tol 0.05)\n",
            n_a + n_b, dr.n_elec_to_vbm);
        return 1;
    }
    // M=0: each spin channel holds ~ half the electrons.
    if (!(std::abs(n_a - 0.5 * kNelec) <= kNelecTol
          && std::abs(n_b - 0.5 * kNelec) <= kNelecTol)) {
        std::printf(
            "FAIL: per-spin occupied integral n_a=%.4f n_b=%.4f want ~4 ± %.1f\n",
            n_a, n_b, kNelecTol);
        return 1;
    }

    std::printf(
        "PASS: collinear spin DOS Si a=%.2f A ecut=%.0f Ry scf_k=%d  "
        "nspin=%d mag=%.3e  gap=%.4f Ry  n_elec_to_vbm=%.3f "
        "(ne=%.0f±%.0f; n_a=%.3f n_b=%.3f)  E_scf=%.6f Ry  bins=%zu\n",
        kA, kEcutRy, kMeshScf, dr.nspin, dr.magnetization, dr.gap_ry,
        dr.n_elec_to_vbm, kNelec, kNelecTol, n_a, n_b, dr.scf_energy_ry,
        dr.dos.size());
    return 0;
}
