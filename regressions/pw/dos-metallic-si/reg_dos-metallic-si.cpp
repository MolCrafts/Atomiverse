/// @file regressions/pw/dos-metallic-si/reg_dos-metallic-si.cpp
/// @brief END-TO-END: PW metallic-occupation DOS on diamond Si (FD smear).
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
    // Twin settings — REFERENCE.md (same geometry class as bands-dos-si /
    // dfpt-metallic-si, with Fermi–Dirac occupations on the DOS path).
    constexpr double kA = 5.43;
    constexpr double kEcutRy = 15.0;
    constexpr int kMeshScf = 2;
    constexpr int kNBands = 8;
    constexpr double kDegaussRy = 0.02;
    constexpr int kExtraBands = 4;
    constexpr double kNelec = 8.0;
    constexpr double kNelecTol = 1.0;
    constexpr double kEFermiMaxRy = 1.0e3;

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
    method.smearing = Smearing::fermi_dirac;
    method.degauss = kDegaussRy;
    method.extra_bands = kExtraBands;

    Dos dos;
    dos.k_mesh = 4;
    dos.n_bands = kNBands;
    dos.sigma_ry = 0.05;
    dos.n_bins = 200;

    const DosResult dr = dos.compute(method, frame);

    if (!dr.scf_converged) {
        std::printf(
            "FAIL: metallic DOS SCF did not converge scf_energy_ry=%g\n",
            dr.scf_energy_ry);
        return 1;
    }
    if (!(std::isfinite(dr.e_fermi_ry)
          && std::abs(dr.e_fermi_ry) < kEFermiMaxRy)) {
        std::printf("FAIL: e_fermi_ry got=%g want finite |E_F| < %g Ry\n",
                    dr.e_fermi_ry, kEFermiMaxRy);
        return 1;
    }
    if (dr.dos.empty() || dr.energy_ry.size() != dr.dos.size()) {
        std::printf("FAIL: empty or mismatched DOS grid\n");
        return 1;
    }
    if (!(std::isfinite(dr.n_elec_to_vbm)
          && std::abs(dr.n_elec_to_vbm - kNelec) <= kNelecTol)) {
        std::printf(
            "FAIL: n_elec_to_E_F (n_elec_to_vbm) got=%.4f want=%.1f ± %.1f\n",
            dr.n_elec_to_vbm, kNelec, kNelecTol);
        return 1;
    }
    // Gapped Si + FD: occupations stay near-integer; gap still open on mesh.
    if (!(dr.gap_ry > 0.0)) {
        std::printf("FAIL: expected open gap on Si twin gap_ry=%g\n",
                    dr.gap_ry);
        return 1;
    }
    // E_F sits in the gap (or near VBM/CBM within broadening scale).
    if (!(dr.e_fermi_ry >= dr.e_vbm_ry - 4.0 * kDegaussRy
          && dr.e_fermi_ry <= dr.e_cbm_ry + 4.0 * kDegaussRy)) {
        std::printf(
            "FAIL: e_fermi_ry=%.6f not near [e_vbm,e_cbm]=[%.6f,%.6f] ±4σ\n",
            dr.e_fermi_ry, dr.e_vbm_ry, dr.e_cbm_ry);
        return 1;
    }

    std::printf("PASS: metallic DOS Si a=%.2f A ecut=%.0f Ry scf_k=%d  "
                "FD degauss=%.3f Ry extra_bands=%d  e_fermi=%.6f Ry  "
                "gap=%.4f Ry  n_elec_to_E_F=%.3f (ne=%.0f±%.0f)  "
                "E_scf=%.6f Ry  bins=%zu\n",
                kA, kEcutRy, kMeshScf, kDegaussRy, kExtraBands, dr.e_fermi_ry,
                dr.gap_ry, dr.n_elec_to_vbm, kNelec, kNelecTol,
                dr.scf_energy_ry, dr.dos.size());
    return 0;
}
