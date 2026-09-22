/// @file regressions/pw/bands-metallic-si/reg_bands-metallic-si.cpp
/// @brief END-TO-END: PW metallic-occupation band structure on diamond Si.
///
/// See REFERENCE.md. Public API only. Requires ATV_BUILD_PW.
/// Exit non-zero on failure with got / want / tol.

#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

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

bool near_k(const std::array<double, 3>& a, const std::array<double, 3>& b,
            double tol = 1e-8) {
    return std::abs(a[0] - b[0]) < tol && std::abs(a[1] - b[1]) < tol
           && std::abs(a[2] - b[2]) < tol;
}

bool sorted_asc(const std::vector<double>& row) {
    for (std::size_t b = 1; b < row.size(); ++b) {
        if (row[b] + 1e-10 < row[b - 1])
            return false;
    }
    return true;
}

} // namespace

int main() {
    // Twin settings — REFERENCE.md (same geometry class as bands-dos-si /
    // dos-metallic-si, with Fermi–Dirac occupations on the bands path).
    constexpr double kA = 5.43;
    constexpr double kEcutRy = 15.0;
    constexpr int kMeshScf = 2;
    constexpr int kNBands = 8;
    constexpr int kNOcc = 4; // Si: n_e = 8 → 4 occupied bands (integer fill)
    constexpr double kDegaussRy = 0.02;
    constexpr int kExtraBands = 4;
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

    Bands bands;
    bands.n_bands = kNBands;
    bands.set_diamond_path(4);

    const BandsResult br = bands.compute(method, frame);

    if (!br.scf_converged) {
        std::printf("FAIL: metallic bands SCF did not converge energy_ry=%g\n",
                    br.energy_ry);
        return 1;
    }
    if (br.nspin != 1) {
        std::printf("FAIL: nspin got=%d want=1\n", br.nspin);
        return 1;
    }
    if (br.eigenvalues_ry.empty()
        || br.kfrac.size() != br.eigenvalues_ry.size()) {
        std::printf("FAIL: empty or mismatched path eigenvalues\n");
        return 1;
    }

    bool has_G = false, has_X = false;
    const std::array<double, 3> G{0, 0, 0};
    const std::array<double, 3> X{0.5, 0.0, 0.5};
    for (const auto& k : br.kfrac) {
        if (near_k(k, G))
            has_G = true;
        if (near_k(k, X))
            has_X = true;
    }
    if (!has_G || !has_X) {
        std::printf("FAIL: path missing Gamma and/or X (G=%d X=%d)\n",
                    has_G ? 1 : 0, has_X ? 1 : 0);
        return 1;
    }

    for (std::size_t ik = 0; ik < br.eigenvalues_ry.size(); ++ik) {
        const auto& row = br.eigenvalues_ry[ik];
        if (static_cast<int>(row.size()) != kNBands) {
            std::printf("FAIL: ik=%zu n_bands got=%zu want=%d\n", ik,
                        row.size(), kNBands);
            return 1;
        }
        if (!sorted_asc(row)) {
            std::printf("FAIL: eigenvalues not sorted ascending at ik=%zu\n",
                        ik);
            return 1;
        }
        // Local same-k gap for integer valence fill (Si twin stays gapped).
        if (!(row[static_cast<std::size_t>(kNOcc)]
              > row[static_cast<std::size_t>(kNOcc - 1)])) {
            std::printf("FAIL: local gap closed at ik=%zu e_v=%.6f e_c=%.6f\n",
                        ik, row[static_cast<std::size_t>(kNOcc - 1)],
                        row[static_cast<std::size_t>(kNOcc)]);
            return 1;
        }
    }

    if (!(br.gap_ry > 0.0)) {
        std::printf("FAIL: expected open integer-fill gap gap_ry=%g\n",
                    br.gap_ry);
        return 1;
    }
    if (!(std::isfinite(br.e_fermi_ry)
          && std::abs(br.e_fermi_ry) < kEFermiMaxRy)) {
        std::printf("FAIL: e_fermi_ry got=%g want finite |E_F| < %g Ry\n",
                    br.e_fermi_ry, kEFermiMaxRy);
        return 1;
    }
    if (!(br.e_fermi_ry >= br.e_vbm_ry - 4.0 * kDegaussRy
          && br.e_fermi_ry <= br.e_cbm_ry + 4.0 * kDegaussRy)) {
        std::printf(
            "FAIL: e_fermi_ry=%.6f not near [e_vbm,e_cbm]=[%.6f,%.6f] ±4σ\n",
            br.e_fermi_ry, br.e_vbm_ry, br.e_cbm_ry);
        return 1;
    }

    std::printf("PASS: metallic bands Si a=%.2f A ecut=%.0f Ry scf_k=%d  "
                "FD degauss=%.3f Ry extra_bands=%d  e_fermi=%.6f Ry  "
                "gap=%.4f Ry  e_vbm=%.6f e_cbm=%.6f  nk_path=%zu  "
                "E_scf=%.6f Ry\n",
                kA, kEcutRy, kMeshScf, kDegaussRy, kExtraBands, br.e_fermi_ry,
                br.gap_ry, br.e_vbm_ry, br.e_cbm_ry, br.kfrac.size(),
                br.energy_ry);
    return 0;
}
