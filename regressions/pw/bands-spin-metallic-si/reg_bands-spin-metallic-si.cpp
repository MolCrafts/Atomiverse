/// @file regressions/pw/bands-spin-metallic-si/reg_bands-spin-metallic-si.cpp
/// @brief END-TO-END: collinear spin + metallic-smear PW Si band structure.
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
#include "atomiverse/cpu/planewave/qe_units.h"
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
    // Twin settings — REFERENCE.md (spin-si + metallic-si combo).
    constexpr double kA = 5.43;
    constexpr double kEcutRy = 15.0;
    constexpr int kMeshScf = 2;
    constexpr int kNBands = 8;
    constexpr int kNOcc = 4; // Si: n_e = 8, M seed 0 → N↑ = N↓ = 4 integer-fill
    constexpr double kDegaussRy = 0.02;
    constexpr int kExtraBands = 4;
    constexpr double kEFermiMaxRy = 1.0e3;
    constexpr double kMagTol = 1e-3;
    constexpr double kAbTolRy = 1e-3;
    // Yin & Cohen PRB 26, 5668 (1982): LDA indirect ~0.45 eV.
    constexpr double kGapLoEv = 0.05;
    constexpr double kGapHiEv = 2.50;

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
    method.smearing = Smearing::fermi_dirac;
    method.degauss = kDegaussRy;
    method.extra_bands = kExtraBands;

    Bands bands;
    bands.n_bands = kNBands;
    bands.set_diamond_path(4);

    const BandsResult br = bands.compute(method, frame);

    if (!br.scf_converged) {
        std::printf(
            "FAIL: spin+metallic bands SCF did not converge energy_ry=%g\n",
            br.energy_ry);
        return 1;
    }
    if (br.nspin != 2) {
        std::printf("FAIL: nspin got=%d want=2\n", br.nspin);
        return 1;
    }
    if (!(std::abs(br.magnetization) <= kMagTol)) {
        std::printf("FAIL: magnetization got=%.3e want |M|<=%.1e\n",
                    br.magnetization, kMagTol);
        return 1;
    }
    if (br.eigenvalues_ry.empty() || br.eigenvalues_beta_ry.empty()
        || br.kfrac.size() != br.eigenvalues_ry.size()
        || br.eigenvalues_ry.size() != br.eigenvalues_beta_ry.size()) {
        std::printf("FAIL: empty or mismatched α/β path eigenvalues\n");
        return 1;
    }

    bool has_G = false, has_X = false;
    const std::array<double, 3> G{0, 0, 0};
    const std::array<double, 3> X{0.5, 0.0, 0.5};
    int ik_g = -1, ik_x = -1;
    for (std::size_t ik = 0; ik < br.kfrac.size(); ++ik) {
        if (near_k(br.kfrac[ik], G)) {
            has_G = true;
            if (ik_g < 0)
                ik_g = static_cast<int>(ik);
        }
        if (near_k(br.kfrac[ik], X)) {
            has_X = true;
            if (ik_x < 0)
                ik_x = static_cast<int>(ik);
        }
    }
    if (!has_G || !has_X || ik_g < 0 || ik_x < 0) {
        std::printf("FAIL: path missing Gamma and/or X (G=%d X=%d)\n",
                    has_G ? 1 : 0, has_X ? 1 : 0);
        return 1;
    }

    for (std::size_t ik = 0; ik < br.eigenvalues_ry.size(); ++ik) {
        for (int spin = 0; spin < 2; ++spin) {
            const auto& row = (spin == 0) ? br.eigenvalues_ry[ik]
                                          : br.eigenvalues_beta_ry[ik];
            if (static_cast<int>(row.size()) != kNBands) {
                std::printf("FAIL: ik=%zu spin=%d n_bands got=%zu want=%d\n",
                            ik, spin, row.size(), kNBands);
                return 1;
            }
            if (!sorted_asc(row)) {
                std::printf(
                    "FAIL: eigenvalues not sorted ascending at ik=%zu spin=%d\n",
                    ik, spin);
                return 1;
            }
            if (!(row[static_cast<std::size_t>(kNOcc)]
                  > row[static_cast<std::size_t>(kNOcc - 1)])) {
                std::printf(
                    "FAIL: local gap closed at ik=%zu spin=%d e_v=%.6f e_c=%.6f\n",
                    ik, spin, row[static_cast<std::size_t>(kNOcc - 1)],
                    row[static_cast<std::size_t>(kNOcc)]);
                return 1;
            }
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

    const auto& eg_a = br.eigenvalues_ry[static_cast<std::size_t>(ik_g)];
    const auto& ex_a = br.eigenvalues_ry[static_cast<std::size_t>(ik_x)];
    const auto& eg_b = br.eigenvalues_beta_ry[static_cast<std::size_t>(ik_g)];
    const auto& ex_b = br.eigenvalues_beta_ry[static_cast<std::size_t>(ik_x)];

    const double e_v_g_a = eg_a[static_cast<std::size_t>(kNOcc - 1)];
    const double e_c_g_a = eg_a[static_cast<std::size_t>(kNOcc)];
    const double e_c_x_a = ex_a[static_cast<std::size_t>(kNOcc)];
    const double e_v_g_b = eg_b[static_cast<std::size_t>(kNOcc - 1)];
    const double e_c_x_b = ex_b[static_cast<std::size_t>(kNOcc)];

    const double gap_gx_ev = (e_c_x_a - e_v_g_a) * qe_units::rydberg_to_ev;
    if (!(gap_gx_ev >= kGapLoEv && gap_gx_ev <= kGapHiEv)) {
        std::printf("FAIL: Gamma->X gap_ev (α) got=%.6f want [%.2f, %.2f] eV "
                    "(Yin–Cohen LDA ~0.45 eV; see REFERENCE.md)\n",
                    gap_gx_ev, kGapLoEv, kGapHiEv);
        return 1;
    }
    if (!(e_c_x_a < e_c_g_a)) {
        std::printf(
            "FAIL: e_c^α(X)=%.6f Ry not below e_c^α(Γ)=%.6f Ry (Si indirect)\n",
            e_c_x_a, e_c_g_a);
        return 1;
    }

    const double d_vbm = std::abs(e_v_g_a - e_v_g_b);
    const double d_cbm = std::abs(e_c_x_a - e_c_x_b);
    if (!(d_vbm <= kAbTolRy && d_cbm <= kAbTolRy)) {
        std::printf("FAIL: α≈β |Δε| VBM_Γ=%.3e CBM_X=%.3e Ry want <= %.1e\n",
                    d_vbm, d_cbm, kAbTolRy);
        return 1;
    }

    std::printf(
        "PASS: Si collinear spin+metallic bands a=%.2f A ecut=%.0f Ry scf_k=%d  "
        "FD degauss=%.3f Ry extra_bands=%d  nspin=%d mag=%.3e  "
        "e_fermi=%.6f Ry  gap=%.4f Ry  e_vbm=%.6f e_cbm=%.6f  "
        "Gamma->X(α)=%.4f eV  e_c^α(X)<e_c^α(G)  "
        "|α−β|_VBM=%.2e |α−β|_CBM=%.2e Ry  nk_path=%zu  E_scf=%.6f Ry\n",
        kA, kEcutRy, kMeshScf, kDegaussRy, kExtraBands, br.nspin,
        br.magnetization, br.e_fermi_ry, br.gap_ry, br.e_vbm_ry, br.e_cbm_ry,
        gap_gx_ev, d_vbm, d_cbm, br.kfrac.size(), br.energy_ry);
    return 0;
}
