/// @file regressions/pw/bands-spin-si/reg_bands-spin-si.cpp
/// @brief END-TO-END: collinear spin-polarized PW Si band structure.
///
/// See REFERENCE.md. Public API only. Requires ATV_BUILD_PW.
/// Exit non-zero on failure with got / want / tol.

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
    constexpr double kA = 5.43;
    constexpr double kEcutRy = 15.0;
    constexpr int kMeshScf = 2;
    constexpr int kNBands = 8;
    constexpr int kNOcc = 4; // Si: n_e = 8, M = 0 → N↑ = N↓ = 4
    // Yin & Cohen PRB 26, 5668 (1982): LDA indirect ~0.45 eV.
    constexpr double kGapLoEv = 0.05;
    constexpr double kGapHiEv = 2.50;
    constexpr double kMagTol = 1e-6;
    constexpr double kAbTolRy = 5e-4;

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
    method.xc = cpu::scf::XCType::LSDA;
    method.spin_polarized = true;
    method.multiplicity = 1;

    Bands bands;
    bands.n_bands = kNBands;
    bands.set_diamond_path(4);
    const BandsResult br = bands.compute(method, frame);

    if (!br.scf_converged) {
        std::printf("FAIL: spin bands SCF did not converge energy_ry=%g\n",
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
        || static_cast<int>(br.eigenvalues_ry[0].size()) != kNBands
        || static_cast<int>(br.eigenvalues_beta_ry[0].size()) != kNBands
        || br.eigenvalues_ry.size() != br.eigenvalues_beta_ry.size()) {
        std::printf("FAIL: expected %d α/β bands per path k\n", kNBands);
        return 1;
    }
    for (const auto& row : br.eigenvalues_ry) {
        if (!sorted_asc(row)) {
            std::printf("FAIL: α eigenvalues not sorted at a path k\n");
            return 1;
        }
    }
    for (const auto& row : br.eigenvalues_beta_ry) {
        if (!sorted_asc(row)) {
            std::printf("FAIL: β eigenvalues not sorted at a path k\n");
            return 1;
        }
    }

    // Local same-k gaps positive for both spins.
    double min_local_ev = 1e300;
    for (std::size_t ik = 0; ik < br.eigenvalues_ry.size(); ++ik) {
        for (int spin = 0; spin < 2; ++spin) {
            const auto& row =
                (spin == 0) ? br.eigenvalues_ry[ik] : br.eigenvalues_beta_ry[ik];
            const double lg =
                (row[static_cast<std::size_t>(kNOcc)]
                 - row[static_cast<std::size_t>(kNOcc - 1)])
                * qe_units::rydberg_to_ev;
            if (lg < min_local_ev)
                min_local_ev = lg;
        }
    }
    if (!(min_local_ev > 0.0)) {
        std::printf("FAIL: min local gap_ev got=%.4f want > 0\n", min_local_ev);
        return 1;
    }

    const std::array<double, 3> G{0, 0, 0};
    const std::array<double, 3> X{0.5, 0.0, 0.5};
    int ik_g = -1, ik_x = -1;
    for (std::size_t ik = 0; ik < br.kfrac.size(); ++ik) {
        if (near_k(br.kfrac[ik], G) && ik_g < 0)
            ik_g = static_cast<int>(ik);
        if (near_k(br.kfrac[ik], X) && ik_x < 0)
            ik_x = static_cast<int>(ik);
    }
    if (ik_g < 0 || ik_x < 0) {
        std::printf("FAIL: path missing Γ or X (ik_g=%d ik_x=%d)\n", ik_g,
                    ik_x);
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
        "PASS: Si collinear spin bands a=%.2f A ecut=%.0f Ry scf_k=%d  "
        "nspin=%d mag=%.3e  Gamma->X(α)=%.4f eV (window [%.2f,%.2f])  "
        "e_c^α(X)<e_c^α(G)  |α−β|_VBM=%.2e |α−β|_CBM=%.2e Ry  "
        "min_local=%.3f eV  path_pts=%zu  E_scf=%.6f Ry\n",
        kA, kEcutRy, kMeshScf, br.nspin, br.magnetization, gap_gx_ev, kGapLoEv,
        kGapHiEv, d_vbm, d_cbm, min_local_ev, br.kfrac.size(), br.energy_ry);
    return 0;
}
