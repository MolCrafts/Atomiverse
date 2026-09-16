/// @file regressions/pw/bands-dos-si/reg_bands-dos-si.cpp
/// @brief END-TO-END: PW Si band structure + DOS vs literature Γ→X gap.
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

} // namespace

int main() {
    constexpr double kA = 5.43;
    constexpr double kEcutRy = 15.0;
    constexpr int kMeshScf = 2;
    constexpr int kNBands = 8;
    constexpr int kNOcc = 4; // Si: n_e = 8
    // Yin & Cohen PRB 26, 5668 (1982): LDA indirect ~0.45 eV (Γ_v → near-X_c).
    // Twin uses tutorial ecut/k → window, not a freeze of Atomiverse-only output.
    constexpr double kGapLoEv = 0.05;
    constexpr double kGapHiEv = 2.50;
    constexpr double kNelec = 8.0;
    constexpr double kNelecTol = 1.0;

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
    method.max_iter = 50;
    method.xc = cpu::scf::XCType::LSDA;

    // ── Bands ──────────────────────────────────────────────────────────
    Bands bands;
    bands.n_bands = kNBands;
    bands.set_diamond_path(4);
    const BandsResult br = bands.compute(method, frame);

    if (!br.scf_converged) {
        std::printf("FAIL: bands SCF did not converge energy_ry=%g\n",
                    br.energy_ry);
        return 1;
    }
    if (br.eigenvalues_ry.empty()
        || static_cast<int>(br.eigenvalues_ry[0].size()) != kNBands) {
        std::printf("FAIL: expected %d bands per k\n", kNBands);
        return 1;
    }
    for (const auto& row : br.eigenvalues_ry) {
        for (std::size_t b = 1; b < row.size(); ++b) {
            if (row[b] + 1e-10 < row[b - 1]) {
                std::printf("FAIL: eigenvalues not sorted at a path k\n");
                return 1;
            }
        }
    }

    // All local (same-k) gaps positive — no inverted spectrum at a k-point.
    double min_local_ev = 1e300;
    for (const auto& row : br.eigenvalues_ry) {
        const double lg = (row[static_cast<std::size_t>(kNOcc)]
                           - row[static_cast<std::size_t>(kNOcc - 1)])
                          * qe_units::rydberg_to_ev;
        if (lg < min_local_ev)
            min_local_ev = lg;
    }
    if (!(min_local_ev > 0.0)) {
        std::printf("FAIL: min local gap_ev got=%.4f want > 0\n", min_local_ev);
        return 1;
    }

    // Locate Γ and X on the diamond path.
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

    const auto& eg = br.eigenvalues_ry[static_cast<std::size_t>(ik_g)];
    const auto& ex = br.eigenvalues_ry[static_cast<std::size_t>(ik_x)];
    const double e_v_g = eg[static_cast<std::size_t>(kNOcc - 1)];
    const double e_c_g = eg[static_cast<std::size_t>(kNOcc)];
    const double e_c_x = ex[static_cast<std::size_t>(kNOcc)];
    const double direct_g_ev = (e_c_g - e_v_g) * qe_units::rydberg_to_ev;
    // Textbook Si indirect transition (Γ valence → X conduction).
    const double gap_gx_ev = (e_c_x - e_v_g) * qe_units::rydberg_to_ev;

    if (!(gap_gx_ev >= kGapLoEv && gap_gx_ev <= kGapHiEv)) {
        std::printf("FAIL: Gamma->X gap_ev got=%.6f want [%.2f, %.2f] eV "
                    "(Yin–Cohen LDA ~0.45 eV; incomplete ecut/k window; see "
                    "REFERENCE.md)\n",
                    gap_gx_ev, kGapLoEv, kGapHiEv);
        return 1;
    }

    // Indirect character: CBM at X below conduction at Γ.
    if (!(e_c_x < e_c_g)) {
        std::printf(
            "FAIL: e_c(X)=%.6f Ry not below e_c(Γ)=%.6f Ry (Si indirect)\n",
            e_c_x, e_c_g);
        return 1;
    }
    // Direct Γ gap larger than Γ→X (Si).
    if (direct_g_ev + 0.05 < gap_gx_ev) {
        std::printf("FAIL: direct_Gamma=%.4f eV < Gamma->X=%.4f eV - 0.05\n",
                    direct_g_ev, gap_gx_ev);
        return 1;
    }

    // ── DOS ────────────────────────────────────────────────────────────
    Dos dos;
    dos.k_mesh = 4;
    dos.n_bands = kNBands;
    dos.sigma_ry = 0.05;
    dos.n_bins = 200;
    const DosResult dr = dos.compute(method, frame);
    if (!dr.scf_converged) {
        std::printf("FAIL: DOS SCF did not converge\n");
        return 1;
    }
    if (!(std::isfinite(dr.n_elec_to_vbm)
          && std::abs(dr.n_elec_to_vbm - kNelec) <= kNelecTol)) {
        std::printf("FAIL: n_elec_to_vbm got=%.4f want=%.1f ± %.1f\n",
                    dr.n_elec_to_vbm, kNelec, kNelecTol);
        return 1;
    }

    std::printf(
        "PASS: Si bands+DOS a=%.2f A ecut=%.0f Ry scf_k=%d  "
        "Gamma->X=%.4f eV (window [%.2f,%.2f]; Yin-Cohen ~0.45)  "
        "direct_G=%.3f eV  min_local=%.3f eV  e_c(X)<e_c(G)  "
        "n_elec_to_vbm=%.3f (ne=%.0f±%.0f)  path_pts=%zu  E_scf=%.6f Ry\n",
        kA, kEcutRy, kMeshScf, gap_gx_ev, kGapLoEv, kGapHiEv, direct_g_ev,
        min_local_ev, dr.n_elec_to_vbm, kNelec, kNelecTol, br.kfrac.size(),
        br.energy_ry);
    return 0;
}
