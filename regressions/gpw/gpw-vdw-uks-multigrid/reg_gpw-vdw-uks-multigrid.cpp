/// @file regressions/gpw/gpw-vdw-uks-multigrid/reg_gpw-vdw-uks-multigrid.cpp
/// @brief Γ collinear-UKS nonlocal vdW-DF (DRSLL) on the REL_CUTOFF multigrid
///        (NGRIDS 4): O₂ triplet LSD vs offline CP2K 2026.1 matched twin.
///
/// Public-API consumer only. Product pin for the triple combo
/// (UKS × DRSLL × ngrids>1) already wired in build_uks_H: V_nl of the total
/// density is added to both spin potentials on the fine grid before per-level
/// restriction. Oracle numbers in REFERENCE.md (offline CP2K only; never live
/// in ctest).
///
/// Exit non-zero on failure. Wall ~5 min Release (8 threads) — SC nonlocal
/// RPS every SCF iter on O₂@400 Ry multigrid; family
/// `gpw-vdw-uks` ~196 s single-grid / this multigrid twin ~290 s.

#include <chrono>
#include <cmath>
#include <cstdio>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/evaluate/single_point.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

namespace {

// O₂, R = 1.2075 Å centered in a 12 Å cubic PERIODIC cell (Å) — same twin
// as gpw-vdw-uks / CP2K o2_uks_pbe_drsll_mg.inp.
ATV::cpu::Frame make_o2() {
    ATV::cpu::Frame frame(2);
    constexpr double L = 12.0;
    constexpr double R = 1.2075;
    for (int i = 0; i < 2; ++i) {
        frame.span(ATV::kAtom_type)[i] = 8;
        frame.span(ATV::kAtom_x)[i] = 0.5 * L;
        frame.span(ATV::kAtom_y)[i] = 0.5 * L;
    }
    frame.span(ATV::kAtom_z)[0] = 0.5 * L - 0.5 * R;
    frame.span(ATV::kAtom_z)[1] = 0.5 * L + 0.5 * R;
    frame.box = ATV::cpu::Box::cubic(L);
    return frame;
}

} // namespace

int main() {
    using namespace ATV;
    using clock = std::chrono::steady_clock;

    // Offline CP2K 2026.1 (git:5e54ba2) NGRIDS 4 / REL_CUTOFF 60 + DRSLL —
    // REFERENCE.md. Fair twin: ATV also uses ngrids=4 / rel_cutoff=60.
    constexpr double kEtotCp2k = -31.815654475057819;  // Ha
    constexpr double kEDispCp2k = 1.1311984819209e-01; // Ha (SC E_c^nl)
    constexpr double kTolEtot = 1.0e-3;                // Ha
    constexpr double kTolDisp = 1.0e-4;                // Ha
    // CP2K timer TOTAL TIME on the same twin job (1 MPI × OMP=8).
    constexpr double kCp2kWall = 34.247; // s

    const cpu::gpw::Kind O{
        molrs::Element::O,
        cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-GTH"),
        cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};

    cpu::Frame frame = make_o2();
    cpu::gpw::GPWEvaluator eval({O});
    eval.cutoff_ry = 400.0;
    eval.ngrids = 4;
    eval.rel_cutoff = 60.0;
    eval.max_iter = 200;
    eval.tol = 1e-7;
    eval.mix = 0.4;
    eval.xc = cpu::gpw::XcKind::PBE;
    eval.n_alpha = 7; // multiplicity 3, 12 valence e⁻
    eval.n_beta = 5;
    eval.vdw_df = true;

    const auto t0 = clock::now();
    cpu::single_point(frame, eval);
    const auto t1 = clock::now();
    const double wall_s = std::chrono::duration<double>(t1 - t0).count();

    const cpu::gpw::GpwResult& r = eval.last_result();
    const double e_ha = eval.last_energy_hartree();
    const double e_frame = frame.meta.get(kEnergy) / units::kHartreeInEv;

    std::printf("E(O2 UKS PBE+DRSLL NGRIDS=4) = %.12f Ha (CP2K %.12f, "
                "|dE|=%.3e, tol %.0e)\n",
                e_ha, kEtotCp2k, std::fabs(e_ha - kEtotCp2k), kTolEtot);
    std::printf("E_disp                       = %.12e Ha (CP2K %.12e, "
                "|dD|=%.3e, tol %.0e)\n",
                r.e_disp, kEDispCp2k, std::fabs(r.e_disp - kEDispCp2k),
                kTolDisp);
    std::printf("converged=%d iters=%d wall=%.1f s  "
                "T_atv/T_cp2k=%.2f (cp2k=%.1f s)\n",
                static_cast<int>(r.converged), r.iters, wall_s,
                wall_s / kCp2kWall, kCp2kWall);

    bool ok = r.converged && std::isfinite(e_ha) && std::isfinite(r.e_disp);
    ok = ok && std::fabs(e_ha - kEtotCp2k) <= kTolEtot;
    ok = ok && std::fabs(r.e_disp - kEDispCp2k) <= kTolDisp;
    ok = ok && std::fabs(e_frame - e_ha) < 1e-9;
    ok = ok && r.e_disp != 0.0; // E_c^nl actually folded on multigrid path

    if (!ok) {
        std::fprintf(stderr, "FAIL gpw-vdw-uks-multigrid\n");
        return 1;
    }
    std::printf("PASS gpw-vdw-uks-multigrid\n");
    return 0;
}
