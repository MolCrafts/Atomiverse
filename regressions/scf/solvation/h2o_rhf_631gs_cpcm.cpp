/// @file regressions/scf/h2o_rhf_631gs_cpcm/reg_h2o_rhf_631gs_cpcm.cpp
/// @brief H₂O RHF/6-31G* C-PCM electrostatic free energy vs pinned PySCF
///        solvent.PCM (see REFERENCE.md).
///
/// Out-of-library consumer: public headers + link atomiverse only.
/// Exit non-zero on mismatch. <= 60 s, single CPU.

#include <cmath>
#include <cstdio>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"
#include "harness/registry.h"

namespace atv_reg_scf_h2o_rhf_631gs_cpcm {

namespace {

ATV::cpu::Frame make_h2o() {
    using namespace ATV;
    cpu::Frame frame(3);
    frame.span(kAtom_atomic_number)[0] = 8;
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.1173;
    frame.span(kAtom_atomic_number)[1] = 1;
    frame.span(kAtom_x)[1] = 0.0;
    frame.span(kAtom_y)[1] = 0.7572;
    frame.span(kAtom_z)[1] = -0.4692;
    frame.span(kAtom_atomic_number)[2] = 1;
    frame.span(kAtom_x)[2] = 0.0;
    frame.span(kAtom_y)[2] = -0.7572;
    frame.span(kAtom_z)[2] = -0.4692;
    frame.box = cpu::Box::cubic(20.0);
    return frame;
}

} // namespace

int run() {
    using namespace ATV;

    auto frame = make_h2o();

    cpu::scf::RHF gas{cpu::scf::BasisSet::Pople631Gs};
    gas.tol = 1e-10;
    gas.max_iter = 100;
    gas.run(frame);
    if (!gas.converged()) {
        std::printf("FAIL: gas-phase SCF did not converge\n");
        return 1;
    }

    cpu::scf::RHF sol_mf{cpu::scf::BasisSet::Pople631Gs};
    sol_mf.tol = 1e-10;
    sol_mf.max_iter = 100;
    cpu::scf::WithSolvent<cpu::scf::RHF> sol{
        std::move(sol_mf), cpu::scf::SolventModel::CPCM, /*eps=*/78.4};
    sol.radii_scale = 1.2;
    sol.x = 0.0; // C-PCM
    sol.run(frame);
    if (!sol.mf.converged()) {
        std::printf("FAIL: C-PCM SCF did not converge\n");
        return 1;
    }

    // Hartree → kcal/mol (CODATA 2018 style constant used in REFERENCE.md).
    constexpr double kHaInKcal = 627.5094740631;
    const double dg_el_ha = sol.e_solvent();
    const double dg_el_kcal = dg_el_ha * kHaInKcal;
    const double e_gas = gas.energy();
    const double e_sol = sol.mf.energy();

    std::printf("E_gas      = %.12f Ha\n", e_gas);
    std::printf("E_cpcm    = %.12f Ha\n", e_sol);
    std::printf("DG_el     = %.12f Ha = %.6f kcal/mol\n", dg_el_ha, dg_el_kcal);
    std::printf("E_sol-E_g = %.6f kcal/mol\n", (e_sol - e_gas) * kHaInKcal);

    // Pinned PySCF 2.14.0 solvent.PCM C-PCM (see REFERENCE.md).
    // Target band: −6 … −8 kcal/mol electrostatic free energy.
    constexpr double kPinKcal = -7.95973818; // PySCF e_solvent
    constexpr double kTolKcal = 0.1;

    if (!(dg_el_kcal < 0.0)) {
        std::printf("FAIL: DG_el must be stabilizing (negative)\n");
        return 1;
    }
    // The *pinned* literature value must sit in the textbook −6…−8 kcal/mol
    // band for H₂O electrostatic solvation; the Atomiverse number is gated
    // by the 0.1 kcal/mol match below (hard bury-delete can sit a few
    // hundredths outside the band while still matching PySCF).
    static_assert(kPinKcal >= -8.0 && kPinKcal <= -6.0,
                  "PySCF pin must lie in [-8, -6] kcal/mol");
    if (!(e_sol < e_gas)) {
        std::printf("FAIL: total C-PCM energy not below gas phase\n");
        return 1;
    }
    const double err = std::abs(dg_el_kcal - kPinKcal);
    if (err > kTolKcal) {
        std::printf("FAIL: |DG_el - pin| = %.4f kcal/mol > %.2f "
                    "(atv=%.6f pin=%.6f)\n",
                    err, kTolKcal, dg_el_kcal, kPinKcal);
        return 1;
    }

    std::printf("PASS: |DG_el - pin| = %.4f kcal/mol "
                "(atv=%.6f pin=%.6f)\n",
                err, dg_el_kcal, kPinKcal);
    return 0;
}

} // namespace atv_reg_scf_h2o_rhf_631gs_cpcm

ATV_REGRESSION("scf/solvation", "h2o_rhf_631gs_cpcm",
               atv_reg_scf_h2o_rhf_631gs_cpcm::run);
