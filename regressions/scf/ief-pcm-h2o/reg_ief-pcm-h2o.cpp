/// @file regressions/scf/ief-pcm-h2o/reg_ief-pcm-h2o.cpp
/// @brief H₂O RHF/6-31G* IEF-PCM electrostatic free energy vs pinned PySCF
///        solvent.PCM (see REFERENCE.md).
///
/// Out-of-library consumer: public headers + link atomiverse only.
/// Exit non-zero on mismatch. <= 60 s, single CPU.

#include <cmath>
#include <cstdio>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

namespace {

ATV::cpu::Frame make_h2o() {
    using namespace ATV;
    cpu::Frame frame(3);
    frame.span(kAtom_type)[0] = 8;
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.1173;
    frame.span(kAtom_type)[1] = 1;
    frame.span(kAtom_x)[1] = 0.0;
    frame.span(kAtom_y)[1] = 0.7572;
    frame.span(kAtom_z)[1] = -0.4692;
    frame.span(kAtom_type)[2] = 1;
    frame.span(kAtom_x)[2] = 0.0;
    frame.span(kAtom_y)[2] = -0.7572;
    frame.span(kAtom_z)[2] = -0.4692;
    frame.box = cpu::Box::cubic(20.0);
    return frame;
}

} // namespace

int main() {
    using namespace ATV;

    auto frame = make_h2o();

    cpu::scf::RHF gas;
    gas.basis = cpu::scf::BasisSet::Pople631Gs;
    gas.tol = 1e-10;
    gas.max_iter = 100;
    gas.run(frame);
    if (!gas.converged()) {
        std::printf("FAIL: gas-phase SCF did not converge\n");
        return 1;
    }

    cpu::scf::RHF sol;
    sol.basis = cpu::scf::BasisSet::Pople631Gs;
    sol.tol = 1e-10;
    sol.max_iter = 100;
    sol.solvent_eps = 78.4;
    sol.solvent_radii_scale = 1.2;
    sol.solvent_model = cpu::scf::SolventModel::IEFPCM;
    sol.run(frame);
    if (!sol.converged()) {
        std::printf("FAIL: IEF-PCM SCF did not converge\n");
        return 1;
    }

    constexpr double kHaInKcal = 627.5094740631;
    const double e_solv = sol.e_solvent(); // ΔG_el [Hartree]
    const double e_gas = gas.energy();
    const double e_sol = sol.energy();

    std::printf("E_gas   = %.12f Ha\n", e_gas);
    std::printf("E_ief   = %.12f Ha\n", e_sol);
    std::printf("e_solv  = %.12f Ha = %.6f kcal/mol\n", e_solv,
                e_solv * kHaInKcal);
    std::printf("E_sol-E_g = %.6f kcal/mol\n", (e_sol - e_gas) * kHaInKcal);

    // Pinned PySCF 2.14.0 solvent.PCM IEF-PCM (see REFERENCE.md).
    constexpr double kPinHa = -0.01265803490853;
    // Hard bury-delete + point-charge AO potentials vs PySCF SWIG + int3c2e:
    // same class of residual as h2o_rhf_631gs_cpcm (~0.06 kcal/mol).
    constexpr double kTolHa = 1e-4;

    if (!(e_solv < 0.0)) {
        std::printf("FAIL: e_solv must be stabilizing (negative)\n");
        return 1;
    }
    if (!(e_sol < e_gas)) {
        std::printf("FAIL: total IEF-PCM energy not below gas phase\n");
        return 1;
    }
    const double err = std::abs(e_solv - kPinHa);
    if (err > kTolHa) {
        std::printf("FAIL: |e_solv - pin| = %.6e Ha > %.1e "
                    "(atv=%.12f pin=%.12f)\n",
                    err, kTolHa, e_solv, kPinHa);
        return 1;
    }

    std::printf("PASS: |e_solv - pin| = %.6e Ha "
                "(atv=%.12f pin=%.12f)\n",
                err, e_solv, kPinHa);
    return 0;
}
