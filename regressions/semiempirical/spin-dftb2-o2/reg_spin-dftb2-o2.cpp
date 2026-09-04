/// @file regressions/semiempirical/spin-dftb2-o2/reg_spin-dftb2-o2.cpp
/// @brief O₂ collinear spin-polarized SCC-DFTB2 — energy + Mulliken spin gate.
///
/// Method: collinear spin-DFTB2 (Elstner SCC + Köhler / DFTB+ spin channel).
/// oracle: dftb+ 24.1 (tools/tb_oracle/inputs/dftbplus/spin-dftb2-o2/).
/// Published PBE W via spin_w_ha; ShellResolvedSpin = No (HOAO collapse).
///
/// Exit non-zero on failure. Public API only. <= 60 s.
#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"

using namespace ATV;

namespace {

cpu::Frame make_o2(double r_ang = 1.21) {
    cpu::Frame f(2);
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 8;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_x)[1] = r_ang;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = 0.0;
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_mass)[i] = 15.999;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box::cubic(30.0);
    f.meta.get(kEnergy) = 0;
    return f;
}

} // namespace

int main() {
    auto frame = make_o2();

    cpu::Evaluate eval;
    cpu::semiempirical::DFTB2 dftb;
    // Isolated molecule in an isolation box — the DFTB+ twin is a `C`
    // geometry, so say so rather than letting a cutoff hide the images.
    dftb.boundary = cpu::semiempirical::DftbBoundary::cluster;
    dftb.charge = 0;
    dftb.multiplicity = 3; // ground-state triplet O₂
    dftb.spin_w_ha = cpu::semiempirical::basis_data::kDftbSpinConstantsPbeHa;
    dftb.with_forces = false;
    dftb.scc_tol = 1e-10;
    dftb.max_scc = 80;
    // Keep a reference to the component for Mulliken spin after run: register
    // via add_ref so ownership stays here.
    eval.on(evaluate::Phase::evaluate).add_ref(dftb);
    eval.build();
    eval.run(frame);

    const double e = frame.meta.get(kEnergy);
    std::printf("spin-DFTB2 O2 triplet: E=%.12f eV\n", e);
    if (!std::isfinite(e)) {
        std::printf("FAIL non-finite energy\n");
        return 1;
    }
    if (std::fabs(e) < 1e-3) {
        std::printf("FAIL energy near zero\n");
        return 1;
    }

    const auto& m = dftb.last_mulliken_spin();
    if (m.size() != 2) {
        std::printf("FAIL spin population size %zu\n", m.size());
        return 1;
    }
    const double m0 = m[0];
    const double m1 = m[1];
    const double m_tot = m0 + m1;
    std::printf("  m_Mulliken = [%.12f, %.12f]  sum=%.12f\n", m0, m1, m_tot);
    if (std::fabs(m_tot - 2.0) > 0.25) {
        std::printf("FAIL total spin population (expect ~2 for triplet)\n");
        return 1;
    }
    if (std::fabs(m0) < 0.3 || std::fabs(m1) < 0.3) {
        std::printf("FAIL per-atom spin too small\n");
        return 1;
    }
    if (std::fabs(m0 - m1) > 0.2) {
        std::printf("FAIL spin asymmetry on O2\n");
        return 1;
    }

    // E_spin from the public Mulliken spin; there is no last_e_spin().
    // W_pp(O, PBE) = -0.028 Ha = -0.761918814888 eV. O2 triplet m_A = 1
    // by symmetry, so E_spin = 1/2 Σ_A W m_A² = W.
    constexpr double kW_O_Pbe_eV = -0.761918814888;
    constexpr double kEspin_eV = -0.761918814888;
    constexpr double kEspinTol_eV = 1e-9;
    const double e_spin = 0.5 * kW_O_Pbe_eV * (m0 * m0 + m1 * m1);
    std::printf("  E_spin=0.5*W*(m0^2+m1^2)=%.12f eV (pin %.12f)\n", e_spin,
                kEspin_eV);
    if (std::fabs(e_spin - kEspin_eV) > kEspinTol_eV) {
        std::printf("FAIL E_spin pin\n");
        return 1;
    }

    // Total energy vs the superseded sketch-W ATV twin. E = Tr(ρH0)+E2+E_spin
    // +E_rep; O2 homonuclear ⇒ Δq≡0 ⇒ uniform spin shift leaves ρ unchanged,
    // so ΔE = ΔE_spin = +0.238081185112 eV exactly (1e-4 is numerical noise).
    constexpr double kE_old_eV = -177.2425823576;
    constexpr double kDeltaEspin_eV = 0.238081185112;
    constexpr double kE_pin_eV = -177.004501172488;
    constexpr double kDeltaTol_eV = 1e-4;
    const double shift = e - kE_old_eV;
    std::printf("  E-E_old=%.12f eV (expect ΔE_spin=%.12f, tol %g)\n", shift,
                kDeltaEspin_eV, kDeltaTol_eV);
    if (std::fabs(shift - kDeltaEspin_eV) > kDeltaTol_eV) {
        std::printf("FAIL |E_new - E_old - ΔE_spin|=%.12f eV > %g "
                    "(iron law 9 — do not widen; E_new=%.12f E_old=%.12f)\n",
                    std::fabs(shift - kDeltaEspin_eV), kDeltaTol_eV, e,
                    kE_old_eV);
        return 1;
    }
    const double de = std::fabs(e - kE_pin_eV);
    std::printf("  |E - pin|=%.6e eV (tol %g)\n", de, kDeltaTol_eV);
    if (de > kDeltaTol_eV) {
        std::printf("FAIL energy pin: got %.12f want %.12f\n", e, kE_pin_eV);
        return 1;
    }

    const auto& dq = dftb.last_mulliken_dq();
    if (dq.size() == 2) {
        std::printf("  dq_Mulliken = [%.6f, %.6f]\n", dq[0], dq[1]);
        if (!std::isfinite(dq[0]) || !std::isfinite(dq[1])) {
            std::printf("FAIL non-finite charge\n");
            return 1;
        }
    }

    std::printf("PASS\n");
    return 0;
}
