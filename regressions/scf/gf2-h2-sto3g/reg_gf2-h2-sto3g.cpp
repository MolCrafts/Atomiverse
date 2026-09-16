/// @file regressions/scf/gf2-h2-sto3g/reg_gf2-h2-sto3g.cpp
/// @brief H₂ RHF/STO-3G one-shot GF2 (G0F2) quasiparticle gap vs PySCF pin.
///
/// Public-API consumer only. Geometry, basis, and oracle numbers are pinned
/// in REFERENCE.md (PySCF 2.14.0 offline; no live import in ctest).
///
/// Observable: HOMO–LUMO QP gap [eV] (and IP / EA) from `RHF::gf2()` —
/// one-shot second-order Green's function (PySCF `agf2.RAGF2` one-shot twin).
#include <cmath>
#include <cstdio>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

namespace {

bool check(const char* what, double got, double want, double tol) {
    if (!std::isfinite(got) || std::abs(got - want) > tol) {
        std::fprintf(stderr, "FAIL: %s got=%.12f want=%.12f tol=%.1e\n", what,
                     got, want, tol);
        return false;
    }
    std::printf("  %-36s %14.8f (want %14.8f +/- %.1e)\n", what, got, want,
                tol);
    return true;
}

} // namespace

int main() {
    using namespace ATV;

    // Geometry [Å] — REFERENCE.md (bond on +z).
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_x)[1] = 0.0;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = 0.74;
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 1;
    f.box = cpu::Box::cubic(20.0);

    // PySCF 2.14.0 one-shot AGF2 / G0F2 pins (REFERENCE.md).
    constexpr double kERhfHa = -1.116759307396;
    constexpr double kGapEv = 34.717288266166; // primary gate
    constexpr double kIpEv = 16.098898012595;
    constexpr double kEaEv = 18.618390253572;
    constexpr double kQpHomoHa = -0.591623589738;
    constexpr double kQpLumoHa = 0.684213221821;
    constexpr double kECorrHa = -0.013138073590; // Galitskii–Migdal = MP2
    constexpr double kTolGapEv = 1e-4;           // job gate
    constexpr double kTolEHa = 1e-6;
    constexpr double kTolQpHa = 1e-5;

    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-12;
    rhf.max_iter = 100;
    rhf.run(f);
    if (!rhf.converged()) {
        std::fprintf(stderr, "FAIL: RHF did not converge\n");
        return 1;
    }

    std::printf("H2 RHF/STO-3G one-shot GF2 (G0F2):\n");
    if (!check("E_RHF [Ha]", rhf.energy(), kERhfHa, kTolEHa))
        return 1;

    const cpu::scf::GF2Result gf = rhf.gf2();
    if (!gf.converged) {
        std::fprintf(stderr,
                     "FAIL: GF2 Dyson spectrum did not yield HOMO/LUMO\n");
        return 1;
    }
    if (gf.n_occ != 1 || gf.qp_energy_ha.size() != 2) {
        std::fprintf(stderr, "FAIL: unexpected MO counts n_occ=%d n_mo=%zu\n",
                     gf.n_occ, gf.qp_energy_ha.size());
        return 1;
    }

    std::printf("  QP HOMO = %.10f Ha  LUMO = %.10f Ha\n", gf.qp_energy_ha[0],
                gf.qp_energy_ha[1]);
    std::printf("  gap = %.8f eV  IP = %.8f eV  EA = %.8f eV\n", gf.gap_ev,
                gf.ip_ev, gf.ea_ev);
    std::printf("  e_corr (GM) = %.12f Ha\n", gf.e_corr);

    if (!check("QP gap [eV]", gf.gap_ev, kGapEv, kTolGapEv))
        return 1;
    if (!check("IP [eV]", gf.ip_ev, kIpEv, kTolGapEv))
        return 1;
    if (!check("EA [eV]", gf.ea_ev, kEaEv, kTolGapEv))
        return 1;
    if (!check("QP HOMO [Ha]", gf.qp_energy_ha[0], kQpHomoHa, kTolQpHa))
        return 1;
    if (!check("QP LUMO [Ha]", gf.qp_energy_ha[1], kQpLumoHa, kTolQpHa))
        return 1;
    if (!check("e_corr [Ha]", gf.e_corr, kECorrHa, kTolEHa))
        return 1;

    // Gap consistency with Ha values via CODATA constant.
    const double gap_from_ha =
        (gf.qp_energy_ha[1] - gf.qp_energy_ha[0]) * units::kHartreeInEv;
    if (!check("gap from Ha [eV]", gap_from_ha, gf.gap_ev, 1e-9))
        return 1;

    // Physical sanity: GF2 gap is finite and positive.
    if (!(gf.gap_ev > 0.0) || !std::isfinite(gf.gap_ev)) {
        std::fprintf(stderr, "FAIL: GF2 gap %.6f eV not positive/finite\n",
                     gf.gap_ev);
        return 1;
    }

    std::printf("PASS\n");
    return 0;
}
