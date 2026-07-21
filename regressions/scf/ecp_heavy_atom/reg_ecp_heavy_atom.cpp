/// @file regressions/scf/ecp_heavy_atom/reg_ecp_heavy_atom.cpp
/// @brief Semi-local ECP heavy-element RHF (def2-SVP + def2-ECP): I2 and
///        AuH total energies vs externally pinned PySCF 2.14.0 values.
///
/// Out-of-library consumer: public headers + link atomiverse only. Heavy
/// elements replace their chemically inert core (I: 28, Au: 60 electrons)
/// by a semi-local effective core potential whose parameterization bakes in
/// scalar relativity (Kahn, Baybutt, Truhlar, J. Chem. Phys. 65, 3826
/// (1976), DOI 10.1063/1.432900; integrals per McMurchie, Davidson,
/// J. Comput. Phys. 44, 289 (1981)). def2-ECP pedigree: Weigend, Ahlrichs,
/// PCCP 7, 3297 (2005); I ECP28MWB: Bergner et al., Mol. Phys. 80, 1431
/// (1993); Au ECP60MWB: Andrae et al., Theor. Chim. Acta 77, 123 (1990).
///
/// Assertions (PySCF pins in REFERENCE.md; conv_tol = 1e-12, spherical —
/// the def2 sets run 5d/7f in both engines):
///   - I2  (R = 2.666 A): E = -593.3161818794323 Ha to 1e-6 Ha; 50 valence
///     electrons (2 x (53 - 28)).
///   - AuH (R = 1.524 A): E = -135.34256700791678 Ha to 1e-6 Ha; 20 valence
///     electrons (79 - 60 + 1).
///
/// Exit non-zero on mismatch, printing got / want / tol. <= 60 s, single CPU.

#include <cmath>
#include <cstdio>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

namespace {

/// One RHF run with basis def2-SVP + ecp def2-ECP; returns E_total [Ha].
double run_rhf(ATV::cpu::Frame frame) {
    ATV::cpu::scf::RHF rhf;
    rhf.basis = ATV::cpu::scf::BasisSet::Def2SVP;
    rhf.ecp = "def2-ECP";
    rhf.tol = 1e-12;
    rhf.max_iter = 200;
    rhf.run(frame);
    if (!rhf.converged()) {
        std::printf("FAIL: SCF did not converge (%d iterations)\n",
                    rhf.iterations());
        std::exit(1);
    }
    return rhf.energy();
}

bool check(const char* label, double got, double want, double tol) {
    const double diff = std::fabs(got - want);
    std::printf("%-24s got=%.12f want=%.12f |diff|=%.3e tol=%.0e\n", label,
                got, want, diff, tol);
    return diff < tol;
}

} // namespace

int main() {
    bool ok = true;

    // ── I2, R = 2.666 A along z ─────────────────────────────────────────
    ATV::cpu::Frame i2(2);
    i2.span(ATV::kAtom_type)[0] = 53;
    i2.span(ATV::kAtom_x)[0] = 0.0;
    i2.span(ATV::kAtom_y)[0] = 0.0;
    i2.span(ATV::kAtom_z)[0] = 0.0;
    i2.span(ATV::kAtom_type)[1] = 53;
    i2.span(ATV::kAtom_x)[1] = 0.0;
    i2.span(ATV::kAtom_y)[1] = 0.0;
    i2.span(ATV::kAtom_z)[1] = 2.666;
    i2.simbox = ATV::cpu::Box::cubic(20.0);
    ok &= check("E(I2, def2-SVP+ECP)", run_rhf(i2), -593.3161818794323, 1e-6);

    // ── AuH, R = 1.524 A along z ────────────────────────────────────────
    ATV::cpu::Frame auh(2);
    auh.span(ATV::kAtom_type)[0] = 79;
    auh.span(ATV::kAtom_x)[0] = 0.0;
    auh.span(ATV::kAtom_y)[0] = 0.0;
    auh.span(ATV::kAtom_z)[0] = 0.0;
    auh.span(ATV::kAtom_type)[1] = 1;
    auh.span(ATV::kAtom_x)[1] = 0.0;
    auh.span(ATV::kAtom_y)[1] = 0.0;
    auh.span(ATV::kAtom_z)[1] = 1.524;
    auh.simbox = ATV::cpu::Box::cubic(20.0);
    ok &= check("E(AuH, def2-SVP+ECP)", run_rhf(auh), -135.34256700791678,
                1e-6);

    if (!ok) {
        std::printf("FAIL: heavy-atom ECP regression off the PySCF oracle\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
