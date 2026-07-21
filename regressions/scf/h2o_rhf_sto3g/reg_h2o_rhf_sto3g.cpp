/// @file regressions/scf/h2o_rhf_sto3g/reg_h2o_rhf_sto3g.cpp
/// @brief H2O / STO-3G restricted Hartree-Fock total energy vs the cited
///        LibreTexts literature value — the chain's only Gaussian-SCF e2e.
///
/// Out-of-library consumer: includes only the public `cpu/scf/methods.h` +
/// `cpu/frame/frame.h` + `core/units.h` and links `atomiverse` (public API
/// only; no internal headers and no intermediate config/integral objects).
/// Builds water at R(OH) = 0.95 Å, ∠HOH = 104.5°, runs the concrete
/// `cpu::scf::RHF` method (parameters are members; Frame is the sole I/O
/// currency), and checks the Frame energy `frame.meta.get(kEnergy)` [eV],
/// converted to Hartree, against E_RHF/STO-3G(H2O) = −74.961754 Ha
/// (LibreTexts "Live Textbook of Physical Chemistry", §28.2; cross-checked
/// against PySCF 2.13) within 1e-3 Ha. See REFERENCE.md.
///
/// Exit non-zero on mismatch, printing got / want / tol. <= 60 s, single CPU.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

int main() {
    using namespace ATV;

    // ── Water geometry: O at the origin; the two H symmetric about +z in the
    //    xz-plane, R(OH) = 0.95 Å and ∠HOH = 104.5° (the LibreTexts §28.2
    //    STO-3G water example — the citation geometry).
    constexpr double kPi = 3.14159265358979323846;
    const double r_oh = 0.95;                      // Å
    const double half = 0.5 * 104.5 * kPi / 180.0; // half bond angle [rad]
    const double hx = r_oh * std::sin(half);
    const double hz = r_oh * std::cos(half);

    cpu::Frame frame(3);
    frame.span(kAtom_type)[0] = 8; // O (Z = 8)
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.0;
    frame.span(kAtom_type)[1] = 1; // H (Z = 1)
    frame.span(kAtom_x)[1] = hx;
    frame.span(kAtom_y)[1] = 0.0;
    frame.span(kAtom_z)[1] = hz;
    frame.span(kAtom_type)[2] = 1; // H (Z = 1)
    frame.span(kAtom_x)[2] = -hx;
    frame.span(kAtom_y)[2] = 0.0;
    frame.span(kAtom_z)[2] = hz;
    frame.simbox = cpu::Box::cubic(20.0);

    // ── Concrete RHF method, STO-3G. Parameters are members and Frame is the
    //    sole I/O currency. `run(frame)` writes the eV energy to
    //    `frame.meta[kEnergy]`; loop state via `converged()`.
    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.run(frame);

    // ── Literature check: total RHF/STO-3G energy [Hartree], read from the
    //    Frame (eV) and converted back to Hartree.
    constexpr double kEnergyRef = -74.961754; // Ha — LibreTexts PChem §28.2
    constexpr double kTol = 1.0e-3; // Ha — end-to-end sanity magnitude
    const double got = frame.meta.get(kEnergy) / units::kHartreeInEv;

    if (!rhf.converged()) {
        std::fprintf(stderr, "FAIL: RHF/STO-3G(H2O) did not converge\n");
        return 1;
    }
    if (!std::isfinite(got) || std::abs(got - kEnergyRef) > kTol) {
        std::fprintf(stderr,
                     "FAIL: E_RHF/STO-3G(H2O) got=%.6f want=%.6f tol=%.1e Ha\n",
                     got, kEnergyRef, kTol);
        return 1;
    }

    std::printf("PASS: E_RHF/STO-3G(H2O) = %.6f Ha (want %.6f +/- %.1e) "
                "[LibreTexts Live Textbook of Physical Chemistry, §28.2; "
                "PySCF 2.13 cross-check]\n",
                got, kEnergyRef, kTol);
    return 0;
}
