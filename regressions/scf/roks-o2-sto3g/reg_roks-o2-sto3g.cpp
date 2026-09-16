/// @file regressions/scf/roks-o2-sto3g/reg_roks-o2-sto3g.cpp
/// @brief O2 triplet ROKS/LSDA/STO-3G — restricted open-shell KS vs the
///        externally pinned PySCF 2.14.0 `dft.ROKS` total energy, plus the
///        exact-spin-purity identity of the single-MO-set solution.
///
/// Out-of-library consumer: public headers + link atomiverse only.
///   - E(ROKS-LSDA, O2 R = 1.2075 Å, (9,7), STO-3G cart) vs the PySCF pin
///     −147.193715448518 Ha to 1e-6 Ha (see REFERENCE.md for command).
///   - Spin purity: |⟨S²⟩ − S(S+1)| ≤ 1e-10 (S = 1 → ⟨S²⟩ = 2 exactly) —
///     Roothaan, Rev. Mod. Phys. 32, 179 (1960).
///
/// Exit non-zero on failure. <= 60 s, single CPU.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

namespace {

ATV::cpu::Frame o2_frame() {
    ATV::cpu::Frame frame(2);
    for (int i = 0; i < 2; ++i) {
        frame.span(ATV::kAtom_type)[i] = 8;
        frame.span(ATV::kAtom_mass)[i] = 15.999;
        frame.span(ATV::kAtom_x)[i] = 0.0;
        frame.span(ATV::kAtom_y)[i] = 0.0;
    }
    frame.span(ATV::kAtom_z)[0] = 0.0;
    frame.span(ATV::kAtom_z)[1] = 1.2075;
    frame.box = ATV::cpu::Box::cubic(20.0);
    return frame;
}

} // namespace

int main() {
    using namespace ATV;

    // ── O2 triplet, ROKS(9,7)/LSDA/STO-3G ─────────────────────────────
    auto frame = o2_frame();
    cpu::scf::ROKS roks(9, 7, cpu::scf::XCType::LSDA);
    roks.basis = cpu::scf::BasisSet::STO3G;
    roks.tol = 1e-12;
    roks.max_iter = 200;
    roks.run(frame);

    const double e_roks = roks.energy();
    // PySCF 2.14.0 pin [Ha] — REFERENCE.md / pin_scf.py roks --xc LDA,VWN
    constexpr double want = -147.193715448518;
    constexpr double kTol = 1.0e-6; // Ha
    std::printf("E(O2, ROKS-LSDA/STO-3G) = %.12f Ha (PySCF %.12f, dE = %.3e)\n",
                e_roks, want, e_roks - want);
    bool ok = roks.converged() && std::fabs(e_roks - want) < kTol;

    // Frame energy must track the method energy (eV ↔ Ha).
    const double from_frame = frame.meta.get(kEnergy) / units::kHartreeInEv;
    ok = ok && std::isfinite(from_frame)
         && std::fabs(from_frame - e_roks) < 1e-10;

    // Spin purity: the single-MO-set identity, S = 1 → ⟨S²⟩ = 2 exactly.
    const double contamination = roks.determinant().spin_contamination();
    std::printf("ROKS <S2> - S(S+1) = %.3e (want |.| <= 1e-10)\n",
                contamination);
    ok = ok && std::fabs(contamination) <= 1e-10;

    // Pure functional metadata: LSDA has no exact exchange.
    if (cpu::scf::hybrid_exchange_fraction(cpu::scf::XCType::LSDA) != 0.0) {
        std::fprintf(stderr, "FAIL: LSDA hybrid_exchange_fraction != 0\n");
        return 1;
    }

    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
