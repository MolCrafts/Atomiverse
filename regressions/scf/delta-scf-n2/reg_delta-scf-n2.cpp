/// @file regressions/scf/delta-scf-n2/reg_delta-scf-n2.cpp
/// @brief N₂ UHF/STO-3G ΔSCF (α HOMO→LUMO) with IMOM vs offline PySCF pin.
///
/// Public-API consumer only. Geometry, basis, and oracle numbers are pinned
/// in REFERENCE.md (PySCF 2.14.0 `scf.addons.mom_occ_`; no live import in
/// ctest).
///
/// Product path: `UHF::delta_scf(frame, from, to, MomStyle::IMOM)` — ground
/// UHF (RHF warm-start) + non-Aufbau re-SCF with initial maximum-overlap
/// occupation locking (Barca, Gilbert & Gill 2018; PySCF twin of Gilbert–
/// Besley–Gill MOM 2008 for the frozen-reference variant).

#include <cmath>
#include <cstdio>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

namespace {

ATV::cpu::Frame n2_frame() {
    ATV::cpu::Frame frame(2);
    for (int i = 0; i < 2; ++i) {
        frame.span(ATV::kAtom_type)[i] = 7;
        frame.span(ATV::kAtom_mass)[i] = 14.0067;
        frame.span(ATV::kAtom_x)[i] = 0.0;
        frame.span(ATV::kAtom_y)[i] = 0.0;
    }
    frame.span(ATV::kAtom_z)[0] = 0.0;
    frame.span(ATV::kAtom_z)[1] = 1.0977;
    frame.box = ATV::cpu::Box::cubic(20.0);
    return frame;
}

bool check(const char* what, double got, double want, double tol) {
    if (!std::isfinite(got) || std::abs(got - want) > tol) {
        std::fprintf(stderr, "FAIL: %s got=%.12f want=%.12f tol=%.1e\n", what,
                     got, want, tol);
        return false;
    }
    std::printf("  %-28s %14.8f (want %14.8f +/- %.1e)\n", what, got, want,
                tol);
    return true;
}

} // namespace

int main() {
    using namespace ATV;

    auto frame = n2_frame();

    // Closed-shell N2: 14 e → n_α = n_β = 7. HOMO=6, LUMO=7 (0-based).
    cpu::scf::UHF uhf(/*n_alpha=*/7, /*n_beta=*/7);
    uhf.basis = cpu::scf::BasisSet::STO3G;
    uhf.tol = 1e-12;
    uhf.max_iter = 200;

    const cpu::scf::DeltaScfResult r =
        uhf.delta_scf(frame, /*from=*/6, /*to=*/7, cpu::scf::MomStyle::IMOM);
    if (!r.converged) {
        std::fprintf(stderr, "FAIL: delta_scf did not converge\n");
        return 1;
    }

    // PySCF 2.14.0 pin (REFERENCE.md).
    constexpr double kEGroundHa = -107.495893307834;
    constexpr double kEExcitedHa = -107.192823826964;
    constexpr double kDeltaEv = 8.246940703328;
    constexpr double kTolEv = 1e-4; // product gate
    constexpr double kTolHa = 1e-6; // energy sanity

    std::printf("N2 UHF/STO-3G ΔSCF IMOM (α HOMO→LUMO):\n");
    bool ok = true;
    ok = check("E_ground [Ha]", r.e_ground, kEGroundHa, kTolHa) && ok;
    ok = check("E_excited [Ha]", r.e_excited, kEExcitedHa, kTolHa) && ok;
    ok = check("ΔE [eV]", r.delta_e_ev, kDeltaEv, kTolEv) && ok;

    if (r.e_excited <= r.e_ground) {
        std::fprintf(stderr,
                     "FAIL: excited energy must sit above ground (got "
                     "E_ex=%.10f E_gs=%.10f)\n",
                     r.e_excited, r.e_ground);
        ok = false;
    }

    if (!ok) {
        std::printf("FAIL\n");
        return 1;
    }
    std::printf("PASS (iters=%d)\n", r.iterations);
    return 0;
}
