/// @file regressions/scf/h2o_charges_pyscf/reg_h2o_charges_pyscf.cpp
/// @brief H2O / 6-31G* RHF atomic charges — Löwdin population plus MK and
///        CHELPG ESP fits — vs externally pinned PySCF 2.14.0 values.
///
/// Out-of-library consumer: public headers + link atomiverse only. Runs the
/// concrete RHF once, then
///   - Löwdin charges vs the PySCF S^{1/2} D S^{1/2} pin to 1e-6 e;
///   - MK / CHELPG charges vs the PySCF-density ESP fitted on the SAME
///     documented deterministic grids (tools/pin_charges_oracle.py) to
///     1e-3 e, with Σq = 0 to 1e-10 and the O charge inside the
///     Breneman-Wiberg / Besler-Merz-Kollman literature bands;
///   - the two H charges agree to the grid-discretization level (2e-3 e —
///     the deterministic grids are not exactly mirror-symmetric, and the
///     same asymmetry is present in the pinned oracle).
///
/// References in REFERENCE.md (Löwdin 1950; Besler-Merz-Kollman 1990;
/// Breneman-Wiberg 1990). Exit non-zero on mismatch. <= 60 s, single CPU.

#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/observer/esp_charges.h"
#include "atomiverse/cpu/observer/lowdin.h"
#include "atomiverse/cpu/scf/methods.h"

namespace {

bool check(const char* name, const std::vector<double>& got,
           const double (&want)[3], double tol) {
    bool ok = true;
    for (int a = 0; a < 3; ++a) {
        const double diff = std::fabs(got[static_cast<std::size_t>(a)]
                                      - want[a]);
        std::printf("%s q[%d] got=%.8f want=%.8f |diff|=%.3e\n", name, a,
                    got[static_cast<std::size_t>(a)], want[a], diff);
        ok = ok && diff < tol;
    }
    return ok;
}

} // namespace

int main() {
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
    frame.simbox = cpu::Box::cubic(20.0);

    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::Pople631Gs;
    rhf.tol = 1e-12;
    rhf.max_iter = 200;
    rhf.run(frame);
    std::printf("E(RHF) = %.10f Ha (PySCF -76.010504988257)\n", rhf.energy());

    const auto handle = rhf.result_handle();
    bool ok = true;

    // ── Löwdin vs PySCF S^{1/2} D S^{1/2} in the unit-normalized AO
    //    basis (pinned, 2.14.0, cart; see REFERENCE.md on the convention) ─
    const double want_lowdin[3] = {-0.7284015337, 0.3642007669, 0.3642007669};
    const auto q_lowdin = cpu::lowdin_charge(handle, frame);
    ok = check("lowdin", q_lowdin, want_lowdin, 1e-6) && ok;

    // ── MK on the documented Fibonacci-shell grid (364 pts pinned) ─────
    const double want_mk[3] = {-0.8070921421, 0.4040081384, 0.4030840037};
    cpu::MKCharges mk;
    const auto q_mk = mk.fit(handle, frame);
    ok = check("mk", q_mk, want_mk, 1e-3) && ok;

    // ── CHELPG on the documented cubic grid (8485 pts pinned) ──────────
    const double want_chelpg[3] = {-0.8050061239, 0.4020088594, 0.4029972644};
    cpu::ChelpgCharges chelpg;
    const auto q_chelpg = chelpg.fit(handle, frame);
    ok = check("chelpg", q_chelpg, want_chelpg, 1e-3) && ok;

    // ── Invariants: Σq, H equivalence, literature bands ────────────────
    for (const auto* q : {&q_lowdin, &q_mk, &q_chelpg}) {
        const double sum = (*q)[0] + (*q)[1] + (*q)[2];
        if (std::fabs(sum) > 1e-10) {
            std::printf("FAIL: sum(q) = %.3e > 1e-10\n", sum);
            ok = false;
        }
    }
    // Deterministic grids are not exactly mirror-symmetric: the two H
    // agree to the grid-discretization level (same asymmetry as the pin).
    if (std::fabs(q_mk[1] - q_mk[2]) > 2e-3
        || std::fabs(q_chelpg[1] - q_chelpg[2]) > 2e-3) {
        std::printf("FAIL: H equivalence beyond grid level (MK %.2e, "
                    "CHELPG %.2e)\n",
                    std::fabs(q_mk[1] - q_mk[2]),
                    std::fabs(q_chelpg[1] - q_chelpg[2]));
        ok = false;
    }
    if (std::fabs(q_lowdin[1] - q_lowdin[2]) > 1e-6) {
        std::printf("FAIL: Lowdin H equivalence %.2e > 1e-6\n",
                    std::fabs(q_lowdin[1] - q_lowdin[2]));
        ok = false;
    }
    // Literature bands (HF/6-31G* class): CHELPG O ~ -0.74..-0.81
    // (Breneman-Wiberg 1990); MK O ~ -0.8 (Besler-Merz-Kollman 1990).
    if (q_chelpg[0] < -0.85 || q_chelpg[0] > -0.64) {
        std::printf("FAIL: CHELPG O charge %.4f outside [-0.85, -0.64]\n",
                    q_chelpg[0]);
        ok = false;
    }
    if (q_mk[0] < -0.90 || q_mk[0] > -0.70) {
        std::printf("FAIL: MK O charge %.4f outside [-0.90, -0.70]\n",
                    q_mk[0]);
        ok = false;
    }

    if (!ok) {
        std::printf("FAIL: charges off the PySCF oracle\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
