/// @file regressions/pw/pw-uks-fe-smearing/reg_pw-uks-fe-smearing.cpp
/// @brief END-TO-END: BCC Fe collinear UKS + Fermi–Dirac metallic smearing.
///
/// See REFERENCE.md. Public API only. Requires ATV_BUILD_PW.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"

using namespace ATV;
using namespace ATV::cpu::pw;

namespace {

/// Primitive BCC cell from cubic lattice constant a [Å], one Fe at origin.
cpu::Frame make_fe_bcc_primitive(double a_ang) {
    cpu::Frame f(1);
    f.span(kAtom_type)[0] = 26;
    f.span(kAtom_mass)[0] = 55.845;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    const double h = a_ang / 2.0;
    Mat3 cell;
    // Primitive BCC: (-h,h,h), (h,-h,h), (h,h,-h)
    cell(0, 0) = -h;
    cell(0, 1) = h;
    cell(0, 2) = h;
    cell(1, 0) = h;
    cell(1, 1) = -h;
    cell(1, 2) = h;
    cell(2, 0) = h;
    cell(2, 1) = h;
    cell(2, 2) = -h;
    f.box.set_cell(cell);
    return f;
}

/// @brief One collinear-UKS + FD-smearing solve at the fixed case settings.
PWResult solve(const char* upf, double degauss_ry) {
    auto frame = make_fe_bcc_primitive(/*a_ang=*/2.87);
    PW pw;
    pw.upf_path = upf;
    pw.ecut_ry = 12.0; // modest algorithm pin (≤ 60 s)
    pw.k_mesh = 2;
    pw.tol = 2e-3;
    pw.max_iter = 30;
    pw.mix_beta = 0.3;
    pw.pulay_depth = 5;
    pw.xc = cpu::scf::XCType::PBE;
    pw.spin_polarized = true;
    pw.multiplicity = 5; // seed M≈4; free under smear
    pw.smearing = Smearing::fermi_dirac;
    pw.degauss = degauss_ry; // Ry
    pw.extra_bands = 4;
    pw.build(frame);
    return pw.run(frame);
}

} // namespace

int main() {
    // ── Gates (REFERENCE.md) ───────────────────────────────────────────
    // Two solves at σ and σ/4. Every gate below is an exact statement about
    // the Mermin free energy F = E − TS, the electron count, or the ferro
    // ground state — never a frozen ATV number. The absolute free energy at
    // E_cut = 12 Ry / 2×2×2 k has no citable external value, and the number
    // this case used to carry was a self-freeze recorded on a broken
    // Hamiltonian (see REFERENCE.md, "Void pin").
    constexpr double sigma_hi = 0.02;  // Ry
    constexpr double sigma_lo = 0.005; // Ry
    constexpr double want_ne = 16.0;   // SG15 Fe Z_val (exact)
    constexpr double tol_ne = 1e-4;
    constexpr double mag_min = 0.1; // ferro ground state
    // dF/dT = −S ≤ 0 must hold with a margin far above the SCF wobble
    // (≈4×10⁻³ Ry at tol = 2×10⁻³); the measured gap is ≈5×10⁻² Ry.
    constexpr double dF_margin_ry = 1.0e-2;

    // Compiled-in absolute path (ATV_REG_DATA_DIR): resolves identically from
    // the build tree, the source tree, or a bare ./reg_<case>.
    const char* upf = ATV_REG_DATA_DIR "/pw/Fe_ONCV_PBE-1.2.upf";
    if (std::FILE* f = std::fopen(upf, "r")) {
        std::fclose(f);
    } else {
        std::printf("FAIL: cannot open %s\n", upf);
        return 1;
    }

    const PWResult hi = solve(upf, sigma_hi);
    const PWResult lo = solve(upf, sigma_lo);

    for (const auto& p : {std::pair{sigma_hi, &hi}, std::pair{sigma_lo, &lo}}) {
        const PWResult& r = *p.second;
        std::printf("sig=%.4f Ry: F=%.8f Ry  E=F+TS=%.8f Ry  -TS=%.8f Ry  "
                    "mag=%.6f  e_fermi=%.6f Ry  n_occ=%.8f  conv=%d it=%d\n",
                    p.first, r.energy, r.energy - r.e_entropy, r.e_entropy,
                    r.magnetization, r.e_fermi, r.n_elec_occ, (int)r.converged,
                    r.iterations);
    }

    bool ok = true;
    for (const auto* r : {&hi, &lo}) {
        if (!std::isfinite(r->energy) || !std::isfinite(r->energy_ha)
            || !std::isfinite(r->e_fermi) || !std::isfinite(r->magnetization)
            || !std::isfinite(r->e_entropy)) {
            std::printf("FAIL: non-finite energy/fermi/mag/entropy\n");
            ok = false;
        }
        if (!r->converged) {
            std::printf("FAIL: SCF not converged\n");
            ok = false;
        }
        // Exact: Σ_{σkb} w_k f = Z_val. Gates k-weight normalisation and the
        // ONE shared chemical potential across both spin channels.
        if (std::fabs(r->n_elec_occ - want_ne) > tol_ne) {
            std::printf("FAIL: n_elec_occ=%.8f want=%.1f ± %g\n", r->n_elec_occ,
                        want_ne, tol_ne);
            ok = false;
        }
        // Exact: S ≥ 0, so the reported −TS contribution is never positive.
        if (!(r->e_entropy <= 0.0)) {
            std::printf("FAIL: e_entropy=%.8f Ry > 0 — S < 0 is impossible\n",
                        r->e_entropy);
            ok = false;
        }
        if (!(r->magnetization > mag_min)) {
            std::printf("FAIL: magnetization=%.6f expected > %g (ferro)\n",
                        r->magnetization, mag_min);
            ok = false;
        }
        if (!(std::fabs(r->e_fermi) < 1e3)) {
            std::printf("FAIL: absurd e_fermi=%.6f Ry\n", r->e_fermi);
            ok = false;
        }
    }

    // Exact: ∂F/∂T = −S < 0. Widening the smearing must LOWER the Mermin
    // free energy — a code that reports the internal energy E, or that gets
    // the sign of the entropy term wrong, fails here.
    const double dF = hi.energy - lo.energy;
    std::printf("F(sig=%.3f) - F(sig=%.3f) = %.8f Ry (must be < -%g)\n",
                sigma_hi, sigma_lo, dF, dF_margin_ry);
    if (!(dF < -dF_margin_ry)) {
        std::printf("FAIL: F is not decreasing in the smearing width — "
                    "dF/dT = -S violated\n");
        ok = false;
    }

    // Exact: TS → 0 as T → 0 (S is bounded), so |−TS| must shrink with σ.
    std::printf("|-TS| : %.8f (sig=%.3f) -> %.8f (sig=%.3f)\n",
                std::fabs(hi.e_entropy), sigma_hi, std::fabs(lo.e_entropy),
                sigma_lo);
    if (!(std::fabs(lo.e_entropy) < std::fabs(hi.e_entropy))) {
        std::printf("FAIL: |TS| did not shrink with the smearing width\n");
        ok = false;
    }

    if (!ok)
        return 1;
    std::printf("PASS: pw-uks-fe-smearing (BCC Fe UKS + FD smear: N_e, ferro, "
                "Mermin thermodynamics)\n");
    return 0;
}
