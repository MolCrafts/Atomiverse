/// @file regressions/pw/pw-smearing-al/reg_pw-smearing-al.cpp
/// @brief END-TO-END: FCC Al plane-wave SCF with Fermi–Dirac smearing.
///
/// See REFERENCE.md. Public API only. Requires ATV_BUILD_PW.
/// Pins: electron count, finite Fermi level, free energy vs offline pin.

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

/// FCC primitive cell with lattice constant a [Å], one Al at the origin.
cpu::Frame make_al_fcc_primitive(double a_ang) {
    cpu::Frame f(1);
    f.span(kAtom_type)[0] = 13;
    f.span(kAtom_mass)[0] = 26.9815385;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    const double h = a_ang / 2.0;
    Mat3 cell;
    // Primitive FCC vectors: (0,h,h), (h,0,h), (h,h,0)
    cell(0, 0) = 0;
    cell(0, 1) = h;
    cell(0, 2) = h;
    cell(1, 0) = h;
    cell(1, 1) = 0;
    cell(1, 2) = h;
    cell(2, 0) = h;
    cell(2, 1) = h;
    cell(2, 2) = 0;
    f.box.set_cell(cell);
    return f;
}

} // namespace

int main() {
    // Offline pin — see REFERENCE.md (frozen ATV free energy on the settings
    // below; CP2K GTH-GPW twin documents SMEAR algorithm, not absolute E).
    const double e_pin_ry = -7.54535571; // Ry free energy, ATV offline freeze
    const double e_tol_ry = 0.05;        // Ry
    const double ne_tol = 1e-5;
    const double a_ang = 4.05;

    auto frame = make_al_fcc_primitive(a_ang);

    PW pw;
    pw.upf_path = ATV_REG_DATA_DIR "/pw/Al_pz_nc.upf";
    pw.ecut_ry = 20.0;
    pw.k_mesh = 2;
    pw.tol = 1e-5;
    pw.max_iter = 60;
    pw.mix_beta = 0.4;
    pw.pulay_depth = 5;
    pw.smearing = Smearing::fermi_dirac;
    pw.degauss = 0.02; // Ry
    pw.extra_bands = 4;

    // No toy fallback: REFERENCE.md pins the free energy for THIS NC-LDA UPF,
    // so a missing file is a failure, not a reason to solve a different
    // Hamiltonian and still print PASS.
    if (std::FILE* fp = std::fopen(pw.upf_path.c_str(), "r")) {
        std::fclose(fp);
    } else {
        std::printf("FAIL: cannot open %s\n", pw.upf_path.c_str());
        return 1;
    }

    pw.build(frame);
    const auto& r = pw.run(frame);

    if (!r.converged) {
        std::printf("FAIL: SCF not converged after %d iters  E=%.8f Ry\n",
                    r.iterations, r.energy);
        return 1;
    }
    if (!std::isfinite(r.energy) || !std::isfinite(r.e_fermi)) {
        std::printf("FAIL: non-finite energy or e_fermi\n");
        return 1;
    }
    if (std::fabs(r.n_elec_occ - 3.0) > ne_tol) {
        std::printf("FAIL: n_elec_occ=%.8f  expected 3 ± %g\n", r.n_elec_occ,
                    ne_tol);
        return 1;
    }
    // Energy pin (loose until REFERENCE records the frozen offline value).
    if (std::fabs(r.energy - e_pin_ry) > e_tol_ry) {
        std::printf("FAIL: E=%.8f Ry  pin=%.8f ± %g\n", r.energy, e_pin_ry,
                    e_tol_ry);
        return 1;
    }
    // Fermi level must sit among bands (not NaN, not absurd).
    if (!(std::fabs(r.e_fermi) < 1e3)) {
        std::printf("FAIL: absurd e_fermi=%.6f Ry\n", r.e_fermi);
        return 1;
    }

    std::printf(
        "ok: E=%.8f Ry  e_fermi=%.6f Ry  n_occ=%.8f  e_entropy=%.6f Ry  "
        "iters=%d  upf=%s\n",
        r.energy, r.e_fermi, r.n_elec_occ, r.e_entropy, r.iterations,
        pw.upf_path.empty() ? "toy" : pw.upf_path.c_str());
    std::printf("PASS: pw-smearing-al (FCC Al Fermi–Dirac)\n");
    return 0;
}
