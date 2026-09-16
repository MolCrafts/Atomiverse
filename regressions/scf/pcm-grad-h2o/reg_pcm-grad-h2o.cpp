/// @file regressions/scf/pcm-grad-h2o/reg_pcm-grad-h2o.cpp
/// @brief H₂O RHF/6-31G* C-PCM free-energy forces vs finite difference of
///        the solvent SCF energy (see REFERENCE.md).
///
/// Implicit-solvent analytic gradient chain link (`pcm-grad-h2o`).
/// Out-of-library consumer: public headers + link atomiverse only.
/// Gate: element-wise |g_analytic − g_FD| ≤ 1e-5 Ha/Bohr. ≤ 60 s, single CPU.
///
/// PySCF C-PCM forces are pinned in REFERENCE.md for literature context;
/// SWIG surface + Gaussian tessera charges differ from Atomiverse hard
/// bury-delete + point-charge AO potentials, so the acceptance oracle is
/// FD of *this* free-energy functional (same pattern as field_h2o_grad).

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/gradient_result.h"
#include "atomiverse/cpu/scf/methods.h"

namespace {

using namespace ATV;

cpu::Frame water() {
    cpu::Frame f(3);
    f.span(kAtom_type)[0] = 8;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.1173;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_x)[1] = 0.0;
    f.span(kAtom_y)[1] = 0.7572;
    f.span(kAtom_z)[1] = -0.4692;
    f.span(kAtom_type)[2] = 1;
    f.span(kAtom_x)[2] = 0.0;
    f.span(kAtom_y)[2] = -0.7572;
    f.span(kAtom_z)[2] = -0.4692;
    f.box = cpu::Box::cubic(20.0);
    return f;
}

void apply_solvent(cpu::scf::RHF& rhf) {
    rhf.basis = cpu::scf::BasisSet::Pople631Gs;
    rhf.tol = 1e-12;
    rhf.max_iter = 200;
    rhf.solvent_eps = 78.4;
    rhf.solvent_radii_scale = 1.2;
    rhf.solvent_x = 0.0; // C-PCM
}

double solvent_energy(const cpu::Frame& frame) {
    cpu::scf::RHF rhf;
    apply_solvent(rhf);
    cpu::Frame f = frame;
    rhf.run(f);
    return rhf.energy(); // Ha
}

cpu::Frame displaced(const cpu::Frame& frame, int A, int d, double dx_ang) {
    cpu::Frame f = frame;
    if (d == 0)
        f.span(kAtom_x)[A] += dx_ang;
    else if (d == 1)
        f.span(kAtom_y)[A] += dx_ang;
    else
        f.span(kAtom_z)[A] += dx_ang;
    return f;
}

} // namespace

int main() {
    using namespace ATV;
    const cpu::Frame frame = water();

    cpu::scf::RHF rhf;
    apply_solvent(rhf);
    cpu::Frame ff = frame;
    const cpu::scf::GradientResult res = rhf.forces(ff);
    if (!rhf.converged()) {
        std::printf("FAIL: C-PCM SCF did not converge\n");
        return 1;
    }
    std::printf("E_cpcm = %.12f Ha   DG_el = %.12f Ha\n", rhf.energy(),
                rhf.e_solvent());

    auto grad_ha_bohr = [&](int A, int d) {
        return -units::convert(res.forces(A, d), units::Force::eVPerAngstrom,
                               units::Force::HartreePerBohr);
    };

    // Finite-difference dG/dR of the C-PCM free energy [Ha/Bohr].
    const double h_bohr = 1e-3;
    const double h_ang = h_bohr * units::kBohrInAngstrom;
    constexpr double kTol = 1e-5;

    double max_err = 0.0;
    for (int A = 0; A < 3; ++A)
        for (int d = 0; d < 3; ++d) {
            const double fd = (solvent_energy(displaced(frame, A, d, +h_ang))
                               - solvent_energy(displaced(frame, A, d, -h_ang)))
                              / (2.0 * h_bohr);
            const double an = grad_ha_bohr(A, d);
            const double err = std::fabs(an - fd);
            max_err = std::max(max_err, err);
            std::printf("g[%d][%d] analytic=% .10e fd=% .10e |err|=%.2e\n", A,
                        d, an, fd, err);
        }

    // PySCF 2.14.0 pin (REFERENCE.md) — literature context only; not the gate.
    const double pyscf[3][3] = {
        {0.0, 0.0, 9.3696733492e-03},
        {0.0, 7.0215556208e-03, -4.6848366746e-03},
        {0.0, -7.0215556208e-03, -4.6848366746e-03},
    };
    double max_pyscf = 0.0;
    for (int A = 0; A < 3; ++A)
        for (int d = 0; d < 3; ++d)
            max_pyscf = std::max(max_pyscf,
                                 std::fabs(grad_ha_bohr(A, d) - pyscf[A][d]));
    std::printf("max_vs_pyscf_pin=%.3e (informational; SWIG vs hard bury)\n",
                max_pyscf);
    std::printf("max_elementwise_err=%.3e tol=%.0e\n", max_err, kTol);

    if (!(max_err < kTol)) {
        std::printf("FAIL: C-PCM force != FD of C-PCM free energy\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
