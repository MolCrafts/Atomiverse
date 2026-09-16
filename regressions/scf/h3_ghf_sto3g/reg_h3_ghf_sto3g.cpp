/// @file regressions/scf/h3_ghf_sto3g/reg_h3_ghf_sto3g.cpp
/// @brief Equilateral H3/STO-3G generalized Hartree-Fock — the collapse
///        theorem and the noncollinear spin-frustration win, vs externally
///        pinned PySCF 2.14.0 `scf.GHF` / `scf.UHF` values (REFERENCE.md).
///
/// Out-of-library consumer: public headers + link atomiverse only.
///   - Collapse sentinel (H2, R = 0.74 A, unfrustrated):
///     |E_GHF - E_UHF| <= 1e-8 Ha — without spin-orbit terms every UHF
///     solution is a GHF stationary point, and the unfrustrated GHF
///     minimum is collinear (Fukutome, Int. J. Quantum Chem. 20, 955
///     (1981), DOI 10.1002/qua.560200502; Helgaker-Jorgensen-Olsen,
///     *Molecular Electronic-Structure Theory*, sec. 10).
///   - Noncollinear win (equilateral H3, side 1.0 A, spin-frustrated):
///     E_GHF = -1.34044034354119 Ha (PySCF pin, 1e-8), below the best
///     collinear UHF (-1.33598005401819 Ha) by >= 1e-6 Ha, with the
///     120-degree Neel moment guess; ||P_ab||_F ~ 0.9780588 witnesses
///     genuine noncollinearity.
///   - Collinear trap: guess_spin_rotation = 0 stays on the collinear
///     UHF stationary energy (the saddle DIIS converges onto).
///
/// Exit non-zero on failure. <= 60 s, single CPU.

#include <cmath>
#include <cstdio>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

namespace {

ATV::cpu::Frame h2_frame() {
    ATV::cpu::Frame frame(2);
    for (int i = 0; i < 2; ++i) {
        frame.span(ATV::kAtom_type)[i] = 1;
        frame.span(ATV::kAtom_mass)[i] = 1.008;
        frame.span(ATV::kAtom_y)[i] = 0.0;
        frame.span(ATV::kAtom_z)[i] = 0.0;
    }
    frame.span(ATV::kAtom_x)[0] = 0.0;
    frame.span(ATV::kAtom_x)[1] = 0.74;
    frame.box = ATV::cpu::Box::cubic(20.0);
    return frame;
}

// Equilateral triangle, side 1.0 A, xy-plane — the PySCF oracle geometry.
ATV::cpu::Frame h3_frame() {
    ATV::cpu::Frame frame(3);
    const double hx[3] = {0.0, 1.0, 0.5};
    const double hy[3] = {0.0, 0.0, 0.8660254037844386};
    for (int i = 0; i < 3; ++i) {
        frame.span(ATV::kAtom_type)[i] = 1;
        frame.span(ATV::kAtom_mass)[i] = 1.008;
        frame.span(ATV::kAtom_x)[i] = hx[i];
        frame.span(ATV::kAtom_y)[i] = hy[i];
        frame.span(ATV::kAtom_z)[i] = 0.0;
    }
    frame.box = ATV::cpu::Box::cubic(20.0);
    return frame;
}

} // namespace

int main() {
    using namespace ATV;
    using Spin = cpu::scf::SlaterDeterminant::Spin;

    // PySCF 2.14.0 pins [Ha] — REFERENCE.md records the exact commands.
    const double pin_ghf_h3 = -1.34044034354119;    // scf.GHF noncollinear
    const double pin_uhf_bs_h3 = -1.33598005401819; // scf.UHF (MINAO guess)
    const double pin_h2 = -1.11675930739643;        // scf.UHF = scf.GHF
    const double pin_pab = 0.9780588;               // ||P_ab||_F witness

    bool ok = true;

    // ── Collapse sentinel: H2 (no frustration) ─────────────────────────
    {
        auto frame = h2_frame();
        cpu::scf::UHF uhf(1, 1);
        uhf.basis = cpu::scf::BasisSet::STO3G;
        uhf.tol = 1e-12;
        uhf.max_iter = 200;
        uhf.run(frame);

        cpu::scf::GHF ghf(2);
        ghf.basis = cpu::scf::BasisSet::STO3G;
        ghf.tol = 1e-12;
        ghf.max_iter = 200;
        ghf.run(frame); // default small nonzero guess_spin_rotation

        std::printf("H2: E_UHF = %.12f, E_GHF = %.12f, |dE| = %.3e Ha\n",
                    uhf.energy(), ghf.energy(),
                    std::fabs(ghf.energy() - uhf.energy()));
        ok = ok && uhf.converged() && ghf.converged();
        ok = ok && std::fabs(ghf.energy() - uhf.energy()) <= 1e-8;
        ok = ok && std::fabs(ghf.energy() - pin_h2) <= 1e-8;
    }

    // ── Noncollinear win: equilateral H3 (spin-frustrated) ─────────────
    {
        auto frame = h3_frame();
        cpu::scf::GHF ghf(3);
        ghf.basis = cpu::scf::BasisSet::STO3G;
        ghf.tol = 1e-12;
        ghf.max_iter = 200;
        // Geometry-adapted Neel angle: 120-degree per-AO moment fan.
        ghf.guess_spin_rotation = 2.0 * M_PI / 3.0;
        ghf.run(frame);

        const double pab = [&] {
            const auto Pab =
                ghf.determinant().spin_block(Spin::alpha, Spin::beta);
            double s = 0.0;
            for (std::size_t i = 0; i < Pab.shape()[0]; ++i)
                for (std::size_t j = 0; j < Pab.shape()[1]; ++j)
                    s += Pab(i, j) * Pab(i, j);
            return std::sqrt(s);
        }();
        std::printf("H3: E_GHF = %.12f Ha (PySCF %.12f, dE = %.3e), "
                    "||P_ab|| = %.7f\n",
                    ghf.energy(), pin_ghf_h3, ghf.energy() - pin_ghf_h3, pab);
        ok = ok && ghf.converged();
        ok = ok && std::fabs(ghf.energy() - pin_ghf_h3) <= 1e-8;
        ok = ok && (ghf.energy() <= pin_uhf_bs_h3 - 1e-6);
        ok = ok && std::fabs(pab - pin_pab) <= 1e-3;

        // ── Collinear trap: theta = 0 stays on the UHF saddle ──────────
        cpu::scf::GHF trap(3);
        trap.basis = cpu::scf::BasisSet::STO3G;
        trap.tol = 1e-12;
        trap.max_iter = 200;
        trap.guess_spin_rotation = 0.0;
        trap.run(frame);
        std::printf("H3 trap (theta=0): E = %.12f Ha (collinear UHF "
                    "%.12f)\n",
                    trap.energy(), pin_uhf_bs_h3);
        ok = ok && trap.converged();
        ok = ok && std::fabs(trap.energy() - pin_uhf_bs_h3) <= 1e-8;
        ok = ok && (trap.energy() > pin_ghf_h3 + 1e-3);
    }

    std::printf("h3_ghf_sto3g: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
