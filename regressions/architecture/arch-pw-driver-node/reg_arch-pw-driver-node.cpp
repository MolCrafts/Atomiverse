/// @file reg_arch-pw-driver-node.cpp
/// @brief Architecture gate: `cpu::pw::PW` is a `Driver<pw::Phase>` and the
///        restricted fixed-occupation public `run(frame)` still hits the
///        offline ABINIT-class Si LDA pin (same twin as `pw-lda-si`).
///
/// Public API only. No free-loop reimplementation. Spin / metallic residual
/// paths are out of this slug. See REFERENCE.md.

#include <cmath>
#include <cstdio>
#include <type_traits>

#include "atomiverse/core/mat3.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/driver/driver.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/scf/xc_functional.h"

#ifndef ATV_PW_UPF_PATH
#error "ATV_PW_UPF_PATH must be set to tests/data/pw/Si_lda_nc.upf"
#endif

using namespace ATV;
using namespace ATV::cpu::pw;

int main() {
    static_assert(std::is_base_of_v<cpu::Driver<ATV::pw::Phase>, PW>,
                  "PW must be cpu::Driver<pw::Phase>");
    if (ATV::slot_count<ATV::pw::Phase>() != 5) {
        std::printf(
            "FAIL: pw::Phase slot_count got=%zu want=5\n",
            static_cast<std::size_t>(ATV::slot_count<ATV::pw::Phase>()));
        return 1;
    }
    if (ATV::pw::kind_of(ATV::pw::Phase{}) != ATV::PhaseKind::SCF) {
        std::printf("FAIL: pw::Phase kind is not PhaseKind::SCF\n");
        return 1;
    }

    // Same twin as regressions/pw/pw-lda-si (ABINIT-class E_min band).
    constexpr double kEminAbinitRy = -16.916316;
    constexpr double kBandHeadroomRy = 1.50;
    constexpr double kA = 5.43; // Å
    constexpr double kEcutRy = 15.0;
    constexpr int kMesh = 2;

    cpu::Frame frame(2);
    frame.span(kAtom_type)[0] = 14;
    frame.span(kAtom_type)[1] = 14;
    frame.span(kAtom_mass)[0] = 28.0855;
    frame.span(kAtom_mass)[1] = 28.0855;
    frame.span(kAtom_x)[0] = 0.0;
    frame.span(kAtom_y)[0] = 0.0;
    frame.span(kAtom_z)[0] = 0.0;
    frame.span(kAtom_x)[1] = kA / 4.0;
    frame.span(kAtom_y)[1] = kA / 4.0;
    frame.span(kAtom_z)[1] = kA / 4.0;
    const double h = kA / 2.0;
    Mat3 cell;
    cell(0, 0) = 0;
    cell(0, 1) = h;
    cell(0, 2) = h;
    cell(1, 0) = h;
    cell(1, 1) = 0;
    cell(1, 2) = h;
    cell(2, 0) = h;
    cell(2, 1) = h;
    cell(2, 2) = 0;
    frame.box.set_cell(cell);

    PW method;
    method.upf_path = ATV_PW_UPF_PATH;
    method.ecut_ry = kEcutRy;
    method.k_mesh = kMesh;
    method.tol = 1e-6;
    method.max_iter = 50;
    method.xc = cpu::scf::XCType::LSDA; // restricted fixed-occ LDA path
    method.spin_polarized = false;
    method.smearing = Smearing::none;

    method.build(frame);
    const PWResult& r = method.run(frame);

    if (!r.converged) {
        std::printf(
            "FAIL: PW Driver fixed-occ SCF did not converge (iters=%d)\n",
            r.iterations);
        return 1;
    }
    if (!std::isfinite(r.energy)) {
        std::printf("FAIL: non-finite energy_ry=%g\n", r.energy);
        return 1;
    }

    const double lo = kEminAbinitRy;
    const double hi = kEminAbinitRy + kBandHeadroomRy;
    if (!(r.energy >= lo && r.energy <= hi)) {
        std::printf("FAIL: E_tot got=%.8f Ry  want band [%.6f, %.6f] Ry "
                    "(ABINIT-class offline pin; Driver-node reshape gate)\n",
                    r.energy, lo, hi);
        return 1;
    }

    std::printf(
        "PASS: arch-pw-driver-node  PW : Driver<pw::Phase>  "
        "E=%.8f Ry  band [%.4f, %.4f]  iters=%d  (offline ABINIT-class)\n",
        r.energy, lo, hi, r.iterations);
    return 0;
}
