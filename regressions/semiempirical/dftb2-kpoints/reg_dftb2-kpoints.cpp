/// @file regressions/semiempirical/dftb2-kpoints/reg_dftb2-kpoints.cpp
/// @brief Multi-k SCC-DFTB2 energy on a 1-D carbon chain (4×1×1 MP mesh).
///
/// Domain: Elstner et al., Phys. Rev. B 58, 7260 (1998). Bloch-phased lattice
/// sums of H0/S at Monkhorst–Pack k-points (MioSketch tables). Hard pin is the
/// ATV Hamiltonian twin at the geometry in REFERENCE.md. Public API only.
/// <= 60 s. Never shells to DFTB+.
#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"

using namespace ATV;

namespace {

// 1-D C dimer crystal: r(C–C)=1.4 Å, Lx=2.8 Å (same as dftb2-periodic-gamma).
cpu::Frame make_c_chain() {
    cpu::Frame f(2);
    f.span(kAtom_type)[0] = 6;
    f.span(kAtom_type)[1] = 6;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_x)[1] = 1.4;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = 0.0;
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_mass)[i] = 12.011;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box::orthorhombic(2.8, 12.0, 12.0);
    f.meta.get(kEnergy) = 0;
    return f;
}

double run_energy(cpu::Frame frame, std::array<int, 3> mesh) {
    cpu::Evaluate eval;
    cpu::semiempirical::DFTB2 dftb;
    dftb.with_forces = false;
    dftb.scc_tol = 1e-10;
    dftb.k_mesh = mesh;
    dftb.k_time_reversal = true;
    eval.on(evaluate::Phase::evaluate).add(std::move(dftb));
    eval.build();
    eval.run(frame);
    return frame.meta.get(kEnergy);
}

} // namespace

int main() {
    auto frame = make_c_chain();
    const double e = run_energy(std::move(frame), {{4, 1, 1}});
    std::printf("DFTB2 multi-k (4x1x1) C-chain: E=%.10f eV\n", e);
    if (!std::isfinite(e)) {
        std::printf("FAIL non-finite energy\n");
        return 1;
    }
    if (std::fabs(e) < 1e-3) {
        std::printf("FAIL energy near zero\n");
        return 1;
    }

    // Hard pin: ATV twin with full mio-1-1 C–C (re-pinned 2026-07-29).
    constexpr double kE_ref_eV = -91.5579236054;
    constexpr double kE_tol_eV = 0.05;
    if (std::fabs(e - kE_ref_eV) > kE_tol_eV) {
        std::printf("FAIL energy pin: got %.10f want %.10f ± %.3f eV\n", e,
                    kE_ref_eV, kE_tol_eV);
        return 1;
    }

    // Sanity: multi-k must separate from Γ (same cell, mesh 1x1x1).
    const double e_gamma = run_energy(make_c_chain(), {{1, 1, 1}});
    std::printf("  Γ (1x1x1): E=%.10f eV  Δ=%.6f eV\n", e_gamma, e - e_gamma);
    if (!std::isfinite(e_gamma) || std::fabs(e - e_gamma) < 0.01) {
        std::printf("FAIL multi-k energy does not separate from Γ\n");
        return 1;
    }

    // Γ must still match the published periodic-gamma pin within tol.
    constexpr double kE_gamma_ref = -94.5030857189;
    if (std::fabs(e_gamma - kE_gamma_ref) > 0.05) {
        std::printf("FAIL Γ control pin: got %.10f want %.10f\n", e_gamma,
                    kE_gamma_ref);
        return 1;
    }

    std::printf("PASS\n");
    return 0;
}
