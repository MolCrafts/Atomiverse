/// @file regressions/semiempirical/dftb2-periodic-gamma/reg_dftb2-periodic-gamma.cpp
/// @brief Γ-point periodic SCC-DFTB2 energy on a 1-D carbon chain cell.
///
/// Domain: Elstner et al., Phys. Rev. B 58, 7260 (1998). Lattice-summed
/// H0/S/γ/repulsive within static mio-style SK cutoffs (in-tree tables, not
/// DFTB.org mio-1-1 bit-identical). Hard pin is the ATV Hamiltonian twin
/// at the geometry in REFERENCE.md. Public API only. <= 60 s.
#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"

using namespace ATV;

namespace {

// 1-D C dimer crystal: r(C–C)=1.4 Å, Lx=2.8 Å → second-neighbor images in range.
// Ly=Lz=12 Å isolate the chain from its periodic images in y/z.
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

} // namespace

int main() {
    auto frame = make_c_chain();

    cpu::Evaluate eval;
    cpu::semiempirical::DFTB2 dftb;
    dftb.with_forces = false;
    dftb.scc_tol = 1e-10;
    eval.on(evaluate::Phase::evaluate).add(std::move(dftb));
    eval.build();
    eval.run(frame);

    const double e = frame.meta.get(kEnergy);
    std::printf("DFTB2 Γ-PBC C-chain: E=%.10f eV\n", e);
    if (!std::isfinite(e)) {
        std::printf("FAIL non-finite energy\n");
        return 1;
    }
    if (std::fabs(e) < 1e-3) {
        std::printf("FAIL energy near zero\n");
        return 1;
    }

    // Hard pin: ATV Hamiltonian twin with full mio-1-1 C–C tables
    // (re-pinned 2026-07-29; Δq≡0 so γ form is inert — shift is SK, not SCC).
    constexpr double kE_ref_eV = -94.5030857189;
    constexpr double kE_tol_eV = 0.05;
    if (std::fabs(e - kE_ref_eV) > kE_tol_eV) {
        std::printf("FAIL energy pin: got %.10f want %.10f ± %.3f eV\n", e,
                    kE_ref_eV, kE_tol_eV);
        return 1;
    }

    // Sanity: isolated dimer energy differs (lattice images matter).
    auto isol = make_c_chain();
    isol.box = cpu::Box::cubic(20.0);
    isol.meta.get(kEnergy) = 0;
    cpu::Evaluate eval2;
    cpu::semiempirical::DFTB2 d2;
    d2.with_forces = false;
    d2.scc_tol = 1e-10;
    eval2.on(evaluate::Phase::evaluate).add(std::move(d2));
    eval2.build();
    eval2.run(isol);
    const double e_mol = isol.meta.get(kEnergy);
    std::printf("  isolated dimer (20 Å cubic): E=%.10f eV  Δ=%.6f eV\n", e_mol,
                e - e_mol);
    if (!std::isfinite(e_mol) || std::fabs(e - e_mol) < 0.05) {
        std::printf("FAIL PBC energy does not separate from molecular dimer\n");
        return 1;
    }

    std::printf("PASS\n");
    return 0;
}
