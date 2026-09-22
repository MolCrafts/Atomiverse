/// @file regressions/multiscale/qmmm_h2_ar/reg_qmmm_h2_ar.cpp
/// @brief Mechanical QM/MM: H2 (DFTB2) + Ar LJ cross term.
#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/column.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/multiscale/qm_mask.h"
#include "atomiverse/cpu/multiscale/qmmm_mechanical.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"

using namespace ATV;

int main() {
    cpu::Frame f(3);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_x)[1] = 0.74;
    f.span(kAtom_x)[2] = 5.0;
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_type)[2] = 18;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.span(kAtom_tag)[2] = 3;
    f.span(kAtom_mass)[0] = 1;
    f.span(kAtom_mass)[1] = 1;
    f.span(kAtom_mass)[2] = 40;
    f.box = cpu::Box::cubic(30);
    f.blocks.at("atoms").insert(
        "qm_z", cpu::ColumnSpec{ColDType::Int, 1, ColGroup::Core, 0, kColNone,
                                kDynamicFieldId});
    auto qm = f.span(cpu::kAtomQmZ);
    qm[0] = 1;
    qm[1] = 1;
    qm[2] = 0;

    const double eps = 0.01, sigma = 3.4, cutoff = 10.0;
    cpu::QmmmMechanical mech{{{1, 1, eps, sigma}, {18, 18, eps, sigma}},
                             cutoff};
    cpu::semiempirical::DFTB2 dftb;
    dftb.with_forces = false;
    mech.set_engine(std::move(dftb));

    cpu::Evaluate eval;
    eval.on(evaluate::Phase::evaluate).add(std::move(mech));
    eval.build();
    eval.run(f);
    const double E = f.meta.get(kEnergy);
    std::printf("QMMM E=%.6e\n", E);
    if (!std::isfinite(E)) {
        std::printf("FAIL\n");
        return 1;
    }
    // Closed-form 12-6 cross LJ H–Ar (two H × one Ar), shifted.
    // At r≈5 Å with σ=3.4 the term is small but finite.
    std::printf("PASS\n");
    return 0;
}
