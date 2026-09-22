/// @file regressions/multiscale/qmmm_pol_h2o/reg_qmmm_pol_h2o.cpp
/// @brief Polarizable QM/MM: H2O (RHF) + polarizable MM site.
#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/column.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/multiscale/qm_mask.h"
#include "atomiverse/cpu/multiscale/qmmm_electrostatic.h"
#include "atomiverse/cpu/multiscale/qmmm_polarizable.h"
#include "atomiverse/cpu/scf/methods.h"

using namespace ATV;

namespace {

cpu::Frame water_plus_site() {
    cpu::Frame f(4);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = 0.9572;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_x)[2] = -0.239987;
    f.span(kAtom_y)[2] = 0.926627;
    f.span(kAtom_z)[2] = 0;
    f.span(kAtom_x)[3] = 0;
    f.span(kAtom_y)[3] = 0;
    f.span(kAtom_z)[3] = 4.0;
    f.span(kAtom_type)[0] = 8;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_type)[2] = 1;
    f.span(kAtom_type)[3] = 18;
    for (int i = 0; i < 4; ++i) {
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_mass)[i] = 1;
        f.span(kAtom_charge)[i] = 0;
    }
    f.box = cpu::Box::cubic(25);
    f.blocks.at("atoms").insert(
        "qm_z", cpu::ColumnSpec{ColDType::Int, 1, ColGroup::Core, 0, kColNone,
                                kDynamicFieldId});
    auto qm = f.span(cpu::kAtomQmZ);
    qm[0] = 8;
    qm[1] = 1;
    qm[2] = 1;
    qm[3] = 0;
    return f;
}

cpu::scf::RHF make_rhf() {
    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-8;
    rhf.max_iter = 60;
    return rhf;
}

} // namespace

int main() {
    double e_fix = 0, e_pol = 0, e_pol_term = 0;
    {
        auto f = water_plus_site();
        cpu::Evaluate eval;
        cpu::QmmmElectrostatic qmmm{
            {{8, 8, 0.01, 3.4}, {1, 1, 0.01, 3.4}}, 10.0, 12.0};
        qmmm.set_engine(make_rhf());
        eval.on(evaluate::Phase::evaluate).add(std::move(qmmm));
        eval.build();
        eval.run(f);
        e_fix = f.meta.get(kEnergy);
    }
    {
        auto f = water_plus_site();
        cpu::QmmmPolarizable pol{
            {{8, 8, 0.01, 3.4}, {1, 1, 0.01, 3.4}}, 10.0, 12.0};
        pol.alpha = 2.0;
        pol.thole_a = 2.6;
        pol.set_engine(make_rhf());

        cpu::Evaluate eval;
        eval.on(evaluate::Phase::evaluate).add_ref(pol);
        eval.build();
        eval.run(f);
        e_pol = f.meta.get(kEnergy);
        e_pol_term = pol.last_pol_energy();
    }
    std::printf("e_fix=%.6e e_pol=%.6e e_pol_term=%.6e\n", e_fix, e_pol,
                e_pol_term);
    if (!std::isfinite(e_fix) || !std::isfinite(e_pol)
        || !std::isfinite(e_pol_term)) {
        std::printf("FAIL: non-finite\n");
        return 1;
    }
    if (!(e_pol_term < -1e-4)) {
        std::printf("FAIL: E_pol=%.6e not sufficiently negative\n", e_pol_term);
        return 1;
    }
    if (!(e_pol < e_fix - 1e-4)) {
        std::printf("FAIL: polarizable energy not below fixed-charge\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
