/// @file regressions/multiscale/qmmm_link_ch3oh/reg_qmmm_link_ch3oh.cpp
/// @brief Textbook H-link QM/MM on a methanol-like C–O cut (RHF/STO-3G).
///
/// QM = OH + link H; MM = methyl. Gate on last_qm_energy (not LJ total):
/// finite, deep negative, and sensitive to link scale g. See REFERENCE.md.
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/column.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/multiscale/qm_mask.h"
#include "atomiverse/cpu/multiscale/qmmm_electrostatic.h"
#include "atomiverse/cpu/scf/methods.h"

using namespace ATV;

namespace {

cpu::Frame make_methanol_like() {
    cpu::Frame f(6);
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_x)[1] = -0.96;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = 0.0;
    f.span(kAtom_x)[2] = 1.43;
    f.span(kAtom_y)[2] = 0.0;
    f.span(kAtom_z)[2] = 0.0;
    f.span(kAtom_x)[3] = 1.8;
    f.span(kAtom_y)[3] = 1.0;
    f.span(kAtom_z)[3] = 0.0;
    f.span(kAtom_x)[4] = 1.8;
    f.span(kAtom_y)[4] = -0.5;
    f.span(kAtom_z)[4] = 0.87;
    f.span(kAtom_x)[5] = 1.8;
    f.span(kAtom_y)[5] = -0.5;
    f.span(kAtom_z)[5] = -0.87;
    const int Z[] = {8, 1, 6, 1, 1, 1};
    for (int i = 0; i < 6; ++i) {
        f.span(kAtom_type)[i] = Z[i];
        f.span(kAtom_tag)[i] = static_cast<TagInt>(i + 1);
        f.span(kAtom_mass)[i] = 1.0;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
        f.span(kAtom_charge)[i] = 0;
    }
    f.box = cpu::Box::cubic(30.0);
    f.blocks.at("atoms").insert(
        "qm_z", cpu::ColumnSpec{ColDType::Int, 1, ColGroup::Core, 0, kColNone,
                                kDynamicFieldId});
    auto qm = f.span(cpu::kAtomQmZ);
    qm[0] = 8; // O
    qm[1] = 1; // H
    // C and methyl H stay 0 = MM
    // Covalent QM/MM cut is derived from topology (O–C bond), not set_links.
    f.rebuild_tag_index(0, 6);
    f.add_bond(/*O tag=*/1, /*C tag=*/3, /*type=*/1);
    return f;
}

struct Energies {
    double total;
    double qm;
};

Energies energy_link(double scale) {
    auto f = make_methanol_like();
    // Soft LJ so the gate focuses on the QM energy response to g.
    cpu::QmmmElectrostatic qmmm{
        {{1, 1, 0.01, 3.4}, {6, 6, 0.01, 3.4}}, 8.0, 15.0};
    qmmm.link_scale() = scale;
    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-8;
    rhf.max_iter = 80;
    qmmm.set_engine(std::move(rhf));

    cpu::Evaluate eval;
    eval.on(evaluate::Phase::evaluate).add_ref(qmmm);
    eval.build();
    eval.run(f);
    if (qmmm.last_n_links() != 1)
        throw std::runtime_error("expected one link");
    return {f.meta.get(kEnergy), qmmm.last_qm_energy()};
}

} // namespace

int main() {
    try {
        const auto a = energy_link(0.709);
        const auto b = energy_link(0.55);
        std::printf("qmmm_link_ch3oh E_qm(g=0.709)=%.6f E_qm(g=0.55)=%.6f eV "
                    "dE_qm=%.6f  E_tot=%.6f\n",
                    a.qm, b.qm, a.qm - b.qm, a.total);
        if (!std::isfinite(a.qm) || !std::isfinite(b.qm)
            || !std::isfinite(a.total)) {
            std::printf("FAIL: non-finite energy\n");
            return 1;
        }
        if (std::fabs(a.qm - b.qm) < 1e-3) {
            std::printf("FAIL: link scale did not shift QM energy\n");
            return 1;
        }
        if (!(a.qm < -100.0)) {
            std::printf("FAIL: E_qm=%.6f not below -100 eV gate\n", a.qm);
            return 1;
        }
        std::printf("PASS\n");
        return 0;
    } catch (const std::exception& ex) {
        std::printf("FAIL: %s\n", ex.what());
        return 1;
    }
}
