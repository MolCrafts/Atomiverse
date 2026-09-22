/// @file regressions/multiscale/qmmm_es_h2o_pyscf/reg_qmmm_es_h2o_pyscf.cpp
/// @brief Electrostatic QM/MM coupling vs offline PySCF qmmm.mm_charge pin
///        (esmm-qmmm-03). Public API only; zero cross-LJ so the coupling
///        reproduces the bare embedding SCF of esmm-qmmm-01.

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/column.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/multiscale/qm_mask.h"
#include "atomiverse/cpu/multiscale/qmmm_electrostatic.h"
#include "atomiverse/cpu/scf/methods.h"

int main() {
    using namespace ATV;

    // 3 QM water + 2 MM point-charge atoms (same geometry as embed_h2o_mm_forces)
    cpu::Frame frame(5);
    frame.span(kAtom_type)[0] = 8;
    frame.span(kAtom_z)[0] = 0.1173;
    frame.span(kAtom_type)[1] = 1;
    frame.span(kAtom_y)[1] = 0.7572;
    frame.span(kAtom_z)[1] = -0.4692;
    frame.span(kAtom_type)[2] = 1;
    frame.span(kAtom_y)[2] = -0.7572;
    frame.span(kAtom_z)[2] = -0.4692;
    frame.span(kAtom_type)[3] = 99;
    frame.span(kAtom_z)[3] = 3.0;
    frame.span(kAtom_charge)[3] = 0.5;
    frame.span(kAtom_type)[4] = 99;
    frame.span(kAtom_z)[4] = -3.0;
    frame.span(kAtom_charge)[4] = -0.5;
    for (int i = 0; i < 5; ++i)
        frame.span(kAtom_tag)[i] = i + 1;
    frame.box = cpu::Box::cubic(20.0);
    frame.blocks.at("atoms").insert(
        "qm_z", cpu::ColumnSpec{ColDType::Int, 1, ColGroup::Core, 0, kColNone,
                                kDynamicFieldId});
    auto qm = frame.span(cpu::kAtomQmZ);
    qm[0] = 8;
    qm[1] = 1;
    qm[2] = 1;
    qm[3] = 0;
    qm[4] = 0;

    // Zero cross-LJ → pure electrostatic embedding path.
    cpu::QmmmElectrostatic qmmm{{}, 10.0, 20.0};
    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-12;
    rhf.max_iter = 200;
    qmmm.set_engine(std::move(rhf));

    cpu::Evaluate eval;
    eval.on(evaluate::Phase::evaluate).add_ref(qmmm);
    eval.build();
    eval.run(frame);

    auto g_ha = [&](int i, int d) {
        const double f = d == 0 ? frame.span(kAtom_fx)[i]
                                : (d == 1 ? frame.span(kAtom_fy)[i]
                                          : frame.span(kAtom_fz)[i]);
        return -units::convert(f, units::Force::eVPerAngstrom,
                               units::Force::HartreePerBohr);
    };

    // PySCF 2.14 pins (embed_h2o_grad + embed_h2o_mm_forces, self-energy off)
    const double want_qm[3][3] = {
        {0.0, 0.0, -6.3053441384e-02},
        {0.0, -1.7882928801e-02, 3.1195488026e-02},
        {0.0, 1.7882928801e-02, 3.1195488026e-02},
    };
    const double want_mm[2][3] = {
        {0.0, 0.0, 4.053776692103e-03},
        {0.0, 0.0, -3.391311764744e-03},
    };
    const double tol = 1e-5;

    double max_err = 0.0;
    for (int A = 0; A < 3; ++A)
        for (int d = 0; d < 3; ++d) {
            const double err = std::fabs(g_ha(A, d) - want_qm[A][d]);
            max_err = std::max(max_err, err);
            std::printf("g_qm[%d][%d] got=% .10e want=% .10e |err|=%.2e\n", A,
                        d, g_ha(A, d), want_qm[A][d], err);
        }
    for (int c = 0; c < 2; ++c)
        for (int d = 0; d < 3; ++d) {
            const int i = 3 + c;
            const double err = std::fabs(g_ha(i, d) - want_mm[c][d]);
            max_err = std::max(max_err, err);
            std::printf("g_mm[%d][%d] got=% .10e want=% .10e |err|=%.2e\n", c,
                        d, g_ha(i, d), want_mm[c][d], err);
        }
    std::printf("max_elementwise_err=%.3e tol=%.0e\n", max_err, tol);

    double sum[3] = {0, 0, 0};
    for (int i = 0; i < 5; ++i) {
        sum[0] += frame.span(kAtom_fx)[i];
        sum[1] += frame.span(kAtom_fy)[i];
        sum[2] += frame.span(kAtom_fz)[i];
    }
    const double net =
        std::sqrt(sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]);
    std::printf("net_force_eVA=%.3e (tol 1e-9)\n", net);
    std::printf("E_qm_eV=%.6f residual=(%.3e,%.3e,%.3e)\n",
                qmmm.last_qm_energy(), qmmm.last_force_residual()[0],
                qmmm.last_force_residual()[1], qmmm.last_force_residual()[2]);

    if (!(max_err < tol) || !(net < 1e-9)) {
        std::printf("FAIL\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
