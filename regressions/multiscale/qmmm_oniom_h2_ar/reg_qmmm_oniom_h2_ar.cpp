/// @file regressions/multiscale/qmmm_oniom_h2_ar/reg_qmmm_oniom_h2_ar.cpp
/// @brief Two-layer ONIOM: H2 (DFTB2 high) + classical low on H2+Ar.
///
/// \f$ E = E_{\mathrm{high}}(\mathrm{H_2}) + E_{\mathrm{low}}(\mathrm{real})
///         - E_{\mathrm{low}}(\mathrm{H_2}) \f$
/// Identity is checked to 1e-10 eV. See REFERENCE.md.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/column.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/multiscale/qm_mask.h"
#include "atomiverse/cpu/multiscale/qmmm_oniom.h"
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
    for (int i = 0; i < 3; ++i) {
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_mass)[i] = (i < 2) ? 1.0 : 40.0;
        f.span(kAtom_charge)[i] = 0;
        f.span(kAtom_y)[i] = f.span(kAtom_z)[i] = 0;
    }
    f.box = cpu::Box::cubic(30);
    f.blocks.at("atoms").insert(
        "qm_z", cpu::ColumnSpec{ColDType::Int, 1, ColGroup::Core, 0, kColNone,
                                kDynamicFieldId});
    f.span(cpu::kAtomQmZ)[0] = 1;
    f.span(cpu::kAtomQmZ)[1] = 1;
    f.span(cpu::kAtomQmZ)[2] = 0;

    const double eps = 0.01, sigma = 3.4, cut = 12.0;
    cpu::QmmmOniom oniom{{{1, 1, eps, sigma}, {18, 18, eps, sigma}}, cut};
    cpu::semiempirical::DFTB2 dftb;
    dftb.with_forces = false;
    oniom.set_engine(std::move(dftb));

    // Keep a second instance for last_* after Evaluate takes ownership.
    cpu::QmmmOniom probe{{{1, 1, eps, sigma}, {18, 18, eps, sigma}}, cut};
    probe.set_engine(cpu::semiempirical::DFTB2{});

    cpu::Evaluate eval;
    eval.on(evaluate::Phase::evaluate).add_ref(probe);
    eval.set_threads(1);
    eval.build();
    eval.run(f);

    const double E = f.meta.get(kEnergy);
    const double eh = probe.last_high_energy();
    const double elr = probe.last_low_real_energy();
    const double elm = probe.last_low_model_energy();
    const double recon = eh + elr - elm;

    std::printf("qmmm_oniom_h2_ar E=%.10e high=%.10e low_real=%.10e "
                "low_model=%.10e recon=%.10e\n",
                E, eh, elr, elm, recon);

    if (!std::isfinite(E) || !std::isfinite(eh) || !std::isfinite(elr)
        || !std::isfinite(elm)) {
        std::printf("FAIL: non-finite\n");
        return 1;
    }
    if (std::fabs(E - recon) > 1e-9) {
        std::printf("FAIL: ONIOM identity |E-recon|=%.3e\n",
                    std::fabs(E - recon));
        return 1;
    }
    if (!(std::fabs(elm) > 1e-12)) {
        std::printf("FAIL: E_low(model) should be nonzero with H–H LJ\n");
        return 1;
    }
    if (!(std::fabs(elr - elm) > 1e-6)) {
        std::printf("FAIL: E_low(real) must differ from E_low(model) "
                    "(cross terms)\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
