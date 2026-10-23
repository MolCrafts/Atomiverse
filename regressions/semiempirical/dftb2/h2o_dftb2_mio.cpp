/// @file regressions/semiempirical/h2o_dftb2_mio/reg_h2o_dftb2_mio.cpp
/// @brief H2O SCC-DFTB2 with transcribed mio-1-1 SK — energy + force gate.
///
/// Domain: Elstner et al., Phys. Rev. B 58, 7260 (1998). Static H/C/N/O
/// mio-1-1 tables (`dftb_mio_sk_tables.h`). Absolute energy pin vs DFTB+ 24.1
/// Matches DFTB+ to ~5e-8 eV; see REFERENCE.md.
///
/// Exit non-zero on failure. Public API only. <= 60 s.
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"
#include "harness/registry.h"

namespace atv_reg_semiempirical_h2o_dftb2_mio {

using namespace ATV;

namespace {

struct LoopAudit : cpu::Component {
    cpu::Runtime*& runtime;
    IterationStatus*& status;
    int& calls;
    int mode;

    LoopAudit(cpu::Runtime*& rt, IterationStatus*& st, int& count, int action)
        : runtime(rt), status(st), calls(count), mode(action) {}
    std::string name() const override { return "dftb2_loop_audit"; }
    void launch(const cpu::LaunchContext& ctx) override {
        ++calls;
        if (mode == 0) {
            runtime = ctx.runtime.exec.runtime;
            status = ctx.runtime.loop.status;
        } else if (!runtime || runtime != ctx.runtime.exec.runtime
                   || (mode == 1 && status == ctx.runtime.loop.status)
                   || (mode == 2 && status != ctx.runtime.loop.status)) {
            throw std::runtime_error(
                "DFTB2 nested Runtime/loop isolation failed");
        }
    }
};

cpu::Frame make_water() {
    cpu::Frame f(3);
    f.span(kAtom_type)[0] = 8;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_x)[1] = 0.96;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = 0.0;
    f.span(kAtom_type)[2] = 1;
    f.span(kAtom_x)[2] = -0.24;
    f.span(kAtom_y)[2] = 0.93;
    f.span(kAtom_z)[2] = 0.0;
    for (int i = 0; i < 3; ++i) {
        f.span(kAtom_mass)[i] = (i == 0) ? 15.999 : 1.008;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box::cubic(20.0);
    f.meta.get(kEnergy) = 0;
    return f;
}

} // namespace

int run() {
    auto frame = make_water();

    cpu::Evaluate eval;
    cpu::semiempirical::DFTB2 dftb{cpu::semiempirical::DftbSkSet::MioSketch};
    // Isolated molecule in an isolation box — the DFTB+ twin is a `C`
    // geometry, so say so rather than letting a cutoff hide the images.
    dftb.boundary = cpu::semiempirical::DftbBoundary::cluster;

    dftb.scc_tol = 1e-10;
    eval.on(evaluate::Phase::evaluate).add(std::move(dftb));
    eval.build();
    eval.run(frame);

    const double e = frame.meta.get(kEnergy);
    std::printf("DFTB2 H2O: E=%.8f eV\n", e);
    if (!std::isfinite(e)) {
        std::printf("FAIL non-finite energy\n");
        return 1;
    }
    // DFTB+ mio-1-1 SCC at this geometry, at the precision DFTB+ prints:
    //   Total energy = −4.0776668457 Ha × 27.211386245988 eV/Ha
    // (tools/tb_oracle/inputs/dftbplus/h2o_mio_scc/). Identical on 24.1 and on
    // 25.1 (commit a23bfb2), so the pin is not version-bound.
    constexpr double kDftbPlusEV = -110.9589675208;
    constexpr double kAbsTolEV = 1e-4;
    if (std::fabs(e - kDftbPlusEV) > kAbsTolEV) {
        std::printf("FAIL |E-DFTB+| = %.3e > tol %.1e eV (got %.10f)\n",
                    std::fabs(e - kDftbPlusEV), kAbsTolEV, e);
        return 1;
    }
    std::printf("  vs DFTB+: pin %.10f  |dE|=%.3e eV (tol %.1e)\n", kDftbPlusEV,
                std::fabs(e - kDftbPlusEV), kAbsTolEV);

    double f2 = 0;
    for (int i = 0; i < 3; ++i) {
        const double fxi = static_cast<double>(frame.span(kAtom_fx)[i]);
        const double fyi = static_cast<double>(frame.span(kAtom_fy)[i]);
        const double fzi = static_cast<double>(frame.span(kAtom_fz)[i]);
        std::printf("  F[%d]=(%.6f, %.6f, %.6f) eV/A\n", i, fxi, fyi, fzi);
        f2 += fxi * fxi + fyi * fyi + fzi * fzi;
        if (!std::isfinite(fxi) || !std::isfinite(fyi) || !std::isfinite(fzi)) {
            std::printf("FAIL non-finite force\n");
            return 1;
        }
    }
    if (f2 < 1e-8) {
        std::printf("FAIL all forces ~0 at off-eq geometry\n");
        return 1;
    }

    // Central FD on O_x vs analytic (loose window for product gate).
    const double h = 1e-4;
    auto energy_only = [](cpu::Frame g) {
        g.meta.get(kEnergy) = 0;
        for (int i = 0; i < 3; ++i)
            g.span(kAtom_fx)[i] = g.span(kAtom_fy)[i] = g.span(kAtom_fz)[i] = 0;

        cpu::semiempirical::DFTB2 d{cpu::semiempirical::DftbSkSet::MioSketch};
        // Isolated molecule in an isolation box — the DFTB+ twin is a `C`
        // geometry, so say so rather than letting a cutoff hide the images.
        d.boundary = cpu::semiempirical::DftbBoundary::cluster;

        d.scc_tol = 1e-10;
        d.run(g);
        return g.meta.get(kEnergy);
    };
    auto fp = make_water();
    fp.span(kAtom_x)[0] += static_cast<StorageReal>(h);
    auto fm = make_water();
    fm.span(kAtom_x)[0] -= static_cast<StorageReal>(h);
    const double fd = -(energy_only(fp) - energy_only(fm)) / (2.0 * h);
    const double fx0 = static_cast<double>(frame.span(kAtom_fx)[0]);
    std::printf("  F_O_x analytic=%.6f  fd=%.6f eV/A\n", fx0, fd);
    // A converged SCC-DFTB2 gradient is stationary, so fixed-Δq analytic is
    // FD-exact to the central-difference truncation error. Measured residual
    // 4.5e-08 eV/Å (2026-07-28); the old 1.5 eV/Å window was masking the
    // γ-gradient sign bug and the double-counting error.
    constexpr double kForceTol = 1e-5;
    if (std::fabs(fx0 - fd) > kForceTol) {
        std::printf("FAIL force vs FD (tol %.1f eV/A)\n", kForceTol);
        return 1;
    }

    // Exercise the concrete SCC node through the same pinned water case.
    // Observe the actual Runtime and loop state at the parent/child boundary.
    cpu::Runtime* runtime = nullptr;
    IterationStatus* parent_status = nullptr;
    int before = 0, inner = 0, after = 0;
    cpu::semiempirical::DFTB2 nested{cpu::semiempirical::DftbSkSet::MioSketch};
    nested.boundary = cpu::semiempirical::DftbBoundary::cluster;
    nested.scc_tol = 1e-10;
    nested.on(scc::Phase::check)
        .add(LoopAudit{runtime, parent_status, inner, 1});
    cpu::Evaluate parent;
    parent.on(evaluate::Phase::evaluate)
        .add(LoopAudit{runtime, parent_status, before, 0}, -1);
    parent.on(evaluate::Phase::evaluate).add_ref(nested);
    parent.on(evaluate::Phase::evaluate)
        .add(LoopAudit{runtime, parent_status, after, 2}, 1);
    parent.build();
    auto nested_frame = make_water();
    parent.run(nested_frame);
    if (before != 1 || after != 1 || inner != nested.last_scc_iterations()
        || !nested.converged()
        || std::fabs(nested_frame.meta.get(kEnergy) - e) > 1e-10) {
        std::printf("FAIL DFTB2 nested stage execution: before=%d after=%d "
                    "inner=%d iterations=%d converged=%d dE=%.3e\n",
                    before, after, inner, nested.last_scc_iterations(),
                    nested.converged(), nested_frame.meta.get(kEnergy) - e);
        return 1;
    }

    nested.max_scc = 1;
    bool rejected = false;
    try {
        parent.run(nested_frame);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    if (!rejected || nested.converged() || after != 1) {
        std::printf("FAIL DFTB2 exhausted budget published success\n");
        return 1;
    }
    nested.max_scc = 50;
    parent.run(nested_frame);
    if (after != 2 || !nested.converged()
        || std::fabs(nested_frame.meta.get(kEnergy) - e) > 1e-10) {
        std::printf("FAIL DFTB2 retry after SCC failure\n");
        return 1;
    }

    cpu::semiempirical::DFTB2 direct{cpu::semiempirical::DftbSkSet::MioSketch};
    direct.boundary = cpu::semiempirical::DftbBoundary::cluster;
    direct.scc_tol = 1e-10;
    auto result = make_water();
    result.meta.get(kEnergy) = 123.0;
    for (int a = 0; a < 3; ++a) {
        result.span(kAtom_fx)[a] = 17.0;
        result.span(kAtom_charge)[a] = 23.0;
    }
    direct.run(static_cast<const cpu::Frame&>(result));
    if (result.meta.get(kEnergy) != 123.0
        || result.span(kAtom_charge)[0] != 23.0) {
        std::printf("FAIL DFTB2 const Frame publication\n");
        return 1;
    }
    direct.run(result);
    for (int a = 0; a < 3; ++a)
        if (result.span(kAtom_fx)[a] != 17.0
            || std::fabs(result.span(kAtom_charge)[a]
                         + direct.last_mulliken_dq()[a])
                   > 1e-12) {
            std::printf("FAIL DFTB2 run force preservation or charge sign\n");
            return 1;
        }
    direct.forces(result);
    if (std::fabs(result.meta.get(kEnergy) - e) > 1e-10) {
        std::printf("FAIL DFTB2 Frame energy overwrite\n");
        return 1;
    }
    const std::array force_keys{kAtom_fx, kAtom_fy, kAtom_fz};
    for (auto key : force_keys)
        for (int a = 0; a < 3; ++a)
            if (std::fabs(result.span(key)[a] - frame.span(key)[a]) > 1e-10) {
                std::printf("FAIL DFTB2 standalone/nested force\n");
                return 1;
            }

    const auto direct_stress = direct.stress(result);
    for (int k = 0; k < 6; ++k)
        if (std::fabs(result.meta.get(kStress)[k] - frame.meta.get(kStress)[k])
                > 1e-10
            || std::fabs(direct_stress[k] * result.box.volume()
                         - result.meta.get(kStress)[k])
                   > 1e-10) {
            std::printf("FAIL DFTB2 direct/nested stress units or overwrite\n");
            return 1;
        }

    // A failure after SCC must leave the previous Frame result intact.
    // Invalid D3 cutoff exercises the post-SCC failure path deliberately.
    direct.d3 = true;
    direct.d3_cutoff = -1;
    rejected = false;
    try {
        direct.forces(result);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    if (!rejected || direct.converged() || std::isfinite(direct.last_energy())
        || std::fabs(result.meta.get(kEnergy) - e) > 1e-10) {
        std::printf("FAIL DFTB2 post-SCC failure publication\n");
        return 1;
    }
    for (auto key : force_keys)
        for (int a = 0; a < 3; ++a)
            if (std::fabs(result.span(key)[a] - frame.span(key)[a]) > 1e-10) {
                std::printf("FAIL DFTB2 failed force publication\n");
                return 1;
            }
    direct.d3 = false;
    direct.forces(result);
    if (!direct.converged()
        || std::fabs(result.meta.get(kEnergy) - e) > 1e-10) {
        std::printf("FAIL DFTB2 post-SCC failure retry\n");
        return 1;
    }

    // Reusing scientific buffers must reproduce a fresh solve after geometry,
    // species (even at the same AO count), and atom-count changes. The water
    // endpoint retains the independent DFTB+ pin checked above.
    for (int variant = 0; variant < 4; ++variant) {
        auto changed = make_water();
        if (variant == 0) {
            changed.span(kAtom_x)[1] += 0.08;
            changed.span(kAtom_z)[2] += 0.03;
        } else if (variant == 1) {
            changed.span(kAtom_type)[0] = 6;
        } else if (variant == 2) {
            changed = cpu::Frame(2);
            changed.box = cpu::Box::cubic(20.0);
            for (int a = 0; a < 2; ++a) {
                changed.span(kAtom_type)[a] = 1;
                changed.span(kAtom_x)[a] = 0.74 * a;
                changed.span(kAtom_y)[a] = changed.span(kAtom_z)[a] = 0.0;
            }
        }
        auto fresh_frame = changed;
        cpu::semiempirical::DFTB2 fresh{
            cpu::semiempirical::DftbSkSet::MioSketch};
        fresh.boundary = cpu::semiempirical::DftbBoundary::cluster;
        fresh.scc_tol = 1e-10;
        fresh.forces(fresh_frame);
        direct.forces(changed);
        if (std::fabs(changed.meta.get(kEnergy) - fresh_frame.meta.get(kEnergy))
            > 1e-10) {
            std::printf("FAIL DFTB2 rebuilt workspace energy (variant %d)\n",
                        variant);
            return 1;
        }
        for (auto key : {kAtom_fx, kAtom_fy, kAtom_fz, kAtom_charge})
            for (int a = 0; a < changed.nlocal(); ++a)
                if (std::fabs(changed.span(key)[a] - fresh_frame.span(key)[a])
                    > 1e-10) {
                    std::printf(
                        "FAIL DFTB2 rebuilt workspace output (variant %d)\n",
                        variant);
                    return 1;
                }
    }
    auto invalid_species = make_water();
    invalid_species.span(kAtom_type)[0] = 2;
    rejected = false;
    try {
        direct.forces(invalid_species);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    if (!rejected || std::isfinite(direct.last_energy())) {
        std::printf("FAIL DFTB2 invalid species rebuild\n");
        return 1;
    }
    direct.forces(result);
    if (std::fabs(result.meta.get(kEnergy) - kDftbPlusEV) > kAbsTolEV) {
        std::printf("FAIL DFTB2 retry after invalid species\n");
        return 1;
    }

    cpu::semiempirical::DFTB2 extra{cpu::semiempirical::DftbSkSet::MioSketch};
    extra.boundary = cpu::semiempirical::DftbBoundary::cluster;
    extra.scc_tol = 1e-10;
    cpu::Evaluate combined;
    combined.on(evaluate::Phase::evaluate).add_ref(direct);
    combined.on(evaluate::Phase::evaluate).add_ref(extra);
    combined.build();
    combined.run(result);
    if (std::fabs(result.meta.get(kEnergy) - 2.0 * e) > 1e-10) {
        std::printf("FAIL DFTB2 additive energy\n");
        return 1;
    }
    for (auto key : force_keys)
        for (int a = 0; a < 3; ++a)
            if (std::fabs(result.span(key)[a] - 2.0 * frame.span(key)[a])
                > 1e-10) {
                std::printf("FAIL DFTB2 additive force\n");
                return 1;
            }

    for (int k = 0; k < 6; ++k)
        if (std::fabs(result.meta.get(kStress)[k]
                      - 2.0 * frame.meta.get(kStress)[k])
            > 1e-10) {
            std::printf("FAIL DFTB2 additive stress\n");
            return 1;
        }

    std::printf("PASS\n");
    return 0;
}

} // namespace atv_reg_semiempirical_h2o_dftb2_mio

ATV_REGRESSION("semiempirical/dftb2", "h2o_dftb2_mio",
               atv_reg_semiempirical_h2o_dftb2_mio::run);
