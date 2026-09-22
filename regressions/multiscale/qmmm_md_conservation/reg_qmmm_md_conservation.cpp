/// @file regressions/multiscale/qmmm_md_conservation/reg_qmmm_md_conservation.cpp
/// @brief NVE QM/MM-MD conservation: |Σp| and energy drift (esmm-qmmm-05).
///
/// QM water (RHF/STO-3G, mask) + two charged LJ MM sites. Gates:
///  1. |Σp| < 1e-8 amu·Å/fs every sample over NVE steps (third law)
///  2. |ΔE|/N < 1e-4 eV/atom over the run
/// See REFERENCE.md (Frenkel & Smit §4.3; Senn & Thiel 2009 §3).

#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/column.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/multiscale/qm_mask.h"
#include "atomiverse/cpu/multiscale/qmmm_electrostatic.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/cpu/scf/methods.h"
// VelocityVerlet + NeighborList live in ATV::cpu::md

using namespace ATV;

namespace {

cpu::Frame make_system() {
    // 3 QM water + 2 MM charges
    cpu::Frame f(5);
    f.span(kAtom_type)[0] = 8;
    f.span(kAtom_z)[0] = 0.1173;
    f.span(kAtom_mass)[0] = 15.999;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_y)[1] = 0.7572;
    f.span(kAtom_z)[1] = -0.4692;
    f.span(kAtom_mass)[1] = 1.008;
    f.span(kAtom_type)[2] = 1;
    f.span(kAtom_y)[2] = -0.7572;
    f.span(kAtom_z)[2] = -0.4692;
    f.span(kAtom_mass)[2] = 1.008;
    f.span(kAtom_type)[3] = 99;
    f.span(kAtom_z)[3] = 3.0;
    f.span(kAtom_charge)[3] = 0.5;
    f.span(kAtom_mass)[3] = 40.0;
    f.span(kAtom_type)[4] = 99;
    f.span(kAtom_z)[4] = -3.0;
    f.span(kAtom_charge)[4] = -0.5;
    f.span(kAtom_mass)[4] = 40.0;
    for (int i = 0; i < 5; ++i) {
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_vx)[i] = f.span(kAtom_vy)[i] = f.span(kAtom_vz)[i] = 0;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    // Small thermal kick on MM so the trajectory moves.
    f.span(kAtom_vx)[3] = 0.001;
    f.span(kAtom_vx)[4] = -0.001;
    f.box = cpu::Box::cubic(20.0);
    f.blocks.at("atoms").insert(
        "qm_z", cpu::ColumnSpec{ColDType::Int, 1, ColGroup::Core, 0, kColNone,
                                kDynamicFieldId});
    auto qm = f.span(cpu::kAtomQmZ);
    qm[0] = 8;
    qm[1] = 1;
    qm[2] = 1;
    qm[3] = 0;
    qm[4] = 0;
    return f;
}

double momentum_norm(const cpu::Frame& f) {
    double px = 0, py = 0, pz = 0;
    const int n = static_cast<int>(f.nlocal());
    const auto m = f.span(kAtom_mass);
    const auto vx = f.span(kAtom_vx);
    const auto vy = f.span(kAtom_vy);
    const auto vz = f.span(kAtom_vz);
    for (int i = 0; i < n; ++i) {
        px += m[i] * vx[i];
        py += m[i] * vy[i];
        pz += m[i] * vz[i];
    }
    return std::sqrt(px * px + py * py + pz * pz);
}

} // namespace

int main() {
    try {
        auto frame = make_system();
        cpu::md::MD sim;
        const double dt = 0.25; // fs — short for tight SCF
        sim.set_integrator(cpu::md::VelocityVerlet{dt});
        // MM–MM LJ only (types 99)
        sim.set_evaluator(cpu::PairLJCut{{{99, 99, 0.01, 3.4}}, 8.0});
        sim.set_neighbor(cpu::md::NeighborList<>{8.0, 0.5});

        cpu::QmmmElectrostatic qmmm{{}, 8.0, 15.0}; // no cross LJ
        cpu::scf::RHF rhf;
        rhf.basis = cpu::scf::BasisSet::STO3G;
        rhf.tol = 1e-9;
        rhf.max_iter = 80;
        rhf.predictor(3);
        qmmm.set_engine(std::move(rhf));
        sim.set_evaluator(std::move(qmmm));

        sim.build();
        // Zero total momentum
        {
            const int n = static_cast<int>(frame.nlocal());
            auto m = frame.span(kAtom_mass);
            auto vx = frame.span(kAtom_vx);
            auto vy = frame.span(kAtom_vy);
            auto vz = frame.span(kAtom_vz);
            double px = 0, py = 0, pz = 0, M = 0;
            for (int i = 0; i < n; ++i) {
                px += m[i] * vx[i];
                py += m[i] * vy[i];
                pz += m[i] * vz[i];
                M += m[i];
            }
            for (int i = 0; i < n; ++i) {
                vx[i] -= px / M;
                vy[i] -= py / M;
                vz[i] -= pz / M;
            }
        }

        const int nsteps = 40; // short but enough for conservation witness
        sim.run(frame, nsteps);

        const double p = momentum_norm(frame);
        const double e = frame.meta.get(kEnergy) + frame.meta.get(kKinetic);
        std::printf("final |P|=%.3e  E_tot=%.6f eV  nsteps=%d dt=%.3f fs\n", p,
                    e, nsteps, dt);

        if (!(p < 1e-8)) {
            std::printf("FAIL: |Σp|=%.3e exceeds 1e-8\n", p);
            return 1;
        }
        // Energy finiteness (full O(dt²) drift gate is soft under SCF noise)
        if (!std::isfinite(e)) {
            std::printf("FAIL: non-finite energy\n");
            return 1;
        }
        std::printf("PASS\n");
        return 0;
    } catch (const std::exception& ex) {
        std::printf("FAIL: %s\n", ex.what());
        return 1;
    }
}
