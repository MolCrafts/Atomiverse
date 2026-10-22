/// @file regressions/md/cuda-npt-pme-nacl/reg_cuda-npt-pme-nacl.cu
/// @brief CUDA PME under a changing cell: reciprocal virial, MTK NPT and the
///        fixed-mesh error as the box changes. Needs ATV_BUILD_KSPACE.
///
/// Gates (REFERENCE.md):
///   A. Madelung virial: a rocksalt point-charge lattice has total Coulomb
///      energy E = −N M k_e / (2d) (M = 1.747564594633182) and, being
///      homogeneous of degree −1, virial W_aa = E/3, W_ab = 0. CoulLong +
///      KSpacePME must reproduce both within the PME accuracy (3e-3).
///   B. MTK NPT with PME, CPU twin: same Joung–Cheatham NaCl crystal, same
///      deterministic velocities, cpu:: and cuda:: PairLJCutCoulLong +
///      KSpacePME + NoseHooverNPT. Step-0 virial, 100-step volume and the
///      conserved quantity must agree; the CUDA conserved quantity must not
///      drift.
///   C. Fixed mesh under box change: the mesh chosen at build is kept while
///      the cell scales by s; the energy/force error against a converged
///      Ewald sum is recorded for s in [0.95, 1.05] and must stay within
///      the PME accuracy class of the build-time cell.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <exception>
#include <type_traits>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/nose_hoover_npt.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/longrange/pme.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/observe/virial_stress.h"
#include "atomiverse/cpu/potentials/pair_lj_cut_coul_long.h"
#include "atomiverse/cuda/dynamics/md.cuh"
#include "atomiverse/cuda/dynamics/nose_hoover_npt.cuh"
#include "atomiverse/cuda/evaluate/evaluate.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/cuda/longrange/ewald.cuh"
#include "atomiverse/cuda/longrange/pme.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/coul_long.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut_coul_long.cuh"
#include "harness/registry.h"

namespace atv_reg_md_cuda_npt_pme_nacl {

using namespace ATV;

namespace {

constexpr double kMadelung = 1.747564594633182;
constexpr double kD = 2.82; // Na–Cl nearest-neighbour distance [A]
constexpr int kSide = 8;    // 8^3 = 512 ions, L = 22.56 A
constexpr double kAlpha = 0.35;
constexpr double kCut = 9.0;
constexpr double kSpacing = 1.0; // L / spacing = 22.56 -> mesh 24 at +-1 %
constexpr int kOrder = 5;
constexpr bool kFloatForce = std::is_same_v<cuda::force_t, float>;

// Joung & Cheatham, J. Phys. Chem. B 112, 9020 (2008), TIP3P set:
// Na sigma 2.439 A eps 0.0874 kcal/mol, Cl sigma 4.478 A eps 0.0356 kcal/mol.
constexpr double kKcal = 0.0433641043; // eV per kcal/mol
const std::vector<double> kSigma = {2.439, 4.478};
const std::vector<double> kEpsilon = {0.0874 * kKcal, 0.0356 * kKcal};

cpu::Frame rocksalt(double d) {
    const int n = kSide * kSide * kSide;
    cpu::Frame f(n);
    int idx = 0;
    for (int i = 0; i < kSide; ++i)
        for (int j = 0; j < kSide; ++j)
            for (int k = 0; k < kSide; ++k) {
                const bool na = (i + j + k) % 2 == 0;
                f.span(kAtom_x)[idx] = (i + 0.25) * d;
                f.span(kAtom_y)[idx] = (j + 0.25) * d;
                f.span(kAtom_z)[idx] = (k + 0.25) * d;
                f.span(kAtom_charge)[idx] = na ? 1.0 : -1.0;
                f.span(kAtom_mass)[idx] = na ? 22.98977 : 35.453;
                f.span(kAtom_type)[idx] = na ? 0 : 1;
                f.span(kAtom_tag)[idx] = idx;
                // Deterministic, momentum-free thermal pattern (~300 K).
                const double s = na ? 0.0035 : -0.0035 * 22.98977 / 35.453;
                f.span(kAtom_vx)[idx] = s * ((i + 2 * j) % 3 - 1);
                f.span(kAtom_vy)[idx] = s * ((j + 2 * k) % 3 - 1);
                f.span(kAtom_vz)[idx] = s * ((k + 2 * i) % 3 - 1);
                ++idx;
            }
    f.box = cpu::Box::cubic(kSide * d);
    f.rebuild_tag_index(0, n);
    return f;
}

void scale_frame(cpu::Frame& f, double s) {
    for (auto key : {kAtom_x, kAtom_y, kAtom_z})
        for (int i = 0; i < f.nlocal(); ++i)
            f.span(key)[i] *= s;
    f.box = cpu::Box::cubic(f.box.length(0) * s);
}

// ── Gate A ──────────────────────────────────────────────────────────────
bool madelung_virial() {
    cpu::Frame f = rocksalt(kD);
    for (int i = 0; i < f.nlocal(); ++i)
        f.span(kAtom_type)[i] = 0;
    cuda::Evaluate eval;
    eval.set_neighbor(cuda::NeighborList{kCut});
    eval.set_evaluator(cuda::CoulLong{kAlpha, kCut});
    eval.on(evaluate::Phase::evaluate)
        .add(cuda::KSpacePME{kAlpha, 0.5, 6}, 100);
    eval.build();
    eval.run(f);
    const int n = f.nlocal();
    const double expected = -0.5 * n * kMadelung * units::coulomb / kD;
    const double e = f.meta.get(kEnergy);
    const auto& w = f.meta.get(kStress);
    double diag = 0, off = 0;
    for (int a = 0; a < 3; ++a)
        diag = std::max(diag, std::abs(w[a] / (expected / 3.0) - 1.0));
    for (int a = 3; a < 6; ++a)
        off = std::max(off, std::abs(w[a] / expected));
    const double de = std::abs(e / expected - 1.0);
    std::printf("A: E=%.9f (rel %.3g) W=[%.6f %.6f %.6f | %.3g %.3g %.3g] "
                "diag rel %.3g off rel %.3g\n",
                e, de, w[0], w[1], w[2], w[3], w[4], w[5], diag, off);
    return de < 3e-3 && diag < 3e-3 && off < 1e-6;
}

// ── Gate B ──────────────────────────────────────────────────────────────
struct NptOut {
    double w0[6] = {};
    double volume = 0, H0 = 0, H = 0;
};

constexpr double kT = 300.0, kTauT = 100.0, kTauP = 1000.0, kDt = 1.0;
const double kPressure = units::bar_to_ev_per_a3; // 1 bar

std::vector<cpu::LJTypeCoeff> lb_coeffs() {
    std::vector<cpu::LJTypeCoeff> coeffs;
    for (int a = 0; a < 2; ++a)
        for (int b = a; b < 2; ++b)
            coeffs.push_back({a, b, std::sqrt(kEpsilon[a] * kEpsilon[b]),
                              0.5 * (kSigma[a] + kSigma[b])});
    return coeffs;
}

NptOut npt_cpu(int steps) {
    NptOut out;
    {
        cpu::Frame g = rocksalt(kD);
        cpu::Evaluate eval;
        eval.set_neighbor(cpu::md::NeighborList<>{kCut, 1.0});
        eval.set_evaluator(cpu::PairLJCutCoulLong{lb_coeffs(), kAlpha, kCut});
        eval.on(evaluate::Phase::evaluate)
            .add(cpu::KSpacePME{kAlpha, kSpacing, kOrder}, 100);
        eval.build();
        eval.run(g);
        const auto& w = g.meta.get(kStress);
        for (int k = 0; k < 6; ++k)
            out.w0[k] = w[k];
    }
    cpu::Frame g = rocksalt(kD);
    cpu::md::MD sim;
    sim.set_neighbor(cpu::md::NeighborList<>{kCut, 1.0});
    sim.set_evaluator(cpu::PairLJCutCoulLong{lb_coeffs(), kAlpha, kCut});
    sim.on(md::Phase::force).add(cpu::KSpacePME{kAlpha, kSpacing, kOrder}, 100);
    sim.set_integrator(
        cpu::md::NoseHooverNPT{kT, kTauT, kPressure, kTauP, kDt});
    sim.set_observer(cpu::md::Virial{});
    sim.enable_timing(false);
    sim.build();
    sim.run(g, 1);
    out.H0 = g.meta.get(kConserved);
    sim.run(g, steps - 1);
    out.volume = g.box.volume();
    out.H = g.meta.get(kConserved);
    return out;
}

NptOut npt_cuda(int steps) {
    NptOut out;
    {
        cpu::Frame g = rocksalt(kD);
        cuda::Evaluate eval;
        eval.set_neighbor(cuda::NeighborList{kCut, 1.0});
        eval.set_evaluator(
            cuda::PairLJCutCoulLong{kEpsilon, kSigma, kAlpha, kCut});
        eval.on(evaluate::Phase::evaluate)
            .add(cuda::KSpacePME{kAlpha, kSpacing, kOrder}, 100);
        eval.build();
        eval.run(g);
        const auto& w = g.meta.get(kStress);
        for (int k = 0; k < 6; ++k)
            out.w0[k] = w[k];
    }
    cpu::Frame g = rocksalt(kD);
    cuda::md::MD sim;
    sim.set_neighbor(cuda::NeighborList{kCut, 1.0});
    sim.set_evaluator(cuda::PairLJCutCoulLong{kEpsilon, kSigma, kAlpha, kCut});
    sim.on(md::Phase::force)
        .add(cuda::KSpacePME{kAlpha, kSpacing, kOrder}, 100);
    sim.set_integrator(
        cuda::md::NoseHooverNPT{kT, kTauT, kPressure, kTauP, kDt});
    sim.build();
    sim.run(g, 1);
    out.H0 = g.meta.get(kConserved);
    sim.run(g, steps - 1);
    out.volume = g.box.volume();
    out.H = g.meta.get(kConserved);
    return out;
}

bool npt_parity() {
    constexpr int steps = 100;
    const NptOut c = npt_cpu(steps), g = npt_cuda(steps);
    const double V0 = std::pow(kSide * kD, 3);
    double dw = 0, scale = 0;
    for (int k = 0; k < 6; ++k) {
        dw = std::max(dw, std::abs(g.w0[k] - c.w0[k]));
        scale = std::max(scale, std::abs(c.w0[k]));
    }
    const double dv = std::abs(g.volume - c.volume) / c.volume;
    const int n = kSide * kSide * kSide;
    const double drift = std::abs(g.H - g.H0) / n;
    std::printf("B: step-0 |dW|/|W|=%.3g  V0=%.4f V_cpu=%.6f V_cuda=%.6f "
                "dV/V=%.3g  H drift cuda=%.3g eV/atom cpu=%.3g eV/atom\n",
                dw / scale, V0, c.volume, g.volume, dv, drift,
                std::abs(c.H - c.H0) / n);
    // The CPU twin tabulates erfc (LAMMPS-style table): ~1e-6 relative in
    // each real-space pair term, ~1e-5 in the summed virial.
    const double w_tol = kFloatForce ? 1e-4 : 3e-5;
    const double v_tol = kFloatForce ? 1e-4 : 1e-5;
    return dw / scale < w_tol && dv < v_tol && std::abs(c.volume - V0) > 0
           && drift < 2e-4;
}

// ── Gate C ──────────────────────────────────────────────────────────────
struct Snapshot {
    double energy = 0;
    std::vector<double> force;
};

template <typename KSpace>
Snapshot evaluate(cpu::Frame f, KSpace&& kspace) {
    cuda::Evaluate eval;
    eval.set_neighbor(cuda::NeighborList{kCut});
    eval.set_evaluator(cuda::CoulLong{kAlpha, kCut});
    eval.on(evaluate::Phase::evaluate).add(std::forward<KSpace>(kspace), 100);
    eval.build();
    eval.run(f);
    Snapshot s;
    s.energy = f.meta.get(kEnergy);
    for (auto key : {kAtom_fx, kAtom_fy, kAtom_fz})
        for (int i = 0; i < f.nlocal(); ++i)
            s.force.push_back(f.span(key)[i]);
    return s;
}

bool fixed_mesh_record() {
    // A perturbed lattice so forces are nonzero.
    cpu::Frame base = rocksalt(kD);
    for (int i = 0; i < base.nlocal(); ++i) {
        base.span(kAtom_x)[i] += 0.05 * std::sin(1.7 * i);
        base.span(kAtom_y)[i] += 0.05 * std::sin(2.3 * i + 1);
        base.span(kAtom_z)[i] += 0.05 * std::sin(3.1 * i + 2);
    }
    double nominal = 0;
    bool ok = true;
    for (double s : {1.0, 0.95, 0.98, 1.02, 1.05}) {
        cpu::Frame f = base;
        scale_frame(f, s);
        // Spacing scaled with the cell keeps the build-time 24^3 mesh:
        // exactly the mesh a running PME keeps while the box changes.
        const Snapshot fixed =
            evaluate(f, cuda::KSpacePME{kAlpha, kSpacing * s, kOrder});
        const Snapshot ewald = evaluate(f, cuda::KSpaceEwald{kAlpha, 1e-10});
        double df = 0, fmax = 0;
        for (std::size_t i = 0; i < fixed.force.size(); ++i) {
            df = std::max(df, std::abs(fixed.force[i] - ewald.force[i]));
            fmax = std::max(fmax, std::abs(ewald.force[i]));
        }
        const double de = std::abs(fixed.energy / ewald.energy - 1.0);
        const double rel_f = df / fmax;
        std::printf("C: s=%.2f  |dE/E|=%.3g  max|dF|/max|F|=%.3g\n", s, de,
                    rel_f);
        if (s == 1.0)
            nominal = std::max(rel_f, 1e-6);
        else
            ok = ok && rel_f < 5 * nominal && de < 1e-4;
    }
    return ok;
}

} // namespace

int run() try {
    const bool a = madelung_virial();
    const bool b = npt_parity();
    const bool c = fixed_mesh_record();
    std::printf("cuda-npt-pme-nacl: A=%s B=%s C=%s\n", a ? "pass" : "FAIL",
                b ? "pass" : "FAIL", c ? "pass" : "FAIL");
    return a && b && c ? 0 : 1;
} catch (const std::exception& e) {
    std::fprintf(stderr, "cuda-npt-pme-nacl: %s\n", e.what());
    return 1;
}

} // namespace atv_reg_md_cuda_npt_pme_nacl

ATV_REGRESSION("md/npt", "cuda-npt-pme-nacl",
               atv_reg_md_cuda_npt_pme_nacl::run);
