/// @file tests/cuda/potentials/test_energy_terms.cu
/// @brief Every CUDA force-field style reports its energy in its own
///        per-term scalar (`energy_pair` / `_bond` / `_angle` / `_dihedral` /
///        `_improper`) exactly as the same-settings CPU style writes
///        `kEnergyPair` / `kEnergyBond` / …, so `energy` is the sum of the
///        terms on both backends (LAMMPS thermo `epair ebond eangle edihed
///        eimp`). Reciprocal-space self/mesh/background and LJ tail energies
///        count as pair energy, as on CPU.
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/lj_type_coeff.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/frame/transfer.h"
#include "atomiverse/cpu/longrange/ewald.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/angle_cosine.h"
#include "atomiverse/cpu/potentials/angle_harmonic.h"
#include "atomiverse/cpu/potentials/bond_fene.h"
#include "atomiverse/cpu/potentials/bond_harmonic.h"
#include "atomiverse/cpu/potentials/coul_long.h"
#include "atomiverse/cpu/potentials/dihedral_multi_harmonic.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/cpu/potentials/pair_lj_cut_coul_long.h"
#include "atomiverse/cuda/component/context.cuh"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/evaluate/evaluate.cuh"
#include "atomiverse/cuda/frame/transfer.cuh"
#include "atomiverse/cuda/longrange/ewald.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/angle_cosine.cuh"
#include "atomiverse/cuda/potentials/angle_harmonic.cuh"
#include "atomiverse/cuda/potentials/bond_fene.cuh"
#include "atomiverse/cuda/potentials/bond_harmonic.cuh"
#include "atomiverse/cuda/potentials/coul_long.cuh"
#include "atomiverse/cuda/potentials/dihedral_multi_harmonic.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut_coul_long.cuh"
#ifdef ATV_HAS_KSPACE
#include "atomiverse/cpu/longrange/pme.h"
#include "atomiverse/cpu/longrange/pppm.h"
#include "atomiverse/cuda/longrange/pme.cuh"
#endif

using namespace ATV;

namespace {

constexpr bool kFloat = std::is_same_v<cuda::force_t, float>;
/// CUDA vs CPU of the same style (FP64 work: summation order only).
constexpr double kParity = kFloat ? 1e-4 : 1e-9;
/// CUDA vs CPU through a reciprocal mesh (FP32 mesh in float mode).
constexpr double kMeshParity = kFloat ? 1e-3 : 1e-7;
/// A term against its total on one backend (same addends, other order).
constexpr double kSelf = 1e-12;

struct TermKey {
    const char* name;
    const MetaKey<double>* key;
};
const TermKey kTerms[5] = {{"pair", &kEnergyPair},
                           {"bond", &kEnergyBond},
                           {"angle", &kEnergyAngle},
                           {"dihedral", &kEnergyDihedral},
                           {"improper", &kEnergyImproper}};

double rel(double ref, double tol) { return tol * (1.0 + std::abs(ref)); }

/// @brief One style alone: its term carries the whole energy on both
///        backends and matches CPU; every other term stays exactly 0.
void expect_only_term(const cpu::Frame& gpu, const cpu::Frame& cpu,
                      const MetaKey<double>& term, double parity) {
    const double e_gpu = gpu.meta.get(kEnergy);
    const double e_cpu = cpu.meta.get(kEnergy);
    ASSERT_GT(std::abs(e_cpu), 1e-6) << "vacuous: the style has no energy";
    for (const TermKey& t : kTerms) {
        const double g = gpu.meta.get(*t.key);
        const double c = cpu.meta.get(*t.key);
        if (t.key == &term) {
            EXPECT_NEAR(g, e_gpu, rel(e_gpu, kSelf)) << t.name << " (CUDA)";
            EXPECT_NEAR(c, e_cpu, rel(e_cpu, kSelf)) << t.name << " (CPU)";
            EXPECT_NEAR(g, c, rel(c, parity)) << t.name << " CUDA vs CPU";
        } else {
            EXPECT_EQ(g, 0.0) << t.name << " (CUDA)";
            EXPECT_EQ(c, 0.0) << t.name << " (CPU)";
        }
    }
    EXPECT_NEAR(e_gpu, e_cpu, rel(e_cpu, parity));
}

// ── Bonded styles, driven directly (one launch on a fresh Frame) ──────────

constexpr double kBondedBox = 14.0;

/// Three five-atom chains with off-plane kinks: 4 bonds, 3 angles and 2
/// dihedrals per chain, two types each; chain 2 is wrapped through x = L.
cpu::Frame chains_frame() {
    constexpr int kChains = 3, kPer = 5, n = kChains * kPer;
    cpu::Frame f(n);
    for (int c = 0; c < kChains; ++c)
        for (int k = 0; k < kPer; ++k) {
            const int i = c * kPer + k;
            const double x = 1.0 + 4.0 * c + 1.1 * k + (c == 2 ? 3.0 : 0.0);
            f.span(kAtom_x)[i] = std::fmod(x, kBondedBox);
            f.span(kAtom_y)[i] = 3.0 + 3.5 * c + 0.9 * (k % 2) + 0.07 * k * k;
            // Bond lengths stay in 1.1–1.8 Å, inside both FENE R0.
            f.span(kAtom_z)[i] = 5.0 + 0.5 * std::sin(1.9 * k + c);
        }
    for (int i = 0; i < n; ++i) {
        f.span(kAtom_tag)[i] = i;
        f.span(kAtom_mass)[i] = 1.0;
        f.span(kAtom_type)[i] = i % 2;
        f.span(kAtom_charge)[i] = 0.0;
    }
    f.box = cpu::Box::cubic(kBondedBox);
    f.clear_tag_index();
    f.rebuild_tag_index(0, n);
    for (int c = 0; c < kChains; ++c) {
        const int b = c * kPer;
        for (int k = 0; k + 1 < kPer; ++k)
            f.add_bond(b + k, b + k + 1, k % 2);
        for (int k = 0; k + 2 < kPer; ++k)
            f.add_angle(b + k, b + k + 1, b + k + 2, k % 2);
        for (int k = 0; k + 3 < kPer; ++k)
            f.add_dihedral(b + k, b + k + 1, b + k + 2, b + k + 3, k % 2);
    }
    f.build_special_list();
    return f;
}

struct DeviceInput {
    cuda::Frame frame;
    cuda::Context context;
    explicit DeviceInput(const cpu::Frame& source)
        : frame(cpu::to_cuda(source)) {
        cuda::geom_t cell[9] = {kBondedBox, 0, 0, 0,         kBondedBox,
                                0,          0, 0, kBondedBox};
        context.host_box.assign_cell(cell);
    }
    ~DeviceInput() { cuda::destroy(frame); }
};

template <class CpuStyle, class CudaStyle>
void expect_bonded_term(CpuStyle cpu_style, CudaStyle cuda_style,
                        const MetaKey<double>& term) {
    cpu::Frame cpu_frame = chains_frame();
    cpu::Context cpu_ctx;
    cpu_style.build({cpu_ctx, cpu_frame});
    cpu_style.launch({cpu_ctx, cpu_frame, 0, 0, true, true});
    cpu_style.destroy({cpu_ctx, cpu_frame});

    cpu::Frame host = chains_frame();
    DeviceInput input(host);
    cuda::BuildContext build{input.context, input.frame};
    cuda_style.build(build);
    cuda_style.launch({input.context, input.frame, 0, 0});
    ATV_CUDA_CHECK(cudaDeviceSynchronize());
    cuda::to_cpu(input.frame, host);
    cuda_style.destroy({input.context, input.frame});
    expect_only_term(host, cpu_frame, term, kParity);
}

const std::vector<double> kBondK = {12.0, 7.5};
const std::vector<double> kBondR0 = {1.0, 1.3};
const std::vector<double> kFeneR0 = {2.2, 2.6};
const std::vector<double> kAngleK = {1.4, 0.8};
const std::vector<double> kTheta0 = {1.9, 2.3}; // rad
const std::vector<std::vector<double>> kDihCpu = {
    {0.1, -0.2, 0.3, 0.05, -0.04}, {0.2, 0.1, -0.15, 0.02, 0.01}};
const std::vector<std::array<double, 5>> kDihCuda = {
    {0.1, -0.2, 0.3, 0.05, -0.04}, {0.2, 0.1, -0.15, 0.02, 0.01}};

// ── Pair / reciprocal styles, driven through Evaluate on both backends ────

constexpr double kPairBox = 12.0;
constexpr double kCut = 5.0;
constexpr double kSkin = 0.5;
constexpr double kAlpha = 0.32;

/// 64 atoms on a jittered 4×4×4 lattice (spacing 3 Å), two LJ types,
/// alternating ±0.4 e (neutral).
cpu::Frame lattice_frame() {
    constexpr int side = 4, n = side * side * side;
    cpu::Frame f(n);
    int i = 0;
    for (int a = 0; a < side; ++a)
        for (int b = 0; b < side; ++b)
            for (int c = 0; c < side; ++c, ++i) {
                const double j = 0.18 * std::sin(1.7 * i + 0.3);
                f.span(kAtom_x)[i] = 0.4 + 3.0 * a + j;
                f.span(kAtom_y)[i] = 0.7 + 3.0 * b - 0.8 * j;
                f.span(kAtom_z)[i] = 1.1 + 3.0 * c + 0.5 * j;
                f.span(kAtom_charge)[i] = ((a + b + c) % 2) ? -0.4 : 0.4;
                f.span(kAtom_type)[i] = (a + c) % 2;
                f.span(kAtom_mass)[i] = 10.0;
                f.span(kAtom_tag)[i] = i;
            }
    f.box = cpu::Box::cubic(kPairBox);
    f.clear_tag_index();
    f.rebuild_tag_index(0, n);
    f.build_special_list();
    return f;
}

std::vector<LJTypeCoeff> lj_rows() {
    return {{0, 0, 0.012, 2.9}, {1, 1, 0.020, 3.2}, {0, 1, 0.015, 3.05}};
}

/// CPU Evaluate over the given styles (moved in).
template <class... Styles>
cpu::Frame run_cpu(cpu::Frame f, Styles&&... styles) {
    cpu::Evaluate eval;
    eval.set_neighbor(cpu::md::NeighborList<>{kCut, kSkin, 1, 0, true});
    (eval.set_evaluator(std::forward<Styles>(styles)), ...);
    eval.enable_timing(false);
    eval.build();
    eval.run(f);
    return f;
}

/// CUDA Evaluate over the given styles (moved in).
template <class... Styles>
cpu::Frame run_cuda(cpu::Frame f, Styles&&... styles) {
    cuda::Evaluate eval;
    eval.set_neighbor(cuda::NeighborList{kCut, kSkin});
    (eval.set_evaluator(std::forward<Styles>(styles)), ...);
    eval.build();
    eval.run(f);
    return f;
}

} // namespace

// ── Bond / angle / dihedral ────────────────────────────────────────────────

TEST(CudaEnergyTerms, BondHarmonicIsBondEnergy) {
    expect_bonded_term(cpu::BondHarmonic(kBondK, kBondR0),
                       cuda::BondHarmonic(kBondK, kBondR0), kEnergyBond);
}

TEST(CudaEnergyTerms, BondFeneIsBondEnergy) {
    expect_bonded_term(cpu::BondFene(kBondK, kFeneR0),
                       cuda::BondFene(kBondK, kFeneR0), kEnergyBond);
}

TEST(CudaEnergyTerms, AngleHarmonicIsAngleEnergy) {
    expect_bonded_term(cpu::AngleHarmonic(kAngleK, kTheta0),
                       cuda::AngleHarmonic(kAngleK, kTheta0), kEnergyAngle);
}

TEST(CudaEnergyTerms, AngleCosineIsAngleEnergy) {
    expect_bonded_term(cpu::AngleCosine(kAngleK), cuda::AngleCosine(kAngleK),
                       kEnergyAngle);
}

TEST(CudaEnergyTerms, DihedralMultiHarmonicIsDihedralEnergy) {
    expect_bonded_term(cpu::DihedralMultiHarmonic(kDihCpu),
                       cuda::DihedralMultiHarmonic(kDihCuda), kEnergyDihedral);
}

// A step without eflag leaves the total and every term untouched.
TEST(CudaEnergyTerms, UnrequestedEnergyLeavesEveryTermAtZero) {
    cpu::Frame host = chains_frame();
    DeviceInput input(host);
    cuda::BondHarmonic bond(kBondK, kBondR0);
    cuda::BuildContext build{input.context, input.frame};
    bond.build(build);
    cuda::LaunchContext ctx{input.context, input.frame, 0, 0};
    ctx.energy_requested = false;
    bond.launch(ctx);
    ATV_CUDA_CHECK(cudaDeviceSynchronize());
    cuda::to_cpu(input.frame, host);
    bond.destroy({input.context, input.frame});
    EXPECT_EQ(host.meta.get(kEnergy), 0.0);
    for (const TermKey& t : kTerms)
        EXPECT_EQ(host.meta.get(*t.key), 0.0) << t.name;
}

// ── Pair styles (real space + tail) ────────────────────────────────────────

TEST(CudaEnergyTerms, PairLJCutIsPairEnergy) {
    const cpu::Frame c =
        run_cpu(lattice_frame(), cpu::PairLJCut(0.012, 2.9, kCut));
    const cpu::Frame g =
        run_cuda(lattice_frame(), cuda::PairLJCut(0.012, 2.9, kCut));
    expect_only_term(g, c, kEnergyPair, kParity);
}

// Typed table + LAMMPS tail correction: the tail is pair energy.
TEST(CudaEnergyTerms, TypedPairLJCutWithTailIsPairEnergy) {
    const cpu::Frame c =
        run_cpu(lattice_frame(),
                cpu::PairLJCut(lj_rows(), kCut).set_tail_correction(true));
    const cpu::Frame g =
        run_cuda(lattice_frame(),
                 cuda::PairLJCut(lj_rows(), kCut).set_tail_correction(true));
    expect_only_term(g, c, kEnergyPair, kParity);
}

// Real-space Coulomb + Ewald (self, reciprocal and background terms).
TEST(CudaEnergyTerms, CoulLongWithEwaldIsPairEnergy) {
    const cpu::Frame c = run_cpu(lattice_frame(), cpu::CoulLong(kAlpha, kCut),
                                 cpu::KSpaceEwald(kAlpha, 1e-10));
    const cpu::Frame g = run_cuda(lattice_frame(), cuda::CoulLong(kAlpha, kCut),
                                  cuda::KSpaceEwald(kAlpha, 1e-10));
    expect_only_term(g, c, kEnergyPair, kMeshParity);
}

#ifdef ATV_HAS_KSPACE
TEST(CudaEnergyTerms, CoulLongWithPmeIsPairEnergy) {
    const cpu::Frame c = run_cpu(lattice_frame(), cpu::CoulLong(kAlpha, kCut),
                                 cpu::KSpacePME(kAlpha, 0.5, 6));
    const cpu::Frame g = run_cuda(lattice_frame(), cuda::CoulLong(kAlpha, kCut),
                                  cuda::KSpacePME(kAlpha, 0.5, 6));
    expect_only_term(g, c, kEnergyPair, kMeshParity);
}

// The fused LJ + real-space Coulomb style with a tail, under PPPM.
TEST(CudaEnergyTerms, LJCutCoulLongWithPppmIsPairEnergy) {
    const cpu::Frame c = run_cpu(lattice_frame(),
                                 cpu::PairLJCutCoulLong(lj_rows(), kAlpha, kCut)
                                     .set_tail_correction(true),
                                 cpu::KSpacePPPM(kAlpha, 0.5, 5));
    const cpu::Frame g =
        run_cuda(lattice_frame(),
                 cuda::PairLJCutCoulLong(lj_rows(), kAlpha, kCut)
                     .set_tail_correction(true),
                 cuda::KSpacePPPM(kAlpha, 0.5, 5));
    expect_only_term(g, c, kEnergyPair, kMeshParity);
}
#endif

// ── Several families at once: energy is the sum of the terms ───────────────

TEST(CudaEnergyTerms, MixedForceFieldEnergyIsTheSumOfTheTerms) {
    // The chains frame is small for a 5 Å neighbor list, so the pair style
    // runs at a 2.5 Å cutoff; bonded partners are excluded (special 0 0 0).
    auto cpu_run = [] {
        cpu::Frame f = chains_frame();
        cpu::Evaluate eval;
        eval.set_neighbor(cpu::md::NeighborList<>{2.5, 0.3, 1, 0, true});
        eval.set_evaluator(cpu::PairLJCut(0.02, 1.0, 2.5));
        eval.set_evaluator(cpu::BondHarmonic(kBondK, kBondR0));
        eval.set_evaluator(cpu::AngleHarmonic(kAngleK, kTheta0));
        eval.set_evaluator(cpu::DihedralMultiHarmonic(kDihCpu));
        eval.enable_timing(false);
        eval.build();
        eval.run(f);
        return f;
    };
    auto cuda_run = [] {
        cpu::Frame f = chains_frame();
        cuda::Evaluate eval;
        eval.set_neighbor(cuda::NeighborList{2.5, 0.3});
        eval.set_evaluator(cuda::PairLJCut(0.02, 1.0, 2.5));
        eval.set_evaluator(cuda::BondHarmonic(kBondK, kBondR0));
        eval.set_evaluator(cuda::AngleHarmonic(kAngleK, kTheta0));
        eval.set_evaluator(cuda::DihedralMultiHarmonic(kDihCuda));
        eval.build();
        eval.run(f);
        return f;
    };
    const cpu::Frame c = cpu_run();
    const cpu::Frame g = cuda_run();
    for (const cpu::Frame* f : {&c, &g}) {
        double sum = 0;
        for (const TermKey& t : kTerms)
            sum += f->meta.get(*t.key);
        EXPECT_NEAR(sum, f->meta.get(kEnergy),
                    rel(f->meta.get(kEnergy), kSelf));
    }
    for (const TermKey& t : kTerms)
        EXPECT_NEAR(g.meta.get(*t.key), c.meta.get(*t.key),
                    rel(c.meta.get(*t.key), kParity))
            << t.name;
    EXPECT_NE(c.meta.get(kEnergyBond), 0.0);
    EXPECT_NE(c.meta.get(kEnergyAngle), 0.0);
    EXPECT_NE(c.meta.get(kEnergyDihedral), 0.0);
    EXPECT_EQ(c.meta.get(kEnergyImproper), 0.0);
}
