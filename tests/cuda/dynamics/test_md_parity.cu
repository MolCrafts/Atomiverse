/// @file tests/cuda/dynamics/test_md_parity.cu
/// @brief CPU <-> CUDA parity of every classical MD component through the
///        public drivers, one case per component:
///        - one step: forces, energy, kinetic energy, virial, positions,
///          velocities and cell against the same-settings CPU run;
///        - a 50-step trajectory against the CPU run;
///        - the replayed step graphs against the eager launch path
///          (`enable_timing()` keeps every step eager) for the same CUDA
///          run, which catches host state baked into a captured launch.
///        Components without a deterministic CPU twin (stochastic noise
///        from a different generator) run the graph-vs-eager check only.
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/lj_type_coeff.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/sllod_flow.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/barostat.h"
#include "atomiverse/cpu/dynamics/deform.h"
#include "atomiverse/cpu/dynamics/fix_langevin.h"
#include "atomiverse/cpu/dynamics/fix_rattle.h"
#include "atomiverse/cpu/dynamics/fix_settle.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/nose_hoover.h"
#include "atomiverse/cpu/dynamics/nose_hoover_npt.h"
#include "atomiverse/cpu/dynamics/nve_limit.h"
#include "atomiverse/cpu/dynamics/nvt_sllod.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/longrange/ewald.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/observe/virial_stress.h"
#include "atomiverse/cpu/potentials/angle_cosine.h"
#include "atomiverse/cpu/potentials/angle_harmonic.h"
#include "atomiverse/cpu/potentials/bond_fene.h"
#include "atomiverse/cpu/potentials/bond_harmonic.h"
#include "atomiverse/cpu/potentials/dihedral_charmm.h"
#include "atomiverse/cpu/potentials/dihedral_fourier.h"
#include "atomiverse/cpu/potentials/dihedral_multi_harmonic.h"
#include "atomiverse/cpu/potentials/improper_cvff.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/cpu/potentials/pair_lj_cut_coul_long.h"
#include "atomiverse/cuda/dynamics/barostat.cuh"
#include "atomiverse/cuda/dynamics/brownian.cuh"
#include "atomiverse/cuda/dynamics/deform.cuh"
#include "atomiverse/cuda/dynamics/fix_langevin.cuh"
#include "atomiverse/cuda/dynamics/fix_rattle.cuh"
#include "atomiverse/cuda/dynamics/fix_settle.cuh"
#include "atomiverse/cuda/dynamics/md.cuh"
#include "atomiverse/cuda/dynamics/nose_hoover.cuh"
#include "atomiverse/cuda/dynamics/nose_hoover_npt.cuh"
#include "atomiverse/cuda/dynamics/nve_limit.cuh"
#include "atomiverse/cuda/dynamics/nvt_sllod.cuh"
#include "atomiverse/cuda/dynamics/velocity_verlet.cuh"
#include "atomiverse/cuda/longrange/ewald.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/observe/stress.cuh"
#include "atomiverse/cuda/potentials/angle_cosine.cuh"
#include "atomiverse/cuda/potentials/angle_harmonic.cuh"
#include "atomiverse/cuda/potentials/bond_fene.cuh"
#include "atomiverse/cuda/potentials/bond_harmonic.cuh"
#include "atomiverse/cuda/potentials/dihedral_multi_harmonic.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut_coul_long.cuh"
#include "atomiverse/cuda/potentials/periodic_torsion.cuh"
#include "atomiverse/cuda/record/pressure_series.cuh"
#ifdef ATV_HAS_KSPACE
#include "atomiverse/cpu/longrange/pme.h"
#include "atomiverse/cuda/longrange/pme.cuh"
#endif

using namespace ATV;

namespace {

constexpr bool kDouble = std::is_same_v<cuda::force_t, double>
                         && std::is_same_v<cuda::pos_t, double>;

/// Relative tolerances of one case (scaled by 1 + |reference|).
struct Tolerance {
    double step; ///< After one step (summation order only in FP64).
    double traj; ///< After the trajectory (order noise grows with time).
};
constexpr Tolerance kExact{kDouble ? 1e-9 : 1e-4, kDouble ? 1e-7 : 1e-2};
constexpr Tolerance kMesh{kDouble ? 1e-6 : 1e-3, kDouble ? 1e-5 : 1e-2};
/// Graphed vs eager: the same kernels in the same order; only FP64 atomic
/// accumulation order may differ between two runs.
constexpr double kReplay = kDouble ? 1e-10 : 1e-4;
constexpr int kTrajectorySteps = 50;

// ── Systems ───────────────────────────────────────────────────────────────

enum class System { lj, ions, chains, water, ethane };

constexpr double kLjEps = 0.0104, kLjSigma = 3.4, kLjCut = 6.5, kLjSkin = 1.0;
constexpr double kIonAlpha = 0.34;
// Bonded neighbors (1.3-1.9 Å) are not excluded from LJ: sigma stays below
// the bond lengths, as in the per-term energy test.
constexpr double kChainEps = 0.004, kChainSigma = 1.1, kChainCut = 4.0;

/// 4x4x4 jittered lattice, 3.9 Å spacing, small thermal velocities with
/// zero total momentum. `ions` alternates +-0.5 e on two LJ types.
cpu::Frame lattice(bool ions) {
    constexpr int side = 4, n = side * side * side;
    constexpr double a = 3.9, L = side * a;
    cpu::Frame f(n);
    double p[3] = {};
    int i = 0;
    for (int x = 0; x < side; ++x)
        for (int y = 0; y < side; ++y)
            for (int z = 0; z < side; ++z, ++i) {
                const double j = 0.12 * std::sin(1.7 * i + 0.3);
                f.span(kAtom_x)[i] = (x + 0.5) * a + j;
                f.span(kAtom_y)[i] = (y + 0.5) * a - 0.7 * j;
                f.span(kAtom_z)[i] = (z + 0.5) * a + 0.4 * j;
                f.span(kAtom_vx)[i] = 2e-3 * std::sin(1.3 * i);
                f.span(kAtom_vy)[i] = 2e-3 * std::cos(0.7 * i + 0.2);
                f.span(kAtom_vz)[i] = 2e-3 * std::sin(2.1 * i + 0.4);
                p[0] += f.span(kAtom_vx)[i];
                p[1] += f.span(kAtom_vy)[i];
                p[2] += f.span(kAtom_vz)[i];
                f.span(kAtom_mass)[i] = 39.948;
                f.span(kAtom_tag)[i] = i + 1;
                f.span(kAtom_type)[i] = ions ? (x + y + z) % 2 : 0;
                f.span(kAtom_charge)[i] =
                    ions ? ((x + y + z) % 2 ? -0.5 : 0.5) : 0.0;
            }
    for (int k = 0; k < n; ++k) {
        f.span(kAtom_vx)[k] -= p[0] / n;
        f.span(kAtom_vy)[k] -= p[1] / n;
        f.span(kAtom_vz)[k] -= p[2] / n;
    }
    f.box = cpu::Box::cubic(L);
    f.clear_tag_index();
    f.rebuild_tag_index(0, n);
    f.build_special_list();
    return f;
}

/// Eight six-bead kinked chains (two bond / angle / dihedral / improper
/// types) in a 16 Å box; chain 3 crosses the x face.
cpu::Frame chains() {
    constexpr int kChains = 8, kPer = 6, n = kChains * kPer;
    constexpr double L = 16.0;
    cpu::Frame f(n);
    for (int c = 0; c < kChains; ++c)
        for (int k = 0; k < kPer; ++k) {
            const int i = c * kPer + k;
            const double x =
                1.0 + 1.25 * k + (c == 3 ? 9.5 : 0.0) + 4.0 * (c % 2);
            f.span(kAtom_x)[i] = std::fmod(x, L);
            f.span(kAtom_y)[i] = 1.5 + 3.8 * (c / 2) + 0.75 * (k % 2);
            f.span(kAtom_z)[i] =
                2.0 + 7.0 * (c % 2) + 0.15 * std::sin(1.9 * k + c);
            f.span(kAtom_vx)[i] = 3e-3 * std::sin(1.1 * i);
            f.span(kAtom_vy)[i] = 3e-3 * std::cos(0.9 * i);
            f.span(kAtom_vz)[i] = 3e-3 * std::sin(0.5 * i + 1.0);
            f.span(kAtom_mass)[i] = 12.011;
            // Topology refers to tags; the CUDA transfer resolves tag = row.
            f.span(kAtom_tag)[i] = i;
            f.span(kAtom_type)[i] = 0;
            f.span(kAtom_charge)[i] = 0.0;
        }
    f.box = cpu::Box::cubic(L);
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
        // Out-of-plane terms around beads 1 and 3 (owner = second atom).
        f.add_improper(b + 0, b + 1, b + 2, b + 3, 0);
        f.add_improper(b + 2, b + 3, b + 4, b + 5, 1);
    }
    f.build_special_list();
    return f;
}

/// One femtosecond in the engine's eV·Å·amu time unit (10.18 fs).
constexpr double kFs = 1.0 / 10.180505;
/// LJ-unit charge of one elementary charge: qqr2e = 1 holds e²/(4πε₀) =
/// 14.3996 eV·Å in the charge itself.
const double kQ = std::sqrt(14.399645);

// TIP3P geometry and charges; LJ on the oxygen only.
constexpr double kDOH = 0.9572, kDHH = 1.5139;
constexpr double kWaterCut = 6.0, kWaterAlpha = 0.3;

std::vector<LJTypeCoeff> water_lj() {
    return {{0, 0, 0.00660, 3.1507}, {1, 1, 0.0, 1.0}, {0, 1, 0.0, 1.0}};
}

/// Rotate `v` by Rz(c) Ry(b) Rx(a).
std::array<double, 3> rotate(std::array<double, 3> v, double a, double b,
                             double c) {
    auto rx = [&](double t) {
        v = {v[0], std::cos(t) * v[1] - std::sin(t) * v[2],
             std::sin(t) * v[1] + std::cos(t) * v[2]};
    };
    auto ry = [&](double t) {
        v = {std::cos(t) * v[0] + std::sin(t) * v[2], v[1],
             -std::sin(t) * v[0] + std::cos(t) * v[2]};
    };
    auto rz = [&](double t) {
        v = {std::cos(t) * v[0] - std::sin(t) * v[1],
             std::sin(t) * v[0] + std::cos(t) * v[1], v[2]};
    };
    rx(a);
    ry(b);
    rz(c);
    return v;
}

/// 5x5x5 rigid TIP3P waters (3.1 Å O-O spacing, ~1 g/cm³), each with a
/// jittered common orientation and a rigid-body velocity (translation plus
/// rotation), so the constraints hold at the start. Tags equal rows; the
/// O-H bonds and H-O-H angle carry no potential, only the exclusions.
cpu::Frame water() {
    constexpr int side = 5, n_w = side * side * side, n = 3 * n_w;
    constexpr double a = 3.1, L = side * a;
    const double half = 0.5 * kDHH;
    const double h = std::sqrt(kDOH * kDOH - half * half);
    const double m_o = 15.9994, m_h = 1.008, m_t = m_o + 2 * m_h;
    // Body frame with the centre of mass at the origin.
    const double yo = 2 * m_h * h / m_t;
    const std::array<std::array<double, 3>, 3> body = {
        {{0.0, yo, 0.0}, {-half, yo - h, 0.0}, {half, yo - h, 0.0}}};
    cpu::Frame f(n);
    int w = 0;
    for (int x = 0; x < side; ++x)
        for (int y = 0; y < side; ++y)
            for (int z = 0; z < side; ++z, ++w) {
                const std::array<double, 3> com = {(x + 0.5) * a, (y + 0.5) * a,
                                                   (z + 0.5) * a};
                const double ra = 0.3 * std::sin(1.3 * w),
                             rb = 0.3 * std::cos(0.7 * w + 0.4),
                             rc = 0.3 * std::sin(2.3 * w + 1.1);
                const std::array<double, 3> vcom = {
                    0.02 * std::sin(0.9 * w), 0.02 * std::cos(1.7 * w),
                    0.02 * std::sin(0.4 * w + 0.5)};
                const std::array<double, 3> omega = {
                    0.1 * std::cos(1.1 * w), 0.1 * std::sin(0.6 * w + 0.2),
                    0.1 * std::cos(2.9 * w)};
                for (int s = 0; s < 3; ++s) {
                    const int i = 3 * w + s;
                    const auto r = rotate(body[s], ra, rb, rc);
                    f.span(kAtom_x)[i] = std::fmod(com[0] + r[0] + L, L);
                    f.span(kAtom_y)[i] = std::fmod(com[1] + r[1] + L, L);
                    f.span(kAtom_z)[i] = std::fmod(com[2] + r[2] + L, L);
                    f.span(kAtom_vx)[i] =
                        vcom[0] + omega[1] * r[2] - omega[2] * r[1];
                    f.span(kAtom_vy)[i] =
                        vcom[1] + omega[2] * r[0] - omega[0] * r[2];
                    f.span(kAtom_vz)[i] =
                        vcom[2] + omega[0] * r[1] - omega[1] * r[0];
                    f.span(kAtom_mass)[i] = s == 0 ? m_o : m_h;
                    f.span(kAtom_tag)[i] = i;
                    f.span(kAtom_type)[i] = s == 0 ? 0 : 1;
                    f.span(kAtom_charge)[i] = (s == 0 ? -0.834 : 0.417) * kQ;
                }
            }
    f.box = cpu::Box::cubic(L);
    f.clear_tag_index();
    f.rebuild_tag_index(0, n);
    for (int k = 0; k < n_w; ++k) {
        f.add_bond(3 * k, 3 * k + 1, 0);
        f.add_bond(3 * k, 3 * k + 2, 0);
        f.add_angle(3 * k + 1, 3 * k, 3 * k + 2, 0);
    }
    f.build_special_list();
    return f;
}

// Ethane, OPLS-AA-like: C-C / C-H bonds (types 0 / 1), H-C-C / H-C-H
// angles (0 / 1), H-C-C-H torsions, LJ + charges with Amber exclusions.
constexpr double kCC = 1.529, kCH = 1.09, kHCC = 1.9321; // 110.7 deg
constexpr double kEthaneCut = 6.0, kEthaneAlpha = 0.3;
const std::vector<double> kEthBondK = {11.62, 14.74}, kEthBondR0 = {kCC, kCH};
const std::vector<double> kEthAngleK = {1.626, 1.431},
                          kEthTheta0 = {kHCC, 1.8814};
const std::vector<std::vector<TorsionTerm>> kEthTorsion = {{{0.0065, 3, 0.0}}};

std::vector<LJTypeCoeff> ethane_lj() {
    return {
        {0, 0, 0.002862, 3.5}, {1, 1, 0.001301, 2.5}, {0, 1, 0.001930, 2.958}};
}

/// 3x3x3 staggered ethanes on a 5 Å lattice, jittered orientations and
/// small velocities (not constraint-consistent: the first step projects
/// them). Tags equal rows: C1 C2 H1..H3 (on C1) H4..H6 (on C2).
cpu::Frame ethane() {
    constexpr int side = 3, n_m = side * side * side, per = 8, n = per * n_m;
    constexpr double a = 5.0, L = side * a;
    const double tp = M_PI - kHCC; // H tilt from the -C-C axis
    std::array<std::array<double, 3>, per> body{};
    body[0] = {-0.5 * kCC, 0.0, 0.0};
    body[1] = {0.5 * kCC, 0.0, 0.0};
    for (int k = 0; k < 3; ++k) {
        const double p1 = 2.0 * M_PI * k / 3.0, p2 = p1 + M_PI / 3.0;
        body[2 + k] = {-0.5 * kCC - kCH * std::cos(tp),
                       kCH * std::sin(tp) * std::cos(p1),
                       kCH * std::sin(tp) * std::sin(p1)};
        body[5 + k] = {0.5 * kCC + kCH * std::cos(tp),
                       kCH * std::sin(tp) * std::cos(p2),
                       kCH * std::sin(tp) * std::sin(p2)};
    }
    cpu::Frame f(n);
    int m = 0;
    for (int x = 0; x < side; ++x)
        for (int y = 0; y < side; ++y)
            for (int z = 0; z < side; ++z, ++m) {
                const std::array<double, 3> c = {(x + 0.5) * a, (y + 0.5) * a,
                                                 (z + 0.5) * a};
                const double ra = 1.1 * std::sin(1.3 * m),
                             rb = 0.9 * std::cos(0.7 * m + 0.4),
                             rc = 1.3 * std::sin(2.3 * m + 1.1);
                for (int s = 0; s < per; ++s) {
                    const int i = per * m + s;
                    const auto r = rotate(body[s], ra, rb, rc);
                    f.span(kAtom_x)[i] = std::fmod(c[0] + r[0] + L, L);
                    f.span(kAtom_y)[i] = std::fmod(c[1] + r[1] + L, L);
                    f.span(kAtom_z)[i] = std::fmod(c[2] + r[2] + L, L);
                    f.span(kAtom_vx)[i] = 0.03 * std::sin(1.1 * i);
                    f.span(kAtom_vy)[i] = 0.03 * std::cos(0.9 * i);
                    f.span(kAtom_vz)[i] = 0.03 * std::sin(0.5 * i + 1.0);
                    f.span(kAtom_mass)[i] = s < 2 ? 12.011 : 1.008;
                    f.span(kAtom_tag)[i] = i;
                    f.span(kAtom_type)[i] = s < 2 ? 0 : 1;
                    f.span(kAtom_charge)[i] = (s < 2 ? -0.18 : 0.06) * kQ;
                }
            }
    f.box = cpu::Box::cubic(L);
    f.clear_tag_index();
    f.rebuild_tag_index(0, n);
    for (int k = 0; k < n_m; ++k) {
        const int b = per * k;
        f.add_bond(b, b + 1, 0);
        for (int h = 0; h < 3; ++h) {
            f.add_bond(b, b + 2 + h, 1);
            f.add_bond(b + 1, b + 5 + h, 1);
            f.add_angle(b + 2 + h, b, b + 1, 0);
            f.add_angle(b + 5 + h, b + 1, b, 0);
            for (int g = h + 1; g < 3; ++g) {
                f.add_angle(b + 2 + h, b, b + 2 + g, 1);
                f.add_angle(b + 5 + h, b + 1, b + 5 + g, 1);
            }
            for (int g = 0; g < 3; ++g)
                f.add_dihedral(b + 2 + h, b, b + 1, b + 5 + g, 0);
        }
    }
    f.build_special_list();
    return f;
}

/// H-C-H 1-3 distance of the built ethane.
double ethane_hh() {
    const double tp = M_PI - kHCC;
    return 2.0 * kCH * std::sin(tp) * std::sin(M_PI / 3.0);
}

cpu::Frame make_system(System s) {
    switch (s) {
    case System::lj:
        return lattice(false);
    case System::ions:
        return lattice(true);
    case System::chains:
        return chains();
    case System::water:
        return water();
    case System::ethane:
        return ethane();
    }
    return lattice(false);
}

std::vector<LJTypeCoeff> ion_lj() {
    return {{0, 0, 0.010, 3.2}, {1, 1, 0.006, 3.6}, {0, 1, 0.008, 3.4}};
}

// Near the built geometry (bonds 1.46-1.5 Å, angles ~2.07 rad), so 50 steps
// stay close to equilibrium.
const std::vector<double> kBondK = {6.0, 4.0}, kBondR0 = {1.45, 1.5};
const std::vector<double> kFeneK = {0.05, 0.04}, kFeneR0 = {2.4, 2.6};
const std::vector<double> kCosineK = {0.05, 0.03};
const std::vector<double> kAngleK = {1.2, 0.9}, kTheta0 = {2.0, 2.2};
const std::vector<std::vector<double>> kDihCpu = {
    {0.03, -0.02, 0.04, 0.01, -0.005}, {0.02, 0.01, -0.03, 0.004, 0.002}};
const std::vector<std::array<double, 5>> kDihCuda = {
    {0.03, -0.02, 0.04, 0.01, -0.005}, {0.02, 0.01, -0.03, 0.004, 0.002}};
// Amber-like multi-term torsions (n up to 4, nonzero phases) and cvff
// impropers (d = -1 / +1, n = 2 / 3).
const std::vector<std::vector<TorsionTerm>> kFourier = {
    {{0.02, 1, 0.0}, {0.015, 2, 3.14159265358979}, {0.004, 3, 0.3}},
    {{0.01, 4, 1.1}, {0.006, 2, -0.4}}};
const std::vector<double> kCharmmK = {0.03, 0.012};
const std::vector<int> kCharmmN = {3, 2};
const std::vector<double> kCharmmDelta = {0.0, 3.14159265358979};
const std::vector<double> kCvffK = {0.05, 0.02};
const std::vector<int> kCvffD = {-1, 1};
const std::vector<int> kCvffN = {2, 3};

// ── Cases ─────────────────────────────────────────────────────────────────

/// SETTLE over every water of `water()` (tags 3k, 3k+1, 3k+2).
template <typename Settle>
Settle settle_waters() {
    Settle settle{2.0 * kFs, kDOH, kDHH};
    for (int k = 0; k < 125; ++k)
        settle.add_water(3 * k, 3 * k + 1, 3 * k + 2);
    return settle;
}

struct Case {
    std::string name;
    System system;
    /// Empty: the component has no deterministic CPU twin.
    std::function<void(cpu::md::MD&)> cpu;
    std::function<void(cuda::md::MD&)> gpu;
    Tolerance tol = kExact;
};

double cut_of(System s) {
    switch (s) {
    case System::chains:
        return kChainCut;
    case System::water:
        return kWaterCut;
    case System::ethane:
        return kEthaneCut;
    default:
        return kLjCut;
    }
}

void mount_cpu_forces(cpu::md::MD& sim, System s) {
    sim.set_neighbor(cpu::md::NeighborList<>{cut_of(s), kLjSkin, 1, 0, true});
    sim.on(md::Phase::post_force).add(cpu::md::Virial{}, -50);
    switch (s) {
    case System::lj:
        sim.set_evaluator(cpu::PairLJCut{kLjEps, kLjSigma, kLjCut});
        break;
    case System::ions:
        sim.set_evaluator(cpu::PairLJCutCoulLong{ion_lj(), kIonAlpha, kLjCut});
        break;
    case System::chains:
        sim.set_evaluator(cpu::PairLJCut{kChainEps, kChainSigma, kChainCut});
        break;
    case System::water:
        sim.set_evaluator(
            cpu::PairLJCutCoulLong{water_lj(), kWaterAlpha, kWaterCut});
        sim.set_evaluator(cpu::KSpaceEwald{kWaterAlpha, 1e-8});
        break;
    case System::ethane:
        sim.set_evaluator(
            cpu::PairLJCutCoulLong{ethane_lj(), kEthaneAlpha, kEthaneCut});
        sim.set_evaluator(cpu::KSpaceEwald{kEthaneAlpha, 1e-8});
        sim.set_evaluator(cpu::BondHarmonic{kEthBondK, kEthBondR0});
        sim.set_evaluator(cpu::AngleHarmonic{kEthAngleK, kEthTheta0});
        sim.set_evaluator(cpu::DihedralFourier{kEthTorsion});
        break;
    }
}

void mount_cuda_forces(cuda::md::MD& sim, System s) {
    sim.set_neighbor(cuda::NeighborList{cut_of(s), kLjSkin});
    sim.on(md::Phase::post_force).add(cuda::Virial{}, -50);
    switch (s) {
    case System::lj:
        sim.set_evaluator(cuda::PairLJCut{kLjEps, kLjSigma, kLjCut});
        break;
    case System::ions:
        sim.set_evaluator(cuda::PairLJCutCoulLong{ion_lj(), kIonAlpha, kLjCut});
        break;
    case System::chains:
        sim.set_evaluator(cuda::PairLJCut{kChainEps, kChainSigma, kChainCut});
        break;
    case System::water:
        sim.set_evaluator(
            cuda::PairLJCutCoulLong{water_lj(), kWaterAlpha, kWaterCut});
        sim.set_evaluator(cuda::KSpaceEwald{kWaterAlpha, 1e-8});
        break;
    case System::ethane:
        sim.set_evaluator(
            cuda::PairLJCutCoulLong{ethane_lj(), kEthaneAlpha, kEthaneCut});
        sim.set_evaluator(cuda::KSpaceEwald{kEthaneAlpha, 1e-8});
        sim.set_evaluator(cuda::BondHarmonic{kEthBondK, kEthBondR0});
        sim.set_evaluator(cuda::AngleHarmonic{kEthAngleK, kEthTheta0});
        sim.set_evaluator(cuda::DihedralFourier{kEthTorsion});
        break;
    }
}

constexpr double kDt = 2.0;
// The classical engine works in LJ units (kB = 1): thermostat targets are
// energies, on the scale of the LJ well depth (~0.4-1 eps).
constexpr double kT = 0.005, kTHot = 0.0075, kTRamp = 0.01;

std::vector<Case> cases() {
    std::vector<Case> out;
    out.push_back(
        {"VelocityVerletLJ", System::lj,
         [](cpu::md::MD& s) { s.set_integrator(cpu::md::VelocityVerlet{kDt}); },
         [](cuda::md::MD& s) {
             s.set_integrator(cuda::md::VelocityVerlet{kDt});
         }});
    out.push_back({"LJCutCoulLongEwald", System::ions,
                   [](cpu::md::MD& s) {
                       s.set_evaluator(cpu::KSpaceEwald{kIonAlpha, 1e-8});
                       s.set_integrator(cpu::md::VelocityVerlet{1.0});
                   },
                   [](cuda::md::MD& s) {
                       s.set_evaluator(cuda::KSpaceEwald{kIonAlpha, 1e-8});
                       s.set_integrator(cuda::md::VelocityVerlet{1.0});
                   },
                   kMesh});
#ifdef ATV_HAS_KSPACE
    out.push_back({"LJCutCoulLongPME", System::ions,
                   [](cpu::md::MD& s) {
                       s.set_evaluator(cpu::KSpacePME{kIonAlpha, 0.6, 5});
                       s.set_integrator(cpu::md::VelocityVerlet{1.0});
                   },
                   [](cuda::md::MD& s) {
                       s.set_evaluator(cuda::KSpacePME{kIonAlpha, 0.6, 5});
                       s.set_integrator(cuda::md::VelocityVerlet{1.0});
                   },
                   kMesh});
#endif
    out.push_back({"BondAngleDihedralHarmonic", System::chains,
                   [](cpu::md::MD& s) {
                       s.set_evaluator(cpu::BondHarmonic{kBondK, kBondR0});
                       s.set_evaluator(cpu::AngleHarmonic{kAngleK, kTheta0});
                       s.set_evaluator(cpu::DihedralMultiHarmonic{kDihCpu});
                       s.set_integrator(cpu::md::VelocityVerlet{0.5});
                   },
                   [](cuda::md::MD& s) {
                       s.set_evaluator(cuda::BondHarmonic{kBondK, kBondR0});
                       s.set_evaluator(cuda::AngleHarmonic{kAngleK, kTheta0});
                       s.set_evaluator(cuda::DihedralMultiHarmonic{kDihCuda});
                       s.set_integrator(cuda::md::VelocityVerlet{0.5});
                   }});
    out.push_back(
        {"DihedralFourierImproperCvff", System::chains,
         [](cpu::md::MD& s) {
             s.set_evaluator(cpu::BondHarmonic{kBondK, kBondR0});
             s.set_evaluator(cpu::DihedralFourier{kFourier});
             s.set_evaluator(cpu::ImproperCvff{kCvffK, kCvffD, kCvffN});
             s.set_integrator(cpu::md::VelocityVerlet{0.5});
         },
         [](cuda::md::MD& s) {
             s.set_evaluator(cuda::BondHarmonic{kBondK, kBondR0});
             s.set_evaluator(cuda::DihedralFourier{kFourier});
             s.set_evaluator(cuda::ImproperCvff{kCvffK, kCvffD, kCvffN});
             s.set_integrator(cuda::md::VelocityVerlet{0.5});
         }});
    out.push_back({"DihedralCharmm", System::chains,
                   [](cpu::md::MD& s) {
                       s.set_evaluator(cpu::BondHarmonic{kBondK, kBondR0});
                       s.set_evaluator(cpu::DihedralCharmm{kCharmmK, kCharmmN,
                                                           kCharmmDelta});
                       s.set_integrator(cpu::md::VelocityVerlet{0.5});
                   },
                   [](cuda::md::MD& s) {
                       s.set_evaluator(cuda::BondHarmonic{kBondK, kBondR0});
                       s.set_evaluator(cuda::DihedralCharmm{kCharmmK, kCharmmN,
                                                            kCharmmDelta});
                       s.set_integrator(cuda::md::VelocityVerlet{0.5});
                   }});
    out.push_back({"BondFeneAngleCosine", System::chains,
                   [](cpu::md::MD& s) {
                       s.set_evaluator(cpu::BondFene{kFeneK, kFeneR0});
                       s.set_evaluator(cpu::AngleCosine{kCosineK});
                       s.set_integrator(cpu::md::VelocityVerlet{0.5});
                   },
                   [](cuda::md::MD& s) {
                       s.set_evaluator(cuda::BondFene{kFeneK, kFeneR0});
                       s.set_evaluator(cuda::AngleCosine{kCosineK});
                       s.set_integrator(cuda::md::VelocityVerlet{0.5});
                   }});
    out.push_back({"NoseHoover", System::lj,
                   [](cpu::md::MD& s) {
                       s.set_integrator(cpu::md::VelocityVerlet{kDt});
                       s.set_thermostat(cpu::md::NoseHoover{kT, 40.0, kDt});
                   },
                   [](cuda::md::MD& s) {
                       s.set_integrator(cuda::md::VelocityVerlet{kDt});
                       s.set_thermostat(cuda::md::NoseHoover{kT, 40.0, kDt});
                   }});
    out.push_back({"NoseHooverNPT", System::lj,
                   [](cpu::md::MD& s) {
                       s.set_integrator(cpu::md::NoseHooverNPT{
                           kT, 40.0, 50 * units::bar_to_ev_per_a3, 200.0, kDt});
                   },
                   [](cuda::md::MD& s) {
                       s.set_integrator(cuda::md::NoseHooverNPT{
                           kT, 40.0, 50 * units::bar_to_ev_per_a3, 200.0, kDt});
                   }});
    // A temperature ramp over the run span (host-side target today).
    out.push_back(
        {"NoseHooverNPTRamp", System::lj,
         [](cpu::md::MD& s) {
             cpu::md::NoseHooverNPT npt{
                 kT,   40.0, 0.0, 200.0, kDt, 0.0, 3, 3, {true, true, true},
                 kTHot};
             npt.set_run_span(0, kTrajectorySteps);
             s.set_integrator(std::move(npt));
         },
         [](cuda::md::MD& s) {
             cuda::md::NoseHooverNPT npt{
                 kT,   40.0, 0.0, 200.0, kDt, 0.0, 3, 3, {true, true, true},
                 kTHot};
             npt.set_run_span(0, kTrajectorySteps);
             s.set_integrator(std::move(npt));
         }});
    out.push_back({"BerendsenBarostat", System::lj,
                   [](cpu::md::MD& s) {
                       s.set_integrator(cpu::md::VelocityVerlet{kDt});
                       s.set_barostat(cpu::md::Barostat{
                           100 * units::bar_to_ev_per_a3, 100.0, kDt, 10.0});
                   },
                   [](cuda::md::MD& s) {
                       s.set_integrator(cuda::md::VelocityVerlet{kDt});
                       s.set_barostat(cuda::md::Barostat{
                           100 * units::bar_to_ev_per_a3, 100.0, kDt, 10.0});
                   }});
    // xmax below the fastest atom's |v| dt: the limiter engages.
    out.push_back(
        {"NVELimit", System::lj,
         [](cpu::md::MD& s) { s.set_integrator(cpu::md::NVELimit{kDt, 2e-3}); },
         [](cuda::md::MD& s) {
             s.set_integrator(cuda::md::NVELimit{kDt, 2e-3});
         }});
    out.push_back({"NvtSllod", System::lj,
                   [](cpu::md::MD& s) {
                       s.set_integrator(cpu::md::NvtSllod{kT, 40.0, kDt, 2e-4,
                                                          md::kFlowXY, 3});
                   },
                   [](cuda::md::MD& s) {
                       s.set_integrator(cuda::md::NvtSllod{kT, 40.0, kDt, 2e-4,
                                                           md::kFlowXY, 3});
                   }});
    out.push_back({"Deform", System::lj,
                   [](cpu::md::MD& s) {
                       s.set_integrator(cpu::md::VelocityVerlet{kDt});
                       s.on(md::Phase::initial_integrate)
                           .add(cpu::md::Deform{cpu::md::Deform::Mode::erate,
                                                2e-5, kDt, 1},
                                200);
                   },
                   [](cuda::md::MD& s) {
                       s.set_integrator(cuda::md::VelocityVerlet{kDt});
                       s.on(md::Phase::initial_integrate)
                           .add(cuda::md::Deform{cuda::md::Deform::Mode::erate,
                                                 2e-5, kDt, 1},
                                200);
                   }});
    // T = 0: drag only, so the CPU twin is deterministic.
    out.push_back(
        {"LangevinDrag", System::lj,
         [](cpu::md::MD& s) {
             s.set_integrator(cpu::md::VelocityVerlet{kDt});
             s.set_thermostat(cpu::md::FixLangevin{0.0, 0.0, 80.0, kDt, 7u});
         },
         [](cuda::md::MD& s) {
             s.set_integrator(cuda::md::VelocityVerlet{kDt});
             s.set_thermostat(cuda::md::FixLangevin{0.0, 0.0, 80.0, kDt, 7u});
         }});
    // Noise keyed on (seed, tag, step): only graph-vs-eager is meaningful.
    out.push_back({"LangevinNoise", System::lj, {}, [](cuda::md::MD& s) {
                       s.set_integrator(cuda::md::VelocityVerlet{kDt});
                       s.set_thermostat(
                           cuda::md::FixLangevin{kT, kT, 80.0, kDt, 7u});
                   }});
    out.push_back({"LangevinNoiseRamp", System::lj, {}, [](cuda::md::MD& s) {
                       s.set_integrator(cuda::md::VelocityVerlet{kDt});
                       cuda::md::FixLangevin langevin{kT, kTRamp, 80.0, kDt,
                                                      7u};
                       langevin.set_run_span(0, kTrajectorySteps);
                       s.set_thermostat(std::move(langevin));
                   }});
    out.push_back({"Brownian", System::lj, {}, [](cuda::md::MD& s) {
                       s.set_integrator(cuda::md::Brownian{kDt, 1e-4, kT, 11u});
                   }});
    // Constraints at a 2 fs step: rigid TIP3P through SETTLE, X-H bonds
    // (and, separately, rigid H-C-H angles) through RATTLE clusters.
    constexpr double kDt2fs = 2.0 * kFs;
    out.push_back({"SettleWater", System::water,
                   [](cpu::md::MD& s) {
                       s.set_integrator(cpu::md::VelocityVerlet{kDt2fs});
                       s.set_observer(settle_waters<cpu::md::FixSettle>());
                   },
                   [](cuda::md::MD& s) {
                       s.set_integrator(cuda::md::VelocityVerlet{kDt2fs});
                       s.set_observer(settle_waters<cuda::md::FixSettle>());
                   },
                   kMesh});
    // Ndof = 6 per water reaches the thermostat's target.
    out.push_back(
        {"SettleWaterNoseHoover", System::water,
         [](cpu::md::MD& s) {
             s.set_integrator(cpu::md::VelocityVerlet{kDt2fs});
             s.set_thermostat(cpu::md::NoseHoover{0.02, 20 * kDt2fs, kDt2fs});
             s.set_observer(settle_waters<cpu::md::FixSettle>());
         },
         [](cuda::md::MD& s) {
             s.set_integrator(cuda::md::VelocityVerlet{kDt2fs});
             s.set_thermostat(cuda::md::NoseHoover{0.02, 20 * kDt2fs, kDt2fs});
             s.set_observer(settle_waters<cuda::md::FixSettle>());
         },
         kMesh});
    out.push_back({"RattleXH", System::ethane,
                   [](cpu::md::MD& s) {
                       s.set_integrator(cpu::md::VelocityVerlet{kDt2fs});
                       cpu::md::FixRattle rattle{kDt2fs};
                       rattle.constrain_bond(1, kCH);
                       s.set_observer(std::move(rattle));
                   },
                   [](cuda::md::MD& s) {
                       s.set_integrator(cuda::md::VelocityVerlet{kDt2fs});
                       cuda::md::FixRattle rattle{kDt2fs};
                       rattle.constrain_bond(1, kCH);
                       s.set_observer(std::move(rattle));
                   },
                   kMesh});
    out.push_back(
        {"RattleXHAngles", System::ethane,
         [](cpu::md::MD& s) {
             s.set_integrator(cpu::md::VelocityVerlet{kDt2fs});
             s.set_thermostat(cpu::md::NoseHoover{0.02, 20 * kDt2fs, kDt2fs});
             cpu::md::FixRattle rattle{kDt2fs};
             rattle.constrain_bond(1, kCH);
             rattle.constrain_angle(1, ethane_hh());
             s.set_observer(std::move(rattle));
         },
         [](cuda::md::MD& s) {
             s.set_integrator(cuda::md::VelocityVerlet{kDt2fs});
             s.set_thermostat(cuda::md::NoseHoover{0.02, 20 * kDt2fs, kDt2fs});
             cuda::md::FixRattle rattle{kDt2fs};
             rattle.constrain_bond(1, kCH);
             rattle.constrain_angle(1, ethane_hh());
             s.set_observer(std::move(rattle));
         },
         kMesh});
    return out;
}

// ── Runs and comparisons ──────────────────────────────────────────────────

cpu::Frame run_cpu(const Case& c, int steps) {
    cpu::Frame f = make_system(c.system);
    cpu::md::MD sim;
    mount_cpu_forces(sim, c.system);
    c.cpu(sim);
    sim.build();
    sim.run(f, steps);
    return f;
}

cpu::Frame run_cuda(const Case& c, int steps, bool eager) {
    cpu::Frame f = make_system(c.system);
    cuda::md::MD sim;
    mount_cuda_forces(sim, c.system);
    c.gpu(sim);
    if (eager)
        sim.enable_timing();
    sim.build();
    sim.run(f, steps);
    return f;
}

struct Row {
    std::array<double, 3> unwrapped, v, f;
};

std::map<std::int64_t, Row> rows(const cpu::Frame& f) {
    std::map<std::int64_t, Row> out;
    const auto& h = f.box.cell();
    for (int i = 0; i < f.nlocal(); ++i) {
        Row r;
        const double x[3] = {f.span(kAtom_x)[i], f.span(kAtom_y)[i],
                             f.span(kAtom_z)[i]};
        for (int a = 0; a < 3; ++a) {
            r.unwrapped[a] = x[a];
            for (int b = 0; b < 3; ++b)
                r.unwrapped[a] +=
                    h(a, b) * static_cast<double>(f.span(kAtom_image)(i, b));
        }
        r.v = {f.span(kAtom_vx)[i], f.span(kAtom_vy)[i], f.span(kAtom_vz)[i]};
        r.f = {f.span(kAtom_fx)[i], f.span(kAtom_fy)[i], f.span(kAtom_fz)[i]};
        out[f.span(kAtom_tag)[i]] = r;
    }
    return out;
}

double scale_of(const std::map<std::int64_t, Row>& rows,
                std::array<double, 3> Row::* field) {
    double m = 0.0;
    for (const auto& [tag, r] : rows)
        for (double x : r.*field)
            m = std::max(m, std::abs(x));
    return m;
}

/// Every compared quantity within `tol * (1 + scale)`, where the scale of a
/// per-atom field is its largest reference magnitude.
void expect_same_state(const cpu::Frame& got, const cpu::Frame& want,
                       double tol, bool forces) {
    const auto g = rows(got), w = rows(want);
    ASSERT_EQ(g.size(), w.size());
    const double sx = 1.0 + scale_of(w, &Row::unwrapped);
    const double sv = 1.0 + scale_of(w, &Row::v);
    const double sf = 1.0 + scale_of(w, &Row::f);
    int reported = 0;
    for (const auto& [tag, wr] : w) {
        const auto it = g.find(tag);
        ASSERT_NE(it, g.end()) << "tag " << tag;
        const Row& gr = it->second;
        for (int a = 0; a < 3; ++a) {
            const bool ok =
                std::abs(gr.unwrapped[a] - wr.unwrapped[a]) <= tol * sx
                && std::abs(gr.v[a] - wr.v[a]) <= tol * sv
                && (!forces || std::abs(gr.f[a] - wr.f[a]) <= tol * sf);
            if (!ok && reported++ < 4) {
                ADD_FAILURE() << "tag " << tag << " axis " << a << ": x "
                              << gr.unwrapped[a] << " vs " << wr.unwrapped[a]
                              << ", v " << gr.v[a] << " vs " << wr.v[a]
                              << ", f " << gr.f[a] << " vs " << wr.f[a];
            }
        }
    }
    EXPECT_EQ(reported, 0) << "atoms out of tolerance";
    auto near = [&](double a, double b, const char* what) {
        EXPECT_NEAR(a, b, tol * (1.0 + std::abs(b))) << what;
    };
    near(got.meta.get(kEnergy), want.meta.get(kEnergy), "energy");
    near(got.meta.get(kKinetic), want.meta.get(kKinetic), "kinetic");
    const auto& sg = got.meta.get(kStress);
    const auto& sw = want.meta.get(kStress);
    double ss = 0.0;
    for (int k = 0; k < 6; ++k)
        ss = std::max(ss, std::abs(sw[k]));
    for (int k = 0; k < 6; ++k)
        EXPECT_NEAR(sg[k], sw[k], tol * (1.0 + ss)) << "virial " << k;
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            EXPECT_NEAR(got.box.cell()(a, b), want.box.cell()(a, b),
                        tol * (1.0 + std::abs(want.box.cell()(a, b))))
                << "cell(" << a << "," << b << ")";
}

class CudaMdParity : public testing::TestWithParam<Case> {};

} // namespace

TEST_P(CudaMdParity, OneStepMatchesCpu) {
    const Case& c = GetParam();
    if (!c.cpu)
        GTEST_SKIP() << c.name << " has no deterministic CPU twin";
    const cpu::Frame want = run_cpu(c, 1);
    const cpu::Frame got = run_cuda(c, 1, /*eager=*/false);
    ASSERT_GT(std::abs(want.meta.get(kEnergy)), 1e-8) << "vacuous energy";
    expect_same_state(got, want, c.tol.step, /*forces=*/true);
}

TEST_P(CudaMdParity, TrajectoryMatchesCpu) {
    const Case& c = GetParam();
    if (!c.cpu)
        GTEST_SKIP() << c.name << " has no deterministic CPU twin";
    const cpu::Frame want = run_cpu(c, kTrajectorySteps);
    const cpu::Frame got = run_cuda(c, kTrajectorySteps, /*eager=*/false);
    expect_same_state(got, want, c.tol.traj, /*forces=*/true);
}

TEST_P(CudaMdParity, GraphedStepsMatchEagerSteps) {
    const Case& c = GetParam();
    const cpu::Frame eager = run_cuda(c, kTrajectorySteps, /*eager=*/true);
    const cpu::Frame graphed = run_cuda(c, kTrajectorySteps, /*eager=*/false);
    // A replay that froze the first step's host state would leave the
    // trajectory far from the eager one; make sure the run moved at all.
    const cpu::Frame start = make_system(c.system);
    double moved = 0.0;
    const auto s = rows(start), e = rows(eager);
    for (const auto& [tag, r] : s)
        for (int a = 0; a < 3; ++a)
            moved = std::max(moved,
                             std::abs(e.at(tag).unwrapped[a] - r.unwrapped[a]));
    ASSERT_GT(moved, 1e-6) << "vacuous: the eager run did not move";
    expect_same_state(graphed, eager, kReplay, /*forces=*/true);
}

INSTANTIATE_TEST_SUITE_P(Components, CudaMdParity, testing::ValuesIn(cases()),
                         [](const testing::TestParamInfo<Case>& info) {
                             return info.param.name;
                         });

// ── PressureSeries: rows of the replayed run equal the eager run's ────────

namespace {

std::vector<std::vector<double>> series_rows(bool eager) {
    const std::string path = testing::TempDir() + "atv_parity_series_"
                             + (eager ? "eager" : "graphed") + ".dat";
    {
        cpu::Frame f = lattice(false);
        cuda::md::MD sim;
        mount_cuda_forces(sim, System::lj);
        sim.set_integrator(
            cuda::md::NvtSllod{kT, 40.0, kDt, 2e-4, md::kFlowXY, 1});
        sim.on(md::Phase::end_of_step)
            .add(cuda::md::PressureSeries{path, 5, kLjEps, md::kFlowXY}, 20);
        if (eager)
            sim.enable_timing();
        sim.build();
        sim.run(f, kTrajectorySteps);
    }
    std::vector<std::vector<double>> out;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream row(line);
        std::vector<double> values;
        double v;
        while (row >> v)
            values.push_back(v);
        out.push_back(values);
    }
    std::remove(path.c_str());
    return out;
}

} // namespace

TEST(CudaMdParitySeries, PressureSeriesGraphedRowsMatchEagerRows) {
    const auto eager = series_rows(true);
    const auto graphed = series_rows(false);
    ASSERT_EQ(eager.size(), std::size_t(kTrajectorySteps / 5 + 1));
    ASSERT_EQ(graphed.size(), eager.size());
    for (std::size_t r = 0; r < eager.size(); ++r) {
        ASSERT_EQ(graphed[r].size(), eager[r].size()) << "row " << r;
        for (std::size_t k = 0; k < eager[r].size(); ++k)
            EXPECT_NEAR(graphed[r][k], eager[r][k],
                        1e-8 * (1.0 + std::abs(eager[r][k])))
                << "row " << r << " column " << k;
    }
}

// ── Constraints: the geometry holds, and an unsatisfiable one faults ──────

namespace {

double distance(const cpu::Frame& f, int i, int j) {
    double d[3] = {f.span(kAtom_x)[i] - f.span(kAtom_x)[j],
                   f.span(kAtom_y)[i] - f.span(kAtom_y)[j],
                   f.span(kAtom_z)[i] - f.span(kAtom_z)[j]};
    f.box.min_image(d[0], d[1], d[2]);
    return std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
}

const Case& case_named(const std::string& name) {
    static const std::vector<Case> all = cases();
    for (const Case& c : all)
        if (c.name == name)
            return c;
    throw std::invalid_argument("no case " + name);
}

/// Run `steps` of a case on a caller-built frame (CPU or CUDA driver).
template <typename Sim, typename Mount, typename Setup>
void run_on(cpu::Frame& f, System s, Mount mount, Setup setup, int steps) {
    Sim sim;
    mount(sim, s);
    setup(sim);
    sim.build();
    sim.run(f, steps);
}

} // namespace

TEST(CudaMdConstraints, SettleKeepsEveryWaterRigid) {
    const cpu::Frame f =
        run_cuda(case_named("SettleWater"), kTrajectorySteps, false);
    double worst = 0.0;
    for (int k = 0; k < 125; ++k) {
        worst = std::max(worst, std::abs(distance(f, 3 * k, 3 * k + 1) - kDOH));
        worst = std::max(worst, std::abs(distance(f, 3 * k, 3 * k + 2) - kDOH));
        worst =
            std::max(worst, std::abs(distance(f, 3 * k + 1, 3 * k + 2) - kDHH));
    }
    EXPECT_LT(worst, kDouble ? 1e-10 : 1e-4);
}

TEST(CudaMdConstraints, RattleKeepsEveryXHBond) {
    const cpu::Frame f =
        run_cuda(case_named("RattleXHAngles"), kTrajectorySteps, false);
    double worst_bond = 0.0, worst_hh = 0.0;
    for (int m = 0; m < 27; ++m) {
        const int b = 8 * m;
        for (int h = 0; h < 3; ++h) {
            worst_bond =
                std::max(worst_bond, std::abs(distance(f, b, b + 2 + h) - kCH));
            worst_bond = std::max(
                worst_bond, std::abs(distance(f, b + 1, b + 5 + h) - kCH));
            worst_hh = std::max(
                worst_hh, std::abs(distance(f, b + 2 + h, b + 2 + (h + 1) % 3)
                                   - ethane_hh()));
        }
    }
    EXPECT_LT(worst_bond, kDouble ? 1e-8 : 1e-3);
    EXPECT_LT(worst_hh, kDouble ? 1e-8 : 1e-3);
}

TEST(CudaMdConstraints, SettleOutOfPlaneOxygenFaultsLikeTheCpuThrows) {
    // One oxygen leaves its molecular plane by far more than the
    // molecule's own size in one step (1.6 Å against the 0.066 Å oxygen
    // offset from the centre of mass): the CPU twin throws, the device
    // raises the constraint fault and the run's retrieve throws.
    auto broken = [] {
        cpu::Frame f = water();
        f.span(kAtom_vz)[0] = 8.0;
        return f;
    };
    const Case& c = case_named("SettleWater");
    {
        cpu::Frame f = broken();
        EXPECT_THROW(
            run_on<cpu::md::MD>(f, c.system, mount_cpu_forces, c.cpu, 2),
            std::runtime_error);
    }
    cpu::Frame f = broken();
    try {
        run_on<cuda::md::MD>(f, c.system, mount_cuda_forces, c.gpu, 2);
        ADD_FAILURE() << "the CUDA run did not fault";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("constraint"), std::string::npos)
            << e.what();
    }
}

TEST(CudaMdConstraints, RattleUnconvergedClusterFaultsLikeTheCpuThrows) {
    // One sweep cannot converge a CH3 cluster (three bonds through one
    // carbon) to 1e-10: the CPU twin throws, the device faults.
    auto cpu_setup = [](cpu::md::MD& s) {
        s.set_integrator(cpu::md::VelocityVerlet{2.0 * kFs});
        cpu::md::FixRattle rattle{2.0 * kFs, 1e-10, 1};
        rattle.constrain_bond(1, kCH);
        s.set_observer(std::move(rattle));
    };
    auto cuda_setup = [](cuda::md::MD& s) {
        s.set_integrator(cuda::md::VelocityVerlet{2.0 * kFs});
        cuda::md::FixRattle rattle{2.0 * kFs, 1e-10, 1};
        rattle.constrain_bond(1, kCH);
        s.set_observer(std::move(rattle));
    };
    {
        cpu::Frame f = ethane();
        EXPECT_THROW(run_on<cpu::md::MD>(f, System::ethane, mount_cpu_forces,
                                         cpu_setup, 2),
                     std::runtime_error);
    }
    cpu::Frame f = ethane();
    try {
        run_on<cuda::md::MD>(f, System::ethane, mount_cuda_forces, cuda_setup,
                             2);
        ADD_FAILURE() << "the CUDA run did not fault";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("constraint"), std::string::npos)
            << e.what();
    }
}
