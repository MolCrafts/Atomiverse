/// @file tests/core/test_pppm_accuracy.cpp
/// @brief `longrange::pppm_from_accuracy` against LAMMPS `PPPM::init`.
///
/// Expected values are what LAMMPS printed (`G vector`, `grid`, `estimated
/// absolute RMS force accuracy`) for `units metal`, `atom_style charge`,
/// `pair_style coul/long rc`, `pair_modify table 0`, `kspace_style pppm acc`,
/// `kspace_modify order p` (+ the listed `kspace_modify` overrides), `run 0`,
/// one MPI rank. The data files hold the charges of `make_sites` below at
/// r = H s (H the LAMMPS restricted-triclinic cell). LAMMPS binaries: the
/// GH200 oracle `lab/projects/pzpe/engines/lmp-cu132` (2 Sep 2026, e8c4c278a8)
/// and the x86_64 `~/.local/x86_64/bin/lmp` (30 Mar 2026 development) print
/// the same numbers for every case.
#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/core/pppm_accuracy.h"
#include "atomiverse/core/units.h"

using namespace ATV;

namespace {

struct Case {
    const char* name;
    double lx, ly, lz, xy, xz, yz; ///< LAMMPS box [Å]
    int natoms;
    double q;        ///< |charge| [e], alternating sign
    double cutoff;   ///< coul/long cutoff [Å]
    double accuracy; ///< kspace_style pppm (relative)
    int order;
    double fixed_alpha; ///< kspace_modify gewald, 0 = none
    int fixed_mesh[3];  ///< kspace_modify mesh, 0 = none
    // LAMMPS output
    double alpha;
    int mesh[3];
    double accuracy_absolute;
};

// clang-format off
constexpr Case kCases[] = {
    {"ortho_o5", 12, 13, 14, 0, 0, 0, 40, 1.0, 6.0, 1e-4, 5, 0, {0, 0, 0},
     0.46565994, {12, 15, 15}, 0.00090979614},
    {"ortho_o6", 12, 13, 14, 0, 0, 0, 40, 1.0, 6.0, 1e-4, 6, 0, {0, 0, 0},
     0.46340394, {12, 12, 12}, 0.00098501461},
    {"ortho_o7", 12, 13, 14, 0, 0, 0, 40, 1.0, 6.0, 1e-6, 7, 0, {0, 0, 0},
     0.5799976, {20, 24, 24}, 1.148331e-05},
    {"ortho_o4", 20, 20, 30, 0, 0, 0, 64, 0.5, 8.0, 1e-5, 4, 0, {0, 0, 0},
     0.34215452, {20, 20, 25}, 0.00014236147},
    {"tri_o6", 11, 12, 13, 2.0, -1.5, 1.0, 30, 0.8, 5.0, 1e-5, 6, 0, {0, 0, 0},
     0.63583362, {18, 24, 24}, 6.0316364e-05},
    {"mesh_o6", 12, 13, 14, 0, 0, 0, 40, 1.0, 6.0, 1e-4, 6, 0, {20, 21, 22},
     0.55705724, {20, 24, 24}, 3.0342276e-05},
    {"gewald_o6", 12, 13, 14, 0, 0, 0, 40, 1.0, 6.0, 1e-5, 6, 0.3, {0, 0, 0},
     0.3, {10, 10, 12}, 0.062315461},
};
// clang-format on

longrange::PppmAccuracyRequest request_for(const Case& c) {
    longrange::PppmAccuracyRequest r;
    r.accuracy = c.accuracy;
    r.cutoff = c.cutoff;
    r.order = c.order;
    // Columns a = (lx,0,0), b = (xy,ly,0), c = (xz,yz,lz).
    const double cell[9] = {c.lx, c.xy, c.xz, 0, c.ly, c.yz, 0, 0, c.lz};
    for (int k = 0; k < 9; ++k)
        r.cell[k] = cell[k];
    r.natoms = c.natoms;
    r.qsqsum = c.natoms * c.q * c.q;
    r.alpha = c.fixed_alpha;
    for (int d = 0; d < 3; ++d)
        r.mesh[d] = c.fixed_mesh[d];
    // The printed goldens are `units metal`. The default request is lj.
    r.unit_system = units::from_style("metal");
    return r;
}

} // namespace

TEST(PppmAccuracy, MeshAndAlphaMatchLammpsInit) {
    for (const Case& c : kCases) {
        SCOPED_TRACE(c.name);
        const auto req = request_for(c);
        const auto got = longrange::pppm_from_accuracy(req);
        for (int d = 0; d < 3; ++d)
            EXPECT_EQ(got.mesh[d], c.mesh[d]) << "axis " << d;
        EXPECT_EQ(got.order, c.order);
        // LAMMPS prints 8 significant digits.
        EXPECT_NEAR(got.alpha, c.alpha, 6e-8 * c.alpha);
        EXPECT_NEAR(got.accuracy_absolute, c.accuracy_absolute,
                    6e-8 * c.accuracy_absolute);
        EXPECT_NEAR(got.accuracy_relative,
                    got.accuracy_absolute / req.unit_system.two_charge_force(),
                    1e-18);
    }
}

TEST(PppmAccuracy, FinalEstimateMatchesTheCgElectrolyteOracle) {
    // regressions/forcefield/cg-electrolyte: 360 beads, 144 of them at
    // ±q0, 9 Å cube, kspace_modify gewald 1.1 mesh 32 32 32 order p;
    // lammps_output.txt (GH200 oracle) prints the final estimate per order
    // and per real-space cutoff (coul/long 3.5, and 0.05 for the k-space
    // term).
    constexpr double q0 = 0.073389141347849185;
    struct Row {
        int order;
        double cutoff;
        double estimate;
    };
    const Row rows[] = {{6, 3.5, 2.2276433e-07},
                        {5, 3.5, 1.1257833e-06},
                        {7, 3.5, 5.1262689e-08},
                        {6, 0.05, 0.19439905}};
    for (const Row& row : rows) {
        SCOPED_TRACE("order " + std::to_string(row.order) + " cutoff "
                     + std::to_string(row.cutoff));
        longrange::PppmAccuracyRequest r;
        r.accuracy = 1.0e-8;
        r.cutoff = row.cutoff;
        r.order = row.order;
        r.cell[0] = r.cell[4] = r.cell[8] = 9.0;
        r.natoms = 360;
        r.qsqsum = 144 * q0 * q0;
        r.alpha = 1.1000000000000001;
        r.mesh[0] = r.mesh[1] = r.mesh[2] = 32;
        r.unit_system = units::from_style("metal");
        const auto got = longrange::pppm_from_accuracy(r);
        EXPECT_EQ(got.alpha, r.alpha);
        for (int d = 0; d < 3; ++d)
            EXPECT_EQ(got.mesh[d], 32);
        EXPECT_NEAR(got.accuracy_absolute, row.estimate, 6e-8 * row.estimate);
    }
}

TEST(PppmAccuracy, RotatedCellGivesTheSameChoice) {
    // The choice depends on the LAMMPS restricted-triclinic box only, not
    // on how the lattice vectors are oriented in space.
    const Case& c = kCases[4];
    auto r = request_for(c);
    const auto base = longrange::pppm_from_accuracy(r);
    const double t = 0.4, cs = std::cos(t), sn = std::sin(t);
    const double rot[9] = {cs, -sn, 0, sn, cs, 0, 0, 0, 1};
    double turned[9] = {};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                turned[3 * i + j] += rot[3 * i + k] * r.cell[3 * k + j];
    for (int k = 0; k < 9; ++k)
        r.cell[k] = turned[k];
    const auto rotated = longrange::pppm_from_accuracy(r);
    for (int d = 0; d < 3; ++d)
        EXPECT_EQ(rotated.mesh[d], base.mesh[d]);
    EXPECT_NEAR(rotated.alpha, base.alpha, 1e-12 * base.alpha);
}

TEST(PppmAccuracy, EstimateIkErrorIsTheAconsSeries) {
    // order 3: acons[3] = {1/588, 7/1440, 21/3872}.
    const double h = 0.8, prd = 10.0, q2 = 3.0, alpha = 0.4;
    const long long n = 50;
    const double x = h * alpha;
    const double sum =
        1.0 / 588.0 + 7.0 / 1440.0 * x * x + 21.0 / 3872.0 * x * x * x * x;
    const double expect =
        q2 * x * x * x
        * std::sqrt(alpha * prd * std::sqrt(2.0 * units::pi) * sum / n)
        / (prd * prd);
    EXPECT_NEAR(longrange::pppm_estimate_ik_error(h, prd, n, q2, alpha, 3),
                expect, 1e-15 * expect);
    EXPECT_EQ(longrange::pppm_estimate_ik_error(h, prd, 0, q2, alpha, 3), 0.0);
}

TEST(PppmAccuracy, SmallMeshIsRaisedToTheStencilBound) {
    // LAMMPS would pick a 4-point axis here; the Atomiverse stencil wraps
    // one period, so the axis becomes the smallest 2·3·5 size ≥ order + 1.
    longrange::PppmAccuracyRequest r;
    r.accuracy = 1e-2;
    r.cutoff = 5.0;
    r.order = 6;
    const double cell[9] = {8, 0, 0, 0, 8, 0, 0, 0, 8};
    for (int k = 0; k < 9; ++k)
        r.cell[k] = cell[k];
    r.natoms = 10;
    r.qsqsum = 10.0;
    const auto got = longrange::pppm_from_accuracy(r);
    for (int d = 0; d < 3; ++d)
        EXPECT_GE(got.mesh[d], 8);
}

TEST(PppmAccuracy, LjUnitChargesKeepTheMetalMesh) {
    // qelectron = angstrom = 1 in both rows, so two_charge == qqrd2e and
    // the mesh loop is invariant. The final α is not: LAMMPS
    // adjust_gewald stops on the absolute |f| < SMALL, so a row with a
    // smaller qqr2e stops at a different Newton iterate. With the metal
    // (α, mesh) pinned, the absolute RMS scales with qqr2e.
    for (const Case& c : kCases) {
        SCOPED_TRACE(c.name);
        const auto metal_req = request_for(c);
        auto lj_req = metal_req;
        lj_req.unit_system = units::UnitSystem::lj();
        const auto metal = longrange::pppm_from_accuracy(metal_req);
        const auto lj = longrange::pppm_from_accuracy(lj_req);
        for (int d = 0; d < 3; ++d)
            EXPECT_EQ(lj.mesh[d], metal.mesh[d]);
        EXPECT_NEAR(lj.accuracy_relative, lj.accuracy_absolute, 1e-18);

        auto pinned_req = lj_req;
        pinned_req.alpha = metal.alpha;
        for (int d = 0; d < 3; ++d)
            pinned_req.mesh[d] = metal.mesh[d];
        const auto pinned = longrange::pppm_from_accuracy(pinned_req);
        EXPECT_EQ(pinned.alpha, metal.alpha);
        const double scale = metal_req.unit_system.two_charge_force();
        EXPECT_NEAR(pinned.accuracy_absolute, metal.accuracy_absolute / scale,
                    1e-12 * pinned.accuracy_absolute);
    }
}

TEST(PppmAccuracy, CgLjMeshMatchesTheLammpsProbe) {
    // MIL r=0.90, units lj: N=1368, L=12.027562977106312, 1008 charges at
    // ±√3 (qsqsum=3024), coul cutoff 3, accuracy 1e-4. The GH200 probe
    // (scratch/gpu-bench/mil-r0.90-lmp.kk.log) is order 5 and prints
    // G=1.0742731, grid 45 45 45. Its absolute RMS 9.5201105e-05 includes
    // the default pair table; this estimate is pair_modify table 0.
    // pzpe asks for order 6, which settles on 36³.
    constexpr double L = 12.027562977106312;
    longrange::PppmAccuracyRequest lj;
    lj.accuracy = 1e-4;
    lj.cutoff = 3.0;
    lj.cell[0] = lj.cell[4] = lj.cell[8] = L;
    lj.natoms = 1368;
    lj.qsqsum = 3024.0;
    lj.order = 5;
    const auto o5 = longrange::pppm_from_accuracy(lj);
    EXPECT_EQ(o5.mesh[0], 45);
    EXPECT_EQ(o5.mesh[1], 45);
    EXPECT_EQ(o5.mesh[2], 45);
    EXPECT_NEAR(o5.alpha, 1.0742731, 6e-8 * 1.0742731);
    EXPECT_NEAR(o5.accuracy_absolute, 9.5180268122e-05,
                6e-8 * 9.5180268122e-05);
    EXPECT_NEAR(o5.accuracy_relative, o5.accuracy_absolute, 1e-18);

    lj.order = 6;
    const auto o6 = longrange::pppm_from_accuracy(lj);
    EXPECT_EQ(o6.mesh[0], 36);
    EXPECT_EQ(o6.mesh[1], 36);
    EXPECT_EQ(o6.mesh[2], 36);
    EXPECT_NEAR(o6.alpha, 1.07651986869, 6e-8 * 1.07651986869);
    EXPECT_NEAR(o6.accuracy_absolute, 9.03169756509e-05,
                6e-8 * 9.03169756509e-05);

    // The old metal mapping: same geometry, charges shrunk by
    // sqrt(ε0/qqr2e), accuracy still 1e-4. That hits the g>=1 branch.
    const double energy = 8.617333262e-5 * 300.0;
    longrange::PppmAccuracyRequest metal = lj;
    metal.order = 6;
    metal.qsqsum = 3024.0 * (energy / 14.399645);
    metal.unit_system = units::from_style("metal");
    const auto coarse = longrange::pppm_from_accuracy(metal);
    EXPECT_EQ(coarse.mesh[0], 12);
    EXPECT_EQ(coarse.mesh[1], 12);
    EXPECT_EQ(coarse.mesh[2], 12);
    EXPECT_NEAR(coarse.alpha, 0.720987551267, 6e-8 * 0.720987551267);
    EXPECT_GT(o6.mesh[0], coarse.mesh[0]);
}

TEST(PppmAccuracy, RejectsInvalidRequests) {
    auto ok = request_for(kCases[1]);
    auto bad = ok;
    bad.accuracy = 0.0;
    EXPECT_THROW(longrange::pppm_from_accuracy(bad), std::invalid_argument);
    bad = ok;
    bad.order = 8;
    EXPECT_THROW(longrange::pppm_from_accuracy(bad), std::invalid_argument);
    bad = ok;
    bad.mesh[0] = 16;
    EXPECT_THROW(longrange::pppm_from_accuracy(bad), std::invalid_argument);
    bad = ok;
    bad.qsqsum = 0.0;
    EXPECT_THROW(longrange::pppm_from_accuracy(bad), std::invalid_argument);
    bad.alpha = 0.3; // uncharged is fine with a fixed alpha
    EXPECT_NO_THROW(longrange::pppm_from_accuracy(bad));
    bad = ok;
    bad.natoms = 0;
    EXPECT_THROW(longrange::pppm_from_accuracy(bad), std::invalid_argument);
}
