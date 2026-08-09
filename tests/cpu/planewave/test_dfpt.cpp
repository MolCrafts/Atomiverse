/// @file test_dfpt.cpp
/// @brief Unit tests for the Γ-point DFPT building blocks: the analytic
///        Ewald dynamical matrix, the conduction-space Sternheimer CG on a
///        free-electron Hamiltonian, and the DfptPhonon guard rails.
///        No SCF is run anywhere in this file.

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/vec3.h"
#include "atomiverse/cpu/frame/box.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/dfpt_phonon.h"
#include "atomiverse/cpu/planewave/internal/dfpt.h"
#include "atomiverse/cpu/planewave/internal/ewald.h"
#include "atomiverse/cpu/planewave/internal/fft.h"
#include "atomiverse/cpu/planewave/internal/g_vector.h"
#include "atomiverse/cpu/planewave/internal/pw_workspace.h"

#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

using namespace ATV;
using namespace ATV::cpu::pw;
using namespace ATV::cpu::pw::internal;

namespace {
constexpr double kA2B = 1.8897259886;
}

TEST(EwaldDynamicalMatrix, MatchesForceFd) {
    // Off-symmetry 2-atom periodic cell: analytic Hessian vs central FD of
    // the analytic Ewald forces (same α, same cutoffs).
    cpu::Box box = cpu::Box::cubic(8.0); // Å
    std::vector<Vec3> pos = {{0.0, 0.0, 0.0}, {1.5, 0.7, -0.3}};
    std::vector<AccReal> charges = {4.0, 4.0};
    const AccReal ecut = AccReal(10.0); // Hartree

    const auto D = ewald_dynamical_matrix(pos, charges, box, ecut);
    ASSERT_EQ(D.size(), 36u);

    const double h = 1e-3; // Å
    double max_err = 0;
    for (int i = 0; i < 2; ++i) {
        for (int a = 0; a < 3; ++a) {
            auto pp = pos;
            pp[static_cast<std::size_t>(i)][a] += AccReal(h);
            EwaldSummation ep(pp, charges, box, ecut);
            pp[static_cast<std::size_t>(i)][a] -= AccReal(2 * h);
            EwaldSummation em(pp, charges, box, ecut);
            for (int j = 0; j < 2; ++j) {
                for (int b = 0; b < 3; ++b) {
                    // D = −∂F_jb/∂R_ia, displacement h Å = h·a2b Bohr.
                    const double fd =
                        -(static_cast<double>(
                              ep.forces_bohr()[static_cast<std::size_t>(j)][b])
                          - static_cast<double>(
                              em.forces_bohr()[static_cast<std::size_t>(j)][b]))
                        / (2.0 * h * kA2B);
                    const double an =
                        static_cast<double>(D[static_cast<std::size_t>(
                            (3 * i + a) * 6 + 3 * j + b)]);
                    max_err = std::max(max_err, std::abs(fd - an));
                }
            }
        }
    }
    EXPECT_LT(max_err, 5e-6); // Ha/Bohr²; FD truncation dominates
}

TEST(EwaldDynamicalMatrix, AcousticSumRule) {
    // Row sums over atoms vanish exactly by construction (pair accumulation).
    cpu::Box box = cpu::Box::cubic(7.0);
    std::vector<Vec3> pos = {{0.1, 0.0, 0.2}, {1.9, 1.4, 1.6}};
    std::vector<AccReal> charges = {4.0, 4.0};
    const auto D = ewald_dynamical_matrix(pos, charges, box, AccReal(8.0));
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
            double s01 = static_cast<double>(
                             D[static_cast<std::size_t>((0 + a) * 6 + 0 + b)])
                         + static_cast<double>(
                             D[static_cast<std::size_t>((0 + a) * 6 + 3 + b)]);
            EXPECT_NEAR(s01, 0.0, 1e-12);
        }
}

TEST(SternheimerCg, FreeElectronAnalytic) {
    // V_eff = 0, no KB: H = T(G) diagonal. The projected CG solution is
    // Δψ(G) = P_c rhs(G) / (T(G) − ε) exactly.
    cpu::Box box_bohr = cpu::Box::cubic(8.0); // Bohr
    GVectorList gv = enumerate_g_vectors(box_bohr, AccReal(4.0), Vec3{});
    ASSERT_GT(gv.count(), 10);

    PWWorkspace ws;
    ws.gvec = &gv;
    ws.n_g = gv.count();
    ws.fft_nx = gv.nx;
    ws.fft_ny = gv.ny;
    ws.fft_nz = gv.nz;
    ws.nr = gv.nx * gv.ny * gv.nz;
    ws.nc = (gv.nx / 2 + 1) * gv.ny * gv.nz;
    ws.fft = std::make_unique<FFTPlan3D>(gv.nx, gv.ny, gv.nz);
    ws.omega = box_bohr.volume();
    ws.T_G.resize({static_cast<std::size_t>(ws.n_g)});
    for (int ig = 0; ig < ws.n_g; ++ig)
        ws.T_G[ig] = gv.ekin[ig];
    ws.V_eff_r.resize({static_cast<std::size_t>(ws.nr)});
    for (int i = 0; i < ws.nr; ++i)
        ws.V_eff_r[i] = AccReal(0);

    // Occupied manifold: the G = 0 plane wave (lowest T).
    int ig0 = 0;
    for (int ig = 1; ig < ws.n_g; ++ig)
        if (ws.T_G[ig] < ws.T_G[ig0])
            ig0 = ig;
    ASSERT_NEAR(static_cast<double>(ws.T_G[ig0]), 0.0, 1e-14);

    std::vector<std::complex<AccReal>> occ(static_cast<std::size_t>(ws.n_g),
                                           {0, 0});
    occ[static_cast<std::size_t>(ig0)] = {1, 0};
    const AccReal eps = ws.T_G[ig0];

    std::vector<std::complex<AccReal>> rhs(static_cast<std::size_t>(ws.n_g));
    for (int ig = 0; ig < ws.n_g; ++ig)
        rhs[static_cast<std::size_t>(ig)] = {
            AccReal(1) / (AccReal(1) + ws.T_G[ig]),
            AccReal(0.2) / (AccReal(2) + ws.T_G[ig])};

    std::vector<std::complex<AccReal>> x(static_cast<std::size_t>(ws.n_g),
                                         {0, 0});
    FFTWBuffer<fftw_complex> work(static_cast<std::size_t>(ws.nr));
    const int its = sternheimer_cg(ws, occ.data(), 1, eps, rhs.data(), x.data(),
                                   AccReal(1e-12), 500, work.data());
    EXPECT_LE(its, 500);

    for (int ig = 0; ig < ws.n_g; ++ig) {
        std::complex<AccReal> want(0, 0);
        if (ig != ig0)
            want = rhs[static_cast<std::size_t>(ig)] / (ws.T_G[ig] - eps);
        EXPECT_NEAR(static_cast<double>(x[static_cast<std::size_t>(ig)].real()),
                    static_cast<double>(want.real()), 1e-8);
        EXPECT_NEAR(static_cast<double>(x[static_cast<std::size_t>(ig)].imag()),
                    static_cast<double>(want.imag()), 1e-8);
    }
}

namespace {

cpu::Frame si_frame() {
    cpu::Frame frame(2);
    frame.span(kAtom_type)[0] = 14;
    frame.span(kAtom_type)[1] = 14;
    frame.span(kAtom_mass)[0] = 28.0855;
    frame.span(kAtom_mass)[1] = 28.0855;
    frame.span(kAtom_x)[1] = 5.43 / 4.0;
    frame.span(kAtom_y)[1] = 5.43 / 4.0;
    frame.span(kAtom_z)[1] = 5.43 / 4.0;
    frame.box = cpu::Box::cubic(5.43);
    return frame;
}

} // namespace

TEST(DfptPhonon, ThrowsOnEmptyUpf) {
    auto frame = si_frame();
    PW method; // no pseudopotential set at all
    method.xc = cpu::scf::XCType::LSDA;
    DfptPhonon dfpt;
    EXPECT_THROW(dfpt.compute(method, frame), std::invalid_argument);
}

TEST(DfptPhonon, ThrowsOnHybridXc) {
    // Pure GGA is supported; hybrids (exact-exchange response) still throw.
    // Validation of XC precedes UPF I/O for hybrid_exchange_fraction ≠ 0.
    auto frame = si_frame();
    PW method;
    method.upf_path = "unused.upf";
    method.xc = cpu::scf::XCType::PBE0;
    DfptPhonon dfpt;
    EXPECT_THROW(dfpt.compute(method, frame), std::invalid_argument);
}

TEST(DfptPhonon, ThrowsOnEmptyUpfWithPbe) {
    // Pure PBE is accepted at the XC gate; empty UPF still fails first.
    auto frame = si_frame();
    PW method; // no pseudopotential set at all
    method.xc = cpu::scf::XCType::PBE;
    DfptPhonon dfpt;
    EXPECT_THROW(dfpt.compute(method, frame), std::invalid_argument);
}

TEST(DfptPhonon, RestrictedSmearPassesXcGate) {
    // Restricted metallic smearing is accepted at the XC/guard layer; the
    // missing UPF fails during SCF setup (runtime_error from the reader).
    auto frame = si_frame();
    PW method;
    method.upf_path = "unused.upf";
    method.xc = cpu::scf::XCType::LSDA;
    method.smearing = Smearing::fermi_dirac;
    method.degauss = 0.02;
    method.extra_bands = 2;
    DfptPhonon dfpt;
    EXPECT_ANY_THROW(dfpt.compute(method, frame));
}

TEST(DfptPhonon, ThrowsOnSmearWithoutDegauss) {
    auto frame = si_frame();
    PW method;
    method.upf_path = "unused.upf";
    method.xc = cpu::scf::XCType::LSDA;
    method.smearing = Smearing::fermi_dirac;
    method.degauss = 0.0;
    DfptPhonon dfpt;
    EXPECT_THROW(dfpt.compute(method, frame), std::invalid_argument);
}

TEST(DfptPhonon, SpinWithSmearingReachesUpfIo) {
    // Collinear spin + metallic smear at Gamma is supported; empty/missing
    // UPF fails inside the spin+smear path (I/O), not at a spin+smear gate.
    auto frame = si_frame();
    PW method;
    method.upf_path = "unused.upf";
    method.xc = cpu::scf::XCType::LSDA;
    method.spin_polarized = true;
    method.multiplicity = 1;
    method.smearing = Smearing::fermi_dirac;
    method.degauss = 0.02;
    DfptPhonon dfpt;
    EXPECT_THROW(dfpt.compute(method, frame), std::exception);
}

TEST(DfptPhonon, SpinSmearIncommensurateQReachesUpfIo) {
    // Direct finite-q Sternheimer accepts collinear spin + metallic smear;
    // missing UPF fails inside setup (I/O), not at a spin+smear gate.
    auto frame = si_frame();
    PW method;
    method.upf_path = "unused.upf";
    method.xc = cpu::scf::XCType::LSDA;
    method.spin_polarized = true;
    method.multiplicity = 1;
    method.smearing = Smearing::fermi_dirac;
    method.degauss = 0.02;
    DfptPhonon dfpt;
    dfpt.q_frac = {0.3, 0.0, 0.0};
    EXPECT_THROW(dfpt.compute(method, frame), std::exception);
}

TEST(DfptPhonon, CommensurateFiniteQSpinSmearReachesUpfIo) {
    // X = (1/2,0,0) uses supercell Γ-DFPT; spin+smear is accepted there.
    auto frame = si_frame();
    PW method;
    method.upf_path = "unused.upf";
    method.xc = cpu::scf::XCType::LSDA;
    method.spin_polarized = true;
    method.multiplicity = 1;
    method.smearing = Smearing::fermi_dirac;
    method.degauss = 0.02;
    DfptPhonon dfpt;
    dfpt.q_frac = {0.5, 0.0, 0.0};
    EXPECT_THROW(dfpt.compute(method, frame), std::exception);
}

TEST(DfptPhonon, FiniteQWithSmearingReachesUpfIo) {
    // Restricted metallic smear is accepted on direct finite-q Sternheimer;
    // missing UPF fails inside setup (I/O), not at a smear gate.
    auto frame = si_frame();
    PW method;
    method.upf_path = "unused.upf";
    method.xc = cpu::scf::XCType::LSDA;
    method.smearing = Smearing::fermi_dirac;
    method.degauss = 0.02;
    DfptPhonon dfpt;
    dfpt.q_frac = {0.3, 0.0, 0.0};
    EXPECT_THROW(dfpt.compute(method, frame), std::exception);
}

TEST(DfptPhonon, CommensurateFiniteQWithSmearingReachesUpfIo) {
    // X = (1/2,0,0) uses supercell Γ-DFPT; smear is accepted there too.
    auto frame = si_frame();
    PW method;
    method.upf_path = "unused.upf";
    method.xc = cpu::scf::XCType::LSDA;
    method.smearing = Smearing::fermi_dirac;
    method.degauss = 0.02;
    DfptPhonon dfpt;
    dfpt.q_frac = {0.5, 0.0, 0.0};
    EXPECT_THROW(dfpt.compute(method, frame), std::exception);
}

TEST(DfptPhonon, SpinIncommensurateQReachesUpfIo) {
    // Collinear spin at incommensurate q is supported; empty/missing UPF
    // still fails inside dfpt_q_force_constants (I/O), not at a spin gate.
    auto frame = si_frame();
    PW method;
    method.upf_path = "unused.upf";
    method.xc = cpu::scf::XCType::LSDA;
    method.spin_polarized = true;
    method.multiplicity = 1;
    DfptPhonon dfpt;
    dfpt.q_frac = {0.3, 0.0, 0.0};
    EXPECT_THROW(dfpt.compute(method, frame), std::exception);
}

TEST(DfptPhonon, ThrowsOnBadMixBeta) {
    auto frame = si_frame();
    PW method;
    method.upf_path = "unused.upf";
    method.xc = cpu::scf::XCType::LSDA;
    DfptPhonon dfpt;
    dfpt.mix_beta = 1.5;
    EXPECT_THROW(dfpt.compute(method, frame), std::invalid_argument);
}

TEST(EwaldDynamicalMatrixQ, MatchesGammaAtZeroQ) {
    cpu::Box box = cpu::Box::cubic(8.0);
    std::vector<Vec3> pos = {{0.0, 0.0, 0.0}, {1.5, 0.7, -0.3}};
    std::vector<AccReal> charges = {4.0, 4.0};
    const AccReal ecut = AccReal(10.0);
    const auto Dg = ewald_dynamical_matrix(pos, charges, box, ecut);
    const auto Dq =
        ewald_dynamical_matrix_q(pos, charges, box, ecut, Vec3{0, 0, 0});
    ASSERT_EQ(Dg.size(), Dq.size());
    double max_err = 0;
    for (std::size_t i = 0; i < Dg.size(); ++i) {
        max_err = std::max(max_err, std::abs(static_cast<double>(Dg[i])
                                             - static_cast<double>(Dq[i].real())));
        EXPECT_NEAR(static_cast<double>(Dq[i].imag()), 0.0, 1e-10);
    }
    EXPECT_LT(max_err, 1e-12);
}

TEST(EwaldDynamicalMatrixQ, HermitianAtFiniteQ) {
    cpu::Box box = cpu::Box::cubic(7.0);
    std::vector<Vec3> pos = {{0.1, 0.0, 0.2}, {1.9, 1.4, 1.6}};
    std::vector<AccReal> charges = {4.0, 4.0};
    const auto D = ewald_dynamical_matrix_q(pos, charges, box, AccReal(8.0),
                                            Vec3{0.5, 0.0, 0.0});
    const int dim = 6;
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j) {
            const auto a = D[static_cast<std::size_t>(i) * dim + j];
            const auto b = D[static_cast<std::size_t>(j) * dim + i];
            EXPECT_NEAR(static_cast<double>(a.real()),
                        static_cast<double>(b.real()), 1e-12);
            EXPECT_NEAR(static_cast<double>(a.imag()),
                        -static_cast<double>(b.imag()), 1e-12);
        }
}

TEST(DfptPhonon, DefaultQIsGamma) {
    DfptPhonon dfpt;
    EXPECT_DOUBLE_EQ(dfpt.q_frac[0], 0.0);
    EXPECT_DOUBLE_EQ(dfpt.q_frac[1], 0.0);
    EXPECT_DOUBLE_EQ(dfpt.q_frac[2], 0.0);
}

// Incommensurate q must not throw at the rational-supercell gate. Empty UPF
// still fails validation inside dfpt_q_force_constants (no SCF).
TEST(DfptPhonon, IncommensurateQReachesUpfGuard) {
    auto frame = si_frame();
    PW method;
    method.xc = cpu::scf::XCType::LSDA;
    // upf_path empty
    DfptPhonon dfpt;
    dfpt.q_frac = {0.3, 0.0, 0.0};
    EXPECT_THROW(dfpt.compute(method, frame), std::invalid_argument);
}

TEST(DfptPhonon, IncommensurateQThrowsOnHybridXc) {
    // Pure GGA is supported at finite q; hybrids still fail the XC gate
    // before UPF I/O.
    auto frame = si_frame();
    PW method;
    method.upf_path = "unused.upf";
    method.xc = cpu::scf::XCType::PBE0;
    DfptPhonon dfpt;
    dfpt.q_frac = {0.3, 0.1, 0.05};
    EXPECT_THROW(dfpt.compute(method, frame), std::invalid_argument);
}

TEST(DfptPhonon, OversizeCommensurateFallsToDirectPathGuard) {
    // q = 1/7 is rational but n=7 > nmax=6 → direct Sternheimer gate (empty UPF).
    auto frame = si_frame();
    PW method;
    method.xc = cpu::scf::XCType::LSDA;
    DfptPhonon dfpt;
    dfpt.q_frac = {1.0 / 7.0, 0.0, 0.0};
    EXPECT_THROW(dfpt.compute(method, frame), std::invalid_argument);
}


TEST(EwaldDynamicalMatrix, SupercellEnergySane) {
    constexpr double kA=5.43, h=kA/2;
    Mat3 cell; cell(0,0)=0; cell(0,1)=2*h; cell(0,2)=2*h;
    cell(1,0)=h; cell(1,1)=0; cell(1,2)=h;
    cell(2,0)=h; cell(2,1)=h; cell(2,2)=0;
    cpu::Box box; box.set_cell(cell);
    std::vector<Vec3> pos={{0,0,0},{kA/4,kA/4,kA/4},{0,2.715,2.715},{1.3575,4.0725,4.0725}};
    std::vector<AccReal> z={4,4,4,4};
    EwaldSummation ew(pos, z, box, AccReal(5.0));
    EXPECT_LT(std::abs(static_cast<double>(ew.total_energy())), 1000.0);
}
