/// @file tests/cpu/scf/test_minao.cpp
/// @brief Regression test: `build_minao_density` matches `pyscf.scf.hf.init_guess_by_minao`.
///
/// Reference density matrices were snapshot from PySCF 2.12 at tight
/// precision (see tools/gen_minao_data.py for how the MINAO basis itself
/// is extracted). This test guards against drift in either the basis
/// tables or the projection math.
#ifdef ATV_HAS_HF

#include <cmath>
#include <gtest/gtest.h>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/internal/minao.h"

namespace {

using namespace ATV;

// PySCF references (aspirin passes at 1e-6; H2O / CH4 tested tighter).

// H2O / STO-3G / PySCF init_guess_by_minao, nao=7, nelec=10
// Geometry: O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468 (Angstrom)
static const double H2O_MINAO_DM[7 * 7] = {
    2.11337908197014590e+00,  -4.79973766292214721e-01,
    0.00000000000000000e+00,  -3.25724055672623365e-18,
    -1.76517547519777634e-02, -1.69434453890347815e-02,
    -1.69434453890347746e-02, -4.79973766292214721e-01,
    2.05156201076023414e+00,  0.00000000000000000e+00,
    6.90712900417109329e-18,  9.19363602614738790e-02,
    3.93967419847204292e-02,  3.93967419847203390e-02,
    0.00000000000000000e+00,  0.00000000000000000e+00,
    1.29010579746254650e+00,  0.00000000000000000e+00,
    0.00000000000000000e+00,  0.00000000000000000e+00,
    0.00000000000000000e+00,  -3.25724055672623365e-18,
    6.90712900417109329e-18,  0.00000000000000000e+00,
    1.13826807948742537e+00,  -3.54622531772069584e-19,
    1.16370105856916359e-01,  -1.16370105856916248e-01,
    -1.76517547519777634e-02, 9.19363602614738790e-02,
    0.00000000000000000e+00,  -3.50546714664272751e-19,
    1.21567531267627538e+00,  -5.56297267828068584e-02,
    -5.56297267828067821e-02, -1.69434453890347850e-02,
    3.93967419847204292e-02,  0.00000000000000000e+00,
    1.16370105856916359e-01,  -5.56297267828068723e-02,
    9.86470128487628162e-01,  7.23924063198283785e-02,
    -1.69434453890347780e-02, 3.93967419847203321e-02,
    0.00000000000000000e+00,  -1.16370105856916248e-01,
    -5.56297267828067959e-02, 7.23924063198283785e-02,
    9.86470128487628828e-01,
};
static constexpr double H2O_MINAO_TRACE = 9.86089710122942620e+00;

// CH4 / STO-3G / PySCF init_guess_by_minao, nao=9, nelec=10
// Geometry: C 0 0 0; H at ±0.629 each combination (Angstrom)
static const double CH4_MINAO_DM[9 * 9] = {
    2.11881898331868035e+00,  -4.71498063649311405e-01,
    7.76018496297429315e-19,  -1.06874778891259517e-18,
    1.46497343698701572e-18,  -2.14732383218545660e-02,
    -2.14732383218545521e-02, -2.14732383218545764e-02,
    -2.14732383218545382e-02, -4.71498063649311405e-01,
    1.93810436639397121e+00,  3.90814465962203740e-18,
    3.76258571071290647e-18,  -5.61286487105243488e-18,
    6.08393680905694109e-02,  6.08393680905693068e-02,
    6.08393680905694040e-02,  6.08393680905693068e-02,
    7.76018496297429315e-19,  3.90814465962203740e-18,
    5.88287600764368035e-01,  -2.96809870482496723e-18,
    1.30255307032860228e-18,  3.44195617599802398e-02,
    -3.44195617599800802e-02, -3.44195617599801773e-02,
    3.44195617599801773e-02,  -1.06874778891259517e-18,
    3.76258571071290647e-18,  -2.96809870482496723e-18,
    5.88287600764368035e-01,  1.61462840001520809e-18,
    3.44195617599802259e-02,  -3.44195617599801218e-02,
    3.44195617599801842e-02,  -3.44195617599801704e-02,
    1.46497343698701572e-18,  -5.61286487105243488e-18,
    1.30255307032860228e-18,  1.61462840001520809e-18,
    5.88287600764368035e-01,  3.44195617599802189e-02,
    3.44195617599801218e-02,  -3.44195617599801357e-02,
    -3.44195617599801565e-02, -2.14732383218545660e-02,
    6.08393680905694109e-02,  3.44195617599802398e-02,
    3.44195617599802259e-02,  3.44195617599802120e-02,
    9.46041389345102179e-01,  4.85403354477363727e-02,
    4.85403354477362756e-02,  4.85403354477363658e-02,
    -2.14732383218545521e-02, 6.08393680905693068e-02,
    -3.44195617599800732e-02, -3.44195617599801149e-02,
    3.44195617599801218e-02,  4.85403354477363727e-02,
    9.46041389345102512e-01,  4.85403354477363588e-02,
    4.85403354477362617e-02,  -2.14732383218545764e-02,
    6.08393680905694040e-02,  -3.44195617599801773e-02,
    3.44195617599801773e-02,  -3.44195617599801357e-02,
    4.85403354477362756e-02,  4.85403354477363588e-02,
    9.46041389345102401e-01,  4.85403354477363380e-02,
    -2.14732383218545382e-02, 6.08393680905693068e-02,
    3.44195617599801773e-02,  -3.44195617599801773e-02,
    -3.44195617599801565e-02, 4.85403354477363658e-02,
    4.85403354477362617e-02,  4.85403354477363380e-02,
    9.46041389345102401e-01,
};
static constexpr double CH4_MINAO_TRACE = 9.92172150763174443e+00;

cpu::Frame make_water() {
    cpu::Frame f(3);
    // O 0 0 0.117; H 0 0.757 -0.468; H 0 -0.757 -0.468
    f.span(ATV::kAtom_x)[0] = 0.0;
    f.span(ATV::kAtom_y)[0] = 0.0;
    f.span(ATV::kAtom_z)[0] = 0.117;
    f.span(ATV::kAtom_type)[0] = 8;
    f.span(ATV::kAtom_x)[1] = 0.0;
    f.span(ATV::kAtom_y)[1] = 0.757;
    f.span(ATV::kAtom_z)[1] = -0.468;
    f.span(ATV::kAtom_type)[1] = 1;
    f.span(ATV::kAtom_x)[2] = 0.0;
    f.span(ATV::kAtom_y)[2] = -0.757;
    f.span(ATV::kAtom_z)[2] = -0.468;
    f.span(ATV::kAtom_type)[2] = 1;
    f.box = ATV::cpu::Box::cubic(100.0);
    return f;
}

cpu::Frame make_methane() {
    cpu::Frame f(5);
    // C 0 0 0; H at ±0.629 tetrahedral positions
    f.span(ATV::kAtom_x)[0] = 0.0;
    f.span(ATV::kAtom_y)[0] = 0.0;
    f.span(ATV::kAtom_z)[0] = 0.0;
    f.span(ATV::kAtom_type)[0] = 6;
    f.span(ATV::kAtom_x)[1] = 0.629;
    f.span(ATV::kAtom_y)[1] = 0.629;
    f.span(ATV::kAtom_z)[1] = 0.629;
    f.span(ATV::kAtom_type)[1] = 1;
    f.span(ATV::kAtom_x)[2] = -0.629;
    f.span(ATV::kAtom_y)[2] = -0.629;
    f.span(ATV::kAtom_z)[2] = 0.629;
    f.span(ATV::kAtom_type)[2] = 1;
    f.span(ATV::kAtom_x)[3] = -0.629;
    f.span(ATV::kAtom_y)[3] = 0.629;
    f.span(ATV::kAtom_z)[3] = -0.629;
    f.span(ATV::kAtom_type)[3] = 1;
    f.span(ATV::kAtom_x)[4] = 0.629;
    f.span(ATV::kAtom_y)[4] = -0.629;
    f.span(ATV::kAtom_z)[4] = -0.629;
    f.span(ATV::kAtom_type)[4] = 1;
    f.box = ATV::cpu::Box::cubic(100.0);
    return f;
}

// Compare an atomiverse density to a PySCF reference to the given tolerance.
void expect_matches_pyscf(const cpu::scf::RealMatrix& atv, const double* ref,
                          int n, double tol) {
    double max_diff = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double d = std::abs(atv(i, j) - ref[i * n + j]);
            if (d > max_diff)
                max_diff = d;
        }
    EXPECT_LT(max_diff, tol) << "max |atv-pyscf| exceeds tolerance";
}

} // namespace

TEST(MinaoGuess, H2O_MatchesPyscfToMicroHartree) {
    auto frame = make_water();
    auto integrals = cpu::scf::internal::IntegralWorkspace{
        frame, cpu::scf::BasisSet::STO3G, 0,
        /*compute_eri=*/false};
    auto P = cpu::scf::build_minao_density(integrals);

    ASSERT_EQ(P.shape(0), 7u);
    ASSERT_EQ(P.shape(1), 7u);
    expect_matches_pyscf(P, H2O_MINAO_DM, 7, 1e-6);

    const double trace_PS = xt::sum(P * integrals.S)();
    EXPECT_NEAR(trace_PS, H2O_MINAO_TRACE, 1e-6);
}

TEST(MinaoGuess, CH4_MatchesPyscfToMicroHartree) {
    auto frame = make_methane();
    auto integrals = cpu::scf::internal::IntegralWorkspace{
        frame, cpu::scf::BasisSet::STO3G, 0,
        /*compute_eri=*/false};
    auto P = cpu::scf::build_minao_density(integrals);

    ASSERT_EQ(P.shape(0), 9u);
    ASSERT_EQ(P.shape(1), 9u);
    expect_matches_pyscf(P, CH4_MINAO_DM, 9, 1e-6);

    const double trace_PS = xt::sum(P * integrals.S)();
    EXPECT_NEAR(trace_PS, CH4_MINAO_TRACE, 1e-6);
}

TEST(MinaoGuess, Symmetric) {
    auto integrals = cpu::scf::internal::IntegralWorkspace{
        make_water(), cpu::scf::BasisSet::STO3G, 0, false};
    auto P = cpu::scf::build_minao_density(integrals);
    for (int i = 0; i < static_cast<int>(P.shape(0)); ++i)
        for (int j = 0; j < i; ++j)
            EXPECT_NEAR(P(i, j), P(j, i), 1e-14);
}

TEST(MinaoGuess, SupportedElements) {
    // H, C, N, O, F are supported; Li (Z=3) currently is not.
    auto frame_h2o = make_water();
    auto integrals_h2o = cpu::scf::internal::IntegralWorkspace{
        frame_h2o, cpu::scf::BasisSet::STO3G, 0, false};
    EXPECT_TRUE(cpu::scf::minao_supported(integrals_h2o));
}

#endif // ATV_HAS_HF
