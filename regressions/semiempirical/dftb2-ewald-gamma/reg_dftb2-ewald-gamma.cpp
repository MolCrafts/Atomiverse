/// @file regressions/semiempirical/dftb2-ewald-gamma/reg_dftb2-ewald-gamma.cpp
/// @brief Ewald-summed periodic SCC γ vs DFTB+ on a polar water cell.
///
/// Domain: SCC-DFTB2 (Elstner et al., Phys. Rev. B 58, 7260 (1998)) under 3-D
/// periodic boundaries, mio-1-1. Every other periodic case in this suite is a
/// homonuclear carbon chain where Δq ≡ 0 by symmetry, so E² vanishes and the
/// charge kernel is never exercised; this cell is polar and is the gate on the
/// Ewald treatment of γ's 1/R half. Public API only. <= 60 s.
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"

using namespace ATV;

namespace {

// 8 waters, random orientations, cubic 7.0 Å (~0.81 g/cm³). Geometry and the
// pinned values are the twin in tools/tb_oracle/inputs/dftbplus/dftb2-ewald-gamma/.
constexpr int kNat = 24;
constexpr int kZnum[] = {8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1,
                         8, 1, 1, 8, 1, 1, 8, 1, 1, 8, 1, 1};
constexpr double kPx[] = {
    -0.0732797495, 0.1809613674,  0.6824432840, -0.0514297404, 0.6141779072,
    0.0535641618,  -0.2407676239, 0.1094438666, -1.1414820823, -0.1626781756,
    -0.5247682562, -0.7327417158, 3.3762864232, 4.0533709385,  3.7169811408,
    3.3358584678,  2.6141476801,  3.2571129125, 3.5344097740,  2.7489353490,
    3.7406766111,  3.4372518187,  3.8256912542, 2.5020007581};
constexpr double kPy[] = {
    -0.0007076845, -0.6347547813, 0.0378056567,  -0.2982686596, -0.4893001502,
    -0.9947695860, 3.6210048464,  2.7588128450,  3.4456516298,  3.6013820956,
    3.9190698903,  2.8693140347,  -0.1267883946, 0.3623014016,  -0.2051387448,
    -0.0462642146, 0.3986494519,  0.2334386697,  3.6528025669,  3.1825548232,
    3.2723932997,  3.3480300544,  2.4794891522,  3.1760506787};
constexpr double kPz[] = {
    0.1101771075,  -0.5643195445, 0.7009388355,  3.2426048348,  2.5777188077,
    3.8948798842,  -0.0838408823, -0.3195883419, 0.1982312961,  3.5350728103,
    2.7046721608,  3.7814544541,  0.1282566371,  -0.3449745226, 1.0223418299,
    3.5186942410,  3.9690112052,  2.6037267913,  -0.1183447434, 0.1706137091,
    -0.9752827699, 3.6877641784,  3.8156371865,  3.5561262356};

// DFTB+ 25.1 (commit a23bfb2) + mio-1-1, Γ, SCCTolerance 1e-10 [eV/Å].
constexpr double kFx[] = {
    -0.264638505, -0.046047818, 0.309966442,  -0.011432186, 0.210742545,
    -0.159727317, 0.280810644,  0.261736852,  -0.615282354, 0.424676569,
    -0.121514977, -0.212320815, -0.501661981, 0.338960100,  0.195778312,
    0.090505097,  -0.330699932, 0.121056135,  0.276951994,  -0.393047008,
    0.176561783,  0.150457063,  0.360115501,  -0.541946142};
constexpr double kFy[] = {
    0.415929812,  -0.413676616, 0.049248883,  0.441671609,  -0.081319088,
    -0.433176859, 0.197715409,  -0.361849090, 0.174147646,  0.051553739,
    0.318400308,  -0.327918516, 0.263033776,  0.059027167,  -0.333208375,
    0.150687116,  0.018541121,  -0.030129047, -0.091432587, -0.012099637,
    0.099836155,  0.004690937,  -0.390360042, 0.230686179};
constexpr double kFz[] = {
    -0.048545725, -0.356394346, 0.283912125,  0.065663446,  -0.374862623,
    0.400718060,  0.094400143,  -0.181871028, 0.131443816,  0.107895727,
    -0.430726331, 0.264380596,  0.011417099,  -0.389532194, 0.301533062,
    0.068626398,  0.412693912,  -0.370486294, 0.294382213,  0.261058515,
    -0.602148534, 0.096144982,  0.035895358,  -0.075598375};

cpu::Frame make_water_cell() {
    cpu::Frame f(kNat);
    for (int i = 0; i < kNat; ++i) {
        f.span(kAtom_type)[i] = kZnum[i];
        f.span(kAtom_x)[i] = kPx[i];
        f.span(kAtom_y)[i] = kPy[i];
        f.span(kAtom_z)[i] = kPz[i];
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_mass)[i] = (kZnum[i] == 8) ? 15.999 : 1.008;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box::cubic(7.0);
    f.meta.get(kEnergy) = 0;
    return f;
}

} // namespace

int main() {
    auto frame = make_water_cell();

    cpu::Evaluate eval;
    cpu::semiempirical::DFTB2 dftb;
    dftb.boundary = cpu::semiempirical::DftbBoundary::lattice;
    dftb.scc_tol = 1e-10;
    dftb.max_scc = 200;
    dftb.with_forces = true;
    eval.on(evaluate::Phase::evaluate).add_ref(dftb);
    eval.build();
    eval.run(frame);

    const double e = dftb.last_energy();

    // DFTB+ prints Total energy = −32.6355809899 Ha and Energy SCC =
    // 0.1379943876 Ha for this cell.
    constexpr double kEPinEV = -888.0593996784;
    constexpr double kESccPinEV = 3.7550185808;
    constexpr double kETolEV = 1e-3;
    constexpr double kESccTolEV = 1e-3;
    constexpr double kFTolEVA = 1e-3;

    std::printf("dftb2-ewald-gamma: 8 H2O, cubic 7.0 A, Gamma\n");
    std::printf("  E       = %.8f eV   pin %.8f  |dE| = %.3e\n", e, kEPinEV,
                std::fabs(e - kEPinEV));
    std::printf("  E_SCC   = %.8f eV   pin %.8f  |dE| = %.3e\n",
                dftb.last_e_scc(), kESccPinEV,
                std::fabs(dftb.last_e_scc() - kESccPinEV));

    if (std::fabs(e - kEPinEV) > kETolEV) {
        std::printf("FAIL total energy vs DFTB+ (tol %.1e eV)\n", kETolEV);
        return 1;
    }

    // E² is the whole point of this case: a truncated 1/R lattice sum makes it
    // change sign, so gate it on its own rather than only inside the total.
    if (std::fabs(dftb.last_e_scc() - kESccPinEV) > kESccTolEV) {
        std::printf("FAIL SCC energy vs DFTB+ (tol %.1e eV)\n", kESccTolEV);
        return 1;
    }

    double max_df = 0;
    for (int i = 0; i < kNat; ++i) {
        max_df = std::max(max_df, std::fabs(frame.span(kAtom_fx)[i] - kFx[i]));
        max_df = std::max(max_df, std::fabs(frame.span(kAtom_fy)[i] - kFy[i]));
        max_df = std::max(max_df, std::fabs(frame.span(kAtom_fz)[i] - kFz[i]));
    }
    std::printf("  max |dF| = %.3e eV/A (tol %.1e)\n", max_df, kFTolEVA);
    if (max_df > kFTolEVA) {
        std::printf("FAIL forces vs DFTB+ (tol %.1e eV/A)\n", kFTolEVA);
        return 1;
    }

    // Δq must be genuinely nonzero, or this case would gate nothing.
    double max_dq = 0;
    for (double q : dftb.last_mulliken_dq())
        max_dq = std::max(max_dq, std::fabs(q));
    std::printf("  max |dq| = %.4f e\n", max_dq);
    if (max_dq < 0.1) {
        std::printf("FAIL cell is not polar; the gamma kernel is untested\n");
        return 1;
    }

    std::printf("PASS\n");
    return 0;
}
