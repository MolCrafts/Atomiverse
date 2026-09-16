/// @file regressions/scf/h2o_rhf_ir/reg_h2o_rhf_ir.cpp
/// @brief H2O RHF/STO-3G dipole-derivative tensor (atomic polar tensor,
///        ∂μ/∂R) vs a pinned PySCF finite-difference reference — the IR-
///        intensity foundation (hf-parity-04-ir-intensities).
///
/// Out-of-library consumer. The APT is the load-bearing physics of IR
/// intensities: I_k ∝ Σ_a (Σ_x ∂μ_a/∂R_x · L_{x,k})². The engine builds ∂μ/∂R
/// analytically via the CPHF density response ∂P/∂R (validated in
/// test_cphf.cpp) plus a semi-numerical dipole-integral derivative. This run
/// checks the full tensor against a third-party finite-difference of the SCF
/// dipole (PySCF has no analytic-dipole-derivative module installed here, so
/// the oracle is FD; that sets the ~1e-4 tolerance floor, not the engine).
///
/// Checks (REFERENCE.md, PySCF 2.14.0, conv_tol=1e-13, cart):
///   1. element-wise |APT − APT_PySCF_FD| ≤ 1e-4 au
///   2. translational invariance |Σ_A ∂μ_a/∂R_{A,j}| ≤ 1e-6 au

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"
#include "atomiverse/cpu/vibration/hessian_result.h"

int main() {
    using namespace ATV;

    // O(0,0,0.1173); H(0,±0.7572,−0.4692) Å — the shared H2O geometry.
    cpu::Frame frame(3);
    frame.span(kAtom_type)[0] = 8;
    frame.span(kAtom_z)[0] = 0.1173;
    frame.span(kAtom_type)[1] = 1;
    frame.span(kAtom_y)[1] = 0.7572;
    frame.span(kAtom_z)[1] = -0.4692;
    frame.span(kAtom_type)[2] = 1;
    frame.span(kAtom_y)[2] = -0.7572;
    frame.span(kAtom_z)[2] = -0.4692;
    frame.box = cpu::Box::cubic(20.0);

    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::STO3G;
    rhf.tol = 1e-12;
    rhf.max_iter = 200;
    const cpu::vib::HessianResult h = rhf.hessian(frame);

    if (h.dipole_derivatives.size() == 0) {
        std::printf("FAIL: no dipole derivatives produced\n");
        return 1;
    }

    // PySCF 2.14.0 FD ∂μ_a/∂R_{A,j} [au], indexed [atom A][axis j][dipole a]
    // (REFERENCE.md). Central difference of mf.dip_moment, h = 1e-3 Bohr.
    const double want[3][3][3] = {
        {{-0.61244432, 0.0, 0.0},
         {0.0, -0.05805634, 0.0},
         {0.0, 0.0, 0.03881088}},
        {{0.30622222, 0.0, 0.0},
         {0.0, 0.02902815, 0.15747119},
         {0.0, 0.21470465, -0.01940554}},
        {{0.30622222, 0.0, 0.0},
         {0.0, 0.02902815, -0.15747119},
         {0.0, -0.21470465, -0.01940554}},
    };
    const double tol = 1e-4;

    double max_err = 0.0;
    double net[3][3] = {{0}}; // [axis j][dipole a] summed over atoms
    for (int A = 0; A < 3; ++A)
        for (int j = 0; j < 3; ++j) {
            const int x = 3 * A + j;
            for (int a = 0; a < 3; ++a) {
                const double got = h.dipole_derivatives(x, a);
                net[j][a] += got;
                const double err = std::fabs(got - want[A][j][a]);
                max_err = std::max(max_err, err);
                std::printf("APT A%d j%d a%d got=% .8f want=% .8f |err|=%.2e\n",
                            A, j, a, got, want[A][j][a], err);
            }
        }
    double net_max = 0.0;
    for (int j = 0; j < 3; ++j)
        for (int a = 0; a < 3; ++a)
            net_max = std::max(net_max, std::fabs(net[j][a]));
    std::printf("max_elementwise_err=%.3e tol=%.0e  |Σ_A APT|_max=%.3e\n",
                max_err, tol, net_max);

    if (!(max_err < tol)) {
        std::printf("FAIL: APT off PySCF FD reference\n");
        return 1;
    }
    if (!(net_max < 1e-6)) {
        std::printf("FAIL: APT not translationally invariant\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
