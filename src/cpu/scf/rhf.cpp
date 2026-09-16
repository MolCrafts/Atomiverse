/// @file cpu/scf/rhf.cpp
/// @brief RHF constructor defaults + occupation resolution + Seeger-Pople
///        stability analysis. The shared driver loop lives in
///        mean_field_method.cpp (explicitly instantiated there).

#include "atomiverse/cpu/scf/rhf.h"

#include <cmath>
#include <stdexcept>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/scf/internal/cphf_pack.h"
#include "atomiverse/cpu/scf/internal/integral_derivatives.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/internal/one_electron.h"
#include "atomiverse/cpu/scf/internal/orbital_hessian.h"
#include "atomiverse/cpu/scf/internal/stability.h"

namespace ATV {
namespace cpu {
namespace scf {

namespace {
/// Numerical-zero guard on the stability eigenvalues [Hartree]: a lowest
/// eigenvalue above −kStabilityZeroTol counts as stable (guards round-off
/// on genuinely zero modes; physical instabilities are orders larger).
constexpr double kStabilityZeroTol = 1e-8;

/// Rotation amplitude [rad] of the leading component of the external
/// instability mode — matches the `apply_homo_lumo_mixing` default (20°)
/// so the rotated guess bridges the existing broken-symmetry machinery.
constexpr double kRotationTheta = 0.3490658503988659;
} // namespace

RHF::RHF() {
    spin_ = SpinType::Restricted;
    method_ = MeanFieldType::HF;
    init_guess = InitGuess::MINAO;
    damping_factor = 0.0;
    tol = 1e-9;
}

RHF::RHF(BasisSet basis_in) : RHF() { basis = basis_in; }

RHF::RHF(BasisSet basis_in, int charge_in) : RHF(basis_in) {
    charge = charge_in;
}

RHF::RHF(BasisSet basis_in, int charge_in, double tol_in, int max_iter_in)
    : RHF(basis_in, charge_in) {
    tol = tol_in;
    max_iter = max_iter_in;
}
void RHF::resolve_occupation(int n_electrons) {
    n_alpha_ = n_electrons / 2;
    n_beta_ = n_electrons / 2;
}

StabilityResult RHF::stability() const {
    if (!has_result())
        throw std::runtime_error(
            "RHF::stability: no SCF result — call run(frame) first");

    const SlaterDeterminant& det = determinant();
    const internal::IntegralView pack = integral_view_();

    // Dense working-dimension ERI: reuse the materialized tensor when the
    // run kept one (compute_eri), otherwise rebuild it from the private
    // shell workspace (Cartesian eri_dense + spherical sandwich).
    const xt::xtensor<double, 4> eri =
        (pack.ERI.size() > 0) ? pack.ERI : pack.eri_working();

    const internal::OrbitalHessian hessian{det, eri};
    const internal::StabilityAnalysis analysis{hessian};

    StabilityResult out;
    out.internal_eigenvalue = analysis.lowest(internal::Channel::singlet);
    out.external_eigenvalue = analysis.lowest(internal::Channel::triplet);
    out.internal_stable = out.internal_eigenvalue >= -kStabilityZeroTol;
    out.external_stable = out.external_eigenvalue >= -kStabilityZeroTol;

    const RealMatrix& C = det.coefficients(SlaterDeterminant::Spin::alpha);
    out.rotated_C = C;
    if (out.external_stable)
        return out;

    // Rotate the coefficients along the external instability mode: build the
    // antisymmetric occ-virt generator K from the mode amplitudes X_ai
    // (scaled so the leading amplitude rotates by kRotationTheta), then
    // C_rot = C · exp(K). Same occupied/virtual split as OrbitalHessian
    // (integer occupations, f = 1 ⇔ occupied).
    const RealVector mode = analysis.lowest_vector(internal::Channel::triplet);
    const RealVector& f = det.occupations(SlaterDeterminant::Spin::alpha);
    const int n_mo = static_cast<int>(C.shape()[1]);
    std::vector<int> occ, virt;
    for (int p = 0; p < n_mo; ++p)
        (f(p) > 0.0 ? occ : virt).push_back(p);
    const int no = static_cast<int>(occ.size());
    const int nv = static_cast<int>(virt.size());

    double x_max = 0.0;
    for (std::size_t k = 0; k < mode.size(); ++k)
        x_max = std::max(x_max, std::abs(mode(k)));
    const double scale = kRotationTheta / x_max;

    RealMatrix K = xt::zeros<double>(
        {static_cast<std::size_t>(n_mo), static_cast<std::size_t>(n_mo)});
    for (int a = 0; a < nv; ++a)
        for (int i = 0; i < no; ++i) {
            const double k_ai = scale * mode(a * no + i);
            K(virt[static_cast<std::size_t>(a)],
              occ[static_cast<std::size_t>(i)]) = k_ai;
            K(occ[static_cast<std::size_t>(i)],
              virt[static_cast<std::size_t>(a)]) = -k_ai;
        }

    // exp(K) by plain Taylor series: ‖K‖ ≤ kRotationTheta < 1, so the series
    // converges to machine precision in a handful of terms, and the exact
    // exponential of an antisymmetric K is orthogonal.
    RealMatrix U = xt::eye<double>(static_cast<std::size_t>(n_mo));
    RealMatrix term = U;
    for (int k = 1; k <= 40; ++k) {
        term = xt::linalg::dot(term, K) / static_cast<double>(k);
        U = U + term;
        double t_max = 0.0;
        for (int r = 0; r < n_mo; ++r)
            for (int c = 0; c < n_mo; ++c)
                t_max = std::max(t_max, std::abs(term(r, c)));
        if (t_max < 1e-16)
            break;
    }
    out.rotated_C = xt::linalg::dot(C, U);
    return out;
}

vib::HessianResult RHF::hessian(Frame& frame) {
    if (ri != RI::None)
        throw std::runtime_error(
            "RHF::hessian: the analytic Hessian is not wired for density "
            "fitting (RI) — derivative three-center integrals are a "
            "follow-up chain link; use ri = RI::None (no silent fallback)");

    // Fresh SCF at this geometry (writes the eV energy to frame.meta).
    run(frame);

    const SlaterDeterminant& det = determinant();
    const internal::IntegralView pack = integral_view_();
    // Shared guard: non-convergence, spherical-harmonic pack, ECP pack.
    check_hessian_preconditions("RHF::hessian", pack);

    // Dense working-dimension ERI (stability() precedent): reuse the
    // materialized tensor when the run kept one, else rebuild.
    const xt::xtensor<double, 4> eri =
        (pack.ERI.size() > 0) ? pack.ERI : pack.eri_working();

    // The shared link-08 kernel — throws std::invalid_argument when
    // n_virt = 0 (no occ→virt response exists, e.g. He/STO-3G).
    const internal::OrbitalHessian oh{det, eri};

    // Skeleton first derivatives: dS plus the derivative Fock
    // F^{(x)} = dT + dV + G^{(x)}[P].
    const IntegralDerivPack dpack{pack};
    const RealMatrix P = det.density_total();
    xt::xtensor<double, 4> dF = dpack.dT + dpack.dV;
    accumulate_two_electron_fock_derivative(dF, pack, P);

    // CPHF solve (3N right-hand sides) + Hessian assembly [Hartree/Bohr²].
    internal::CphfPack cphf{det, oh, pack, eri, dpack.dS, dF};
    cphf.solve();
    const xt::xtensor<double, 2> H_ha = cphf.hessian();

    // Frame/vib working units: eV/Å² (vib::HessianResult contract).
    const int natm = static_cast<int>(frame.nlocal());

    // ── Dipole derivatives (atomic polar tensor) for IR intensities ─────
    // ∂μ_a/∂R_{A,j} = −Tr[∂P/∂R_{A,j}·r_a] − Tr[P·∂r_a/∂R_{A,j}] + Z_A δ_{aj}
    // Reuses the CPHF density response ∂P/∂x (solved above); the dipole-
    // integral derivative ∂⟨μ|r_a|ν⟩/∂R is central-differenced (the dipole
    // matrix is a cheap 1e block — same semi-numerical pattern as the ECP
    // gradient). Cartesian-only, matching the analytic Hessian's own scope.
    // ponytail: FD of the dipole 1e block; analytic ∂⟨μ|r|ν⟩/∂R only if hot.
    xt::xtensor<double, 2> apt = xt::zeros<double>({3 * natm, 3});
    {
        const int n = pack.n_basis;
        const double origin[3] = {0.0, 0.0,
                                  0.0}; // origin-free for neutral dμ/dR
        const MultipoleIntegrals mp0{pack.shells.data(), n, origin};
        const double h = 1e-4; // Bohr
        for (int A = 0; A < natm; ++A) {
            const double* ap = pack.atoms[A].pos;
            const auto on_A = [&](const double* c) {
                return c[0] == ap[0] && c[1] == ap[1] && c[2] == ap[2];
            };
            for (int j = 0; j < 3; ++j) {
                const int x = 3 * A + j;
                const RealMatrix dP = cphf.density_response(x);
                std::vector<native::Shell> shp = pack.shells;
                std::vector<native::Shell> shm = pack.shells;
                for (auto& s : shp)
                    if (on_A(s.center))
                        s.center[j] += h;
                for (auto& s : shm)
                    if (on_A(s.center))
                        s.center[j] -= h;
                const MultipoleIntegrals mpp{shp.data(), n, origin};
                const MultipoleIntegrals mpm{shm.data(), n, origin};
                for (int a = 0; a < 3; ++a) {
                    const xt::xtensor<double, 2> dr =
                        (mpp.dipole(a) - mpm.dipole(a)) / (2.0 * h);
                    apt(x, a) = -xt::sum(dP * mp0.dipole(a))()
                                - xt::sum(P * dr)()
                                + (a == j ? pack.atoms[A].Z : 0.0);
                }
            }
        }
    }

    const double to_ev_ang2 =
        units::kHartreeInEv / (units::kBohrInAngstrom * units::kBohrInAngstrom);
    vib::HessianResult out;
    out.n_atoms = natm;
    out.hessian = H_ha * to_ev_ang2;
    out.dipole_derivatives = std::move(apt);
    return out;
}

xt::xtensor<double, 2> RHF::polarizability() const {
    if (!has_result())
        throw std::runtime_error(
            "RHF::polarizability: no SCF result — call run(frame) first");

    const SlaterDeterminant& det = determinant();
    const internal::IntegralView pack = integral_view_();
    const xt::xtensor<double, 4> eri =
        (pack.ERI.size() > 0) ? pack.ERI : pack.eri_working();

    // Link-08 singlet (A+B) — throws std::invalid_argument when n_virt = 0.
    const internal::OrbitalHessian oh{det, eri};

    const RealMatrix& C = det.coefficients(SlaterDeterminant::Spin::alpha);
    const RealVector& f = det.occupations(SlaterDeterminant::Spin::alpha);
    const int n = static_cast<int>(C.shape()[0]);
    const int n_mo = static_cast<int>(C.shape()[1]);
    std::vector<int> occ, virt;
    for (int p = 0; p < n_mo; ++p)
        (f(p) == 1.0 ? occ : virt).push_back(p);
    const int no = static_cast<int>(occ.size());
    const int nv = static_cast<int>(virt.size());
    if (no != oh.n_occ() || nv != oh.n_virt())
        throw std::runtime_error("RHF::polarizability: occ/virt split "
                                 "disagrees with OrbitalHessian");

    RealMatrix Co = xt::zeros<double>({n, no});
    RealMatrix Cv = xt::zeros<double>({n, nv});
    for (int i = 0; i < no; ++i)
        for (int mu = 0; mu < n; ++mu)
            Co(mu, i) = C(mu, occ[static_cast<std::size_t>(i)]);
    for (int a = 0; a < nv; ++a)
        for (int mu = 0; mu < n; ++mu)
            Cv(mu, a) = C(mu, virt[static_cast<std::size_t>(a)]);

    // Length-gauge dipole about the center of nuclear charge — same origin
    // as ExternalField / MultipoleObserver (origin-free for neutral systems
    // only after response; keep the convention for charged systems).
    double origin[3] = {0.0, 0.0, 0.0};
    double zsum = 0.0;
    for (const auto& A : pack.atoms) {
        const double z = static_cast<double>(A.Z);
        zsum += z;
        origin[0] += z * A.pos[0];
        origin[1] += z * A.pos[1];
        origin[2] += z * A.pos[2];
    }
    if (zsum == 0.0)
        throw std::invalid_argument(
            "RHF::polarizability: total nuclear charge is zero — center of "
            "nuclear charge is undefined");
    origin[0] /= zsum;
    origin[1] /= zsum;
    origin[2] /= zsum;

    const MultipoleIntegrals mi{pack.shells.data(),
                                static_cast<int>(pack.shells.size()), origin};
    const int nov = nv * no;
    // B^a_ai = ⟨a|r_a|i⟩ = (C_vᵀ r_a C_o)_ai  [Bohr]; matches ExternalField
    // ∂h/∂F_a (electron charge −1 a.u. → h += F · r).
    xt::xtensor<double, 2> B = xt::zeros<double>({3, nov});
    for (int axis = 0; axis < 3; ++axis) {
        xt::xtensor<double, 2> D = mi.dipole(axis);
        if (pack.spherical()) {
            const auto& c2s = pack.c2s;
            D = xt::linalg::dot(
                xt::transpose(c2s),
                xt::xtensor<double, 2>(xt::linalg::dot(D, c2s)));
        }
        const RealMatrix r_vo =
            xt::linalg::dot(xt::transpose(Cv), xt::linalg::dot(D, Co));
        for (int a = 0; a < nv; ++a)
            for (int i = 0; i < no; ++i)
                B(axis, a * no + i) = r_vo(a, i);
    }

    // Solve ¹(A+B) U^b = −B^b for b = x,y,z (dense — nov is tiny for the
    // textbook cases that exercise this API; nuclear CPHF reuses CG for 3N).
    const RealMatrix Aplus = oh.dense_plus(internal::Channel::singlet);
    RealMatrix rhs = xt::zeros<double>({nov, 3});
    for (int b = 0; b < 3; ++b)
        for (int p = 0; p < nov; ++p)
            rhs(p, b) = -B(b, p);
    const RealMatrix U = xt::linalg::solve(Aplus, rhs);

    // α_ab = −4 Σ_ai U^b_ai ⟨a|r_a|i⟩  [a.u.].
    xt::xtensor<double, 2> alpha = xt::zeros<double>({3, 3});
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
            double acc = 0.0;
            for (int p = 0; p < nov; ++p)
                acc += U(p, b) * B(a, p);
            alpha(a, b) = -4.0 * acc;
        }
    // Exact symmetry up to solver noise — enforce it.
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < a; ++b) {
            const double s = 0.5 * (alpha(a, b) + alpha(b, a));
            alpha(a, b) = s;
            alpha(b, a) = s;
        }
    return alpha;
}

} // namespace scf
} // namespace cpu
} // namespace ATV
