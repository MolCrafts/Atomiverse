/// @file cpu/planewave/internal/pw_stress.cpp
/// @brief Plane-wave cell stress: Nielsen–Martin one-SCF analytic path +
///        multi-SCF strain-FD oracle.
///
/// Analytic path (production `pw_stress`):
///   1. One full SCF at the reference geometry.
///   2. Hellmann–Feynman cell stress by central finite differences of the
///      *non-self-consistent* total energy under symmetric strain: density is
///      affinely remapped (fixed fractional coordinates, ρ → ρ·Ω₀/Ω), V_H /
///      V_xc / V_loc / Ewald rebuilt, one Davidson diagonalization per
///      strained cell, then iteration_energy. This is the numerical form of
///      the Nielsen–Martin HF theorem (Nielsen & Martin, PRB 32, 3780 (1985))
///      and matches multi-SCF strain-FD when the residual is tight, at
///      ~1 SCF + 12 diagonalizations instead of 12 SCFs.
///
/// Oracle path (`pw_stress_fd`): full multi-SCF strain-FD of E_tot (12 SCFs).
///
/// Sign convention (productization 2026-07-25): **mechanical / CP2K**
/// \(\sigma = -(1/\Omega)\,\partial E/\partial\varepsilon\), matching
/// GPW `gpw_stress` and `AnisotropicCellRelax` / `IsotropicCellRelax`
/// (positive diagonal stress when the cell is compressed → expand).

#include "atomiverse/cpu/planewave/internal/pw_stress.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/log.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/planewave/internal/davidson.h"
#include "atomiverse/cpu/planewave/internal/density_builder.h"
#include "atomiverse/cpu/planewave/internal/fft.h"
#include "atomiverse/cpu/planewave/internal/poisson.h"
#include "atomiverse/cpu/planewave/internal/pw_setup.h"
#include "atomiverse/cpu/planewave/internal/pw_workspace.h"
#include "atomiverse/cpu/planewave/internal/restricted_scf.h"
#include "atomiverse/cpu/planewave/internal/scf_loop.h"
#include "atomiverse/cpu/planewave/internal/veff_builder.h"
#include "atomiverse/cpu/planewave/internal/xc_builder.h"
#include "atomiverse/cpu/planewave/qe_units.h"
#include "atomiverse/cpu/xc/xc_functional.h"
#include "atomiverse/electronic/smearing.h"

#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {
namespace {

#ifdef ATV_HAS_MPI
MPI_Datatype mpi_real_type() {
    return sizeof(AccReal) == sizeof(double) ? MPI_DOUBLE : MPI_FLOAT;
}
#endif

struct MpiLayout {
    int rank = 0;
    int nprocs = 1;
    bool active = false;
};

MpiLayout mpi_layout() {
    MpiLayout layout;
#ifdef ATV_HAS_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized) {
        MPI_Comm_rank(MPI_COMM_WORLD, &layout.rank);
        MPI_Comm_size(MPI_COMM_WORLD, &layout.nprocs);
        layout.active = (layout.nprocs > 1);
    }
#endif
    return layout;
}

void voigt_pair(int v, int& a, int& b) {
    static const int A[6] = {0, 1, 2, 0, 0, 1};
    static const int B[6] = {0, 1, 2, 1, 2, 2};
    a = A[v];
    b = B[v];
}

void apply_strain(cpu::Frame& frame, const Mat3& strain) {
    Mat3 F{};
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            F(a, b) = (a == b ? AccReal(1) : AccReal(0)) + strain(a, b);

    Mat3 cell = frame.box.cell();
    Mat3 cell_new;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            AccReal s = 0;
            for (int k = 0; k < 3; ++k)
                s += F(i, k) * cell(k, j);
            cell_new(i, j) = s;
        }
    frame.box.set_cell(cell_new);

    auto x = frame.span(kAtom_x);
    auto y = frame.span(kAtom_y);
    auto z = frame.span(kAtom_z);
    const int n = frame.nlocal();
    for (int a = 0; a < n; ++a) {
        AccReal rx = x[a], ry = y[a], rz = z[a];
        x[a] = F(0, 0) * rx + F(0, 1) * ry + F(0, 2) * rz;
        y[a] = F(1, 0) * rx + F(1, 1) * ry + F(1, 2) * rz;
        z[a] = F(2, 0) * rx + F(2, 1) * ry + F(2, 2) * rz;
    }
}

double scf_energy_ry(const cpu::Frame& frame, const PW& method) {
    PW solver = copy_pw_params(method);
    solver.build(frame);
    return solver.run(frame).energy;
}

// Fourier coefficients are naturally indexed by fractional lattice modes.
// A homogeneous strain changes their Cartesian metric and volume, not those
// labels. Keep them directly, avoiding FFT-shape interpolation or a uniform
// fallback that discards the inhomogeneous density.
using FrozenDensity = std::span<const std::complex<AccReal>>;

void load_rho_scaled(PWWorkspace& ws, const FrozenDensity& rho0,
                     AccReal scale) {
    if (static_cast<int>(rho0.size()) != ws.n_g)
        throw std::logic_error(
            "PW frozen density requires the reference integer basis");
    for (int ig = 0; ig < ws.n_g; ++ig)
        ws.rho_G[ig] = rho0[static_cast<std::size_t>(ig)] * scale;
}

void load_rho_spin_scaled(PWWorkspace& ws, const FrozenDensity& rho_a0,
                          const FrozenDensity& rho_b0, AccReal scale) {
    const auto ng = static_cast<std::size_t>(ws.n_g);
    if (rho_a0.size() != ng || rho_b0.size() != ng)
        throw std::logic_error(
            "PW frozen spin density requires the reference integer basis");
    ws.nspin = 2;
    ws.rho_spin_G.resize(2);
    ws.rho_spin_G_old.resize(2);
    ws.V_xc_spin_G.resize(2);
    for (int spin = 0; spin < 2; ++spin) {
        ws.rho_spin_G[spin].resize({ng});
        ws.rho_spin_G_old[spin].resize({ng});
        ws.V_xc_spin_G[spin].resize({ng});
    }
    for (std::size_t ig = 0; ig < ng; ++ig) {
        ws.rho_spin_G[0][ig] = rho_a0[ig] * scale;
        ws.rho_spin_G[1][ig] = rho_b0[ig] * scale;
        ws.rho_G[ig] = ws.rho_spin_G[0][ig] + ws.rho_spin_G[1][ig];
    }
}

/// Non-self-consistent collinear spin total / free energy [Ry] under
/// remapped ρ_σ. Metallic paths fill both spin spectra with one shared μ.
double nscf_energy_spin_ry(const cpu::Frame& frame, const PW& method,
                           PWStrainReference reference,
                           const FrozenDensity& rho_a0,
                           const FrozenDensity& rho_b0, AccReal omega0_bohr,
                           PoissonSolver& poisson, XCBuilder& xc,
                           VeffBuilder& veff, DavidsonSolver& davidson,
                           int n_bands, int n_alpha, int n_beta) {
    PWSetup prob;
    PWWorkspace ws;
    setup_pw_problem_fixed_basis(frame, reference, method.ecut_ry,
                                 method.k_mesh, prob, ws);
    ws.mix_beta = AccReal(method.mix_beta);
    ws.pulay_depth = method.pulay_depth;

    xc.prepare(ws);
    veff.prepare(ws);
    davidson.prepare(ws);

    AccReal scale = omega0_bohr / ws.omega;
    load_rho_spin_scaled(ws, rho_a0, rho_b0, scale);

    poisson.step(ws);
    xc.step_spin(ws, method.xc);

    const auto layout = mpi_layout();
    const int n_kpts = static_cast<int>(prob.mesh.kg.kpoints.size());
    const int n_occ[2] = {n_alpha, n_beta};
    const bool use_smear = method.smearing != ATV::electronic::Smearing::none;
    AccReal e_band = 0;
    AccReal entropy_sum = 0;
    // Spectra are indexed by spin, k point, then band. Each rank owns
    // disjoint k points; assemble the complete spectrum before finding μ.
    std::vector<AccReal> spectrum(
        use_smear ? static_cast<std::size_t>(2 * n_kpts * n_bands) : 0,
        AccReal(0));
    const auto saved_point = ws.point();
    auto& gv0 = prob.gv0;
    const int ng0 = gv0.count();

    for (int s = 0; s < 2; ++s) {
        if (!use_smear && n_occ[s] <= 0)
            continue;
        for (int ig = 0; ig < ng0; ++ig)
            ws.V_xc_G[ig] = ws.V_xc_spin_G[s][ig];
        veff.step(ws);

        for (int ik = layout.rank; ik < n_kpts; ik += layout.nprocs) {
            ws.bind(prob.mesh.point(static_cast<std::size_t>(ik)));

            davidson.step(ws);

            if (use_smear) {
                if (static_cast<int>(ws.eigenvalues.size()) < n_bands)
                    throw std::runtime_error(
                        "pw_stress: incomplete spin-smearing spectrum");
                for (int b = 0; b < n_bands; ++b)
                    spectrum[static_cast<std::size_t>(
                        (s * n_kpts + ik) * n_bands + b)] = ws.eigenvalues[b];
            } else {
                const AccReal weight =
                    prob.mesh.kg.weights[static_cast<std::size_t>(ik)];
                for (int b = 0; b < n_occ[s]; ++b)
                    e_band += weight * ws.eigenvalues[b];
            }

            ws.bind(saved_point);
        }
    }

#ifdef ATV_HAS_MPI
    if (layout.active && use_smear)
        MPI_Allreduce(MPI_IN_PLACE, spectrum.data(),
                      static_cast<int>(spectrum.size()), mpi_real_type(),
                      MPI_SUM, MPI_COMM_WORLD);
    else if (layout.active)
        MPI_Allreduce(MPI_IN_PLACE, &e_band, 1, mpi_real_type(), MPI_SUM,
                      MPI_COMM_WORLD);
#endif

    if (use_smear) {
        ATV::electronic::Smearing kind = ATV::electronic::Smearing::fermi_dirac;
        if (method.smearing == ATV::electronic::Smearing::gaussian)
            kind = ATV::electronic::Smearing::gaussian;
        else if (method.smearing
                 == ATV::electronic::Smearing::methfessel_paxton)
            kind = ATV::electronic::Smearing::methfessel_paxton;
        const AccReal width =
            AccReal(method.degauss) * AccReal(qe_units::rydberg_to_hartree);
        std::vector<AccReal> weights(spectrum.size());
        for (int s = 0; s < 2; ++s)
            for (int ik = 0; ik < n_kpts; ++ik)
                for (int b = 0; b < n_bands; ++b)
                    weights[static_cast<std::size_t>((s * n_kpts + ik) * n_bands
                                                     + b)] =
                        prob.mesh.kg.weights[static_cast<std::size_t>(ik)];
        const AccReal mu =
            find_fermi_level(spectrum, weights, AccReal(prob.ions.n_elec),
                             width, kind, method.mp_order, AccReal(1));
        const auto fill = fill_occupations(spectrum, weights, mu, width, kind,
                                           method.mp_order, AccReal(1));
        // All ranks now have the same full spectrum and free-energy sum.
        // A second allreduce here would multiply it by the rank count.
        for (std::size_t i = 0; i < spectrum.size(); ++i)
            e_band += weights[i] * spectrum[i] * fill.occ[i];
        entropy_sum = fill.entropy;
    }

    ws.bind({gv0});
    auto terms =
        iteration_energy(ws, e_band, prob.ions.ew_total, prob.ions.vol_bohr);
    // Collinear: replace Parseval ∫ρ V_xc (last channel) with e_xc_int_spin.
    terms.e_one = e_band - AccReal(2.0) * terms.e_h - ws.e_xc_int_spin;
    terms.total = terms.e_one + terms.e_h + ws.E_xc + prob.ions.ew_total;
    if (use_smear)
        terms.total -= AccReal(method.degauss)
                       * AccReal(qe_units::rydberg_to_hartree) * entropy_sum;
    return static_cast<double>(terms.total) / qe_units::rydberg_to_hartree;
}

/// Non-self-consistent free / total energy [Ry] at `frame` with remapped
/// density from the reference SCF (`rho0` at volume `omega0_bohr`).
///
/// Fixed occupations: band sum with f=2. Restricted metallic smear: shared
/// μ over the spectrum, fractional f_b∈[0,2], Mermin free energy
/// F = E_KS − σ·S (σ = degauss in Ha).
double nscf_energy_ry(const cpu::Frame& frame, const PW& method,
                      PWStrainReference reference, const FrozenDensity& rho0,
                      AccReal omega0_bohr, PoissonSolver& poisson,
                      XCBuilder& xc, VeffBuilder& veff,
                      DavidsonSolver& davidson, int n_bands) {
    PWSetup prob;
    PWWorkspace ws;
    setup_pw_problem_fixed_basis(frame, reference, method.ecut_ry,
                                 method.k_mesh, prob, ws);
    ws.mix_beta = AccReal(method.mix_beta);
    ws.pulay_depth = method.pulay_depth;

    xc.prepare(ws);
    veff.prepare(ws);
    davidson.prepare(ws);

    // Remap density: ρ'(r) = ρ₀(r) · (Ω₀ / Ω) so ∫ρ is conserved under
    // volume change (fixed fractional ionic coordinates).
    AccReal scale = omega0_bohr / ws.omega;
    load_rho_scaled(ws, rho0, scale);

    poisson.step(ws);
    xc.step(ws, method.xc);
    veff.step(ws);

    const auto layout = mpi_layout();
    const int n_kpts = static_cast<int>(prob.mesh.kg.kpoints.size());
    const bool use_smear = (method.smearing != ATV::electronic::Smearing::none);
    AccReal e_band = 0;
    AccReal entropy_sum = AccReal(0);
    const auto saved_point = ws.point();
    auto& gv0 = prob.gv0;

    if (use_smear) {
        ATV::electronic::Smearing smear_kind =
            ATV::electronic::Smearing::fermi_dirac;
        if (method.smearing == ATV::electronic::Smearing::methfessel_paxton)
            smear_kind = ATV::electronic::Smearing::methfessel_paxton;
        else if (method.smearing == ATV::electronic::Smearing::gaussian)
            smear_kind = ATV::electronic::Smearing::gaussian;
        const AccReal degauss_ha =
            static_cast<AccReal>(method.degauss) * AccReal(0.5);
        const int mp_order = method.mp_order;

        std::vector<std::vector<AccReal>> evals_k(
            static_cast<std::size_t>(n_kpts));
        for (int ik = layout.rank; ik < n_kpts; ik += layout.nprocs) {
            ws.bind(prob.mesh.point(static_cast<std::size_t>(ik)));

            davidson.step(ws);

            evals_k[static_cast<std::size_t>(ik)].assign(
                ws.eigenvalues.begin(),
                ws.eigenvalues.begin()
                    + std::min(n_bands,
                               static_cast<int>(ws.eigenvalues.size())));
            if (static_cast<int>(evals_k[static_cast<std::size_t>(ik)].size())
                > n_bands)
                evals_k[static_cast<std::size_t>(ik)].resize(
                    static_cast<std::size_t>(n_bands));

            ws.bind(saved_point);
        }

#ifdef ATV_HAS_MPI
        if (layout.active) {
            std::vector<AccReal> pack(
                static_cast<std::size_t>(n_kpts * n_bands), AccReal(0));
            for (int ik = layout.rank; ik < n_kpts; ik += layout.nprocs) {
                const auto& ev = evals_k[static_cast<std::size_t>(ik)];
                for (int b = 0; b < n_bands && b < static_cast<int>(ev.size());
                     ++b)
                    pack[static_cast<std::size_t>(ik * n_bands + b)] =
                        ev[static_cast<std::size_t>(b)];
            }
            MPI_Allreduce(MPI_IN_PLACE, pack.data(), n_kpts * n_bands,
                          mpi_real_type(), MPI_SUM, MPI_COMM_WORLD);
            for (int ik = 0; ik < n_kpts; ++ik) {
                evals_k[static_cast<std::size_t>(ik)].resize(
                    static_cast<std::size_t>(n_bands));
                for (int b = 0; b < n_bands; ++b)
                    evals_k[static_cast<std::size_t>(ik)][b] =
                        pack[static_cast<std::size_t>(ik * n_bands + b)];
            }
        }
#endif

        std::vector<AccReal> all_evals;
        std::vector<AccReal> all_weights;
        all_evals.reserve(static_cast<std::size_t>(n_kpts * n_bands));
        all_weights.reserve(static_cast<std::size_t>(n_kpts * n_bands));
        for (int ik = 0; ik < n_kpts; ++ik) {
            const AccReal w =
                prob.mesh.kg.weights[static_cast<std::size_t>(ik)];
            const auto& ev = evals_k[static_cast<std::size_t>(ik)];
            for (int b = 0; b < n_bands && b < static_cast<int>(ev.size());
                 ++b) {
                all_evals.push_back(ev[static_cast<std::size_t>(b)]);
                all_weights.push_back(w);
            }
        }
        const AccReal mu_ha =
            find_fermi_level(all_evals, all_weights, AccReal(prob.ions.n_elec),
                             degauss_ha, smear_kind, mp_order);

        for (int ik = 0; ik < n_kpts; ++ik) {
            const AccReal w =
                prob.mesh.kg.weights[static_cast<std::size_t>(ik)];
            const auto& ev = evals_k[static_cast<std::size_t>(ik)];
            std::vector<AccReal> w_one(ev.size(), w);
            auto fill = fill_occupations(ev, w_one, mu_ha, degauss_ha,
                                         smear_kind, mp_order);
            entropy_sum += fill.entropy;
            for (std::size_t b = 0; b < ev.size(); ++b) {
                const AccReal f_b = smear_f_band(ev[b], mu_ha, degauss_ha,
                                                 smear_kind, mp_order);
                e_band += w * ev[b] * f_b;
            }
        }
    } else {
        for (int ik = layout.rank; ik < n_kpts; ik += layout.nprocs) {
            ws.bind(prob.mesh.point(static_cast<std::size_t>(ik)));

            davidson.step(ws);

            const AccReal weight =
                prob.mesh.kg.weights[static_cast<std::size_t>(ik)];
            for (int b = 0; b < n_bands; ++b)
                e_band += weight * ws.eigenvalues[b] * AccReal(2);

            ws.bind(saved_point);
        }

#ifdef ATV_HAS_MPI
        if (layout.active) {
            MPI_Allreduce(MPI_IN_PLACE, &e_band, 1, mpi_real_type(), MPI_SUM,
                          MPI_COMM_WORLD);
        }
#endif
    }

    // Restore Γ sphere for energy Parseval (ρ, V_H, V_xc, V_loc live there).
    ws.bind({gv0});
    const auto terms =
        iteration_energy(ws, e_band, prob.ions.ew_total, prob.ions.vol_bohr);
    AccReal e_ha = terms.total;
    if (use_smear) {
        const AccReal degauss_ha =
            static_cast<AccReal>(method.degauss) * AccReal(0.5);
        e_ha += -degauss_ha * entropy_sum;
    }
    return static_cast<double>(e_ha) / qe_units::rydberg_to_hartree;
}

} // namespace

std::array<double, 6> pw_stress_fd_impl(const cpu::Frame& frame,
                                        const PW& method, double h) {
    if (frame.nlocal() <= 0)
        throw std::invalid_argument("pw_stress_fd: frame has no atoms");
    if (frame.box.volume() <= 0)
        throw std::invalid_argument("pw_stress_fd: non-positive cell volume");
    if (!(h > 0.0))
        throw std::invalid_argument("pw_stress_fd: h must be > 0");

    const double vol = static_cast<double>(frame.box.volume());
    const double ry_to_ev = qe_units::rydberg_to_ev;

    std::array<double, 6> stress{};
    for (int v = 0; v < 6; ++v) {
        int a = 0, b = 0;
        voigt_pair(v, a, b);
        Mat3 ep = Mat3{};
        Mat3 em = Mat3{};
        if (a == b) {
            ep(a, a) = h;
            em(a, a) = -h;
        } else {
            ep(a, b) = ep(b, a) = h;
            em(a, b) = em(b, a) = -h;
        }
        cpu::Frame fp = frame;
        cpu::Frame fm = frame;
        apply_strain(fp, ep);
        apply_strain(fm, em);
        const double ep_ry = scf_energy_ry(fp, method);
        const double em_ry = scf_energy_ry(fm, method);
        const double dE_dev = (ep_ry - em_ry) * ry_to_ev / (2.0 * h);
        // Mechanical / CP2K: σ = −(1/Ω) ∂E/∂ε (matches GPW cell relax).
        // Symmetric shear varies ε_ab and ε_ba together: its energy
        // derivative is 2*∂E/∂ε_ab. Return the physical tensor component.
        stress[static_cast<std::size_t>(v)] =
            -dE_dev / (vol * (a == b ? 1.0 : 2.0));
    }
    return stress;
}

std::array<double, 6> pw_stress_from_density(const cpu::Frame& frame,
                                             const PW& method,
                                             PWStrainReference reference,
                                             PWDensityView density,
                                             const PWResult& result) {
    if (frame.nlocal() <= 0)
        throw std::invalid_argument("pw_stress: frame has no atoms");
    if (frame.box.volume() <= 0)
        throw std::invalid_argument("pw_stress: non-positive cell volume");

    const bool use_smear = (method.smearing != ATV::electronic::Smearing::none);
    if (use_smear && !(method.degauss > 0.0))
        throw std::invalid_argument(
            "pw_stress: degauss must be > 0 [Ry] when smearing is enabled");
    const int n_elec = reference.ions.n_elec;

    int n_alpha = 0;
    int n_beta = 0;
    int n_bands = std::max(n_elec / 2, 1);
    if (method.spin_polarized) {
        if (method.multiplicity < 1)
            throw std::invalid_argument(
                "pw_stress: multiplicity must be >= 1 when spin_polarized");
        const int mag = method.multiplicity - 1;
        if ((n_elec + mag) % 2 != 0 || mag > n_elec)
            throw std::invalid_argument(
                "pw_stress: n_elec and multiplicity have incompatible parity");
        n_alpha = (n_elec + mag) / 2;
        n_beta = (n_elec - mag) / 2;
        n_bands = std::max(std::max(n_alpha, n_beta), 1);
        if (use_smear)
            n_bands = std::max(n_bands, std::max((n_elec + 1) / 2, 1)
                                            + std::max(method.extra_bands, 0));
    } else if (use_smear) {
        const int n_occ_ceil = std::max((n_elec + 1) / 2, 1);
        n_bands = n_occ_ceil + std::max(method.extra_bands, 0);
    } else {
        require_closed_shell(n_elec, "pw_stress");
    }

    PoissonSolver poisson;
    XCBuilder xc;
    VeffBuilder veff;
    DavidsonSolver davidson{n_bands, AccReal(method.tol) * AccReal(0.1), 40};
    if (!result.converged)
        ATV_WARN("[pw] pw_stress_analytic: SCF not converged after {} iters; "
                 "using last density for HF stress",
                 result.iterations);

    const AccReal omega0 = reference.ions.vol_bohr;
    FrozenDensity rho_a0, rho_b0;
    if (method.spin_polarized) {
        rho_a0 = {density.spin[0].data(), density.spin[0].size()};
        rho_b0 = {density.spin[1].data(), density.spin[1].size()};
    }
    const auto rho0 = density.total;

    // Hellmann–Feynman strain-FD of the NSCF free / total energy (one SCF).
    const double h = 2e-3;
    const double vol = static_cast<double>(frame.box.volume());
    const double ry_to_ev = qe_units::rydberg_to_ev;

    std::array<double, 6> stress{};
    for (int v = 0; v < 6; ++v) {
        int a = 0, b = 0;
        voigt_pair(v, a, b);
        Mat3 ep = Mat3{};
        Mat3 em = Mat3{};
        if (a == b) {
            ep(a, a) = h;
            em(a, a) = -h;
        } else {
            ep(a, b) = ep(b, a) = h;
            em(a, b) = em(b, a) = -h;
        }
        cpu::Frame fp = frame;
        cpu::Frame fm = frame;
        apply_strain(fp, ep);
        apply_strain(fm, em);

        double ep_ry = 0.0;
        double em_ry = 0.0;
        if (method.spin_polarized) {
            ep_ry = nscf_energy_spin_ry(fp, method, reference, rho_a0, rho_b0,
                                        omega0, poisson, xc, veff, davidson,
                                        n_bands, n_alpha, n_beta);
            em_ry = nscf_energy_spin_ry(fm, method, reference, rho_a0, rho_b0,
                                        omega0, poisson, xc, veff, davidson,
                                        n_bands, n_alpha, n_beta);
        } else {
            ep_ry = nscf_energy_ry(fp, method, reference, rho0, omega0, poisson,
                                   xc, veff, davidson, n_bands);
            em_ry = nscf_energy_ry(fm, method, reference, rho0, omega0, poisson,
                                   xc, veff, davidson, n_bands);
        }
        const double dE_dev = (ep_ry - em_ry) * ry_to_ev / (2.0 * h);
        // Mechanical / CP2K: σ = −(1/Ω) ∂E/∂ε (matches GPW cell relax).
        // Under smear E is the Mermin free energy F.
        stress[static_cast<std::size_t>(v)] =
            -dE_dev / (vol * (a == b ? 1.0 : 2.0));
    }
    return stress;
}

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
