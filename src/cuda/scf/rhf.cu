/// @file cuda/scf/rhf.cu
/// @brief cuda::scf::RHF — concrete restricted HF on the CUDA backend.
///
/// **Energy path (primary):** STO-3G with l ≤ 1 (H…O, mixed Z, e.g. H₂O):
/// integrals + SCF stay on device (`RHFFockDevice` / `launch_*_sp`) —
/// **CUDA-only HF**, no host ERI tensor. Other bases temporarily fill S /
/// Hcore / ERI on host then upload.
///
/// **Forces path (Wave 6):** STO-3G s/p (l ≤ 1, H…O, e.g. H₂O) analytic on
/// device (`launch_rhf_forces_sp`). Higher L / other bases still use host
/// `cpu::scf::RHF::forces`. Use `with_gradient(false)` for energy-only.

#ifdef ATV_HAS_HF

#include "atomiverse/cuda/scf/rhf.cuh"

#include <cuda_runtime.h>

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/transfer.h"
#include "atomiverse/cpu/scf/basis_data/accessors.h"
#include "atomiverse/cpu/scf/dense_matrix.h"
#include "atomiverse/cpu/scf/grid.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/rhf.h"
#include "atomiverse/cpu/scf/scf_enums.h"
#include "atomiverse/cpu/scf/slater_determinant.h"
#include "atomiverse/cpu/scf/xc_functional.h"
#include "atomiverse/cuda/evaluate/evaluate.cuh"
#include "atomiverse/cuda/frame/transfer.cuh"
#include "atomiverse/cuda/frame/types.cuh"
#include "atomiverse/cuda/scf/device_integrals.cuh"
#include "atomiverse/cuda/scf/internal/scf_solver.cuh"

namespace ATV {
namespace cuda {
namespace scf {
namespace {

/// @brief True when every atom is s/p STO-3G (l ≤ 1) with uniform n_prim.
///
/// Wave 4: mixed Z (H₂O) OK — molecular shell list built from the table.
/// Numbers come from `basis_data::sto3g()`, never retyped here.
bool device_sp_sto3g_ok(cpu::scf::BasisSet basis, const int* h_Z, int natm) {
    if (basis != cpu::scf::BasisSet::STO3G || natm <= 0 || h_Z == nullptr)
        return false;
    int n_prim0 = -1;
    for (int i = 0; i < natm; ++i) {
        const int Z = h_Z[i];
        if (!cpu::scf::basis_data::is_sp_sto3g(Z))
            return false;
        const int np = cpu::scf::basis_data::sp_sto3g_n_prim(Z);
        if (n_prim0 < 0)
            n_prim0 = np;
        else if (np != n_prim0)
            return false;
    }
    return true;
}

/// @brief Reconstruct a host Frame from device positions + nuclear Z.
///
/// Used for force assembly and the transitional host-integral energy path.
cpu::Frame host_frame_from_device(const cuda::Frame& dframe) {
    const int n = dframe.n_atoms();
    if (n <= 0) {
        throw std::invalid_argument(
            "cuda::scf::RHF: device frame has no atoms");
    }
    std::vector<real_t> hx(static_cast<size_t>(n));
    std::vector<real_t> hy(static_cast<size_t>(n));
    std::vector<real_t> hz(static_cast<size_t>(n));
    std::vector<int> htype(static_cast<size_t>(n));
    cudaMemcpy(hx.data(), dframe.atoms.x, sizeof(real_t) * static_cast<size_t>(n),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(hy.data(), dframe.atoms.y, sizeof(real_t) * static_cast<size_t>(n),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(hz.data(), dframe.atoms.z, sizeof(real_t) * static_cast<size_t>(n),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(htype.data(), dframe.atoms.type,
               sizeof(int) * static_cast<size_t>(n), cudaMemcpyDeviceToHost);

    cpu::Frame host(n);
    auto x = host.span(kAtom_x);
    auto y = host.span(kAtom_y);
    auto z = host.span(kAtom_z);
    auto type = host.span(kAtom_type);
    for (int i = 0; i < n; ++i) {
        x[i] = static_cast<double>(hx[static_cast<size_t>(i)]);
        y[i] = static_cast<double>(hy[static_cast<size_t>(i)]);
        z[i] = static_cast<double>(hz[static_cast<size_t>(i)]);
        type[i] = htype[static_cast<size_t>(i)];
    }
    // Isolated-molecule box placeholder (integrals do not use the cell).
    host.box = cpu::Box::cubic(100.0);
    return host;
}

/// @brief D2H MO coefficients (col-major n×n) and orbital energies [Ha].
void download_mo(const SCFWorkspace& ws, cpu::scf::RealMatrix& C,
                 cpu::scf::RealVector& epsilon) {
    const int n = ws.n;
    if (n <= 0 || !ws.C || !ws.eig) {
        throw std::runtime_error(
            "cuda::scf::RHF: empty SCF workspace after device SCF");
    }
    std::vector<prec_t> C_dev(static_cast<size_t>(n) * static_cast<size_t>(n));
    std::vector<prec_t> e_dev(static_cast<size_t>(n));
    cudaMemcpy(C_dev.data(), ws.C,
               sizeof(prec_t) * C_dev.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(e_dev.data(), ws.eig, sizeof(prec_t) * e_dev.size(),
               cudaMemcpyDeviceToHost);

    // RealMatrix is column-major (cuBLAS / LAPACK layout) — matches device C.
    C = xt::zeros<double>({n, n});
    epsilon = xt::zeros<double>({n});
    for (int j = 0; j < n; ++j) {
        epsilon(j) = static_cast<double>(e_dev[static_cast<size_t>(j)]);
        for (int i = 0; i < n; ++i) {
            C(i, j) = static_cast<double>(
                C_dev[static_cast<size_t>(i)
                      + static_cast<size_t>(j) * static_cast<size_t>(n)]);
        }
    }
}

/// @brief Restricted Aufbau occupations: f_i = 1 for i < n_occ, else 0.
cpu::scf::RealVector restricted_occupation(int n_mo, int n_occ) {
    cpu::scf::RealVector f = xt::zeros<double>({n_mo});
    for (int i = 0; i < n_occ && i < n_mo; ++i)
        f(i) = 1.0;
    return f;
}

/// @brief Overwrite Frame force columns + meta energy (standalone forces path).
void write_forces_to_frame(cpu::Frame& frame,
                           const cpu::scf::GradientResult& gr) {
    const int natm = static_cast<int>(frame.nlocal());
    AccReal* __restrict fx = frame.span(kAtom_fx);
    AccReal* __restrict fy = frame.span(kAtom_fy);
    AccReal* __restrict fz = frame.span(kAtom_fz);
    for (int A = 0; A < natm; ++A) {
        fx[A] = gr.forces(A, 0);
        fy[A] = gr.forces(A, 1);
        fz[A] = gr.forces(A, 2);
    }
    frame.meta.get(kEnergy) = gr.energy;
    frame.meta.get(kEnergyElectronic) = gr.energy;
}

/// @brief Transitional host fill of S/Hcore/ERI → device SCF (p shells).
std::unique_ptr<SCFSolver>
make_host_integral_solver(const cpu::Frame& frame, cpu::scf::BasisSet basis,
                          int charge, double tol, int max_iter,
                          double level_shift) {
    xt::xtensor<double, 2> S;
    xt::xtensor<double, 2> Hcore;
    xt::xtensor<double, 4> ERI;
    double V_nuc = 0.0;
    int n_basis = 0;
    int n_electrons = 0;
    cpu::scf::internal::IntegralWorkspace::fill_rhf_matrices(
        frame, basis, charge, /*compute_eri=*/true, S, Hcore, ERI, V_nuc,
        n_basis, n_electrons);
    return std::make_unique<SCFSolver>(std::move(S), std::move(Hcore),
                                       std::move(ERI), V_nuc, n_electrons, tol,
                                       max_iter, level_shift);
}

/// @brief CUDA-only RHF: molecular s/p STO-3G → device 1e/ERI + SCF (Wave 4).
///
/// Builds a full-molecule shell list from `basis_data::sp_sto3g_shells(Z)`
/// per atom (mixed Z OK). n_AO = Σ n_cart(L) with s→1, p→3.
/// @param level_shift Virtual level shift λ [Hartree]; wired on device Fock.
std::unique_ptr<SCFSolver>
make_device_sp_sto3g_solver(const int* h_Z, int natm, int charge, double tol,
                           int max_iter, double level_shift) {
    int n_electrons = -charge;
    for (int i = 0; i < natm; ++i)
        n_electrons += h_Z[i];
    if (n_electrons < 0 || n_electrons % 2 != 0) {
        throw std::invalid_argument(
            "cuda::scf::RHF: closed-shell CUDA path requires even non-negative "
            "electron count (n_electrons="
            + std::to_string(n_electrons) + ", charge=" + std::to_string(charge)
            + ")");
    }
    const int n_occ = n_electrons / 2;

    // Count shells / AOs, then fill molecular SoA from the basis table.
    int n_shells = 0;
    int n_ao = 0;
    int n_prim = -1;
    for (int a = 0; a < natm; ++a) {
        const auto& shells = cpu::scf::basis_data::sp_sto3g_shells(h_Z[a]);
        if (n_prim < 0)
            n_prim = cpu::scf::basis_data::sp_sto3g_n_prim(h_Z[a]);
        for (const auto& sh : shells) {
            ++n_shells;
            n_ao += cpu::scf::basis_data::n_cart_l(sh.l);
        }
    }

    std::vector<double> alpha_mol(
        static_cast<std::size_t>(n_shells) * static_cast<std::size_t>(n_prim));
    std::vector<double> coeff_mol(alpha_mol.size());
    std::vector<int> L_mol(static_cast<std::size_t>(n_shells));
    std::vector<int> atom_of(static_cast<std::size_t>(n_shells));

    int s = 0;
    for (int a = 0; a < natm; ++a) {
        const auto& shells = cpu::scf::basis_data::sp_sto3g_shells(h_Z[a]);
        for (const auto& sh : shells) {
            L_mol[static_cast<std::size_t>(s)] = sh.l;
            atom_of[static_cast<std::size_t>(s)] = a;
            for (int p = 0; p < n_prim; ++p) {
                const std::size_t k =
                    static_cast<std::size_t>(s * n_prim + p);
                alpha_mol[k] = sh.alpha[static_cast<std::size_t>(p)];
                coeff_mol[k] = sh.coeff[static_cast<std::size_t>(p)];
            }
            ++s;
        }
    }

    return std::make_unique<SCFSolver>(
        n_shells, n_ao, n_prim, alpha_mol.data(), coeff_mol.data(),
        L_mol.data(), atom_of.data(), n_occ, tol, max_iter, level_shift);
}

} // namespace

RHF::RHF(cpu::scf::BasisSet basis_in, int charge_in, double tol_in,
         int max_iter_in, double level_shift_in, bool compute_force_in)
    : basis(basis_in), charge(charge_in), tol(tol_in), max_iter(max_iter_in),
      level_shift(level_shift_in), compute_force(compute_force_in) {}
RHF::~RHF() = default;
RHF::RHF(RHF&& other) noexcept
    : basis(other.basis), charge(other.charge), tol(other.tol),
      max_iter(other.max_iter), level_shift(other.level_shift),
      compute_force(other.compute_force), solver_(std::move(other.solver_)),
      energy_ha_(other.energy_ha_), converged_(other.converged_),
      iterations_(other.iterations_), built_(other.built_),
      last_force_device_(other.last_force_device_),
      last_device_sp_(other.last_device_sp_),
      host_Z_(std::move(other.host_Z_)) {
    other.energy_ha_ = 0.0;
    other.converged_ = false;
    other.iterations_ = 0;
    other.built_ = false;
    other.last_force_device_ = false;
    other.last_device_sp_ = false;
}

void RHF::cache_host_types(const cpu::Frame& frame) {
    const int n = static_cast<int>(frame.nlocal());
    if (n <= 0) {
        host_Z_.clear();
        return;
    }
    host_Z_.resize(static_cast<std::size_t>(n));
    const auto type = frame.span(kAtom_type);
    for (int i = 0; i < n; ++i)
        host_Z_[static_cast<std::size_t>(i)] = type[i];
}

void RHF::run(cpu::Frame& frame) {
    // Path select + shell pack use host Z before H2D (no setup type[] D2H).
    cache_host_types(frame);
    // Evaluate owns H2D / component lifecycle / D2H so Frame stays the sole
    // public I/O token and streams stay inside the Component path.
    cuda::Evaluate eval;
    eval.on(ATV::evaluate::Phase::evaluate).add_ref(*this);
    eval.build();
    eval.run(frame);
}

cpu::scf::GradientResult RHF::forces(cpu::Frame& frame) {
    // Wave 6: s/p device path via run()+launch; higher L falls back to host
    // inside launch(). Always enable force assembly for this call.
    // host_Z_ seeded in run() via cache_host_types.
    const bool prev = compute_force;
    compute_force = true;
    run(frame);
    compute_force = prev;

    const int natm = static_cast<int>(frame.nlocal());
    xt::xtensor<double, 2> f = xt::zeros<double>(
        std::array<std::size_t, 2>{static_cast<std::size_t>(natm), 3});
    auto fx = frame.span(kAtom_fx);
    auto fy = frame.span(kAtom_fy);
    auto fz = frame.span(kAtom_fz);
    for (int A = 0; A < natm; ++A) {
        f(A, 0) = static_cast<double>(fx[A]);
        f(A, 1) = static_cast<double>(fy[A]);
        f(A, 2) = static_cast<double>(fz[A]);
    }
    const double e_ev = static_cast<double>(frame.meta.get(kEnergy));
    return cpu::scf::GradientResult{std::move(f), e_ev};
}

const SCFWorkspace& RHF::workspace() const {
    if (!solver_) {
        throw std::runtime_error(
            "cuda::scf::RHF::workspace: call launch()/run() first");
    }
    return solver_->workspace();
}

void RHF::build(const BuildContext& ctx) {
    // Readiness only — do not freeze build-time nuclei into SCFSolver/RHFFock.
    // Multi-step cuda::md::MD moves positions between force launches; the
    // private workspace + device Fock upload must track current geometry
    // (CPU twin: rebuild_integrals_ each nested run). Tear down any leftover
    // solver from a prior teardown/rebuild cycle.
    if (solver_) {
        solver_->destroy(ctx.runtime.stream, ctx.frame);
        solver_.reset();
    }
    built_ = true;
    converged_ = false;
    energy_ha_ = 0.0;
    iterations_ = 0;
}

void RHF::launch(const LaunchContext& ctx) {
    if (!built_) {
        throw std::runtime_error(
            "cuda::scf::RHF::launch: build() must run before launch()");
    }

    // Per-launch geometry rebuild (AIMD nuclei move between force phases).
    if (solver_) {
        solver_->destroy(ctx.runtime.stream, ctx.frame);
        solver_.reset();
    }

    const int natm = ctx.frame.n_atoms();
    if (natm <= 0) {
        throw std::invalid_argument(
            "cuda::scf::RHF::launch: device frame has no atoms");
    }
    auto* stream = static_cast<cudaStream_t>(ctx.runtime.stream);

    // Path select + shell pack from host-side Z (seeded in run()/forces before
    // H2D). Z is run-static; geometry stays on device. Covered s/p systems
    // never D2H type[]. Component-only nested use (e.g. MD without prior
    // cache_host_types) fills the cache once and reuses it.
    if (static_cast<int>(host_Z_.size()) != natm) {
        host_Z_.resize(static_cast<size_t>(natm));
        cudaMemcpyAsync(host_Z_.data(), ctx.frame.atoms.type,
                        sizeof(int) * static_cast<size_t>(natm),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }
    const int* h_Z = host_Z_.data();

    // ── Energy: CUDA-only HF when device s/p kernels cover the molecule ─
    // STO-3G with l≤1 (H…O, mixed Z OK): device S/T/V + ERI + SCF (no host
    // ERI), including virtual level shift λ on the orthonormal Fock.
    // Higher L or other bases: host fill + upload.
    last_device_sp_ = false;
    if (device_sp_sto3g_ok(basis, h_Z, natm)) {
        solver_ = make_device_sp_sto3g_solver(h_Z, natm, charge, tol, max_iter,
                                             level_shift);
        last_device_sp_ = true;
    } else {
        cpu::Frame host = host_frame_from_device(ctx.frame);
        solver_ = make_host_integral_solver(host, basis, charge, tol, max_iter,
                                            level_shift);
    }
    solver_->build(ctx.runtime.stream, ctx.frame);
    solver_->run(ctx.runtime.stream, ctx.frame);
    iterations_ = solver_->iterations();
    converged_ = solver_->workspace().converged;

    // ── Forces on device (s/p l≤1) before any energy D2H ────────────────
    // Entire SCF + forces stay device-resident; host energy_ha_ is filled
    // once at the end (post-calculation bookkeeping only).
    last_force_device_ = false;
    if (compute_force) {
        const auto& ws = solver_->workspace();
        const bool device_forces =
            ws.device_sp_shells && ws.n > 0 && ws.P && ws.C && ws.eig
            && ws.d_shell_alpha && ws.d_shell_atom && ws.d_shell_L
            && ws.d_ao_shell && ws.d_ao_cart && ws.d_atom_pos
            && ctx.frame.atoms.fx && ctx.frame.atoms.fy && ctx.frame.atoms.fz;

        if (device_forces) {
            prec_t* d_W = ws.d_W;
            const bool own_W = (d_W == nullptr);
            const int nn = ws.n * ws.n;
            if (own_W)
                cudaMalloc(&d_W, static_cast<size_t>(nn) * sizeof(prec_t));
            launch_build_W_rhf(d_W, ws.C, ws.eig, ws.n, ws.n_occ, stream);
            launch_rhf_forces_sp(
                ctx.frame.atoms.fx, ctx.frame.atoms.fy, ctx.frame.atoms.fz,
                ws.P, d_W, ws.d_shell_alpha, ws.d_shell_coeff,
                ws.d_shell_center, ws.d_shell_atom, ws.d_shell_L, ws.d_ao_shell,
                ws.d_ao_cart, ws.d_atom_Z, ws.d_atom_pos, ws.n, ws.n_prim,
                ws.natm, stream);
            last_force_device_ = true;
            if (own_W) {
                // Free only after stream completes force kernels — no D2H.
                cudaStreamSynchronize(stream);
                cudaFree(d_W);
            }
        } else {
            cpu::Frame host = host_frame_from_device(ctx.frame);
            cpu::scf::RHF host_rhf{basis, charge, tol, max_iter};
            cpu::scf::GradientResult gr = host_rhf.forces(host);
            write_forces_to_frame(host, gr);
            if (ctx.frame.atoms.fx && ctx.frame.atoms.fy && ctx.frame.atoms.fz) {
                std::vector<real_t> hfx(static_cast<size_t>(natm));
                std::vector<real_t> hfy(static_cast<size_t>(natm));
                std::vector<real_t> hfz(static_cast<size_t>(natm));
                auto fx = host.span(kAtom_fx);
                auto fy = host.span(kAtom_fy);
                auto fz = host.span(kAtom_fz);
                for (int A = 0; A < natm; ++A) {
                    hfx[static_cast<size_t>(A)] = static_cast<real_t>(fx[A]);
                    hfy[static_cast<size_t>(A)] = static_cast<real_t>(fy[A]);
                    hfz[static_cast<size_t>(A)] = static_cast<real_t>(fz[A]);
                }
                cudaMemcpy(ctx.frame.atoms.fx, hfx.data(),
                           sizeof(real_t) * static_cast<size_t>(natm),
                           cudaMemcpyHostToDevice);
                cudaMemcpy(ctx.frame.atoms.fy, hfy.data(),
                           sizeof(real_t) * static_cast<size_t>(natm),
                           cudaMemcpyHostToDevice);
                cudaMemcpy(ctx.frame.atoms.fz, hfz.data(),
                           sizeof(real_t) * static_cast<size_t>(natm),
                           cudaMemcpyHostToDevice);
            }
            energy_ha_ = gr.energy / units::kHartreeInEv;
            return;
        }
    }

    // Post-calculation: one energy D2H for host `energy_ha_` (not mid-SCF).
    prec_t e_ev = 0;
    if (ctx.frame.scalars) {
        cudaMemcpyAsync(&e_ev, &ctx.frame.scalars->energy, sizeof(prec_t),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }
    energy_ha_ = static_cast<double>(e_ev) / units::kHartreeInEv;
}

void RHF::destroy(const DestroyContext& ctx) {
    if (solver_) {
        solver_->destroy(ctx.runtime.stream, ctx.frame);
        solver_.reset();
    }
    built_ = false;
}

} // namespace scf
} // namespace cuda
} // namespace ATV

#endif // ATV_HAS_HF
