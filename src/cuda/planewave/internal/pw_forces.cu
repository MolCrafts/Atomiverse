/// @file cuda/planewave/internal/pw_forces.cu
/// @brief Device analytic plane-wave ionic forces (local HF + Ewald + KB).

#include "atomiverse/cuda/planewave/internal/pw_forces.cuh"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/core/vec3.h"
#include "atomiverse/cpu/planewave/internal/ewald.h"
#include "atomiverse/cpu/planewave/internal/pw_setup.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/electronic/smearing.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace ATV {
namespace cuda {
namespace pw {
namespace internal {
namespace {

constexpr int kBlock = 256;
inline int n_blocks(int n) { return (n + kBlock - 1) / kBlock; }

template <typename T>
T* device_alloc(std::size_t n) {
    T* p = nullptr;
    ATV_CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
    return p;
}

struct StreamFence {
    cudaStream_t stream;
    ~StreamFence() { cudaStreamSynchronize(stream); }
};

struct ForceScratch {
    double* buffers[7] = {};
    int count = 0;
    double* allocate(std::size_t n) {
        double* p = device_alloc<double>(n);
        buffers[count++] = p;
        return p;
    }
    ~ForceScratch() {
        for (int i = count - 1; i >= 0; --i)
            cudaFree(buffers[i]);
    }
};

__global__ void publish_force(cuda::Frame target, const double* force,
                              const double* ewald, int n, double scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    add_force(target.atoms.fx, i, (force[3 * i] + ewald[3 * i]) * scale,
              target.force_fault);
    add_force(target.atoms.fy, i, (force[3 * i + 1] + ewald[3 * i + 1]) * scale,
              target.force_fault);
    add_force(target.atoms.fz, i, (force[3 * i + 2] + ewald[3 * i + 2]) * scale,
              target.force_fault);
}

/// Local Hellmann–Feynman: one thread per G; loops atoms.
/// F_ia += Ω v(G) (ρ_r sin φ + ρ_i cos φ) G  for G ≠ 0.
__global__ void
k_local_hf_forces(const cufftDoubleComplex* __restrict__ rho,
                  const double* __restrict__ gx, const double* __restrict__ gy,
                  const double* __restrict__ gz, const double* __restrict__ vg,
                  const double* __restrict__ pos, int ng, int n_atoms,
                  double omega, double* __restrict__ forces) {
    const int ig = blockIdx.x * blockDim.x + threadIdx.x;
    if (ig >= ng)
        return;
    const double Gx = gx[ig], Gy = gy[ig], Gz = gz[ig];
    const double g2 = Gx * Gx + Gy * Gy + Gz * Gz;
    if (g2 < 1e-16)
        return;
    const cufftDoubleComplex r = rho[ig];
    const double v = vg[ig];
    for (int ia = 0; ia < n_atoms; ++ia) {
        const double ax = pos[3 * ia + 0];
        const double ay = pos[3 * ia + 1];
        const double az = pos[3 * ia + 2];
        const double phase = Gx * ax + Gy * ay + Gz * az;
        const double cosph = cos(phase);
        const double sinph = sin(phase);
        const double scale = omega * v * (r.x * sinph + r.y * cosph);
        atomicAdd(forces + 3 * ia + 0, scale * Gx);
        atomicAdd(forces + 3 * ia + 1, scale * Gy);
        atomicAdd(forces + 3 * ia + 2, scale * Gz);
    }
}

} // namespace

void accumulate_local_forces_device(const cufftDoubleComplex* d_rho_G,
                                    const double* d_gx, const double* d_gy,
                                    const double* d_gz, const double* d_vg,
                                    const double* d_pos, int ng, int n_atoms,
                                    double omega, double* d_forces,
                                    cudaStream_t stream) {
    if (ng <= 0 || n_atoms <= 0)
        return;
    k_local_hf_forces<<<n_blocks(ng), kBlock, 0, stream>>>(
        d_rho_G, d_gx, d_gy, d_gz, d_vg, d_pos, ng, n_atoms, omega, d_forces);
    ATV_CUDA_CHECK(cudaGetLastError());
}

std::vector<std::array<double, 3>> pw_forces_analytic_impl(
    const cpu::Frame& frame, DeviceDensity& den, DeviceBandSolver& solver,
    cpu::pw::internal::PWProblem& prob, cpu::pw::internal::PWWorkspace& ws,
    int n_alpha, int n_beta, bool use_smear, int n_elec,
    ATV::electronic::Smearing smear_kind, double degauss_ha, int mp_order,
    cuda::Frame* accumulate_into) {
    const int n = frame.nlocal();
    if (n <= 0)
        throw std::invalid_argument("cuda::pw::PW::forces: frame has no atoms");

    const int ng0 = den.n_g0();
    const double omega = den.omega();
    if (ng0 <= 0 || !(omega > 0.0))
        throw std::runtime_error(
            "cuda::pw::PW::forces: density sphere empty or omega=0");

    const bool spin = (n_alpha >= 0);
    if (spin && !den.spin_polarized())
        throw std::runtime_error(
            "cuda::pw::PW::forces: spin occupations require a spin "
            "DeviceDensity");
    if (spin && n_beta < 0)
        throw std::invalid_argument(
            "cuda::pw::PW::forces: n_beta must be >= 0 on spin path");
    if (use_smear && !(degauss_ha > 0.0))
        throw std::invalid_argument(
            "cuda::pw::PW::forces: degauss must be > 0 for smear force "
            "path");
    if (use_smear && n_elec <= 0)
        throw std::invalid_argument(
            "cuda::pw::PW::forces: n_elec must be > 0 on smear force "
            "path");

    if (accumulate_into
        && (accumulate_into->n_atoms() != n || !accumulate_into->atoms.fx
            || !accumulate_into->atoms.fy || !accumulate_into->atoms.fz))
        throw std::invalid_argument(
            "cuda::pw::forces: incomplete parent force columns");
    ForceScratch scratch;
    double* d_F = scratch.allocate(static_cast<std::size_t>(3 * n));
    ATV_CUDA_CHECK(cudaMemsetAsync(
        d_F, 0, static_cast<std::size_t>(3 * n) * sizeof(double),
        den.stream()));

    // G vectors + local form factor (host → device once).
    std::vector<double> h_gx(static_cast<std::size_t>(ng0));
    std::vector<double> h_gy(static_cast<std::size_t>(ng0));
    std::vector<double> h_gz(static_cast<std::size_t>(ng0));
    for (int ig = 0; ig < ng0; ++ig) {
        h_gx[static_cast<std::size_t>(ig)] = prob.gv0.g[ig][0];
        h_gy[static_cast<std::size_t>(ig)] = prob.gv0.g[ig][1];
        h_gz[static_cast<std::size_t>(ig)] = prob.gv0.g[ig][2];
    }
    const auto& nc = *prob.potential;
    const std::vector<AccReal> nc_vg =
        cpu::pw::internal::nc_vloc_to_gspace(nc, prob.gv0, AccReal(omega));
    std::vector<double> h_vg(static_cast<std::size_t>(ng0));
    for (int ig = 0; ig < ng0; ++ig)
        h_vg[static_cast<std::size_t>(ig)] =
            static_cast<double>(nc_vg[static_cast<std::size_t>(ig)]);

    std::vector<double> h_pos(static_cast<std::size_t>(3 * n));
    for (int ia = 0; ia < n; ++ia) {
        h_pos[static_cast<std::size_t>(3 * ia + 0)] = static_cast<double>(
            ws.atom_pos_bohr[static_cast<std::size_t>(ia)][0]);
        h_pos[static_cast<std::size_t>(3 * ia + 1)] = static_cast<double>(
            ws.atom_pos_bohr[static_cast<std::size_t>(ia)][1]);
        h_pos[static_cast<std::size_t>(3 * ia + 2)] = static_cast<double>(
            ws.atom_pos_bohr[static_cast<std::size_t>(ia)][2]);
    }

    double* d_gx = scratch.allocate(static_cast<std::size_t>(ng0));
    double* d_gy = scratch.allocate(static_cast<std::size_t>(ng0));
    double* d_gz = scratch.allocate(static_cast<std::size_t>(ng0));
    double* d_vg = scratch.allocate(static_cast<std::size_t>(ng0));
    double* d_pos = scratch.allocate(static_cast<std::size_t>(3 * n));
    StreamFence geometry_fence{den.stream()};
    ATV_CUDA_CHECK(cudaMemcpyAsync(
        d_gx, h_gx.data(), static_cast<std::size_t>(ng0) * sizeof(double),
        cudaMemcpyHostToDevice, den.stream()));
    ATV_CUDA_CHECK(cudaMemcpyAsync(
        d_gy, h_gy.data(), static_cast<std::size_t>(ng0) * sizeof(double),
        cudaMemcpyHostToDevice, den.stream()));
    ATV_CUDA_CHECK(cudaMemcpyAsync(
        d_gz, h_gz.data(), static_cast<std::size_t>(ng0) * sizeof(double),
        cudaMemcpyHostToDevice, den.stream()));
    ATV_CUDA_CHECK(cudaMemcpyAsync(
        d_vg, h_vg.data(), static_cast<std::size_t>(ng0) * sizeof(double),
        cudaMemcpyHostToDevice, den.stream()));
    ATV_CUDA_CHECK(cudaMemcpyAsync(
        d_pos, h_pos.data(), static_cast<std::size_t>(3 * n) * sizeof(double),
        cudaMemcpyHostToDevice, den.stream()));

    // Local HF uses total ρ (restricted or ρ↑+ρ↓ after spin mix).
    if (spin)
        den.sync_total_rho();
    accumulate_local_forces_device(den.device_rho_G(), d_gx, d_gy, d_gz, d_vg,
                                   d_pos, ng0, n, omega, d_F, den.stream());

    // Nonlocal KB: rebuild V_eff from the final mixed density, then re-solve.
    // Restricted fixed: one V_eff, f=2. Restricted smear: fractional f_b.
    // Collinear: V_eff_σ per channel, f=1, N_σ.
    if (!prob.kb_proj_k.empty()) {
        const int n_kpts = static_cast<int>(prob.kg.kpoints.size());
        if (spin && use_smear) {
            den.sync_total_rho();
            den.hartree();
            den.exchange_correlation_spin();
            std::vector<std::vector<AccReal>> evals_sk(
                static_cast<std::size_t>(2 * n_kpts));
            for (int s = 0; s < 2; ++s) {
                den.assemble_veff_spin(s);
                for (int ik = 0; ik < n_kpts; ++ik) {
                    solver.bind_kpoint(ik);
                    solver.solve(ws, 40);
                    evals_sk[static_cast<std::size_t>(s * n_kpts + ik)].assign(
                        ws.eigenvalues.begin(), ws.eigenvalues.end());
                }
            }
            std::vector<AccReal> all_evals;
            std::vector<AccReal> all_weights;
            for (int s = 0; s < 2; ++s) {
                for (int ik = 0; ik < n_kpts; ++ik) {
                    const AccReal w =
                        prob.kg.weights[static_cast<std::size_t>(ik)];
                    const auto& ev =
                        evals_sk[static_cast<std::size_t>(s * n_kpts + ik)];
                    for (AccReal e : ev) {
                        all_evals.push_back(e);
                        all_weights.push_back(w);
                    }
                }
            }
            const AccReal mu_ha = ATV::electronic::find_fermi_level(
                all_evals, all_weights, AccReal(n_elec), AccReal(degauss_ha),
                smear_kind, mp_order, AccReal(1));
            for (int s = 0; s < 2; ++s) {
                den.assemble_veff_spin(s);
                for (int ik = 0; ik < n_kpts; ++ik) {
                    solver.bind_kpoint(ik);
                    solver.solve(ws, 40);
                    const auto& ev =
                        evals_sk[static_cast<std::size_t>(s * n_kpts + ik)];
                    const int nb = static_cast<int>(ev.size());
                    std::vector<double> occ(static_cast<std::size_t>(nb));
                    for (int b = 0; b < nb; ++b)
                        occ[static_cast<std::size_t>(b)] =
                            static_cast<double>(ATV::electronic::smear_f_spinor(
                                ev[static_cast<std::size_t>(b)], mu_ha,
                                AccReal(degauss_ha), smear_kind, mp_order));
                    const double weight = static_cast<double>(
                        prob.kg.weights[static_cast<std::size_t>(ik)]);
                    solver.accumulate_nl_forces(weight, n, d_F, occ.data(), nb);
                }
            }
        } else if (spin) {
            den.sync_total_rho();
            den.hartree();
            den.exchange_correlation_spin();
            const int n_occ[2] = {n_alpha, n_beta};
            for (int s = 0; s < 2; ++s) {
                if (n_occ[s] <= 0)
                    continue;
                den.assemble_veff_spin(s);
                for (int ik = 0; ik < n_kpts; ++ik) {
                    solver.bind_kpoint(ik);
                    solver.solve(ws, 40);
                    const double weight = static_cast<double>(
                        prob.kg.weights[static_cast<std::size_t>(ik)]);
                    solver.accumulate_nl_forces(weight, n, d_F, n_occ[s],
                                                /*occ=*/1.0);
                }
            }
        } else if (use_smear) {
            den.hartree();
            den.exchange_correlation();
            den.assemble_veff();

            std::vector<std::vector<AccReal>> evals_k(
                static_cast<std::size_t>(n_kpts));
            for (int ik = 0; ik < n_kpts; ++ik) {
                solver.bind_kpoint(ik);
                solver.solve(ws, 40);
                evals_k[static_cast<std::size_t>(ik)].assign(
                    ws.eigenvalues.begin(), ws.eigenvalues.end());
                solver.stash_ritz();
            }

            std::vector<AccReal> all_evals;
            std::vector<AccReal> all_weights;
            for (int ik = 0; ik < n_kpts; ++ik) {
                const AccReal w = prob.kg.weights[static_cast<std::size_t>(ik)];
                const auto& ev = evals_k[static_cast<std::size_t>(ik)];
                for (std::size_t b = 0; b < ev.size(); ++b) {
                    all_evals.push_back(ev[b]);
                    all_weights.push_back(w);
                }
            }
            const AccReal mu_ha = ATV::electronic::find_fermi_level(
                all_evals, all_weights, AccReal(n_elec), AccReal(degauss_ha),
                smear_kind, mp_order);

            for (int ik = 0; ik < n_kpts; ++ik) {
                solver.load_ritz(ik);
                const auto& ev = evals_k[static_cast<std::size_t>(ik)];
                const int nb = static_cast<int>(ev.size());
                std::vector<double> occ(static_cast<std::size_t>(nb));
                for (int b = 0; b < nb; ++b)
                    occ[static_cast<std::size_t>(b)] =
                        static_cast<double>(ATV::electronic::smear_f_band(
                            ev[static_cast<std::size_t>(b)], mu_ha,
                            AccReal(degauss_ha), smear_kind, mp_order));
                const double weight = static_cast<double>(
                    prob.kg.weights[static_cast<std::size_t>(ik)]);
                solver.accumulate_nl_forces(weight, n, d_F, occ.data(), nb);
            }
        } else {
            den.hartree();
            den.exchange_correlation();
            den.assemble_veff();
            for (int ik = 0; ik < n_kpts; ++ik) {
                solver.bind_kpoint(ik);
                solver.solve(ws, 40);
                const double weight = static_cast<double>(
                    prob.kg.weights[static_cast<std::size_t>(ik)]);
                solver.accumulate_nl_forces(weight, n, d_F);
            }
        }
    }

    // Ewald (host classical, shared with CPU energy/force path).
    const std::vector<AccReal> ion_charges(static_cast<std::size_t>(n),
                                           static_cast<AccReal>(nc.z_val));
    std::vector<Vec3> pos_ang(static_cast<std::size_t>(n));
    const StorageReal* sx = frame.span(kAtom_x);
    const StorageReal* sy = frame.span(kAtom_y);
    const StorageReal* sz = frame.span(kAtom_z);
    for (int i = 0; i < n; ++i)
        pos_ang[static_cast<std::size_t>(i)] = {static_cast<AccReal>(sx[i]),
                                                static_cast<AccReal>(sy[i]),
                                                static_cast<AccReal>(sz[i])};

    cpu::pw::internal::EwaldSummation ewald(pos_ang, ion_charges, frame.box);
    const auto& few = ewald.forces_bohr();
    const double ha_per_bohr_to_ev_per_ang = units::convert(
        1.0, units::Force::HartreePerBohr, units::Force::eVPerAngstrom);
    if (accumulate_into) {
        // The complete local+KB force never crosses the device boundary.
        // Ewald remains the existing host classical term, uploaded once.
        std::vector<double> h_ewald(static_cast<std::size_t>(3 * n));
        for (int i = 0; i < n; ++i)
            for (int d = 0; d < 3; ++d)
                h_ewald[3 * i + d] = few[static_cast<std::size_t>(i)][d];
        double* d_ewald = scratch.allocate(h_ewald.size());
        StreamFence ewald_fence{den.stream()};
        ATV_CUDA_CHECK(cudaMemcpyAsync(d_ewald, h_ewald.data(),
                                       h_ewald.size() * sizeof(double),
                                       cudaMemcpyHostToDevice, den.stream()));
        publish_force<<<n_blocks(n), kBlock, 0, den.stream()>>>(
            *accumulate_into, d_F, d_ewald, n, ha_per_bohr_to_ev_per_ang);
        ATV_CUDA_CHECK(cudaGetLastError());
        ATV_CUDA_CHECK(cudaStreamSynchronize(den.stream()));
        return {};
    }
    // Standalone result boundary: final vector only, then host Ewald sum.
    std::vector<double> h_F(static_cast<std::size_t>(3 * n));
    ATV_CUDA_CHECK(cudaMemcpyAsync(h_F.data(), d_F, h_F.size() * sizeof(double),
                                   cudaMemcpyDeviceToHost, den.stream()));
    ATV_CUDA_CHECK(cudaStreamSynchronize(den.stream()));
    for (int i = 0; i < n; ++i)
        for (int d = 0; d < 3; ++d)
            h_F[3 * i + d] += few[static_cast<std::size_t>(i)][d];

    std::vector<std::array<double, 3>> out(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        out[static_cast<std::size_t>(i)] = {
            h_F[static_cast<std::size_t>(3 * i + 0)]
                * ha_per_bohr_to_ev_per_ang,
            h_F[static_cast<std::size_t>(3 * i + 1)]
                * ha_per_bohr_to_ev_per_ang,
            h_F[static_cast<std::size_t>(3 * i + 2)]
                * ha_per_bohr_to_ev_per_ang};
    }
    return out;
}

} // namespace internal
} // namespace pw
} // namespace cuda
} // namespace ATV
