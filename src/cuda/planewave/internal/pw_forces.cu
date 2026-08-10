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

template <typename T>
void device_free(T*& p) {
    if (p) {
        cudaFree(p);
        p = nullptr;
    }
}

/// Local Hellmann–Feynman: one thread per G; loops atoms.
/// F_ia += Ω v(G) (ρ_r sin φ + ρ_i cos φ) G  for G ≠ 0.
__global__ void k_local_hf_forces(const cufftDoubleComplex* __restrict__ rho,
                                  const double* __restrict__ gx,
                                  const double* __restrict__ gy,
                                  const double* __restrict__ gz,
                                  const double* __restrict__ vg,
                                  const double* __restrict__ pos, int ng,
                                  int n_atoms, double omega,
                                  double* __restrict__ forces) {
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
                                    double omega, double* d_forces) {
    if (ng <= 0 || n_atoms <= 0)
        return;
    k_local_hf_forces<<<n_blocks(ng), kBlock>>>(d_rho_G, d_gx, d_gy, d_gz,
                                                d_vg, d_pos, ng, n_atoms, omega,
                                                d_forces);
    ATV_CUDA_CHECK(cudaGetLastError());
}

std::vector<std::array<double, 3>>
pw_forces_analytic_impl(const cpu::Frame& frame, const std::string& upf_path,
                        const std::string& gth_name, double ecut_ry,
                        DeviceDensity& den, DeviceBandSolver& solver,
                        cpu::pw::internal::PWProblem& prob,
                        cpu::pw::internal::PWWorkspace& ws) {
    const int n = frame.nlocal();
    if (n <= 0)
        throw std::invalid_argument(
            "cuda::pw::pw_forces_analytic: frame has no atoms");

    const int ng0 = den.n_g0();
    const double omega = den.omega();
    if (ng0 <= 0 || !(omega > 0.0))
        throw std::runtime_error(
            "cuda::pw::pw_forces_analytic: density sphere empty or omega=0");

    double* d_F = device_alloc<double>(static_cast<std::size_t>(3 * n));
    ATV_CUDA_CHECK(
        cudaMemset(d_F, 0, static_cast<std::size_t>(3 * n) * sizeof(double)));

    // G vectors + local form factor (host → device once).
    std::vector<double> h_gx(static_cast<std::size_t>(ng0));
    std::vector<double> h_gy(static_cast<std::size_t>(ng0));
    std::vector<double> h_gz(static_cast<std::size_t>(ng0));
    for (int ig = 0; ig < ng0; ++ig) {
        h_gx[static_cast<std::size_t>(ig)] = prob.gv0.g[ig][0];
        h_gy[static_cast<std::size_t>(ig)] = prob.gv0.g[ig][1];
        h_gz[static_cast<std::size_t>(ig)] = prob.gv0.g[ig][2];
    }
    const auto upf =
        cpu::pw::internal::load_pseudopotential(frame, upf_path, gth_name);
    const std::vector<AccReal> upf_vg =
        cpu::pw::internal::upf_vloc_to_gspace(upf, prob.gv0, AccReal(omega));
    std::vector<double> h_vg(static_cast<std::size_t>(ng0));
    for (int ig = 0; ig < ng0; ++ig)
        h_vg[static_cast<std::size_t>(ig)] =
            static_cast<double>(upf_vg[static_cast<std::size_t>(ig)]);

    std::vector<double> h_pos(static_cast<std::size_t>(3 * n));
    for (int ia = 0; ia < n; ++ia) {
        h_pos[static_cast<std::size_t>(3 * ia + 0)] = static_cast<double>(
            ws.atom_pos_bohr[static_cast<std::size_t>(ia)][0]);
        h_pos[static_cast<std::size_t>(3 * ia + 1)] = static_cast<double>(
            ws.atom_pos_bohr[static_cast<std::size_t>(ia)][1]);
        h_pos[static_cast<std::size_t>(3 * ia + 2)] = static_cast<double>(
            ws.atom_pos_bohr[static_cast<std::size_t>(ia)][2]);
    }

    double* d_gx = device_alloc<double>(static_cast<std::size_t>(ng0));
    double* d_gy = device_alloc<double>(static_cast<std::size_t>(ng0));
    double* d_gz = device_alloc<double>(static_cast<std::size_t>(ng0));
    double* d_vg = device_alloc<double>(static_cast<std::size_t>(ng0));
    double* d_pos = device_alloc<double>(static_cast<std::size_t>(3 * n));
    ATV_CUDA_CHECK(cudaMemcpy(d_gx, h_gx.data(),
                              static_cast<std::size_t>(ng0) * sizeof(double),
                              cudaMemcpyHostToDevice));
    ATV_CUDA_CHECK(cudaMemcpy(d_gy, h_gy.data(),
                              static_cast<std::size_t>(ng0) * sizeof(double),
                              cudaMemcpyHostToDevice));
    ATV_CUDA_CHECK(cudaMemcpy(d_gz, h_gz.data(),
                              static_cast<std::size_t>(ng0) * sizeof(double),
                              cudaMemcpyHostToDevice));
    ATV_CUDA_CHECK(cudaMemcpy(d_vg, h_vg.data(),
                              static_cast<std::size_t>(ng0) * sizeof(double),
                              cudaMemcpyHostToDevice));
    ATV_CUDA_CHECK(cudaMemcpy(d_pos, h_pos.data(),
                              static_cast<std::size_t>(3 * n) * sizeof(double),
                              cudaMemcpyHostToDevice));

    accumulate_local_forces_device(den.device_rho_G(), d_gx, d_gy, d_gz, d_vg,
                                   d_pos, ng0, n, omega, d_F);

    // Nonlocal KB: rebuild V_eff from the final mixed density (CPU twin of
    // poisson.step + xc.step + veff.step after SCF), then re-solve bands.
    if (!prob.kb_proj_k.empty()) {
        den.hartree();
        den.exchange_correlation();
        den.assemble_veff();
        const int n_kpts = static_cast<int>(prob.kg.kpoints.size());
        for (int ik = 0; ik < n_kpts; ++ik) {
            solver.bind_kpoint(ik);
            solver.solve(ws, 40);
            const double weight = static_cast<double>(
                prob.kg.weights[static_cast<std::size_t>(ik)]);
            solver.accumulate_nl_forces(weight, n, d_F);
        }
    }

    std::vector<double> h_F(static_cast<std::size_t>(3 * n));
    ATV_CUDA_CHECK(cudaMemcpy(h_F.data(), d_F,
                              static_cast<std::size_t>(3 * n) * sizeof(double),
                              cudaMemcpyDeviceToHost));

    device_free(d_F);
    device_free(d_gx);
    device_free(d_gy);
    device_free(d_gz);
    device_free(d_vg);
    device_free(d_pos);

    // Ewald (host classical, shared with CPU energy/force path).
    const std::vector<AccReal> ion_charges(static_cast<std::size_t>(n),
                                           static_cast<AccReal>(upf.z_val));
    std::vector<Vec3> pos_ang(static_cast<std::size_t>(n));
    const StorageReal* sx = frame.span(kAtom_x);
    const StorageReal* sy = frame.span(kAtom_y);
    const StorageReal* sz = frame.span(kAtom_z);
    for (int i = 0; i < n; ++i)
        pos_ang[static_cast<std::size_t>(i)] = {
            static_cast<AccReal>(sx[i]), static_cast<AccReal>(sy[i]),
            static_cast<AccReal>(sz[i])};

    const AccReal ecut_ht = static_cast<AccReal>(ecut_ry) * AccReal(0.5);
    cpu::pw::internal::EwaldSummation ewald(pos_ang, ion_charges, frame.box,
                                            ecut_ht);
    const auto& few = ewald.forces_bohr();
    for (int i = 0; i < n; ++i) {
        h_F[static_cast<std::size_t>(3 * i + 0)] +=
            static_cast<double>(few[static_cast<std::size_t>(i)][0]);
        h_F[static_cast<std::size_t>(3 * i + 1)] +=
            static_cast<double>(few[static_cast<std::size_t>(i)][1]);
        h_F[static_cast<std::size_t>(3 * i + 2)] +=
            static_cast<double>(few[static_cast<std::size_t>(i)][2]);
    }

    const double ha_per_bohr_to_ev_per_ang = units::convert(
        1.0, units::Force::HartreePerBohr, units::Force::eVPerAngstrom);

    std::vector<std::array<double, 3>> out(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        out[static_cast<std::size_t>(i)] = {
            h_F[static_cast<std::size_t>(3 * i + 0)] * ha_per_bohr_to_ev_per_ang,
            h_F[static_cast<std::size_t>(3 * i + 1)] * ha_per_bohr_to_ev_per_ang,
            h_F[static_cast<std::size_t>(3 * i + 2)]
                * ha_per_bohr_to_ev_per_ang};
    }
    return out;
}

} // namespace internal
} // namespace pw
} // namespace cuda
} // namespace ATV
