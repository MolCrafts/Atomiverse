/// @file cuda/dynamics/fix_efield.cu
/// @brief CUDA constant uniform electric field (LAMMPS `fix efield`).
///
/// Port of the constant-field charge term of LAMMPS `src/fix_efield.cpp`
/// (`FixEfield::post_force`: `f += q E`, `fsum[0] -= F · unwrap`), checkout
/// SHA e8c4c278a8. LAMMPS: Copyright (2003) Sandia Corporation, GNU GPL.

#include "atomiverse/cuda/dynamics/fix_efield.cuh"

#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/frame/box.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/frame/types.cuh"

#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

namespace ATV {
namespace cuda {
namespace md {
namespace {

// One thread per owned atom. The force is an owned (non-atomic) fixed-point
// add; the energy is reduced in FP64 only when the step asks for it.
__global__ void
efield_k(const pos_t* __restrict__ x, const pos_t* __restrict__ y,
         const pos_t* __restrict__ z, const pos_t* __restrict__ xc,
         const pos_t* __restrict__ yc, const pos_t* __restrict__ zc,
         const force_t* __restrict__ charge, const ImageInt* __restrict__ image,
         fixed_t* __restrict__ fx, fixed_t* __restrict__ fy,
         fixed_t* __restrict__ fz, ForceFault* fault,
         acc_t* __restrict__ energy, cuda::Box box_arg, double ex, double ey,
         double ez, bool forces, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    acc_t u = 0;
    if (i < n) {
        const double qi = static_cast<double>(charge[i]);
        const double fxi = qi * ex;
        const double fyi = qi * ey;
        const double fzi = qi * ez;
        if (forces) {
            add_force_owned(fx, i, fxi, fault);
            add_force_owned(fy, i, fyi, fault);
            add_force_owned(fz, i, fzi, fault);
        }
        if (energy != nullptr) {
            // LAMMPS Domain::unmap: r^u = r + H n (lattice vectors in the
            // columns of the row-major cell).
            const Box box = box_arg.current();
            const double r[3] = {static_cast<double>(load_position(x, xc, i)),
                                 static_cast<double>(load_position(y, yc, i)),
                                 static_cast<double>(load_position(z, zc, i))};
            double nimg[3] = {0, 0, 0};
            if (image != nullptr)
                for (int d = 0; d < 3; ++d)
                    nimg[d] = static_cast<double>(image[3 * i + d]);
            double ru[3];
            for (int a = 0; a < 3; ++a)
                ru[a] = r[a] + box.cell[3 * a] * nimg[0]
                        + box.cell[3 * a + 1] * nimg[1]
                        + box.cell[3 * a + 2] * nimg[2];
            u = -(fxi * ru[0] + fyi * ru[1] + fzi * ru[2]);
        }
    }
    // Null energy: no eflag this step (LAMMPS ev_set); the pointer is uniform.
    if (energy != nullptr) {
        using Reduce = cub::BlockReduce<acc_t, kDefaultBlock>;
        __shared__ typename Reduce::TempStorage reduction;
        const acc_t sum = Reduce(reduction).Sum(u);
        if (threadIdx.x == 0)
            atomicAdd(energy, sum);
    }
}

std::uint64_t bits_of(double v) {
    std::uint64_t b = 0;
    std::memcpy(&b, &v, sizeof b);
    return b;
}

} // namespace

std::uint64_t FixEfield::graph_epoch() const {
    // Any field change re-captures: the components are kernel arguments.
    std::uint64_t h = bits_of(Ex);
    h = h * 0x9E3779B97F4A7C15ull ^ bits_of(Ey);
    h = h * 0x9E3779B97F4A7C15ull ^ bits_of(Ez);
    return h;
}

void FixEfield::launch(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    const int n = f.n_local();
    if (n == 0)
        return;
    const auto stream = static_cast<cudaStream_t>(ctx.runtime.stream);
    const auto& a = f.atoms;
    efield_k<<<grid_size(n), kDefaultBlock, 0, stream>>>(
        a.x, a.y, a.z, a.x_corr, a.y_corr, a.z_corr, a.charge, a.image, a.fx,
        a.fy, a.fz, f.force_fault,
        ctx.energy_requested ? &f.scalars->energy : nullptr,
        ctx.runtime.host_box, Ex, Ey, Ez, ctx.forces_requested, n);
}

} // namespace md
} // namespace cuda
} // namespace ATV
