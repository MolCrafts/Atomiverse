#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/planewave/internal/grid_map.cuh"
#include "atomiverse/cuda/planewave/internal/xc_frozen_virial.cuh"
#include "atomiverse/cuda/xc/xc_lsda.cuh"
#include "atomiverse/cuda/xc/xc_pbe.cuh"
#include <cmath>
#include <stdexcept>
namespace ATV::cuda::pw::internal {
namespace {
constexpr int block = 256, partial_count = ATV::cuda::xc::kXcEnergyPartials;
__global__ void scale_density(const cufftDoubleComplex* in,
                              cufftDoubleComplex* out, int n, double scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = {scale * in[i].x, scale * in[i].y};
}
__global__ void virial_stage1(const double* rho, const double* v, int n,
                              double* out) {
    __shared__ double values[block];
    double sum = 0;
    for (int i = blockIdx.x * block + threadIdx.x; i < n;
         i += gridDim.x * block)
        sum += rho[i] * v[i];
    values[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = block / 2; stride; stride /= 2) {
        if (threadIdx.x < stride)
            values[threadIdx.x] += values[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        out[blockIdx.x] = values[0];
}
__global__ void virial_stage2(const double* partials, const double* energy,
                              double dV, FrozenOperatorVirial* out) {
    __shared__ double values[partial_count];
    int t = threadIdx.x;
    values[t] = partials[t];
    __syncthreads();
    for (int stride = partial_count / 2; stride; stride /= 2) {
        if (t < stride)
            values[t] += values[t + stride];
        __syncthreads();
    }
    if (t == 0) {
        out->energy_ha = *energy;
        for (int i = 0; i < 9; ++i)
            out->virial_ha[i] = (i % 4 == 0) ? values[0] * dV - *energy : 0.;
    }
}
__global__ void derivative(const cufftDoubleComplex* in, const double* g, int n,
                           cufftDoubleComplex* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = {-g[i] * in[i].y, g[i] * in[i].x};
}
__global__ void reduce_w(const double* rho, const double* v, const double* gx,
                         const double* gy, const double* gz, const double* px,
                         const double* py, const double* pz, int n,
                         double* out) {
    __shared__ double shared[block];
    const int lane = blockIdx.y, a = lane / 3, b = lane % 3;
    double sum = 0;
    for (int i = blockIdx.x * block + threadIdx.x; i < n;
         i += gridDim.x * block) {
        const double g[3] = {gx[i], gy[i], gz[i]}, p[3] = {px[i], py[i], pz[i]};
        double value = g[a] * p[b];
        if (a == b)
            value += rho[i] * v[i] + g[0] * p[0] + g[1] * p[1] + g[2] * p[2];
        sum += value;
    }
    shared[threadIdx.x] = sum;
    __syncthreads();
    for (int s = block / 2; s; s /= 2) {
        if (threadIdx.x < s)
            shared[threadIdx.x] += shared[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        out[lane * partial_count + blockIdx.x] = shared[0];
}
__global__ void finish_w(const double* in, const double* energy, double dV,
                         FrozenOperatorVirial* out) {
    __shared__ double shared[partial_count];
    const int lane = blockIdx.x, t = threadIdx.x;
    shared[t] = in[lane * partial_count + t];
    __syncthreads();
    for (int s = partial_count / 2; s; s /= 2) {
        if (t < s)
            shared[t] += shared[t + s];
        __syncthreads();
    }
    if (t == 0) {
        out->virial_ha[lane] = shared[0] * dV - (lane % 4 == 0 ? *energy : 0);
        if (lane == 0)
            out->energy_ha = *energy;
    }
}
__global__ void reduce_spin_w(FrozenSpinWorkspaceView w, int n, bool gga) {
    __shared__ double shared[block];
    int lane = blockIdx.y, a = lane / 3, b = lane % 3;
    double value = 0;
    for (int i = blockIdx.x * block + threadIdx.x; i < n;
         i += gridDim.x * block) {
        double point = 0;
        for (int s = 0; s < 2; ++s) {
            if (a == b)
                point += w.rho[s][i] * w.v[s][i];
            if (gga) {
                point += w.raw_gradient[s][a][i] * w.flux[s][b][i];
                if (a == b)
                    for (int d = 0; d < 3; ++d)
                        point += w.raw_gradient[s][d][i] * w.flux[s][d][i];
            }
        }
        value += point;
    }
    shared[threadIdx.x] = value;
    __syncthreads();
    for (int stride = block / 2; stride; stride /= 2) {
        if (threadIdx.x < stride)
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        w.virial_partials[lane * partial_count + blockIdx.x] = shared[0];
}
} // namespace
void evaluate_lsda_frozen_virial(FFTPlan3D& quadrature, const int* h,
                                 const int* k, const int* l, int ng,
                                 const cufftDoubleComplex* rho_orb,
                                 double density_scale, double omega,
                                 FrozenLsdaWorkspaceView scratch,
                                 cudaStream_t stream) {
    if (ng < 1 || !std::isfinite(omega) || omega <= 0
        || !std::isfinite(density_scale) || density_scale <= 0)
        throw std::invalid_argument("XC prototype density/volume");
    scale_density<<<(ng + block - 1) / block, block, 0, stream>>>(
        rho_orb, scratch.scaled_density, ng, density_scale);
    ATV_CUDA_CHECK(cudaGetLastError());
    scatter_g_to_halfcomplex(scratch.scaled_density, h, k, l, ng, scratch.half,
                             quadrature.nx(), quadrature.ny(), quadrature.nz(),
                             stream);
    quadrature.c2r(scratch.half, scratch.rho);
    ATV::cuda::xc::xc_lsda_grid(scratch.rho, scratch.v, scratch.epsilon,
                                quadrature.nr(), stream);
    ATV_CUDA_CHECK(cudaGetLastError());
    const double dV = omega / quadrature.nr();
    ATV::cuda::xc::xc_energy(scratch.rho, scratch.epsilon, quadrature.nr(), dV,
                             scratch.energy, scratch.energy_partials, stream);
    ATV_CUDA_CHECK(cudaGetLastError());
    virial_stage1<<<partial_count, block, 0, stream>>>(
        scratch.rho, scratch.v, quadrature.nr(), scratch.virial_partials);
    ATV_CUDA_CHECK(cudaGetLastError());
    virial_stage2<<<1, partial_count, 0, stream>>>(
        scratch.virial_partials, scratch.energy, dV, scratch.output);
    ATV_CUDA_CHECK(cudaGetLastError());
}
void evaluate_pbe_frozen_virial(ATV::cpu::xc::XCType type, FFTPlan3D& fft,
                                const int* h, const int* k, const int* l,
                                const double* gx, const double* gy,
                                const double* gz, int ng,
                                const cufftDoubleComplex* rho,
                                double density_scale, double omega,
                                FrozenPbeWorkspaceView w, cudaStream_t stream) {
    if (!ATV::cpu::xc::is_pbe_family(type) || ng < 1 || !std::isfinite(omega)
        || omega <= 0 || !std::isfinite(density_scale) || density_scale <= 0)
        throw std::invalid_argument("PBE frozen XC admission");
    scale_density<<<(ng + block - 1) / block, block, 0, stream>>>(
        rho, w.scaled_density, ng, density_scale);
    ATV_CUDA_CHECK(cudaGetLastError());
    scatter_g_to_halfcomplex(w.scaled_density, h, k, l, ng, w.half, fft.nx(),
                             fft.ny(), fft.nz(), stream);
    fft.c2r(w.half, w.rho);
    const double* physical[3] = {gx, gy, gz};
    for (int d = 0; d < 3; ++d) {
        derivative<<<(ng + block - 1) / block, block, 0, stream>>>(
            w.scaled_density, physical[d], ng, w.derivative);
        ATV_CUDA_CHECK(cudaGetLastError());
        scatter_g_to_halfcomplex(w.derivative, h, k, l, ng, w.half, fft.nx(),
                                 fft.ny(), fft.nz(), stream);
        fft.c2r(w.half, w.raw_gradient[d]);
        ATV_CUDA_CHECK(cudaMemcpyAsync(w.flux[d], w.raw_gradient[d],
                                       fft.nr() * sizeof(double),
                                       cudaMemcpyDeviceToDevice, stream));
    }
    ATV::cuda::xc::xc_pbe_grid(type, w.rho, w.flux[0], w.flux[1], w.flux[2],
                               w.v, w.epsilon, fft.nr(), stream);
    ATV_CUDA_CHECK(cudaGetLastError());
    const double dV = omega / fft.nr();
    ATV::cuda::xc::xc_energy(w.rho, w.epsilon, fft.nr(), dV, w.energy,
                             w.energy_partials, stream);
    ATV_CUDA_CHECK(cudaGetLastError());
    reduce_w<<<dim3(partial_count, 9), block, 0, stream>>>(
        w.rho, w.v, w.raw_gradient[0], w.raw_gradient[1], w.raw_gradient[2],
        w.flux[0], w.flux[1], w.flux[2], fft.nr(), w.virial_partials);
    ATV_CUDA_CHECK(cudaGetLastError());
    finish_w<<<9, partial_count, 0, stream>>>(w.virial_partials, w.energy, dV,
                                              w.output);
    ATV_CUDA_CHECK(cudaGetLastError());
}
void evaluate_spin_frozen_virial(
    cpu::xc::XCType type, FFTPlan3D& fft, const int* h, const int* k,
    const int* l, const double* gx, const double* gy, const double* gz, int ng,
    const cufftDoubleComplex* alpha, const cufftDoubleComplex* beta,
    double density_scale, double omega, FrozenSpinWorkspaceView w,
    cudaStream_t stream) {
    const bool gga = cpu::xc::is_pbe_family(type);
    if ((type != cpu::xc::XCType::LSDA && !gga) || ng < 1
        || !std::isfinite(omega) || omega <= 0 || !std::isfinite(density_scale)
        || density_scale <= 0)
        throw std::invalid_argument("spin frozen XC admission");
    const cufftDoubleComplex* input[2] = {alpha, beta};
    const double* G[3] = {gx, gy, gz};
    for (int s = 0; s < 2; ++s) {
        scale_density<<<(ng + block - 1) / block, block, 0, stream>>>(
            input[s], w.scaled_density[s], ng, density_scale);
        ATV_CUDA_CHECK(cudaGetLastError());
        scatter_g_to_halfcomplex(w.scaled_density[s], h, k, l, ng, w.half,
                                 fft.nx(), fft.ny(), fft.nz(), stream);
        fft.c2r(w.half, w.rho[s]);
        if (gga)
            for (int d = 0; d < 3; ++d) {
                derivative<<<(ng + block - 1) / block, block, 0, stream>>>(
                    w.scaled_density[s], G[d], ng, w.derivative);
                ATV_CUDA_CHECK(cudaGetLastError());
                scatter_g_to_halfcomplex(w.derivative, h, k, l, ng, w.half,
                                         fft.nx(), fft.ny(), fft.nz(), stream);
                fft.c2r(w.half, w.raw_gradient[s][d]);
                ATV_CUDA_CHECK(
                    cudaMemcpyAsync(w.flux[s][d], w.raw_gradient[s][d],
                                    fft.nr() * sizeof(double),
                                    cudaMemcpyDeviceToDevice, stream));
            }
    }
    if (gga)
        ATV::cuda::xc::xc_pbe_spin_grid(
            type, w.rho[0], w.rho[1], w.flux[0][0], w.flux[0][1], w.flux[0][2],
            w.flux[1][0], w.flux[1][1], w.flux[1][2], w.v[0], w.v[1], w.epsilon,
            fft.nr(), stream, ATV::cuda::xc::SpinDensityPolicy::PositivePart);
    else
        ATV::cuda::xc::xc_lsda_spin_grid(w.rho[0], w.rho[1], w.v[0], w.v[1],
                                         w.epsilon, fft.nr(), stream);
    ATV_CUDA_CHECK(cudaGetLastError());
    const double dV = omega / fft.nr();
    ATV::cuda::xc::xc_energy_spin(w.rho[0], w.rho[1], w.epsilon, fft.nr(), dV,
                                  w.energy, w.energy_partials, stream);
    ATV_CUDA_CHECK(cudaGetLastError());
    reduce_spin_w<<<dim3(partial_count, 9), block, 0, stream>>>(w, fft.nr(),
                                                                gga);
    ATV_CUDA_CHECK(cudaGetLastError());
    finish_w<<<9, partial_count, 0, stream>>>(w.virial_partials, w.energy, dV,
                                              w.output);
    ATV_CUDA_CHECK(cudaGetLastError());
}

} // namespace ATV::cuda::pw::internal
