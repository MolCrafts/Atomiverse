/// Device T/H/local/full-D NC virials on immutable input views and caller scratch.
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/planewave/internal/frozen_operator_virial.cuh"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace ATV::cuda::pw::internal {
namespace {
constexpr int block = 256;
constexpr double pi = 3.141592653589793238462643383279502884;
int blocks(std::size_t n) {
    return int(std::min<std::size_t>((n + block - 1) / block, 4096));
}
void checked(cublasStatus_t e) {
    if (e != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("PW frozen operator cuBLAS failure");
}
void require(bool condition, const char* message) {
    if (!condition)
        throw std::invalid_argument(message);
}
void validate(FrozenReciprocalView q) {
    require(q.count >= 0, "PW frozen operator: negative reciprocal count");
    require(q.count == 0 || (q.x && q.y && q.z),
            "PW frozen operator: missing reciprocal coordinates");
}
void validate(FrozenReciprocalView q, FrozenBandDeviceView state) {
    validate(q);
    require(state.bands >= 0, "PW frozen operator: negative band count");
    require(std::isfinite(state.k_weight),
            "PW frozen operator: nonfinite k weight");
    require(q.count == 0 || state.bands == 0 || (state.C && state.f),
            "PW frozen operator: missing coefficients or occupations");
}
void validate_volume(double omega) {
    require(std::isfinite(omega) && omega > 0,
            "PW frozen operator: volume must be finite and positive");
}
__device__ double norm2(cufftDoubleComplex a) { return a.x * a.x + a.y * a.y; }
__device__ double inner(cufftDoubleComplex a, cufftDoubleComplex b) {
    return a.x * b.x + a.y * b.y;
}
__device__ cufftDoubleComplex phase(int l, double dot) {
    const double c = cos(dot), s = sin(dot),
                 re = l == 0 ? 1. : (l == 1 ? 0. : -1.), im = l == 1 ? -1. : 0.;
    return {re * c + im * s, im * c - re * s};
}
__device__ void reduce(FrozenOperatorVirial value,
                       FrozenOperatorVirial* output) {
    __shared__ double part[10][block];
    part[0][threadIdx.x] = value.energy_ha;
    for (int ab = 0; ab < 9; ++ab)
        part[1 + ab][threadIdx.x] = value.virial_ha[ab];
    __syncthreads();
    for (int stride = block / 2; stride; stride /= 2) {
        if (threadIdx.x < stride)
            for (int lane = 0; lane < 10; ++lane)
                part[lane][threadIdx.x] += part[lane][threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(&output->energy_ha, part[0][0]);
        for (int ab = 0; ab < 9; ++ab)
            atomicAdd(output->virial_ha + ab, part[1 + ab][0]);
    }
}
__global__ void kinetic_kernel(FrozenReciprocalView q,
                               FrozenBandDeviceView state,
                               FrozenOperatorVirial* output) {
    FrozenOperatorVirial value{};
    const std::size_t size = std::size_t(q.count) * state.bands;
    for (std::size_t i = std::size_t(blockIdx.x) * block + threadIdx.x;
         i < size; i += std::size_t(block) * gridDim.x) {
        const int g = i % q.count, b = i / q.count;
        const double r[3] = {q.x[g], q.y[g], q.z[g]};
        const double w = state.k_weight * state.f[b] * norm2(state.C[i]);
        value.energy_ha += .5 * w * (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                value.virial_ha[3 * a + b] += w * r[a] * r[b];
    }
    reduce(value, output);
}
__global__ void hartree_kernel(FrozenReciprocalView G,
                               const cufftDoubleComplex* rho, double omega,
                               FrozenOperatorVirial* output) {
    FrozenOperatorVirial value{};
    for (int g = blockIdx.x * block + threadIdx.x; g < G.count;
         g += block * gridDim.x) {
        const double r[3] = {G.x[g], G.y[g], G.z[g]},
                     g2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        if (g2 < 1e-16)
            continue;
        const double e = 2 * pi * omega * norm2(rho[g]) / g2;
        value.energy_ha += e;
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                value.virial_ha[3 * a + b] +=
                    e * ((a == b ? 1. : 0.) - 2 * r[a] * r[b] / g2);
    }
    reduce(value, output);
}
__global__ void local_kernel(FrozenReciprocalView G,
                             const cufftDoubleComplex* rho, double omega,
                             FrozenLocalDeviceView geometry,
                             FrozenOperatorVirial* output) {
    FrozenOperatorVirial value{};
    for (int g = blockIdx.x * block + threadIdx.x; g < G.count;
         g += block * gridDim.x) {
        const double r[3] = {G.x[g], G.y[g], G.z[g]},
                     g2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        double e = 0, radial = 0;
        for (int kind = 0; kind < geometry.species_count; ++kind) {
            const auto i = std::size_t(kind) * G.count + g;
            const double overlap = omega * inner(rho[g], geometry.structure[i]);
            e += overlap * geometry.value[i];
            radial += overlap * geometry.slope[i];
        }
        value.energy_ha += e;
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                value.virial_ha[3 * a + b] +=
                    (a == b ? e : 0.)
                    + (g2 > 0 ? radial * r[a] * r[b] / sqrt(g2) : 0.);
    }
    reduce(value, output);
}
// Exact real-solid-harmonic ordering used by current scalar NC operators.
__device__ void solid(int l, int m, const double q[3], double& h,
                      double dh[3]) {
    h = 0;
    dh[0] = dh[1] = dh[2] = 0;
    const double x = q[0], y = q[1], z = q[2];
    if (l == 0) {
        h = 1 / sqrt(4 * pi);
        return;
    }
    if (l == 1) {
        const double c = sqrt(3 / (4 * pi));
        const int axis = m == 0 ? 1 : (m == 1 ? 2 : 0);
        h = c * q[axis];
        dh[axis] = c;
        return;
    }
    const double c = sqrt(15 / (4 * pi)), d = sqrt(5 / (16 * pi));
    if (m == 0) {
        h = c * x * y;
        dh[0] = c * y;
        dh[1] = c * x;
    } else if (m == 1) {
        h = c * y * z;
        dh[1] = c * z;
        dh[2] = c * y;
    } else if (m == 2) {
        h = d * (2 * z * z - x * x - y * y);
        dh[0] = -2 * d * x;
        dh[1] = -2 * d * y;
        dh[2] = 4 * d * z;
    } else if (m == 3) {
        h = c * x * z;
        dh[0] = c * z;
        dh[2] = c * x;
    } else {
        h = .5 * c * (x * x - y * y);
        dh[0] = c * x;
        dh[1] = -c * y;
    }
}
__global__ void build_operator(FrozenNonlocalDeviceGeometry geometry, int ab,
                               cufftDoubleComplex* B) {
    const std::size_t size =
        std::size_t(geometry.q.count) * geometry.column_count;
    for (std::size_t index = std::size_t(blockIdx.x) * block + threadIdx.x;
         index < size; index += std::size_t(block) * gridDim.x) {
        const int g = index % geometry.q.count, col = index / geometry.q.count;
        const auto c = geometry.columns[col];
        const double q[3] = {geometry.q.x[g], geometry.q.y[g], geometry.q.z[g]};
        double h, dh[3];
        solid(c.l, c.m, q, h, dh);
        const auto kind = geometry.species[c.species];
        const auto radial =
            (kind.radial_offset + c.projector) * geometry.q.count + g;
        double amplitude = geometry.R[radial] * h;
        if (ab >= 0) {
            const int a = ab / 3, b = ab % 3;
            amplitude = q[a]
                        * (geometry.R[radial] * dh[b]
                           + geometry.slope_over_q[radial] * q[b] * h);
        }
        const double* pos = geometry.positions + 3 * c.atom;
        const auto p =
            phase(c.l, q[0] * pos[0] + q[1] * pos[1] + q[2] * pos[2]);
        B[index] = {geometry.prefactor * amplitude * p.x,
                    geometry.prefactor * amplitude * p.y};
    }
}
__global__ void apply_D(FrozenNonlocalDeviceGeometry geometry, int bands,
                        const cufftDoubleComplex* c, cufftDoubleComplex* Dc) {
    const std::size_t size = std::size_t(geometry.column_count) * bands;
    for (std::size_t index = std::size_t(blockIdx.x) * block + threadIdx.x;
         index < size; index += std::size_t(block) * gridDim.x) {
        const int col = index % geometry.column_count,
                  b = index / geometry.column_count;
        const auto i = geometry.columns[col];
        const auto kind = geometry.species[i.species];
        const auto row = kind.coupling_offset
                         + std::size_t(i.projector) * kind.projector_count;
        cufftDoubleComplex sum{};
        for (int colj = i.block_first; colj < i.block_first + i.block_size;
             ++colj) {
            const int j = geometry.columns[colj].projector;
            const double d = geometry.D[row + j];
            const auto z = c[colj + b * geometry.column_count];
            sum.x += d * z.x;
            sum.y += d * z.y;
        }
        Dc[index] = sum;
    }
}
__global__ void nl_reduce(int columns, FrozenBandDeviceView state,
                          const cufftDoubleComplex* c,
                          const cufftDoubleComplex* Dc,
                          const cufftDoubleComplex* eta, int ab,
                          FrozenOperatorVirial* output) {
    FrozenOperatorVirial value{};
    const std::size_t size = std::size_t(columns) * state.bands;
    for (std::size_t index = std::size_t(blockIdx.x) * block + threadIdx.x;
         index < size; index += std::size_t(block) * gridDim.x) {
        const int b = index / columns;
        const double w = state.k_weight * state.f[b];
        if (ab < 0) {
            const double e = w * inner(c[index], Dc[index]);
            value.energy_ha += e;
            for (int a = 0; a < 3; ++a)
                value.virial_ha[4 * a] += e;
        } else
            value.virial_ha[ab] += 2 * w * inner(Dc[index], eta[index]);
    }
    reduce(value, output);
}
} // namespace
void kinetic_frozen_virial(FrozenReciprocalView q, FrozenBandDeviceView state,
                           FrozenOperatorVirial* output, cudaStream_t stream) {
    validate(q, state);
    const auto count = std::size_t(q.count) * state.bands;
    if (!count)
        return;
    require(output, "PW frozen kinetic: missing output");
    kinetic_kernel<<<blocks(count), block, 0, stream>>>(q, state, output);
    ATV_CUDA_CHECK(cudaGetLastError());
}
void hartree_frozen_virial(FrozenReciprocalView G,
                           const cufftDoubleComplex* rho, double omega,
                           FrozenOperatorVirial* output, cudaStream_t stream) {
    validate(G);
    validate_volume(omega);
    if (!G.count)
        return;
    require(rho && output, "PW frozen Hartree: missing density or output");
    hartree_kernel<<<blocks(G.count), block, 0, stream>>>(G, rho, omega,
                                                          output);
    ATV_CUDA_CHECK(cudaGetLastError());
}
void local_potential_frozen_virial(FrozenReciprocalView G,
                                   const cufftDoubleComplex* rho, double omega,
                                   FrozenLocalDeviceView geometry,
                                   FrozenOperatorVirial* output,
                                   cudaStream_t stream) {
    validate(G);
    validate_volume(omega);
    if (!G.count)
        return;
    require(rho && output && geometry.value && geometry.slope
                && geometry.structure && geometry.species_count > 0,
            "PW frozen local: missing density, geometry or output");
    local_kernel<<<blocks(G.count), block, 0, stream>>>(G, rho, omega, geometry,
                                                        output);
    ATV_CUDA_CHECK(cudaGetLastError());
}
void project_frozen_nonlocal(FrozenNonlocalDeviceGeometry geometry,
                             FrozenBandDeviceView state,
                             FrozenNonlocalDeviceScratch scratch,
                             cublasHandle_t blas, cudaStream_t stream) {
    validate(geometry.q, state);
    require(geometry.column_count >= 0 && geometry.species_count >= 0,
            "PW frozen nonlocal: negative projector dimensions");
    if (!geometry.q.count || !geometry.column_count || !state.bands)
        return;
    require(geometry.species_count > 0 && std::isfinite(geometry.prefactor)
                && geometry.prefactor > 0,
            "PW frozen nonlocal: invalid projector count or normalization");
    require(
        geometry.R && geometry.D && geometry.positions && geometry.columns
            && geometry.species && scratch.B && scratch.c && scratch.Dc
            && scratch.eta && blas,
        "PW frozen nonlocal: missing geometry, scratch, output or BLAS handle");
    cudaStream_t handle_stream = nullptr;
    cublasPointerMode_t pointer_mode;
    checked(cublasGetStream(blas, &handle_stream));
    checked(cublasGetPointerMode(blas, &pointer_mode));
    require(handle_stream == stream && pointer_mode == CUBLAS_POINTER_MODE_HOST,
            "PW frozen nonlocal: BLAS stream or pointer mode mismatch");
    const cuDoubleComplex one{1, 0}, zero{0, 0};
    const auto size = std::size_t(geometry.q.count) * geometry.column_count;
    build_operator<<<blocks(size), block, 0, stream>>>(geometry, -1, scratch.B);
    ATV_CUDA_CHECK(cudaGetLastError());
    auto project = [&](cufftDoubleComplex* result) {
        checked(cublasZgemm(
            blas, CUBLAS_OP_C, CUBLAS_OP_N, geometry.column_count, state.bands,
            geometry.q.count, &one, scratch.B, geometry.q.count, state.C,
            geometry.q.count, &zero, result, geometry.column_count));
    };
    project(scratch.c);
    const auto reduced = std::size_t(geometry.column_count) * state.bands;
    apply_D<<<blocks(reduced), block, 0, stream>>>(geometry, state.bands,
                                                   scratch.c, scratch.Dc);
    ATV_CUDA_CHECK(cudaGetLastError());
}
void add_projected_nonlocal_virial(FrozenNonlocalDeviceGeometry geometry,
                                   FrozenBandDeviceView state,
                                   FrozenNonlocalDeviceScratch scratch,
                                   cublasHandle_t blas,
                                   FrozenOperatorVirial* output,
                                   cudaStream_t stream) {
    if (!geometry.q.count || !geometry.column_count || !state.bands)
        return;
    require(geometry.slope_over_q && output,
            "missing nonlocal virial geometry/output");
    const cuDoubleComplex one{1, 0}, zero{0, 0};
    const auto size = std::size_t(geometry.q.count) * geometry.column_count;
    const auto reduced = std::size_t(geometry.column_count) * state.bands;
    auto project = [&](cufftDoubleComplex* result) {
        checked(cublasZgemm(
            blas, CUBLAS_OP_C, CUBLAS_OP_N, geometry.column_count, state.bands,
            geometry.q.count, &one, scratch.B, geometry.q.count, state.C,
            geometry.q.count, &zero, result, geometry.column_count));
    };
    nl_reduce<<<blocks(reduced), block, 0, stream>>>(
        geometry.column_count, state, scratch.c, scratch.Dc, nullptr, -1,
        output);
    ATV_CUDA_CHECK(cudaGetLastError());
    for (int ab = 0; ab < 9; ++ab) {
        build_operator<<<blocks(size), block, 0, stream>>>(geometry, ab,
                                                           scratch.B);
        ATV_CUDA_CHECK(cudaGetLastError());
        project(scratch.eta);
        nl_reduce<<<blocks(reduced), block, 0, stream>>>(
            geometry.column_count, state, scratch.c, scratch.Dc, scratch.eta,
            ab, output);
        ATV_CUDA_CHECK(cudaGetLastError());
    }
}
void nonlocal_frozen_virial(FrozenNonlocalDeviceGeometry geometry,
                            FrozenBandDeviceView state,
                            FrozenNonlocalDeviceScratch scratch,
                            cublasHandle_t blas, FrozenOperatorVirial* output,
                            cudaStream_t stream) {
    if (geometry.q.count > 0 && geometry.column_count > 0 && state.bands > 0)
        require(geometry.slope_over_q && output,
                "missing nonlocal virial geometry/output");
    project_frozen_nonlocal(geometry, state, scratch, blas, stream);
    add_projected_nonlocal_virial(geometry, state, scratch, blas, output,
                                  stream);
}
} // namespace ATV::cuda::pw::internal

namespace ATV::cuda::pw::internal {
namespace {
__global__ void frozen_mul_q(FrozenReciprocalView q, int axis,
                             const cufftDoubleComplex* C, int bands,
                             cufftDoubleComplex* out) {
    int i = blockIdx.x * block + threadIdx.x;
    if (i >= q.count * bands)
        return;
    const double* v = axis == 0 ? q.x : axis == 1 ? q.y : q.z;
    double scale = v[i % q.count];
    out[i] = {scale * C[i].x, scale * C[i].y};
}
__global__ void
frozen_force_overlap(const FrozenProjectorColumn* columns, int ncol, int bands,
                     const cufftDoubleComplex* Dc, const cufftDoubleComplex* s,
                     const double* f, double weight, int axis, double* out) {
    int i = blockIdx.x * block + threadIdx.x;
    if (i >= ncol * bands)
        return;
    const auto a = Dc[i], b = s[i];
    atomicAdd(out + 3 * columns[i % ncol].atom + axis,
              2 * weight * f[i / ncol] * (a.x * b.y - a.y * b.x));
}
} // namespace
void accumulate_projected_nonlocal_forces(FrozenNonlocalDeviceGeometry geometry,
                                          FrozenBandDeviceView state,
                                          FrozenNonlocalDeviceScratch scratch,
                                          cufftDoubleComplex* qC,
                                          cublasHandle_t blas, double* forces,
                                          cudaStream_t stream) {
    if (!geometry.column_count)
        return;
    const cuDoubleComplex one{1, 0}, zero{0, 0};
    for (int axis = 0; axis < 3; ++axis) {
        frozen_mul_q<<<blocks(std::size_t(geometry.q.count) * state.bands),
                       block, 0, stream>>>(geometry.q, axis, state.C,
                                           state.bands, qC);
        ATV_CUDA_CHECK(cudaGetLastError());
        checked(cublasZgemm(
            blas, CUBLAS_OP_C, CUBLAS_OP_N, geometry.column_count, state.bands,
            geometry.q.count, &one, scratch.B, geometry.q.count, qC,
            geometry.q.count, &zero, scratch.eta, geometry.column_count));
        frozen_force_overlap<<<blocks(std::size_t(geometry.column_count)
                                      * state.bands),
                               block, 0, stream>>>(
            geometry.columns, geometry.column_count, state.bands, scratch.Dc,
            scratch.eta, state.f, state.k_weight, axis, forces);
        ATV_CUDA_CHECK(cudaGetLastError());
    }
}
} // namespace ATV::cuda::pw::internal
