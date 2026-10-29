#include "atomiverse/core/units.h"
#include "atomiverse/cuda/planewave/internal/frozen_functional_virial.cuh"
#include <array>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <vector>
namespace {
namespace d = ATV::cuda::pw::internal;
void checked(cudaError_t s) {
    if (s != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(s));
}
struct Resources {
    cudaStream_t stream = nullptr;
    std::vector<void*> p;
    Resources() {
        checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }
    ~Resources() {
        if (stream)
            cudaStreamSynchronize(stream);
        for (auto* q : p)
            cudaFree(q);
        if (stream)
            cudaStreamDestroy(stream);
    }
    template <class T>
    T* alloc(std::size_t n) {
        T* q = nullptr;
        checked(cudaMalloc(&q, n * sizeof(T)));
        p.push_back(q);
        return q;
    }
};
} // namespace
TEST(CudaFrozenDerivativeCertificate,
     DiscriminatesFiniteConvertedAndAntisymmetricCandidates) {
    std::array<d::FrozenOperatorVirial, 7> terms{};
    std::array<double, 3> force{};
    cufftDoubleComplex reference{.2, 0}, current{}, orbital{};
    d::FrozenDerivativeCertificate status{};
    Resources r;
    auto* t = r.alloc<d::FrozenOperatorVirial>(7);
    auto* F = r.alloc<double>(3);
    auto* ref = r.alloc<cufftDoubleComplex>(1);
    auto* cur = r.alloc<cufftDoubleComplex>(1);
    auto* orb = r.alloc<cufftDoubleComplex>(1);
    auto* out = r.alloc<d::FrozenDerivativeCertificate>(1);
    for (int test = 0; test < 8; ++test) {
        SCOPED_TRACE(test);
        terms = {};
        force = {1, 2, 3};
        current = reference;
        orbital = reference;
        if (test == 1)
            force[1] = std::numeric_limits<double>::quiet_NaN();
        if (test == 2)
            terms[6].virial_ha[0] = std::numeric_limits<double>::quiet_NaN();
        if (test == 3)
            terms[6].virial_ha[1] = 1.1e-9;
        if (test == 4)
            force[1] = 1e308;
        if (test == 5)
            terms[6].virial_ha[0] = 1e308;
        if (test == 6)
            current.x += 1e-10;
        if (test == 7)
            terms[2].energy_ha = std::numeric_limits<double>::infinity();
        checked(cudaMemcpyAsync(t, terms.data(), sizeof(terms),
                                cudaMemcpyHostToDevice, r.stream));
        checked(cudaMemcpyAsync(F, force.data(), sizeof(force),
                                cudaMemcpyHostToDevice, r.stream));
        checked(cudaMemcpyAsync(ref, &reference, sizeof(reference),
                                cudaMemcpyHostToDevice, r.stream));
        checked(cudaMemcpyAsync(cur, &current, sizeof(current),
                                cudaMemcpyHostToDevice, r.stream));
        checked(cudaMemcpyAsync(orb, &orbital, sizeof(orbital),
                                cudaMemcpyHostToDevice, r.stream));
        d::certify_frozen_derivatives(
            t, F, ref, cur, orb, 1, 1, true, true, out,
            ATV::units::kHartreePerBohrInEvPerAngstrom,
            ATV::units::kHartreeInEv, 10, r.stream);
        checked(cudaMemcpyAsync(&status, out, sizeof(status),
                                cudaMemcpyDeviceToHost, r.stream));
        checked(cudaStreamSynchronize(r.stream));
        const bool allowed = status.finite && status.reference_unchanged
                             && std::isfinite(status.density_mismatch_squared)
                             && status.max_antisymmetric_ha <= 1e-9;
        EXPECT_EQ(allowed, test == 0);
    }
}
