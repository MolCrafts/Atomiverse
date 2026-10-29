#include "atomiverse/cuda/planewave/internal/frozen_functional_virial.cuh"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <vector>
namespace d = ATV::cuda::pw::internal;
namespace {
constexpr int nr = 65;
struct Input {
    std::vector<double> rho[2];
    std::vector<double> gradient[2][3];
    bool gga = true;
    Input() {
        rho[0].assign(nr, .2);
        rho[1].assign(nr, .1);
        for (int s = 0; s < 2; ++s)
            for (int a = 0; a < 3; ++a)
                gradient[s][a].assign(nr, (s ? -.7 : 1.) * (a + 1) * .01);
    }
};
struct Resources {
    cudaStream_t stream = nullptr;
    std::vector<void*> pointers;
    Resources() {
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)
            != cudaSuccess)
            throw std::runtime_error("domain stream");
    }
    ~Resources() {
        if (stream)
            cudaStreamSynchronize(stream);
        for (void* p : pointers)
            cudaFree(p);
        if (stream)
            cudaStreamDestroy(stream);
    }
    template <class T>
    T* allocate(std::size_t n) {
        T* p = nullptr;
        if (cudaMalloc(&p, n * sizeof(T)) != cudaSuccess)
            throw std::runtime_error("domain cold allocation");
        try {
            pointers.push_back(p);
        } catch (...) {
            cudaFree(p);
            throw;
        }
        return p;
    }
    double* upload(const std::vector<double>& v) {
        auto* p = allocate<double>(v.size());
        if (cudaMemcpyAsync(p, v.data(), v.size() * sizeof(double),
                            cudaMemcpyHostToDevice, stream)
            != cudaSuccess)
            throw std::runtime_error("domain cold upload");
        return p;
    }
};
d::FrozenSpinDomainStatus classify(Input input, bool invalid_shape = false) {
    d::FrozenSpinDomainStatus result{{-7., -8.}, -9., 99u, 77u};
    Resources resources;
    d::FrozenSpinDomainInput view{};
    view.count = nr;
    view.gga = input.gga;
    for (int s = 0; s < 2; ++s) {
        view.rho[s] = resources.upload(input.rho[s]);
        for (int a = 0; a < 3; ++a)
            if (input.gga)
                view.gradient[s][a] = resources.upload(input.gradient[s][a]);
    }
    auto* output = resources.allocate<d::FrozenSpinDomainStatus>(1);
    if (cudaMemcpyAsync(output, &result, sizeof(result), cudaMemcpyHostToDevice,
                        resources.stream)
        != cudaSuccess)
        throw std::runtime_error("domain cold sentinel upload");
    if (invalid_shape) {
        view.count = 0;
        EXPECT_THROW(
            d::classify_frozen_spin_domain(view, output, resources.stream),
            std::invalid_argument);
    } else
        d::classify_frozen_spin_domain(view, output, resources.stream);
    if (cudaMemcpyAsync(&result, output, sizeof(result), cudaMemcpyDeviceToHost,
                        resources.stream)
            != cudaSuccess
        || cudaStreamSynchronize(resources.stream) != cudaSuccess)
        throw std::runtime_error("domain result boundary");
    return result;
}
} // namespace
TEST(CudaFrozenSpinDomain, InteriorAndFiniteUnsupportedBoundaries) {
    static_assert(sizeof(d::FrozenSpinDomainStatus) == 32);
    Input interior;
    auto r = classify(interior);
    EXPECT_EQ(r.finite, 1u);
    EXPECT_EQ(r.violations, 0u);
    EXPECT_DOUBLE_EQ(r.minimum[0], .2);
    EXPECT_DOUBLE_EQ(r.minimum[1], .1);
    EXPECT_NEAR(r.max_abs_zeta, 1. / 3., 1e-15);
    interior.gga = false;
    r = classify(interior);
    EXPECT_EQ(r.finite, 1u);
    EXPECT_EQ(r.violations, 0u);
    // A literal PSD orbital density |0.1*sin(2*pi*x)|^2 has legitimate nodes.
    // The classifier rejects its regulated domain; this does not qualify that
    // domain's point derivative or numerical strain law.
    Input node;
    for (int i = 0; i < nr; ++i) {
        double x = double(i) / (nr - 1),
               s = std::sin(2 * 3.14159265358979323846 * x);
        node.rho[1][i] = .01 * s * s;
        for (int a = 0; a < 3; ++a) {
            node.gradient[0][a][i] = 0;
            node.gradient[1][a][i] =
                a == 0 ? .02 * 3.14159265358979323846
                             * std::sin(4 * 3.14159265358979323846 * x)
                       : 0;
        }
    }
    r = classify(node);
    EXPECT_EQ(r.finite, 1u);
    EXPECT_NE(r.violations & 1u, 0u);
    EXPECT_NE(r.violations & 4u, 0u);
    Input negative;
    negative.rho[1].back() = -.01;
    r = classify(negative);
    EXPECT_EQ(r.finite, 1u);
    EXPECT_NE(r.violations & 1u, 0u);
    EXPECT_DOUBLE_EQ(r.minimum[1], -.01);
    Input cutoff;
    cutoff.rho[0].back() = 4e-13;
    cutoff.rho[1].back() = 4e-13;
    r = classify(cutoff);
    EXPECT_EQ(r.finite, 1u);
    EXPECT_NE(r.violations & 2u, 0u);
    // Both spins are above their floor, yet finite precision reaches the zeta
    // endpoint. Positive density alone is insufficient for native qualification.
    Input endpoint;
    endpoint.rho[0].back() = 1e6;
    endpoint.rho[1].back() = 1e-12;
    r = classify(endpoint);
    EXPECT_EQ(r.finite, 1u);
    EXPECT_EQ(r.violations & 1u, 0u);
    EXPECT_NE(r.violations & 4u, 0u);
    Input invalid;
    invalid.rho[0].back() = std::numeric_limits<double>::quiet_NaN();
    r = classify(invalid);
    EXPECT_EQ(r.finite, 0u);
    invalid = Input{};
    invalid.gradient[1][2].back() = std::numeric_limits<double>::infinity();
    r = classify(invalid);
    EXPECT_EQ(r.finite, 0u);
    r = classify(Input{}, true);
    EXPECT_DOUBLE_EQ(r.minimum[0], -7.);
    EXPECT_DOUBLE_EQ(r.minimum[1], -8.);
    EXPECT_DOUBLE_EQ(r.max_abs_zeta, -9.);
    EXPECT_EQ(r.violations, 99u);
    EXPECT_EQ(r.finite, 77u);
}
