/// @file
/// @brief Actual C++ allocation counts for NDDO primitives and public forces.
/// Standing benchmark only: no production instrumentation and no CTest entry.
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/log.h"
#include "atomiverse/cpu/semiempirical/am1_solver.h"
#include "atomiverse/cpu/semiempirical/internal/nddo_derivative.h"

namespace allocation_count {
// These objects require no dynamic allocation, including during startup.
std::atomic<bool> enabled{false};
std::atomic<std::uint64_t> calls{0}, bytes{0};

void* allocate(std::size_t size, std::size_t alignment = 0) {
    void* result = nullptr;
    for (;;) {
        if (alignment) {
            if (::posix_memalign(&result, alignment, size ? size : 1) != 0)
                result = nullptr;
        } else {
            result = std::malloc(size ? size : 1);
        }
        if (result)
            break;
        const auto handler = std::get_new_handler();
        if (!handler)
            throw std::bad_alloc{};
        handler();
    }
    if (enabled.load(std::memory_order_relaxed)) {
        calls.fetch_add(1, std::memory_order_relaxed);
        bytes.fetch_add(size, std::memory_order_relaxed);
    }
    return result;
}

struct Measurement {
    std::uint64_t new_calls = 0, requested_bytes = 0;
};
struct Scope {
    Measurement& result;
    explicit Scope(Measurement& out) : result(out) {
        calls.store(0, std::memory_order_relaxed);
        bytes.store(0, std::memory_order_relaxed);
        enabled.store(true, std::memory_order_relaxed);
    }
    ~Scope() {
        enabled.store(false, std::memory_order_relaxed);
        result.new_calls = calls.load(std::memory_order_relaxed);
        result.requested_bytes = bytes.load(std::memory_order_relaxed);
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
};
} // namespace allocation_count

// Replace every ordinary/array/aligned throwing and nothrow new form. Matching
// deletes use free because both malloc and posix_memalign allocate these blocks.
// Calls to malloc/calloc/realloc made directly by libraries are NOT counted.
void* operator new(std::size_t n) { return allocation_count::allocate(n); }
void* operator new[](std::size_t n) { return allocation_count::allocate(n); }
void* operator new(std::size_t n, std::align_val_t a) {
    return allocation_count::allocate(n, static_cast<std::size_t>(a));
}
void* operator new[](std::size_t n, std::align_val_t a) {
    return allocation_count::allocate(n, static_cast<std::size_t>(a));
}
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
    try {
        return ::operator new(n);
    } catch (...) {
        return nullptr;
    }
}
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept {
    try {
        return ::operator new[](n);
    } catch (...) {
        return nullptr;
    }
}
void* operator new(std::size_t n, std::align_val_t a,
                   const std::nothrow_t&) noexcept {
    try {
        return ::operator new(n, a);
    } catch (...) {
        return nullptr;
    }
}
void* operator new[](std::size_t n, std::align_val_t a,
                     const std::nothrow_t&) noexcept {
    try {
        return ::operator new[](n, a);
    } catch (...) {
        return nullptr;
    }
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept {
    std::free(p);
}
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept {
    std::free(p);
}
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept {
    std::free(p);
}
void operator delete(void* p, std::align_val_t,
                     const std::nothrow_t&) noexcept {
    std::free(p);
}
void operator delete[](void* p, std::align_val_t,
                       const std::nothrow_t&) noexcept {
    std::free(p);
}

int main(int argc, char** argv) {
    using namespace ATV;
    using namespace ATV::cpu::semiempirical;
    std::FILE* output = nullptr;
    try {
        if (argc != 2)
            throw std::invalid_argument(
                "usage: count_nddo_allocations OUTPUT.json");
        for (const char* key : {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"}) {
            const auto* value = std::getenv(key);
            if (!value || std::strcmp(value, "1") != 0)
                throw std::invalid_argument(
                    "set OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1");
        }
        log_init({.level = spdlog::level::off});
        output = std::fopen(argv[1], "w");
        if (!output)
            throw std::runtime_error("cannot open allocation report");

        allocation_count::Measurement self_check;
        bool aligned = false;
        {
            allocation_count::Scope measured(self_check);
            void* p0 = ::operator new(17);
            void* p1 = ::operator new[](23);
            void* p2 = ::operator new(65, std::align_val_t{64});
            void* p3 = ::operator new[](129, std::align_val_t{64});
            void* p4 = ::operator new(19, std::nothrow);
            void* p5 = ::operator new[](29, std::nothrow);
            void* p6 = ::operator new(67, std::align_val_t{64}, std::nothrow);
            void* p7 =
                ::operator new[](131, std::align_val_t{64}, std::nothrow);
            aligned = reinterpret_cast<std::uintptr_t>(p2) % 64 == 0
                      && reinterpret_cast<std::uintptr_t>(p3) % 64 == 0 && p4
                      && p5 && p6 && p7
                      && reinterpret_cast<std::uintptr_t>(p6) % 64 == 0
                      && reinterpret_cast<std::uintptr_t>(p7) % 64 == 0;
            ::operator delete(p0, std::size_t{17});
            ::operator delete[](p1, std::size_t{23});
            ::operator delete(p2, std::size_t{65}, std::align_val_t{64});
            ::operator delete[](p3, std::size_t{129}, std::align_val_t{64});
            ::operator delete(p4, std::nothrow);
            ::operator delete[](p5, std::nothrow);
            ::operator delete(p6, std::align_val_t{64}, std::nothrow);
            ::operator delete[](p7, std::align_val_t{64}, std::nothrow);
        }
        if (self_check.new_calls != 8 || self_check.requested_bytes != 480
            || !aligned)
            throw std::runtime_error("allocation counter self-check failed");
        std::fprintf(
            output,
            "{\n  \"scope\": \"successful C++ new/new[] calls including "
            "aligned; direct malloc/calloc/realloc excluded\",\n"
            "  \"bytes\": \"requested allocation bytes, not live/peak "
            "memory\",\n"
            "  \"threads\": 1, \"max_iter\": 1000, \"density_tol\": 1e-10, "
            "\"energy_tol_ev\": 1e-12,\n"
            "  \"counter_self_check\": {\"new_calls\": %llu, "
            "\"requested_bytes\": %llu},\n  \"primitive_rows\": [\n",
            static_cast<unsigned long long>(self_check.new_calls),
            static_cast<unsigned long long>(self_check.requested_bytes));

        bool first = true;
        for (const auto zs :
             {std::array{1, 1}, std::array{6, 6}, std::array{35, 53}}) {
            NDDOSystem pair;
            const auto* pi = ATV::semiempirical::nddo_parameter(
                ATV::semiempirical::SemiEmpiricalMethod::AM1, zs[0]);
            const auto* pj = ATV::semiempirical::nddo_parameter(
                ATV::semiempirical::SemiEmpiricalMethod::AM1, zs[1]);
            if (!pi || !pj)
                throw std::runtime_error("benchmark AM1 parameter unavailable");
            pair.atoms = {{zs[1], 0, pj->n_orbitals - 1, pj->n_orbitals,
                           pj->core_charge, pj},
                          {zs[0], pj->n_orbitals,
                           pj->n_orbitals + pi->n_orbitals - 1, pi->n_orbitals,
                           pi->core_charge, pi}};
            pair.n_orbitals = pi->n_orbitals + pj->n_orbitals;
            for (const auto pos :
                 {std::array{0.73, -0.38, 1.17}, std::array{0.0, 0.0, 1.43}}) {
                const std::array<StorageReal, 2> x{0, pos[0]}, y{0, pos[1]},
                    z{0, pos[2]};
                constexpr int repeats = 1000;
                for (int kernel = 0; kernel < 2; ++kernel) {
                    volatile double checksum = 0;
                    // Warm exactly the same callable, outside the count interval.
                    const auto evaluate = [&] {
                        if (kernel == 0) {
                            const auto derivative =
                                ATV::cpu::semiempirical::internal::
                                    hcore_pair_derivative(pair, pos, {0, 0, 0});
                            checksum = checksum
                                       + derivative[2].nuclear_repulsion_ev
                                       + derivative[2].packed_hcore[0];
                        } else {
                            const auto derivative = ATV::cpu::semiempirical::
                                internal::two_center_integral_derivative(
                                    pair, x.data(), y.data(), z.data(), 1, 0);
                            checksum = checksum + derivative[2][0];
                        }
                    };
                    evaluate();
                    checksum = 0;
                    allocation_count::Measurement counts;
                    {
                        allocation_count::Scope measured(counts);
                        for (int repeat = 0; repeat < repeats; ++repeat)
                            evaluate();
                    }
                    if (!std::isfinite(checksum))
                        throw std::runtime_error(
                            "non-finite primitive checksum");
                    std::fprintf(
                        output,
                        "%s    {\"kernel\": \"%s\", \"z_i\": %d, \"z_j\": %d, "
                        "\"position\": [%.17g,%.17g,%.17g], \"calls\": %d, "
                        "\"new_calls\": %llu, \"requested_bytes\": %llu, "
                        "\"checksum\": %.17g}",
                        first ? "" : ",\n",
                        kernel == 0 ? "hcore_pair_derivative"
                                    : "two_center_integral_derivative",
                        zs[0], zs[1], pos[0], pos[1], pos[2], repeats,
                        static_cast<unsigned long long>(counts.new_calls),
                        static_cast<unsigned long long>(counts.requested_bytes),
                        static_cast<double>(checksum));
                    first = false;
                }
            }
        }
        std::fprintf(output, "\n  ],\n  \"public_forces_rows\": [\n");
        first = true;
        for (int waters : {1, 4, 8, 16}) {
            // Same geometry and SCF controls as run.py. This measures the native
            // C++ Frame boundary; Python conversion/allocation is not included.
            cpu::Frame frame(3 * waters);
            frame.box.set_cell(Mat3::cubic(100.0));
            constexpr double monomer[3][3] = {
                {0, 0, 0}, {0.96, 0, 0}, {-0.24, 0.93, 0.1}};
            const std::array coordinates{
                frame.span(kAtom_x), frame.span(kAtom_y), frame.span(kAtom_z)};
            for (int w = 0; w < waters; ++w) {
                const double offset[3] = {4.0 * (w % 4), 4.0 * ((w / 4) % 4),
                                          4.0 * (w / 16)};
                for (int a = 0; a < 3; ++a) {
                    const int i = 3 * w + a;
                    frame.span(kAtom_type)[i] = a == 0 ? 8 : 1;
                    frame.span(kAtom_mass)[i] = a == 0 ? 15.999 : 1.008;
                    for (int d = 0; d < 3; ++d)
                        coordinates[d][i] = monomer[a][d] + offset[d];
                }
            }
            AM1 method{0, 1, 1000, 1e-10, 1e-12};
            for (int warm = 0; warm < 5; ++warm)
                method.forces(frame);
            for (int repeat = 0; repeat < 20; ++repeat) {
                allocation_count::Measurement counts;
                {
                    allocation_count::Scope measured(counts);
                    method.forces(frame);
                }
                const auto& result = method.last_result();
                double force_checksum = 0;
                for (const double force : result.forces_ev_a) {
                    if (!std::isfinite(force))
                        throw std::runtime_error("non-finite force");
                    force_checksum += force * force;
                }
                if (!result.converged || !std::isfinite(result.total_energy_ev))
                    throw std::runtime_error("invalid public force result");
                std::fprintf(
                    output,
                    "%s    {\"atoms\": %d, \"warm_call\": %d, \"calls\": 1, "
                    "\"iterations\": %d, \"new_calls\": %llu, "
                    "\"requested_bytes\": %llu, \"energy_ev\": %.17g, "
                    "\"force_squared_sum\": %.17g}",
                    first ? "" : ",\n", 3 * waters, repeat, result.iterations,
                    static_cast<unsigned long long>(counts.new_calls),
                    static_cast<unsigned long long>(counts.requested_bytes),
                    result.total_energy_ev, force_checksum);
                first = false;
            }
        }
        std::fprintf(output, "\n  ]\n}\n");
        const int write_failed = std::ferror(output);
        const int close_failed = std::fclose(output);
        output = nullptr;
        if (write_failed || close_failed)
            throw std::runtime_error("allocation report write failed");
        return 0;
    } catch (const std::exception& error) {
        allocation_count::enabled.store(false, std::memory_order_relaxed);
        if (output)
            std::fclose(output);
        std::fprintf(stderr, "allocation benchmark failed: %s\n", error.what());
        return 1;
    }
}
