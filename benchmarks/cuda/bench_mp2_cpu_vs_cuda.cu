/// @file bench_mp2_cpu_vs_cuda.cu
/// @brief Complete-call RHF + canonical MP2 walls on STO-3G water chains.
///
/// Manual harness, never CTest. Example (on GH200):
/// OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 bench_mp2_cpu_vs_cuda 4 3 both
/// Arguments: max_waters [1..32], warm_repeats [1..100], cpu|cuda|both,
///            SCF tolerance [default 1e-10], workspace budget MiB [default 4096].
/// The first call uses fresh method instances; subsequent calls reuse those
/// instances, resetting CPU density history. Timers cover MP2.run and CUDA
/// completion, including setup, SCF, correlation, transfers and call cleanup.
/// Construction, input Frame copying and CUDA context initialization are outside
/// the timer. Device-free-memory snapshots are neither allocation peaks nor
/// evidence of device residency. No force calculation is requested.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_OPENMP) || defined(ATV_BENCH_HAS_OPENMP)
#include <omp.h>
#endif

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/post_hf.h"
#include "atomiverse/cpu/scf/rhf.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/scf/mp2.cuh"
#include "atomiverse/cuda/scf/rhf.cuh"

namespace {

using Clock = std::chrono::steady_clock;
using Result = ATV::cpu::scf::MP2Result;
constexpr int kMaxIterations = 200;
constexpr double kEnergyToleranceHa = 1e-6;
constexpr std::size_t kMiB = std::size_t{1024} * 1024;

int parse_integer(const char* text, int low, int high, const char* label) {
    std::size_t end = 0;
    const auto value = std::stoll(text, &end);
    if (text[end] != '\0' || value < low || value > high)
        throw std::invalid_argument(std::string(label) + " outside ["
                                    + std::to_string(low) + ","
                                    + std::to_string(high) + "]");
    return static_cast<int>(value);
}

double parse_tolerance(const char* text) {
    std::size_t end = 0;
    const double value = std::stod(text, &end);
    if (text[end] != '\0' || !std::isfinite(value) || value <= 0
        || value > 1e-6)
        throw std::invalid_argument("SCF tolerance must be in (0, 1e-6]");
    return value;
}

ATV::cpu::Frame water_chain(int count) {
    using namespace ATV;
    cpu::Frame frame(3 * count);
    for (int water = 0; water < count; ++water) {
        // Parallel, well-separated molecules. This is a scaling workload;
        // the one-water geometry also carries the repository's PySCF pin.
        constexpr double y[3] = {0, 0.7572, -0.7572};
        constexpr double z[3] = {0.1173, -0.4692, -0.4692};
        for (int atom = 0; atom < 3; ++atom) {
            const int i = 3 * water + atom;
            frame.span(kAtom_type)[i] = atom == 0 ? 8 : 1;
            frame.span(kAtom_mass)[i] = atom == 0 ? 15.999 : 1.008;
            frame.span(kAtom_x)[i] = 4.0 * water;
            frame.span(kAtom_y)[i] = y[atom];
            frame.span(kAtom_z)[i] = z[atom];
        }
    }
    frame.box = cpu::Box::cubic(std::max(40.0, 4.0 * count + 20.0));
    return frame;
}

std::size_t planned_bytes(int nao) {
    // Deliberately conservative admission estimate for the current in-core
    // CPU/device AO ERI paths plus transformations, matrices and library slack.
    // This is a planning estimate, not a measured allocator high-water mark.
    const std::size_t n = static_cast<std::size_t>(nao);
    return 6 * n * n * n * n * sizeof(double) + 128 * n * n * sizeof(double)
           + 64 * kMiB;
}

void require_result(const Result& result) {
    if (!std::isfinite(result.e_tot) || !std::isfinite(result.e_corr)
        || !std::isfinite(result.e_corr_os) || !std::isfinite(result.e_corr_ss)
        || result.ri || result.frozen != 0
        || std::abs(result.e_corr - result.e_corr_os - result.e_corr_ss)
               > 1e-10)
        throw std::runtime_error(
            "invalid or inconsistent canonical MP2 result");
}

struct Measurement {
    double cold_ms = 0;
    std::vector<double> warm_ms;
    Result result{};
    int cold_iterations = 0;
    int warm_iterations_min = std::numeric_limits<int>::max();
    int warm_iterations_max = 0;
    std::size_t free_before = 0;
    std::size_t free_after = 0;
    std::size_t total_memory = 0;
};

template <typename Solve, typename Reset, typename Iterations>
Measurement measure(const ATV::cpu::Frame& prototype, int repeats, bool on_cuda,
                    Solve&& solve, Reset&& reset, Iterations&& iterations) {
    Measurement measured;
    if (on_cuda) {
        ATV_CUDA_CHECK(cudaDeviceSynchronize());
        ATV_CUDA_CHECK(
            cudaMemGetInfo(&measured.free_before, &measured.total_memory));
    }
    for (int sample = 0; sample <= repeats; ++sample) {
        auto frame = prototype;
        reset();
        const auto start = Clock::now();
        const Result result = solve(frame);
        if (on_cuda)
            ATV_CUDA_CHECK(cudaDeviceSynchronize());
        const double elapsed =
            std::chrono::duration<double, std::milli>(Clock::now() - start)
                .count();
        require_result(result);
        if (sample > 0
            && std::abs(result.e_tot - measured.result.e_tot)
                   > kEnergyToleranceHa)
            throw std::runtime_error("repeated-call total energy changed");
        measured.result = result;
        const int count = iterations();
        if (sample == 0) {
            measured.cold_ms = elapsed;
            measured.cold_iterations = count;
        } else {
            measured.warm_ms.push_back(elapsed);
            measured.warm_iterations_min =
                std::min(measured.warm_iterations_min, count);
            measured.warm_iterations_max =
                std::max(measured.warm_iterations_max, count);
        }
    }
    if (on_cuda)
        ATV_CUDA_CHECK(
            cudaMemGetInfo(&measured.free_after, &measured.total_memory));
    return measured;
}

Measurement measure_cpu(const ATV::cpu::Frame& prototype, int repeats,
                        double tolerance) {
    using namespace ATV;
    cpu::scf::RHF hf{cpu::scf::BasisSet::STO3G, 0, tolerance, kMaxIterations};
    hf.init_guess = cpu::scf::InitGuess::Hcore;
    hf.damping_factor = 0;
    hf.diis_start_iter = 1;
    hf.diis_max_vectors = 8;
    cpu::scf::MP2<cpu::scf::RHF> mp2{hf};
    return measure(
        prototype, repeats, false,
        [&](cpu::Frame& frame) { return mp2.run(frame); },
        [&] { hf.reset_history(); }, [&] { return hf.iterations(); });
}

Measurement measure_cuda(const ATV::cpu::Frame& prototype, int repeats,
                         double tolerance, std::size_t budget) {
    using namespace ATV;
    cuda::scf::RHF hf{
        cpu::scf::BasisSet::STO3G, 0, tolerance, kMaxIterations, 0.0, false};
    cuda::scf::MP2 mp2{hf, budget};
    return measure(
        prototype, repeats, true,
        [&](cpu::Frame& frame) { return mp2.run(frame); }, [] {},
        [&] { return hf.iterations(); });
}

void report(const char* backend, int waters, Measurement measured) {
    std::sort(measured.warm_ms.begin(), measured.warm_ms.end());
    const auto count = measured.warm_ms.size();
    const double median = count % 2 ? measured.warm_ms[count / 2]
                                    : 0.5
                                          * (measured.warm_ms[count / 2 - 1]
                                             + measured.warm_ms[count / 2]);
    std::printf("backend=%s waters=%d atoms=%d nao=%d nocc=%d nvirt=%d "
                "cold_instance_ms=%.6f warm_instance_median_ms=%.6f "
                "warm_min_ms=%.6f warm_max_ms=%.6f warm_repeats=%zu "
                "cold_scf_iterations=%d warm_scf_iterations_min=%d "
                "warm_scf_iterations_max=%d E_tot_Ha=%.14f E_corr_Ha=%.14f "
                "E_os_Ha=%.14f E_ss_Ha=%.14f planned_workspace_bytes=%zu\n",
                backend, waters, 3 * waters, 7 * waters, 5 * waters, 2 * waters,
                measured.cold_ms, median, measured.warm_ms.front(),
                measured.warm_ms.back(), count, measured.cold_iterations,
                measured.warm_iterations_min, measured.warm_iterations_max,
                measured.result.e_tot, measured.result.e_corr,
                measured.result.e_corr_os, measured.result.e_corr_ss,
                planned_bytes(7 * waters));
    if (std::string(backend) == "cuda")
        std::printf("device_memory_snapshot waters=%d free_before_bytes=%zu "
                    "free_after_bytes=%zu total_bytes=%zu peak=unmeasured\n",
                    waters, measured.free_before, measured.free_after,
                    measured.total_memory);
    if (waters == 1) {
        // PySCF 2.14.0, tools/pyscf_oracle/pin_scf.py mp2 --geom water
        // --basis sto-3g --conv-tol 1e-12; scf/mp2/REFERENCE.md.
        constexpr double pin_corr = -0.035545651647;
        constexpr double pin_total = -74.998568790110;
        if (std::abs(measured.result.e_corr - pin_corr) > kEnergyToleranceHa
            || std::abs(measured.result.e_tot - pin_total) > kEnergyToleranceHa)
            throw std::runtime_error(std::string(backend)
                                     + " water MP2 external pin mismatch");
    }
}

const char* environment_value(const char* key) {
    const char* value = std::getenv(key);
    return value ? value : "unset";
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 6)
            throw std::invalid_argument("too many arguments");
        const int max_waters =
            argc > 1 ? parse_integer(argv[1], 1, 32, "max_waters") : 4;
        const int repeats =
            argc > 2 ? parse_integer(argv[2], 1, 100, "warm_repeats") : 3;
        const std::string backend = argc > 3 ? argv[3] : "both";
        if (backend != "cpu" && backend != "cuda" && backend != "both")
            throw std::invalid_argument("backend must be cpu, cuda or both");
        const double tolerance = argc > 4 ? parse_tolerance(argv[4]) : 1e-10;
        const std::size_t budget =
            static_cast<std::size_t>(
                argc > 5
                    ? parse_integer(argv[5], 128, 65536, "workspace_budget_MiB")
                    : 4096)
            * kMiB;
        const bool use_cpu = backend != "cuda";
        const bool use_cuda = backend != "cpu";
        int threads = 1;
#if defined(_OPENMP) || defined(ATV_BENCH_HAS_OPENMP)
        threads = omp_get_max_threads();
#endif
        std::printf("basis=STO-3G charge=0 frozen=0 os_scale=1 ss_scale=1 "
                    "scf_tol=%.3e max_iter=%d level_shift=0 diis_vectors=8 "
                    "diis_start=1 damping=0 OMP_max_threads=%d "
                    "OMP_NUM_THREADS=%s OPENBLAS_NUM_THREADS=%s "
                    "MKL_NUM_THREADS=%s workspace_budget_bytes=%zu "
                    "energy_tolerance_Ha=%.3e\n",
                    tolerance, kMaxIterations, threads,
                    environment_value("OMP_NUM_THREADS"),
                    environment_value("OPENBLAS_NUM_THREADS"),
                    environment_value("MKL_NUM_THREADS"), budget,
                    kEnergyToleranceHa);
        std::puts(
            "initial_state: cpu=Hcore_density_no_history "
            "cuda=zero_density_then_Hcore_diagonalization; "
            "SCF stopping criteria are backend-specific; "
            "compare complete walls and energies, inspect iteration counts");
        std::puts("timing: cold=fresh_instance_first_run "
                  "warm=same_instance_after_one_complete_run; "
                  "construction_and_context_init=excluded; "
                  "Frame_copy=excluded; GPU_completion=included; "
                  "memory_snapshots_are_not_peak_or_residency_evidence");
        if (use_cuda) {
            int device = 0;
            cudaDeviceProp properties{};
            ATV_CUDA_CHECK(cudaGetDevice(&device));
            ATV_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
            ATV_CUDA_CHECK(cudaFree(nullptr));
            int runtime_version = 0;
            ATV_CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
            std::printf("device=%d name=%s compute_capability=%d.%d "
                        "runtime_version=%d\n",
                        device, properties.name, properties.major,
                        properties.minor, runtime_version);
        }
        for (int waters = 1;; waters = std::min(2 * waters, max_waters)) {
            const auto planned = planned_bytes(7 * waters);
            if (planned > budget)
                throw std::runtime_error("planned workspace exceeds requested "
                                         "budget at waters="
                                         + std::to_string(waters));
            if (use_cuda) {
                std::size_t free_bytes = 0, total_bytes = 0;
                ATV_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
                if (planned > free_bytes - free_bytes / 5)
                    throw std::runtime_error("planned workspace exceeds 80% "
                                             "of current free device memory");
            }
            const auto prototype = water_chain(waters);
            Measurement cpu_result, cuda_result;
            if (use_cpu) {
                cpu_result = measure_cpu(prototype, repeats, tolerance);
                report("cpu", waters, cpu_result);
            }
            if (use_cuda) {
                cuda_result =
                    measure_cuda(prototype, repeats, tolerance, budget);
                report("cuda", waters, cuda_result);
            }
            if (use_cpu && use_cuda) {
                const double total_error = std::abs(cpu_result.result.e_tot
                                                    - cuda_result.result.e_tot);
                const double corr_error = std::abs(cpu_result.result.e_corr
                                                   - cuda_result.result.e_corr);
                const double os_error = std::abs(
                    cpu_result.result.e_corr_os - cuda_result.result.e_corr_os);
                const double ss_error = std::abs(
                    cpu_result.result.e_corr_ss - cuda_result.result.e_corr_ss);
                const bool passed =
                    std::max({total_error, corr_error, os_error, ss_error})
                    <= kEnergyToleranceHa;
                std::printf("comparison waters=%d abs_total_error_Ha=%.3e "
                            "abs_corr_error_Ha=%.3e abs_os_error_Ha=%.3e "
                            "abs_ss_error_Ha=%.3e tolerance_Ha=%.3e %s\n",
                            waters, total_error, corr_error, os_error, ss_error,
                            kEnergyToleranceHa, passed ? "PASS" : "FAIL");
                if (!passed)
                    return 1;
            }
            if (waters == max_waters)
                break;
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        std::fprintf(stderr, "usage: bench_mp2_cpu_vs_cuda [max_waters=4] "
                             "[warm_repeats=3] [cpu|cuda|both] "
                             "[scf_tol=1e-10] [workspace_budget_MiB=4096]\n");
        return 1;
    }
}
