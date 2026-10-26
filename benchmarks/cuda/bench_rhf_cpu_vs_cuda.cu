/// @file bench_rhf_cpu_vs_cuda.cu
/// @brief Complete-call RHF walls on Cartesian STO-3G water chains.
///
/// Historical optimization source, no longer a CMake benchmark target.
/// The supported HF/oracle suite is `python -m benchmarks`.
///
/// Manual GH200 harness, never CTest. Example:
/// OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 bench_rhf_cpu_vs_cuda 16 3 both
/// Arguments: max_waters [1..30, default 16], warm_repeats [1..100, default 3],
/// cpu|cuda|both, SCF tolerance [default 1e-10], admission budget MiB [16384], cpu_eri stored|direct [stored].
/// Timers cover RHF.run and final GPU synchronization, including setup,
/// SCF, result output and call teardown. Construction, input Frame copying,
/// CPU history reset and CUDA context initialization are outside the timer.
/// The first call uses a fresh instance; warm calls reuse it with Hcore starts.
/// Device-free-memory snapshots are not peaks or residency evidence.

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
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/rhf.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/scf/rhf.cuh"

namespace {

using Clock = std::chrono::steady_clock;
constexpr int kMaxIterations = 200;
constexpr double kEnergyToleranceHa = 1e-6;
constexpr double kRepeatToleranceHa = 1e-8;
constexpr double kFrameToleranceEv = 1e-8;
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
    constexpr double y[3] = {0, 0.7572, -0.7572};
    constexpr double z[3] = {0.1173, -0.4692, -0.4692};
    for (int water = 0; water < count; ++water)
        for (int atom = 0; atom < 3; ++atom) {
            const int i = 3 * water + atom;
            frame.span(kAtom_type)[i] = atom == 0 ? 8 : 1;
            frame.span(kAtom_mass)[i] = atom == 0 ? 15.999 : 1.008;
            frame.span(kAtom_x)[i] = 4.0 * water;
            frame.span(kAtom_y)[i] = y[atom];
            frame.span(kAtom_z)[i] = z[atom];
        }
    // Same supplied molecular geometry as the MP2 standing benchmark.
    frame.box = cpu::Box::cubic(std::max(40.0, 4.0 * count + 20.0));
    return frame;
}

std::size_t planned_bytes(int nao) {
    const std::size_t n = static_cast<std::size_t>(nao);
    const std::size_t n4 = n * n * n * n;
    if (n4 > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("dense RHF requires nAO^4 <= INT_MAX");
    // Conservative admission estimate: dense AO ERIs, CPU/device work arrays,
    // matrices and library slack. This neither enforces an allocator cap nor
    // measures peak usage; use a full-process allocation trace for that.
    return 6 * n4 * sizeof(double) + 128 * n * n * sizeof(double) + 64 * kMiB;
}

void require_pin(int waters, double energy) {
    double want = 0;
    switch (waters) {
    case 1:
        // regressions/scf/rhf/sto3g_const.cpp, PySCF water pin.
        want = -74.96302313846289;
        break;
    case 8:
        // regressions/scf/mp2/water-chain.h, PySCF 2.11.0, Hcore start.
        want = -599.6949297548834;
        break;
    case 16:
        // Offline pyscf-water16.jsonl, PySCF 2.11.0, Hcore start,
        // Cartesian STO-3G, conv_tol=1e-12, 4 Angstrom water chain.
        want = -1199.388118981794;
        break;
    default:
        return;
    }
    if (std::abs(energy - want) > kEnergyToleranceHa)
        throw std::runtime_error("RHF external pin mismatch at waters="
                                 + std::to_string(waters));
}

struct Measurement {
    double cold_ms = 0;
    std::vector<double> warm_ms;
    double energy_ha = 0;
    double max_frame_error_ev = 0;
    double max_repeat_error_ha = 0;
    int cold_iterations = 0;
    int warm_iterations_min = std::numeric_limits<int>::max();
    int warm_iterations_max = 0;
    std::size_t free_before = 0;
    std::size_t free_after = 0;
    std::size_t total_memory = 0;
};

template <typename Method, typename Reset>
Measurement measure(const ATV::cpu::Frame& prototype, int waters, int repeats,
                    bool on_cuda, Method& hf, Reset&& reset) {
    Measurement measured;
    if (on_cuda) {
        ATV_CUDA_CHECK(cudaDeviceSynchronize());
        ATV_CUDA_CHECK(
            cudaMemGetInfo(&measured.free_before, &measured.total_memory));
    }
    for (int sample = 0; sample <= repeats; ++sample) {
        auto frame = prototype;
        reset();
        // A stale result must not accidentally satisfy the Frame contract.
        frame.meta.get(ATV::kEnergy) = std::numeric_limits<double>::quiet_NaN();
        const auto start = Clock::now();
        hf.run(frame);
        if (on_cuda)
            ATV_CUDA_CHECK(cudaDeviceSynchronize());
        const double elapsed =
            std::chrono::duration<double, std::milli>(Clock::now() - start)
                .count();
        const double energy = hf.energy();
        const int iterations = hf.iterations();
        const double frame_energy = frame.meta.get(ATV::kEnergy);
        if (!hf.converged() || iterations < 1 || iterations > kMaxIterations
            || !std::isfinite(energy) || !std::isfinite(frame_energy)
            || !std::isfinite(elapsed) || elapsed <= 0)
            throw std::runtime_error("nonfinite or unconverged RHF result");
        const double frame_error =
            std::abs(frame_energy - energy * ATV::units::kHartreeInEv);
        if (frame_error > kFrameToleranceEv)
            throw std::runtime_error("Frame RHF total energy mismatch");
        require_pin(waters, energy);
        measured.max_frame_error_ev =
            std::max(measured.max_frame_error_ev, frame_error);
        if (sample == 0) {
            measured.energy_ha = energy;
            measured.cold_ms = elapsed;
            measured.cold_iterations = iterations;
        } else {
            // Compare every warm solve with the first solve, avoiding drift.
            const double repeat_error = std::abs(energy - measured.energy_ha);
            if (repeat_error > kRepeatToleranceHa)
                throw std::runtime_error("repeated RHF total energy changed");
            measured.max_repeat_error_ha =
                std::max(measured.max_repeat_error_ha, repeat_error);
            measured.warm_ms.push_back(elapsed);
            measured.warm_iterations_min =
                std::min(measured.warm_iterations_min, iterations);
            measured.warm_iterations_max =
                std::max(measured.warm_iterations_max, iterations);
        }
    }
    if (on_cuda)
        ATV_CUDA_CHECK(
            cudaMemGetInfo(&measured.free_after, &measured.total_memory));
    return measured;
}

Measurement measure_cpu(const ATV::cpu::Frame& prototype, int waters,
                        int repeats, double tolerance, bool stored_eri) {
    using namespace ATV;
    cpu::scf::RHF hf{cpu::scf::BasisSet::STO3G, 0, tolerance, kMaxIterations};
    hf.with_gradient(false);
    hf.init_guess = cpu::scf::InitGuess::Hcore;
    hf.compute_eri = stored_eri;
    hf.damping_factor = 0;
    hf.level_shift = 0;
    hf.use_diis = true;
    hf.diis_start_iter = 1;
    hf.diis_max_vectors = 8;
    return measure(prototype, waters, repeats, false, hf,
                   [&] { hf.reset_history(); });
}

Measurement measure_cuda(const ATV::cpu::Frame& prototype, int waters,
                         int repeats, double tolerance) {
    using namespace ATV;
    cuda::scf::RHF hf{
        cpu::scf::BasisSet::STO3G, 0, tolerance, kMaxIterations, 0.0, false};
    return measure(prototype, waters, repeats, true, hf, [] {});
}

void report(const char* backend, int waters, Measurement measured) {
    std::sort(measured.warm_ms.begin(), measured.warm_ms.end());
    const auto count = measured.warm_ms.size();
    const double median = count % 2 ? measured.warm_ms[count / 2]
                                    : 0.5
                                          * (measured.warm_ms[count / 2 - 1]
                                             + measured.warm_ms[count / 2]);
    std::printf("backend=%s method=rhf waters=%d atoms=%d nao=%d nocc=%d "
                "cold_instance_ms=%.6f warm_instance_median_ms=%.6f "
                "warm_min_ms=%.6f warm_max_ms=%.6f warm_repeats=%zu "
                "cold_scf_iterations=%d warm_scf_iterations_min=%d "
                "warm_scf_iterations_max=%d converged=1 E_tot_Ha=%.14f "
                "max_frame_error_eV=%.3e max_repeat_error_Ha=%.3e "
                "planned_memory_bytes=%zu\n",
                backend, waters, 3 * waters, 7 * waters, 5 * waters,
                measured.cold_ms, median, measured.warm_ms.front(),
                measured.warm_ms.back(), count, measured.cold_iterations,
                measured.warm_iterations_min, measured.warm_iterations_max,
                measured.energy_ha, measured.max_frame_error_ev,
                measured.max_repeat_error_ha, planned_bytes(7 * waters));
    if (std::string(backend) == "cuda")
        std::printf("device_memory_snapshot waters=%d free_before_bytes=%zu "
                    "free_after_bytes=%zu total_bytes=%zu peak=unmeasured\n",
                    waters, measured.free_before, measured.free_after,
                    measured.total_memory);
}

const char* environment_value(const char* key) {
    const char* value = std::getenv(key);
    return value ? value : "unset";
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 7)
            throw std::invalid_argument("too many arguments");
        // 30 waters = 210 AOs; 31 already exceeds the dense int-index limit.
        const int max_waters =
            argc > 1 ? parse_integer(argv[1], 1, 30, "max_waters") : 16;
        const int repeats =
            argc > 2 ? parse_integer(argv[2], 1, 100, "warm_repeats") : 3;
        const std::string backend = argc > 3 ? argv[3] : "both";
        if (backend != "cpu" && backend != "cuda" && backend != "both")
            throw std::invalid_argument("backend must be cpu, cuda or both");
        const double tolerance = argc > 4 ? parse_tolerance(argv[4]) : 1e-10;
        const std::size_t budget =
            static_cast<std::size_t>(
                argc > 5
                    ? parse_integer(argv[5], 256, 65536, "memory_budget_MiB")
                    : 16384)
            * kMiB;
        const std::string cpu_eri = argc > 6 ? argv[6] : "stored";
        if (cpu_eri != "stored" && cpu_eri != "direct")
            throw std::invalid_argument("cpu_eri must be stored or direct");
        const bool stored_eri = cpu_eri == "stored";
        const bool use_cpu = backend != "cuda";
        const bool use_cuda = backend != "cpu";
        // Validate the largest case before doing any expensive solves.
        if (planned_bytes(7 * max_waters) > budget)
            throw std::runtime_error("planned memory exceeds admission budget");
        int threads = 1;
#if defined(_OPENMP) || defined(ATV_BENCH_HAS_OPENMP)
        threads = omp_get_max_threads();
#endif
        std::printf("basis=STO-3G method=rhf charge=0 scf_tol=%.3e max_iter=%d "
                    "level_shift=0 diis_vectors=8 diis_start=1 damping=0 "
                    "OMP_max_threads=%d OMP_NUM_THREADS=%s "
                    "OPENBLAS_NUM_THREADS=%s MKL_NUM_THREADS=%s "
                    "memory_budget_bytes=%zu cpu_eri=%s compute_eri=%d "
                    "energy_tolerance_Ha=%.3e "
                    "repeat_tolerance_Ha=%.3e frame_tolerance_eV=%.3e\n",
                    tolerance, kMaxIterations, threads,
                    environment_value("OMP_NUM_THREADS"),
                    environment_value("OPENBLAS_NUM_THREADS"),
                    environment_value("MKL_NUM_THREADS"), budget,
                    cpu_eri.c_str(), stored_eri ? 1 : 0, kEnergyToleranceHa,
                    kRepeatToleranceHa, kFrameToleranceEv);
        std::puts(
            "initial_state: cpu=Hcore_density_no_history "
            "cuda=zero_density_then_Hcore_diagonalization; "
            "SCF stopping criteria are backend-specific; "
            "compare complete walls and energies, inspect iteration counts");
        std::puts(
            "timing: cold=fresh_instance_first_run "
            "warm=same_instance_after_one_complete_run; "
            "construction_and_context_init=excluded; "
            "Frame_copy_and_history_reset=excluded; "
            "setup_SCF_final_output_teardown_and_GPU_completion=included; "
            "memory_budget_is_conservative_admission_not_allocator_cap; "
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
            if (use_cuda) {
                std::size_t free_bytes = 0, total_bytes = 0;
                ATV_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
                if (planned > free_bytes - free_bytes / 5)
                    throw std::runtime_error("planned memory exceeds 80% "
                                             "of current free device memory");
            }
            const auto prototype = water_chain(waters);
            Measurement cpu_result, cuda_result;
            if (use_cpu) {
                cpu_result = measure_cpu(prototype, waters, repeats, tolerance,
                                         stored_eri);
                report("cpu", waters, cpu_result);
            }
            if (use_cuda) {
                cuda_result =
                    measure_cuda(prototype, waters, repeats, tolerance);
                report("cuda", waters, cuda_result);
            }
            if (use_cpu && use_cuda) {
                const double error =
                    std::abs(cpu_result.energy_ha - cuda_result.energy_ha);
                const bool passed = error <= kEnergyToleranceHa;
                std::printf("comparison waters=%d abs_total_error_Ha=%.3e "
                            "tolerance_Ha=%.3e %s\n",
                            waters, error, kEnergyToleranceHa,
                            passed ? "PASS" : "FAIL");
                if (!passed)
                    return 1;
            }
            if (waters == max_waters)
                break;
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        std::fprintf(stderr, "usage: bench_rhf_cpu_vs_cuda [max_waters=16] "
                             "[warm_repeats=3] [cpu|cuda|both] "
                             "[scf_tol=1e-10] [memory_budget_MiB=16384] "
                             "[cpu_eri=stored|direct]\n");
        return 1;
    }
}
