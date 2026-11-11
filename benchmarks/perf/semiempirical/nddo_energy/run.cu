/// Standing native benchmark, deliberately outside CTest. No production probes.
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/log.h"
#include "atomiverse/cpu/semiempirical/am1_solver.h"
#include "atomiverse/cpu/semiempirical/rm1_solver.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/evaluate/evaluate.cuh"
#include "atomiverse/cuda/semiempirical/am1_solver.cuh"
#include "atomiverse/cuda/semiempirical/rm1_solver.cuh"

#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace ATV;
using Clock = std::chrono::steady_clock;
using Result = ATV::semiempirical::NDDOResult;
constexpr int cold_calls = 5, nested_warmup_calls = 5, max_iter = 1000;
constexpr double density_tol = 1e-10, energy_tol_ev = 1e-12;
constexpr double compare_energy_ev = 1e-6, compare_charge_e = 1e-7;

struct Sample {
    double seconds = 0;
    int iterations = 0;
    // Populated only outside the timed interval; empty for resident steps.
    Result result;
};
struct Series {
    std::string method, backend, scope;
    int waters = 0;
    std::vector<Sample> cold, warm;
    Result final_result;
};

cpu::Frame water_cluster(int waters) {
    // Same distorted monomers and spacing as nddo_forces/run.py. Default Box
    // is isolated on BOTH backends; CUDA molecular NDDO rejects periodic cells.
    cpu::Frame frame(3 * waters);
    constexpr double xyz[3][3] = {{0, 0, 0}, {.96, 0, 0}, {-.24, .93, .1}};
    constexpr int species[3] = {8, 1, 1};
    constexpr double masses[3] = {15.999, 1.008, 1.008};
    for (int w = 0; w < waters; ++w)
        for (int a = 0; a < 3; ++a) {
            const int i = 3 * w + a;
            frame.span(kAtom_atomic_number)[i] = species[a];
            frame.span(kAtom_mass)[i] = masses[a];
            frame.span(kAtom_x)[i] = xyz[a][0] + 4 * (w % 4);
            frame.span(kAtom_y)[i] = xyz[a][1] + 4 * ((w / 4) % 4);
            frame.span(kAtom_z)[i] = xyz[a][2] + 4 * (w / 16);
        }
    return frame;
}

void check_result(const Result& result, const cpu::Frame& frame) {
    if (!result.converged || result.iterations <= 0
        || !std::isfinite(result.total_energy_ev)
        || result.mulliken_charges.size() != std::size_t(frame.nlocal()))
        throw std::runtime_error("invalid or unconverged benchmark result");
    if (!std::isfinite(frame.meta.get(kEnergy))
        || std::abs(frame.meta.get(kEnergy) - result.total_energy_ev) > 1e-9)
        throw std::runtime_error("Frame/result energy publication mismatch");
    for (int i = 0; i < frame.nlocal(); ++i)
        if (!std::isfinite(result.mulliken_charges[i])
            || !std::isfinite(frame.span(kAtom_charge)[i])
            || std::abs(frame.span(kAtom_charge)[i]
                        - result.mulliken_charges[i])
                   > 1e-12)
            throw std::runtime_error(
                "Frame/result charge publication mismatch");
}

template <class Method>
Series standalone(const char* name, const char* backend, int waters,
                  int repeats) {
    Series series{name, backend, "host_frame_energy_charge", waters};
    series.cold.resize(cold_calls);
    series.warm.resize(repeats);
    auto frame = water_cluster(waters);
    std::unique_ptr<Method> method;
    for (auto& sample : series.cold) {
        method.reset(); // Destruction and Frame construction are not timed.
        const auto start = Clock::now();
        method = std::make_unique<Method>(0, 1, max_iter, density_tol,
                                          energy_tol_ev);
        const auto& result = method->run(frame);
        sample.seconds =
            std::chrono::duration<double>(Clock::now() - start).count();
        sample.iterations = result.iterations;
        check_result(result, frame);
        sample.result = result;
    }
    // Match the force harness: reuse the last cold method and accepted density.
    for (auto& sample : series.warm) {
        const auto start = Clock::now();
        const auto& result = method->run(frame);
        sample.seconds =
            std::chrono::duration<double>(Clock::now() - start).count();
        sample.iterations = result.iterations;
        check_result(result, frame);
        sample.result = result;
    }
    series.final_result = series.warm.back().result;
    return series;
}

// Select the energy/charge timing boundary; default Evaluate also requests
// forces, whose cost belongs in a separate full-force measurement.
class EnergyIntent final : public cuda::Component {
  public:
    std::string name() const override { return "benchmark_energy_intent"; }
    void launch(const cuda::LaunchContext& ctx) override {
        ctx.runtime.exec.forces_requested = false;
        ctx.runtime.exec.stress_requested = false;
        ctx.runtime.exec.initializing_forces = false;
        ctx.runtime.exec.trial_evaluation = false;
    }
};
class Iterations final : public cuda::Component {
  public:
    int count = 0;
    std::string name() const override { return "benchmark_iteration_count"; }
    void launch(const cuda::LaunchContext&) override { ++count; }
};

// Benchmark-only composition of the existing preset lifecycle. The production
// Evaluate API remains one pass. Setup, warm-up, teardown and result downloads
// are outside the captured region; each step still runs the whole Evaluate Plan.
class ResidentEvaluate final : public cuda::Evaluate {
  public:
    void measure(cpu::Frame& frame, Iterations& iterations,
                 std::vector<Sample>& samples, bool profile) {
        bool capturing = false;
        try {
            setup_pass(frame);
            for (int i = 0; i < nested_warmup_calls; ++i)
                step_once(i);
            const auto stream =
                static_cast<cudaStream_t>(this->backend_->context().stream);
            ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
            if (profile) {
                ATV_CUDA_CHECK(cudaProfilerStart());
                capturing = true;
            }
            for (std::size_t i = 0; i < samples.size(); ++i) {
                const int before = iterations.count;
                const auto start = Clock::now();
                step_once(nested_warmup_calls + i);
                // All components in this benchmark use the parent's main stream.
                ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
                samples[i].seconds =
                    std::chrono::duration<double>(Clock::now() - start).count();
                samples[i].iterations = iterations.count - before;
                if (samples[i].iterations <= 0)
                    throw std::runtime_error(
                        "resident solve executed no SCF checks");
            }
            if (capturing) {
                ATV_CUDA_CHECK(cudaProfilerStop());
                capturing = false;
            }
            teardown_pass(frame);
        } catch (...) {
            if (capturing)
                (void)cudaProfilerStop();
            // Release the parent's acquired components before the borrowed
            // method is destroyed. Preserve the original benchmark failure.
            try {
                this->backend_->shutdown();
            } catch (...) {
            }
            throw;
        }
    }
};

template <class Method>
Series resident(const char* name, int waters, int repeats, bool profile) {
    Series series{name, "cuda", "resident_evaluate_energy_intent", waters};
    series.warm.resize(
        repeats); // No benchmark storage allocation in the range.
    auto frame = water_cluster(waters);
    Iterations iterations;
    Method method(0, 1, max_iter, density_tol, energy_tol_ev);
    method.on(nddo::Phase::check).add_ref(iterations, 100);
    EnergyIntent intent;
    ResidentEvaluate outer;
    outer.on(evaluate::Phase::prepare).add_ref(intent);
    outer.on(evaluate::Phase::evaluate).add_ref(method);
    outer.build();
    outer.measure(frame, iterations, series.warm, profile);
    // This explicit read can copy charges. Never put it inside the resident range.
    series.final_result = method.last_result();
    check_result(series.final_result, frame);
    if (series.final_result.iterations != series.warm.back().iterations)
        throw std::runtime_error(
            "SCF check probe and final iteration count disagree");
    return series;
}

void compare(const Series& cpu, const Series& other) {
    const auto& a = cpu.final_result;
    const auto& b = other.final_result;
    if (std::abs(a.total_energy_ev - b.total_energy_ev) > compare_energy_ev)
        throw std::runtime_error(cpu.method
                                 + " CPU/CUDA energy tolerance exceeded");
    for (std::size_t i = 0; i < a.mulliken_charges.size(); ++i)
        if (std::abs(a.mulliken_charges[i] - b.mulliken_charges[i])
            > compare_charge_e)
            throw std::runtime_error(cpu.method
                                     + " CPU/CUDA charge tolerance exceeded");
}

template <class Cpu, class Gpu>
void collect(std::vector<Series>& out, const char* name, int waters,
             int repeats, bool profile) {
    auto cpu = standalone<Cpu>(name, "cpu", waters, repeats);
    auto gpu = standalone<Gpu>(name, "cuda", waters, repeats);
    auto nested = resident<Gpu>(name, waters, repeats, profile);
    compare(cpu, gpu);
    compare(cpu, nested);
    out.push_back(std::move(cpu));
    out.push_back(std::move(gpu));
    out.push_back(std::move(nested));
}

int integer(const char* text) {
    int value = 0;
    const auto end = text + std::strlen(text);
    const auto parsed = std::from_chars(text, end, value);
    if (parsed.ec != std::errc{} || parsed.ptr != end)
        throw std::invalid_argument("expected integer benchmark argument");
    return value;
}
void write_result(std::ostream& out, const Result& result) {
    out << "{\"energy_ev\":" << result.total_energy_ev << ",\"charges_e\":[";
    for (std::size_t i = 0; i < result.mulliken_charges.size(); ++i)
        out << (i ? "," : "") << result.mulliken_charges[i];
    out << "]}";
}
void write_samples(std::ostream& out, const std::vector<Sample>& samples) {
    out << '[';
    for (std::size_t i = 0; i < samples.size(); ++i) {
        const auto& sample = samples[i];
        out << (i ? "," : "") << "{\"seconds\":" << sample.seconds
            << ",\"iterations\":" << sample.iterations;
        if (sample.result.converged) {
            out << ",\"result\":";
            write_result(out, sample.result);
        }
        out << '}';
    }
    out << ']';
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 6)
            throw std::invalid_argument(
                "usage: bench_nddo_energy OUTPUT.json [am1|rm1|both] "
                "[waters:0|1|4|8|16] [warm:1..100] [profile:0|1]");
        const std::string method = argc > 2 ? argv[2] : "both";
        const int selected_waters = argc > 3 ? integer(argv[3]) : 0;
        const int repeats = argc > 4 ? integer(argv[4]) : 20;
        const int profile = argc > 5 ? integer(argv[5]) : 0;
        if ((method != "am1" && method != "rm1" && method != "both")
            || (selected_waters != 0 && selected_waters != 1
                && selected_waters != 4 && selected_waters != 8
                && selected_waters != 16)
            || repeats < 1 || repeats > 100 || profile < 0 || profile > 1
            || (profile && (method == "both" || selected_waters == 0)))
            throw std::invalid_argument(
                "invalid or unbounded benchmark selection");
        if (sizeof(StorageReal) != sizeof(double)
            || sizeof(WorkReal) != sizeof(double)
            || cuda::kPrecisionMode != cuda::PrecisionMode::Double)
            throw std::invalid_argument(
                "this benchmark requires FP64 storage/work/CUDA");
        for (const char* key : {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"}) {
            const char* value = std::getenv(key);
            if (!value || std::strcmp(value, "1") != 0)
                throw std::invalid_argument(
                    "set OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1");
        }
        log_init({.level = spdlog::level::off});
        // Cold refers to the method, not process/CUDA-context startup. Record
        // the latter separately and initialize it before any timed method call.
        const auto context_start = Clock::now();
        ATV_CUDA_CHECK(cudaFree(nullptr));
        const double context_seconds =
            std::chrono::duration<double>(Clock::now() - context_start).count();
        int device = 0, runtime_version = 0, driver_version = 0;
        cudaDeviceProp properties{};
        ATV_CUDA_CHECK(cudaGetDevice(&device));
        ATV_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        ATV_CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
        ATV_CUDA_CHECK(cudaDriverGetVersion(&driver_version));
        std::vector<Series> series;
        series.reserve(24);
        for (int waters : {1, 4, 8, 16}) {
            if (selected_waters && selected_waters != waters)
                continue;
            if (method != "rm1")
                collect<cpu::semiempirical::AM1, cuda::semiempirical::AM1>(
                    series, "am1", waters, repeats, profile);
            if (method != "am1")
                collect<cpu::semiempirical::RM1, cuda::semiempirical::RM1>(
                    series, "rm1", waters, repeats, profile);
        }
        // Reporting does not run in any timed or captured interval.
        std::ofstream out(argv[1]);
        out.exceptions(std::ios::badbit | std::ios::failbit);
        out << std::setprecision(17)
            << "{\n\"schema\":1,\"precision\":\"fp64\",\"boundary\":"
               "\"isolated\","
            << "\"threads\":1,\"cold_calls\":" << cold_calls
            << ",\"nested_warmup_calls\":" << nested_warmup_calls
            << ",\"max_iter\":" << max_iter
            << ",\"density_tol\":" << density_tol
            << ",\"energy_tol_ev\":" << energy_tol_ev
            << ",\"compare_energy_tol_ev\":" << compare_energy_ev
            << ",\"compare_charge_tol_e\":" << compare_charge_e
            << ",\"profile_enabled\":" << (profile ? "true" : "false")
            << ",\"cuda_context_init_seconds\":" << context_seconds
            << ",\"gpu\":" << std::quoted(properties.name)
            << ",\"gpu_total_memory_bytes\":" << properties.totalGlobalMem
            << ",\"cuda_runtime_version\":" << runtime_version
            << ",\"cuda_driver_version\":" << driver_version
            << ",\n\"series\":[\n";
        for (std::size_t i = 0; i < series.size(); ++i) {
            const auto& s = series[i];
            const std::size_t nao = 6 * s.waters;
            out << (i ? ",\n" : "") << "{\"method\":" << std::quoted(s.method)
                << ",\"backend\":" << std::quoted(s.backend)
                << ",\"scope\":" << std::quoted(s.scope)
                << ",\"atoms\":" << 3 * s.waters << ",\"orbitals\":" << nao;
            if (s.backend == "cuda")
                out << ",\"six_dense_matrix_bytes_lower_bound\":"
                    << 6 * nao * nao * sizeof(double);
            out << ",\"cold\":";
            write_samples(out, s.cold);
            out << ",\"warm\":";
            write_samples(out, s.warm);
            out << ",\"final_result\":";
            write_result(out, s.final_result);
            out << '}';
        }
        out << "\n]}\n";
        out.close();
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "NDDO energy benchmark failed: %s\n",
                     error.what());
        return 1;
    }
}
