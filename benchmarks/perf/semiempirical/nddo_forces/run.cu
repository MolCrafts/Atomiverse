/// Standing native benchmark, deliberately outside CTest. No production probes.
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/log.h"
#include "atomiverse/cpu/semiempirical/am1_solver.h"
#include "atomiverse/cpu/semiempirical/mndo_solver.h"
#include "atomiverse/cpu/semiempirical/pm3_solver.h"
#include "atomiverse/cpu/semiempirical/rm1_solver.h"
#include "atomiverse/cuda/device/error.cuh"
#include "atomiverse/cuda/evaluate/evaluate.cuh"
#include "atomiverse/cuda/semiempirical/am1_solver.cuh"
#include "atomiverse/cuda/semiempirical/mndo_solver.cuh"
#include "atomiverse/cuda/semiempirical/pm3_solver.cuh"
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
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <sys/utsname.h>
#include <vector>

namespace {
using namespace ATV;
using Clock = std::chrono::steady_clock;
using Result = ATV::semiempirical::NDDOResult;
constexpr int cold_calls = 5, nested_warmup_calls = 5, max_iter = 1000;
constexpr double density_tol = 1e-10, energy_tol_ev = 1e-12;
constexpr double compare_energy_ev = 1e-6, compare_charge_e = 1e-7,
                 compare_force_ev_a = 1e-5;

struct Sample {
    double seconds = 0;
    int iterations = 0;
    // Populated only outside the timed interval; empty for resident steps.
    Result result;
};
struct Series {
    std::string method, backend, scope, geometry_mode;
    int waters = 0;
    std::vector<Sample> cold, warm;
    Result final_result;
    std::size_t device_free_before = 0, device_free_live = 0,
                device_free_after = 0;
};

std::size_t free_device_memory() {
    std::size_t free = 0, total = 0;
    ATV_CUDA_CHECK(cudaMemGetInfo(&free, &total));
    return free;
}

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
            frame.span(kAtom_tag)[i] = i + 1;
            frame.span(kAtom_mass)[i] = masses[a];
            frame.span(kAtom_x)[i] = xyz[a][0] + 4 * (w % 4);
            frame.span(kAtom_y)[i] = xyz[a][1] + 4 * ((w / 4) % 4);
            frame.span(kAtom_z)[i] = xyz[a][2] + 4 * (w / 16);
        }
    return frame;
}

void update_geometry(cpu::Frame& frame, int step) {
    for (int w = 0; w < frame.nlocal() / 3; ++w) {
        frame.span(kAtom_x)[3 * w + 1] =
            .96 + 4 * (w % 4) + .002 * std::sin(.2 * step);
        frame.span(kAtom_y)[3 * w + 2] =
            .93 + 4 * ((w / 4) % 4) + .002 * std::sin(.17 * step);
        frame.span(kAtom_z)[3 * w + 2] =
            .1 + 4 * (w / 16) + .001 * std::sin(.11 * step);
    }
}

__global__ void moving_geometry(cuda::Frame frame, int waters, int step) {
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= waters)
        return;
    frame.atoms.x[3 * w + 1] = .96 + 4 * (w % 4) + .002 * sin(.2 * step);
    frame.atoms.y[3 * w + 2] = .93 + 4 * ((w / 4) % 4) + .002 * sin(.17 * step);
    frame.atoms.z[3 * w + 2] = .1 + 4 * (w / 16) + .001 * sin(.11 * step);
}

// Benchmark-owned device geometry sequence. Only scalar step/count kernel
// arguments cross the launch boundary; no coordinate array upload is involved.
class MovingGeometry final : public cuda::Component {
  public:
    int step = 0;
    std::string name() const override { return "benchmark_moving_geometry"; }
    void launch(const cuda::LaunchContext& ctx) override {
        // Step zero writes the original geometry while warming the measured
        // kernel and its device trigonometric constants before capture starts.
        const int waters = ctx.frame.n_atoms() / 3;
        moving_geometry<<<(waters + 31) / 32, 32, 0,
                          static_cast<cudaStream_t>(ctx.runtime.stream)>>>(
            ctx.frame, waters, step);
        ATV_CUDA_CHECK(cudaGetLastError());
    }
};

void check_result(const Result& result, const cpu::Frame& frame) {
    if (!result.converged || result.iterations <= 0
        || !std::isfinite(result.total_energy_ev)
        || result.mulliken_charges.size() != std::size_t(frame.nlocal())
        || result.forces_ev_a.size() != 3 * std::size_t(frame.nlocal()))
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
    for (int i = 0; i < frame.nlocal(); ++i) {
        const double force[] = {frame.span(kAtom_fx)[i],
                                frame.span(kAtom_fy)[i],
                                frame.span(kAtom_fz)[i]};
        for (int d = 0; d < 3; ++d)
            if (!std::isfinite(force[d])
                || !std::isfinite(result.forces_ev_a[3 * i + d])
                || std::abs(force[d] - result.forces_ev_a[3 * i + d]) > 1e-7)
                throw std::runtime_error(
                    "Frame/result force publication mismatch");
    }
}

template <class Method>
Series standalone(const char* name, const char* backend, int waters,
                  int repeats, const std::string& mode) {
    Series series{name, backend, "host_frame_energy_charge_force", mode,
                  waters};
    series.cold.resize(cold_calls);
    series.warm.resize(repeats);
    auto frame = water_cluster(waters);
    const bool gpu = std::strcmp(backend, "cuda") == 0;
    if (gpu)
        series.device_free_before = free_device_memory();
    std::unique_ptr<Method> method;
    for (auto& sample : series.cold) {
        method.reset(); // Destruction and Frame construction are not timed.
        const auto start = Clock::now();
        method = std::make_unique<Method>(0, 1, max_iter, density_tol,
                                          energy_tol_ev);
        const auto& result = method->forces(frame);
        sample.seconds =
            std::chrono::duration<double>(Clock::now() - start).count();
        sample.iterations = result.iterations;
        check_result(result, frame);
        sample.result = result;
    }
    // Each geometry mode starts from its own fresh method and identical warm-up.
    method.reset();
    method =
        std::make_unique<Method>(0, 1, max_iter, density_tol, energy_tol_ev);
    for (int i = 0; i < nested_warmup_calls; ++i)
        method->forces(frame);
    int geometry_step = 0;
    for (auto& sample : series.warm) {
        if (mode == "moving")
            update_geometry(frame, ++geometry_step);
        const auto start = Clock::now();
        const auto& result = method->forces(frame);
        sample.seconds =
            std::chrono::duration<double>(Clock::now() - start).count();
        sample.iterations = result.iterations;
        check_result(result, frame);
        sample.result = result;
    }
    series.final_result = series.warm.back().result;
    if (gpu)
        series.device_free_live = free_device_memory();
    method.reset();
    if (gpu)
        series.device_free_after = free_device_memory();
    return series;
}

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
    void measure(cpu::Frame& frame, Iterations& iterations, Series& series,
                 bool profile, MovingGeometry& geometry) {
        auto& samples = series.warm;
        bool capturing = false;
        try {
            setup_pass(frame);
            for (int i = 0; i < nested_warmup_calls; ++i)
                step_once(i);
            const auto stream =
                static_cast<cudaStream_t>(this->backend_->context().stream);
            ATV_CUDA_CHECK(cudaStreamSynchronize(stream));
            series.device_free_live = free_device_memory();
            if (profile) {
                ATV_CUDA_CHECK(cudaProfilerStart());
                capturing = true;
            }
            for (std::size_t i = 0; i < samples.size(); ++i) {
                geometry.step = static_cast<int>(i + 1);
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
Series resident(const char* name, int waters, int repeats, bool profile,
                const std::string& mode) {
    Series series{name, "cuda", "resident_evaluate_force", mode, waters};
    series.warm.resize(
        repeats); // No benchmark storage allocation in the range.
    auto frame = water_cluster(waters);
    series.device_free_before = free_device_memory();
    Iterations iterations;
    MovingGeometry geometry;
    Method method(0, 1, max_iter, density_tol, energy_tol_ev);
    method.on(nddo::Phase::check).add_ref(iterations, 100);
    ResidentEvaluate outer;
    if (mode == "moving")
        outer.on(evaluate::Phase::prepare).add_ref(geometry);
    outer.on(evaluate::Phase::evaluate).add_ref(method);
    outer.build();
    outer.measure(frame, iterations, series, profile, geometry);
    // Charges and full forces are downloaded only after the resident range.
    series.final_result = method.last_result();
    check_result(series.final_result, frame);
    if (series.final_result.iterations != series.warm.back().iterations)
        throw std::runtime_error(
            "SCF check probe and final iteration count disagree");
    method.shutdown();
    series.device_free_after = free_device_memory();
    return series;
}

void compare(const Result& a, const Result& b, const std::string& label) {
    if (a.mulliken_charges.size() != b.mulliken_charges.size()
        || a.forces_ev_a.size() != b.forces_ev_a.size())
        throw std::runtime_error(label + " CPU/CUDA result size mismatch");
    if (std::abs(a.total_energy_ev - b.total_energy_ev) > compare_energy_ev)
        throw std::runtime_error(label + " CPU/CUDA energy tolerance exceeded");
    for (std::size_t i = 0; i < a.mulliken_charges.size(); ++i)
        if (std::abs(a.mulliken_charges[i] - b.mulliken_charges[i])
            > compare_charge_e)
            throw std::runtime_error(label
                                     + " CPU/CUDA charge tolerance exceeded");
    for (std::size_t i = 0; i < a.forces_ev_a.size(); ++i)
        if (std::abs(a.forces_ev_a[i] - b.forces_ev_a[i]) > compare_force_ev_a)
            throw std::runtime_error(label
                                     + " CPU/CUDA force tolerance exceeded");
}

template <class Cpu, class Gpu>
void collect(std::vector<Series>& out, const char* name, int waters,
             int repeats, bool profile, const std::string& mode) {
    auto cpu = standalone<Cpu>(name, "cpu", waters, repeats, mode);
    auto gpu = standalone<Gpu>(name, "cuda", waters, repeats, mode);
    auto nested = resident<Gpu>(name, waters, repeats, profile, mode);
    compare(cpu.final_result, gpu.final_result, name);
    compare(cpu.final_result, nested.final_result, name);
    for (std::size_t i = 0; i < cpu.cold.size(); ++i)
        compare(cpu.cold[i].result, gpu.cold[i].result,
                std::string(name) + " cold=" + std::to_string(i));
    for (std::size_t i = 0; i < cpu.warm.size(); ++i)
        compare(cpu.warm[i].result, gpu.warm[i].result,
                std::string(name) + " warm=" + std::to_string(i));
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
    out << "{\"energy_ev\":" << result.total_energy_ev
        << ",\"hof_kcal_mol\":" << result.heat_of_formation_kcal_mol
        << ",\"charges_e\":[";
    for (std::size_t i = 0; i < result.mulliken_charges.size(); ++i)
        out << (i ? "," : "") << result.mulliken_charges[i];
    out << "],\"forces_ev_a\":[";
    for (std::size_t i = 0; i < result.forces_ev_a.size(); ++i)
        out << (i ? "," : "") << result.forces_ev_a[i];
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
        if (argc < 2 || argc > 7)
            throw std::invalid_argument(
                "usage: bench_nddo_forces OUTPUT.json [am1|rm1|mndo|pm3|all] "
                "[waters:0|1|4|8|16|32] [warm:1..100] [profile:0|1] "
                "[fixed|moving|all]");
        const std::string method = argc > 2 ? argv[2] : "all";
        const int selected_waters = argc > 3 ? integer(argv[3]) : 0;
        const int repeats = argc > 4 ? integer(argv[4]) : 20;
        const int profile = argc > 5 ? integer(argv[5]) : 0;
        const std::string geometry_mode = argc > 6 ? argv[6] : "all";
        if ((method != "am1" && method != "rm1" && method != "all"
             && method != "mndo" && method != "pm3")
            || (selected_waters != 0 && selected_waters != 1
                && selected_waters != 4 && selected_waters != 8
                && selected_waters != 16 && selected_waters != 32)
            || (geometry_mode != "fixed" && geometry_mode != "moving"
                && geometry_mode != "all")
            || repeats < 1 || repeats > 100 || profile < 0 || profile > 1
            || (profile
                && (method == "all" || selected_waters == 0
                    || geometry_mode == "all")))
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
        series.reserve(120);
        for (const std::string mode : {"fixed", "moving"}) {
            if (geometry_mode != "all" && mode != geometry_mode)
                continue;
            for (int waters : {1, 4, 8, 16, 32}) {
                if (!selected_waters && waters == 32)
                    continue;
                if (selected_waters && selected_waters != waters)
                    continue;
                if (method == "all" || method == "am1")
                    collect<cpu::semiempirical::AM1, cuda::semiempirical::AM1>(
                        series, "am1", waters, repeats, profile, mode);
                if (method == "all" || method == "rm1")
                    collect<cpu::semiempirical::RM1, cuda::semiempirical::RM1>(
                        series, "rm1", waters, repeats, profile, mode);
                if (method == "all" || method == "mndo")
                    collect<cpu::semiempirical::MNDO,
                            cuda::semiempirical::MNDO>(series, "mndo", waters,
                                                       repeats, profile, mode);
                if (method == "all" || method == "pm3")
                    collect<cpu::semiempirical::PM3, cuda::semiempirical::PM3>(
                        series, "pm3", waters, repeats, profile, mode);
            }
        }
        struct utsname machine{};
        if (uname(&machine) != 0)
            throw std::runtime_error("cannot read benchmark machine metadata");
        struct rusage usage{};
        if (getrusage(RUSAGE_SELF, &usage) != 0)
            throw std::runtime_error("cannot read benchmark process memory");
        // Linux/GH200 reports ru_maxrss in KiB; this is the complete process,
        // including library initialization and all cases, not method workspace.
        const auto peak_rss_bytes =
            static_cast<unsigned long long>(usage.ru_maxrss) * 1024;
        const auto executable =
            std::filesystem::read_symlink("/proc/self/exe").string();
        // Reporting does not run in any timed or captured interval.
        std::ofstream out(argv[1]);
        out.exceptions(std::ios::badbit | std::ios::failbit);
        out << std::setprecision(17)
            << "{\n\"schema\":2,\"precision\":\"fp64\",\"boundary\":"
               "\"isolated\","
            << "\"threads\":1,\"cold_calls\":" << cold_calls
            << ",\"standalone_warmup_calls\":" << nested_warmup_calls
            << ",\"nested_warmup_calls\":" << nested_warmup_calls
            << ",\"max_iter\":" << max_iter
            << ",\"density_tol\":" << density_tol
            << ",\"energy_tol_ev\":" << energy_tol_ev
            << ",\"compare_energy_tol_ev\":" << compare_energy_ev
            << ",\"compare_charge_tol_e\":" << compare_charge_e
            << ",\"compare_force_tol_ev_a\":" << compare_force_ev_a
            << ",\"profile_enabled\":" << (profile ? "true" : "false")
            << ",\"cuda_context_init_seconds\":" << context_seconds
            << ",\"machine\":" << std::quoted(machine.machine)
            << ",\"hostname\":" << std::quoted(machine.nodename)
            << ",\"os\":" << std::quoted(machine.sysname)
            << ",\"kernel\":" << std::quoted(machine.release)
            << ",\"compiler\":" << std::quoted(__VERSION__)
            << ",\"executable\":" << std::quoted(executable)
            << ",\"executable_size_bytes\":"
            << std::filesystem::file_size(executable)
            << ",\"process_peak_rss_bytes\":" << peak_rss_bytes
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
                << ",\"geometry_mode\":" << std::quoted(s.geometry_mode)
                << ",\"atoms\":" << 3 * s.waters << ",\"orbitals\":" << nao;
            if (s.backend == "cuda")
                out << ",\"seven_dense_matrix_bytes_lower_bound\":"
                    << 7 * nao * nao * sizeof(double)
                    << ",\"device_free_before_bytes\":" << s.device_free_before
                    << ",\"device_free_live_bytes\":" << s.device_free_live
                    << ",\"device_free_after_bytes\":" << s.device_free_after;
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
        std::fprintf(stderr, "NDDO force benchmark failed: %s\n", error.what());
        return 1;
    }
}
