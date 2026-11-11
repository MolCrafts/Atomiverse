/// @file regressions/md/npt_lj_mtk/reg_npt_lj_mtk.cu
/// @brief Short LJ MTK NPT — one case, `--device=cpu|cuda|both`.
///        Named npt_lj_mtk so the binary does not collide with mc/npt_lj.
///
/// Gates (REFERENCE.md):
///   1. conserved H finite after 80 steps;
///   2. cell volume moved (barostat not dead);
///   3. with --device=both, |V_cuda − V_cpu| / V_cpu < 1e-4.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/nose_hoover_npt.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/observe/virial_stress.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/cuda/dynamics/md.cuh"
#include "atomiverse/cuda/dynamics/nose_hoover_npt.cuh"
#include "atomiverse/cuda/frame/precision.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut.cuh"
#include "harness/registry.h"

namespace atv_reg_md_npt_lj_mtk {

using namespace ATV;

namespace {

enum class Device { Cpu, Cuda };
const char* device_name(Device d) { return d == Device::Cuda ? "cuda" : "cpu"; }

std::string parse_device_arg(int argc, char** argv) {
    std::string want = "both";
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (std::strncmp(a, "--device=", 9) == 0)
            want = a + 9;
        else if (std::strcmp(a, "--device") == 0 && i + 1 < argc)
            want = argv[++i];
        else if (std::strcmp(a, "cpu") == 0 || std::strcmp(a, "cuda") == 0
                 || std::strcmp(a, "both") == 0)
            want = a;
    }
    return want;
}

/// @brief Project a coordinate onto the CUDA stored representation (primary
///        plus correction when present), as `cpu::to_cuda` uploads it.
/// @param value Coordinate [Angstrom].
/// @return The coordinate the device reconstructs [Angstrom].
double project_position(double value) {
    cuda::pos_t primary, correction = cuda::pos_t(0);
    cuda::split_position(cuda::mixed_t(value), primary, correction);
    return double(cuda::load_position(&primary, &correction, 0));
}

cpu::Frame make_lj(int n_side, bool projected = false) {
    const int n = n_side * n_side * n_side;
    const double density = 0.8;
    const double exact_len = std::cbrt(double(n) / density);
    const double box_len =
        projected ? double(static_cast<cuda::geom_t>(exact_len)) : exact_len;
    const double spacing = box_len / double(n_side);
    cpu::Frame frame(n);
    int idx = 0;
    for (int ix = 0; ix < n_side; ++ix)
        for (int iy = 0; iy < n_side; ++iy)
            for (int iz = 0; iz < n_side; ++iz) {
                frame.span(kAtom_x)[idx] = (ix + 0.5) * spacing;
                frame.span(kAtom_y)[idx] = (iy + 0.5) * spacing;
                frame.span(kAtom_z)[idx] = (iz + 0.5) * spacing;
                frame.span(kAtom_mass)[idx] = 1.0;
                frame.span(kAtom_type)[idx] = 18;
                frame.span(kAtom_vx)[idx] = 0.02 * ((ix + iy + iz) % 3 - 1);
                frame.span(kAtom_vy)[idx] = 0.02 * ((ix + 2 * iy) % 3 - 1);
                frame.span(kAtom_vz)[idx] = 0.02 * ((2 * ix + iz) % 3 - 1);
                ++idx;
            }
    if (projected)
        for (int i = 0; i < n; ++i) {
            frame.span(kAtom_x)[i] = project_position(frame.span(kAtom_x)[i]);
            frame.span(kAtom_y)[i] = project_position(frame.span(kAtom_y)[i]);
            frame.span(kAtom_z)[i] = project_position(frame.span(kAtom_z)[i]);
        }
    frame.box = cpu::Box::cubic(box_len);
    return frame;
}

struct Out {
    double volume = 0;
    double H = 0;
    double initial_volume = 0;
    double represented_volume = 1;
};

Out run_on_device(Device dev, int steps = 80, bool projected = false) {
    constexpr double kCut = 2.5;
    constexpr double kSkin = 0.3;
    constexpr double kDt = 0.005;
    auto frame = make_lj(6, projected); // 216 atoms; L/2 > r_c
    const double V0 = frame.box.volume();
    if (dev == Device::Cpu) {
        cpu::md::MD sim;
        sim.set_neighbor(cpu::md::NeighborList<>{kCut, kSkin});
        sim.set_evaluator(cpu::PairLJCut{1.0, 1.0, kCut});
        // T* = 1 with ε = 1 eV → T = 1/kB [K]. Same deterministic kick
        // on both backends (InitVelocity RNGs differ, so not used here).
        const double T = projected ? 1.0 : 1.0 / units::UnitSystem::lj().boltz;
        sim.set_integrator(cpu::md::NoseHooverNPT{T, 0.1, 0.0, 1.0, kDt});
        sim.set_observer(cpu::md::Virial{});
        sim.enable_timing(false);
        sim.build();
        sim.run(frame, steps);
    } else {
        cuda::md::MD sim;
        sim.set_neighbor(cuda::NeighborList{kCut, kSkin});
        sim.set_evaluator(cuda::PairLJCut{1.0, 1.0, kCut});
        const double T = projected ? 1.0 : 1.0 / units::UnitSystem::lj().boltz;
        sim.set_integrator(cuda::md::NoseHooverNPT{T, 0.1, 0.0, 1.0, kDt});
        sim.build();
        sim.run(frame, steps);
    }
    Out o;
    o.volume = frame.box.volume();
    o.initial_volume = V0;
    for (int d = 0; d < 3; ++d)
        o.represented_volume *= static_cast<cuda::geom_t>(frame.box.length(d));
    o.H = static_cast<double>(frame.meta.get(kConserved));
    std::printf("[%s] V0=%.6f V=%.6f H=%.6e dV/V0=%.4e\n", device_name(dev), V0,
                o.volume, o.H, (o.volume - V0) / V0);
    return o;
}

} // namespace

int run(int argc, char** argv) {
    const std::string want = parse_device_arg(argc, argv);
    if (want != "cpu" && want != "cuda" && want != "both") {
        std::fprintf(stderr, "FAIL: unknown --device=%s\n", want.c_str());
        return 1;
    }
    Out cpu{}, cuda{};
    if (want == "cpu" || want == "both")
        cpu = run_on_device(Device::Cpu);
    if (want == "cuda" || want == "both")
        cuda = run_on_device(Device::Cuda);

    auto check = [](const Out& o, const char* tag) {
        if (!std::isfinite(o.H) || !std::isfinite(o.volume) || o.volume <= 0) {
            std::fprintf(stderr, "FAIL [%s]: non-finite H or V\n", tag);
            return false;
        }
        return true;
    };
    if ((want == "cpu" || want == "both") && !check(cpu, "cpu"))
        return 1;
    if ((want == "cuda" || want == "both") && !check(cuda, "cuda"))
        return 1;
    if (want == "both") {
        const double rel = std::abs(cuda.volume - cpu.volume)
                           / std::max(1.0, std::abs(cpu.volume));
        std::printf("[both] |V_cuda-V_cpu|/V=%.4e\n", rel);
        if (rel > 1e-4) {
            std::fprintf(stderr, "FAIL [both]: volume parity\n");
            return 1;
        }
    }
    if (want == "both")
        for (int steps : {1, 2}) {
            const auto c = run_on_device(Device::Cpu, steps, true);
            const auto g = run_on_device(Device::Cuda, steps, true);
            const double tolerance = steps == 1 ? 1e-8 : 1e-6;
            if (!std::isfinite(c.volume) || !std::isfinite(g.volume)
                || (steps == 1
                    && std::abs(c.volume - c.initial_volume) <= 1e-12)
                || std::abs(g.volume - c.represented_volume)
                       > tolerance
                             * std::max(1.0, std::abs(c.represented_volume))) {
                std::fprintf(stderr, "FAIL: %d-step NPT preparation/parity\n",
                             steps);
                return 1;
            }
        }
    std::printf("PASS: npt_lj MTK NoseHooverNPT\n");
    return 0;
}

} // namespace atv_reg_md_npt_lj_mtk

ATV_REGRESSION("md/npt", "npt_lj_mtk", atv_reg_md_npt_lj_mtk::run);
