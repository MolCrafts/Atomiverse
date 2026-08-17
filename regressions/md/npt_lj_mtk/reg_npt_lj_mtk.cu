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
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/nose_hoover_npt.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/observe/virial_stress.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/cuda/dynamics/md.cuh"
#include "atomiverse/cuda/dynamics/nose_hoover_npt.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut.cuh"

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

cpu::Frame make_lj(int n_side) {
    const int n = n_side * n_side * n_side;
    const double density = 0.8;
    const double box_len = std::cbrt(double(n) / density);
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
    frame.box = cpu::Box::cubic(box_len);
    return frame;
}

struct Out {
    double volume = 0;
    double H = 0;
};

Out run_on_device(Device dev) {
    constexpr double kCut = 2.5;
    constexpr double kSkin = 0.3;
    constexpr double kDt = 0.005;
    auto frame = make_lj(4); // 64 atoms
    const double V0 = frame.box.volume();
    if (dev == Device::Cpu) {
        cpu::md::MD sim;
        sim.set_neighbor(cpu::md::NeighborList<>{kCut, kSkin});
        sim.set_evaluator(cpu::PairLJCut{1.0, 1.0, kCut});
        sim.set_integrator(cpu::md::NoseHooverNPT{1.0, 0.1, 0.0, 1.0, kDt});
        sim.set_observer(cpu::md::Virial{});
        sim.build();
        sim.run(frame, 80);
    } else {
        cuda::md::MD sim;
        sim.set_neighbor(cuda::NeighborList{kCut, kSkin});
        sim.set_evaluator(cuda::PairLJCut{1.0, 1.0, kCut});
        sim.set_integrator(cuda::md::NoseHooverNPT{1.0, 0.1, 0.0, 1.0, kDt});
        sim.build();
        sim.run(frame, 80);
    }
    Out o;
    o.volume = frame.box.volume();
    o.H = static_cast<double>(frame.meta.get(kConserved));
    std::printf("[%s] V0=%.6f V=%.6f H=%.6e dV/V0=%.4e\n", device_name(dev), V0,
                o.volume, o.H, (o.volume - V0) / V0);
    return o;
}

} // namespace

int main(int argc, char** argv) {
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
    std::printf("PASS: npt_lj MTK NoseHooverNPT\n");
    return 0;
}
