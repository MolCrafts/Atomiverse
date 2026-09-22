/// @file regressions/md/lj_argon/reg_lj_argon.cu
/// @brief END-TO-END REGRESSION: NVE energy conservation of Rahman's LJ argon
///        — **one regression**, multi-device (PyTorch-style
///        `--device=cpu|cuda|both`, default `both`).
///
/// Same literature gates for every device (see REFERENCE.md):
///   1. bounded energy error (symplectic O(dt^2) envelope), and
///   2. non-secular drift (oscillation, not a leak),
/// plus, with `--device=both`,
///   3. same-settings CPU<->CUDA parity of the frozen-lattice pair energy —
///      identical positions on both backends, so the only admissible
///      difference is floating-point summation order.
///
/// Public API only: `cpu::md::MD` or `cuda::md::MD` + `Frame`. No Context, no
/// `internal::`. Built only when `ATV_BUILD_CUDA=ON` (replaces the .cpp
/// target). The CUDA leg has no host fallback to grade silently: every
/// force/integration component here is a `cuda::` component whose launch() is
/// a device kernel, and the host frame is touched only at the explicit
/// H2D/D2H boundaries of `cuda::md::MD::run`.
///
/// Exits non-zero on failure.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <optional>
#include <string>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/init.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
#include "atomiverse/cuda/dynamics/init.cuh"
#include "atomiverse/cuda/dynamics/md.cuh"
#include "atomiverse/cuda/dynamics/velocity_verlet.cuh"
#include "atomiverse/cuda/neighbor/neighbor_list.cuh"
#include "atomiverse/cuda/potentials/pair_lj_cut.cuh"

using namespace ATV;

namespace {

// ── Rahman (1964) liquid argon — shared by every leg ─────────────────────
// eps/kB = 120 K, sigma = 3.4 A, number density 0.02071 A^-3, T = 94.4 K.
const double kEps = 120.0 * units::kB_ev_per_k; // 0.010341 eV
constexpr double kSigma = 3.4;                  // [A]
const double kCutoff = 2.5 * kSigma;            // 8.5 A, the textbook LJ cutoff
constexpr double kSkin = 0.5;                   // [A]
constexpr double kNumberDensity = 0.02071;      // [A^-3]
constexpr double kTemperature = 94.4;           // [K]
constexpr double kDt = 5.0;                     // [fs]  (Rahman used 10 fs)
constexpr int kNSide = 6;                       // 6^3 = 216 atoms
constexpr int kNSteps = 2000;                   // 10 ps
constexpr int kNChunks = 20; // sample the drift 20x along the way
constexpr unsigned kSeed = 12345u;

// Gate thresholds (REFERENCE.md): the symplectic envelope and the
// non-secular ratio — analytic properties, not golden values.
constexpr double kBound = 1.0e-2;     // [eps/atom]
constexpr double kSecularRatio = 4.0; // 2nd half may not exceed 1st by >4x

// Gate 3: frozen-lattice pair-energy parity, per atom [eV]. The two legs sum
// the SAME shifted LJ pair terms at bit-identical positions in double
// precision; ordering/FMA differences are ~1e-13 relative (~1e-14 eV here),
// so 1e-9 eV/atom is generous against FP and seven orders of magnitude below
// the physics scale eps ~ 1e-2 eV. Design-time bound — re-pin from the first
// GH200 run before ever tightening it.
constexpr double kParityTolEvPerAtom = 1.0e-9;

enum class Device { Cpu, Cuda };

const char* device_name(Device d) { return d == Device::Cuda ? "cuda" : "cpu"; }

std::string parse_device_arg(int argc, char** argv) {
    std::string want = "both";
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (std::strncmp(a, "--device=", 9) == 0) {
            want = a + 9;
        } else if (std::strcmp(a, "--device") == 0 && i + 1 < argc) {
            want = argv[++i];
        } else if (std::strcmp(a, "cpu") == 0 || std::strcmp(a, "cuda") == 0
                   || std::strcmp(a, "both") == 0) {
            want = a;
        }
    }
    return want;
}

/// @brief Simple-cubic argon lattice at the Rahman density (v = 0).
///
/// The box must exceed twice the cutoff so the minimum-image convention is
/// unambiguous — checked here once for every leg.
/// @param box_len Out: cubic box edge [A].
/// @return 216-atom frame, or nlocal()==0 on geometry failure.
cpu::Frame make_lattice(double& box_len) {
    const int n = kNSide * kNSide * kNSide;
    box_len = std::cbrt(double(n) / kNumberDensity);
    if (box_len <= 2.0 * kCutoff) {
        std::printf("FAIL: box %.2f A is too small for cutoff %.2f A\n",
                    box_len, kCutoff);
        return cpu::Frame(0);
    }
    const double spacing = box_len / double(kNSide);
    cpu::Frame frame(n);
    int idx = 0;
    for (int ix = 0; ix < kNSide; ++ix)
        for (int iy = 0; iy < kNSide; ++iy)
            for (int iz = 0; iz < kNSide; ++iz) {
                frame.span(kAtom_x)[idx] = (double(ix) + 0.5) * spacing;
                frame.span(kAtom_y)[idx] = (double(iy) + 0.5) * spacing;
                frame.span(kAtom_z)[idx] = (double(iz) + 0.5) * spacing;
                frame.span(kAtom_mass)[idx] = 39.948; // argon [amu]
                frame.span(kAtom_type)[idx] = 0;
                ++idx;
            }
    frame.box = cpu::Box::cubic(box_len);
    return frame;
}

/// @brief Gates 1 + 2 on one backend: run the Rahman NVE trajectory and
///        check the bounded + non-secular drift envelope.
/// @param dev Which backend runs the trajectory.
/// @return 0 on pass, 1 on failure (message already printed).
int check_conservation(Device dev) {
    double box_len = 0.0;
    cpu::Frame frame = make_lattice(box_len);
    if (frame.nlocal() == 0)
        return 1;
    const int n = frame.nlocal();

    // NVE: no thermostat. InitVelocity draws a Maxwell distribution at T once,
    // at build; from then on the total energy is the engine's to conserve.
    // One engine per leg, alive for the whole trajectory so consecutive
    // run() calls extend ONE trajectory on both backends.
    std::optional<cpu::md::MD> cpu_sim;
    std::optional<cuda::md::MD> cuda_sim;
    std::function<void(int)> advance;
    if (dev == Device::Cpu) {
        cpu_sim.emplace();
        cpu_sim->set_evaluator(cpu::PairLJCut{kEps, kSigma, kCutoff});
        cpu_sim->set_neighbor(cpu::md::NeighborList<>{kCutoff, kSkin});
        cpu_sim->set_integrator(cpu::md::VelocityVerlet{kDt});
        cpu_sim->set_observer(cpu::md::InitVelocity{kTemperature, kSeed});
        cpu_sim->enable_timing(false); // correctness check, not a benchmark
        cpu_sim->build();
        advance = [&](int steps) { cpu_sim->run(frame, steps); };
    } else {
        cuda_sim.emplace();
        cuda_sim->set_evaluator(cuda::PairLJCut{kEps, kSigma, kCutoff});
        cuda_sim->set_neighbor(cuda::NeighborList{kCutoff, kSkin});
        cuda_sim->set_integrator(cuda::md::VelocityVerlet{kDt});
        cuda_sim->set_observer(cuda::md::InitVelocity{kTemperature, kSeed});
        cuda_sim->build();
        advance = [&](int steps) { cuda_sim->run(frame, steps); };
    }

    // One step to populate the energies, then take E0 as the reference.
    advance(1);
    const double e0 =
        frame.meta.get(kEnergy) + frame.meta.get(kKinetic); // [eV]

    // Run in chunks so we see the WHOLE drift history, not just the endpoint —
    // a symplectic integrator's energy error is bounded and oscillatory, so an
    // endpoint-only check can pass by luck at a favourable phase.
    double worst_drift = 0.0;
    double drift_first_half = 0.0, drift_second_half = 0.0;
    for (int c = 1; c <= kNChunks; ++c) {
        advance(kNSteps / kNChunks);
        const double e = frame.meta.get(kEnergy) + frame.meta.get(kKinetic);
        const double drift =
            std::fabs(e - e0) / (double(n) * kEps); // [eps/atom]
        if (drift > worst_drift)
            worst_drift = drift;
        if (c <= kNChunks / 2)
            drift_first_half = std::fmax(drift_first_half, drift);
        else
            drift_second_half = std::fmax(drift_second_half, drift);
    }

    // ── The checks (see REFERENCE.md) ────────────────────────────────────
    // (1) Bounded: velocity Verlet is symplectic, so |dE| stays inside an
    //     O(dt^2) envelope. 1e-2 eps/atom is that envelope at dt = 5 fs.
    // (2) Non-secular: the error must NOT grow linearly with time. A symplectic
    //     integrator oscillates; a broken one ramps. Compare the two halves.
    std::printf("[%s] LJ argon NVE, N = %d, rho = %.5f A^-3, T0 = %.1f K, "
                "dt = %.1f fs\n",
                device_name(dev), n, kNumberDensity, kTemperature, kDt);
    std::printf("[%s]   E_total(0)          = %.6f eV\n", device_name(dev), e0);
    std::printf(
        "[%s]   worst |dE|/(N*eps)  = %.3e   want <= %.1e   [bounded]\n",
        device_name(dev), worst_drift, kBound);
    std::printf("[%s]   drift 1st half      = %.3e\n", device_name(dev),
                drift_first_half);
    std::printf("[%s]   drift 2nd half      = %.3e   want <= %.1fx the 1st "
                "[non-secular]\n",
                device_name(dev), drift_second_half, kSecularRatio);

    bool ok = true;
    if (!(worst_drift <= kBound)) {
        std::printf(
            "FAIL [%s]: energy drift %.3e eps/atom exceeds bound %.1e\n",
            device_name(dev), worst_drift, kBound);
        ok = false;
    }
    if (drift_first_half > 0.0
        && drift_second_half > kSecularRatio * drift_first_half) {
        std::printf("FAIL [%s]: drift grows with time (%.3e -> %.3e) — energy "
                    "is leaking, not oscillating\n",
                    device_name(dev), drift_first_half, drift_second_half);
        ok = false;
    }
    if (!std::isfinite(worst_drift)) {
        std::printf("FAIL [%s]: energy is not finite\n", device_name(dev));
        ok = false;
    }
    std::printf("%s [%s]: conservation\n", ok ? "PASS" : "FAIL",
                device_name(dev));
    return ok ? 0 : 1;
}

/// @brief Gate 3 support: shifted-LJ pair energy of the frozen lattice.
///
/// The perfect simple-cubic lattice is inversion-symmetric, so the net force
/// on every atom is exactly zero and one velocity-Verlet step from v = 0
/// moves nothing: kEnergy after that step IS the lattice pair energy, at
/// bit-identical positions on both backends. This isolates the pair kernel +
/// neighbor list + minimum image from the (backend-specific) velocity RNG.
/// @param dev   Which backend evaluates the lattice.
/// @param e_pot Out: lattice pair energy [eV].
/// @return 0 on pass, 1 on failure (message already printed).
int lattice_energy(Device dev, double& e_pot) {
    double box_len = 0.0;
    cpu::Frame frame = make_lattice(box_len);
    if (frame.nlocal() == 0)
        return 1;
    if (dev == Device::Cpu) {
        cpu::md::MD sim;
        sim.set_evaluator(cpu::PairLJCut{kEps, kSigma, kCutoff});
        sim.set_neighbor(cpu::md::NeighborList<>{kCutoff, kSkin});
        sim.set_integrator(cpu::md::VelocityVerlet{kDt});
        sim.enable_timing(false);
        sim.build();
        sim.run(frame, 1);
        e_pot = frame.meta.get(kEnergy);
    } else {
        cuda::md::MD sim;
        sim.set_evaluator(cuda::PairLJCut{kEps, kSigma, kCutoff});
        sim.set_neighbor(cuda::NeighborList{kCutoff, kSkin});
        sim.set_integrator(cuda::md::VelocityVerlet{kDt});
        sim.build();
        sim.run(frame, 1);
        e_pot = frame.meta.get(kEnergy);
    }
    if (!std::isfinite(e_pot)) {
        std::printf("FAIL [%s]: lattice energy is not finite\n",
                    device_name(dev));
        return 1;
    }
    std::printf("[%s]   E_pot(lattice)      = %.12f eV\n", device_name(dev),
                e_pot);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    const std::string want = parse_device_arg(argc, argv);
    if (want != "cpu" && want != "cuda" && want != "both") {
        std::fprintf(stderr, "FAIL: unknown --device=%s (want cpu|cuda|both)\n",
                     want.c_str());
        return 1;
    }
    const bool run_cpu = (want == "cpu" || want == "both");
    const bool run_cuda = (want == "cuda" || want == "both");

    int rc = 0;
    if (run_cpu)
        rc |= check_conservation(Device::Cpu);
    if (run_cuda)
        rc |= check_conservation(Device::Cuda);

    // Gate 3 — frozen-lattice pair-energy parity (both legs requested only).
    if (rc == 0 && run_cpu && run_cuda) {
        double e_cpu = 0.0, e_cuda = 0.0;
        rc |= lattice_energy(Device::Cpu, e_cpu);
        rc |= lattice_energy(Device::Cuda, e_cuda);
        if (rc == 0) {
            const int n = kNSide * kNSide * kNSide;
            const double delta_per_atom = std::fabs(e_cuda - e_cpu) / double(n);
            if (!(delta_per_atom <= kParityTolEvPerAtom)) {
                std::fprintf(stderr,
                             "FAIL [parity]: |E_cuda - E_cpu|/N = %.3e eV/atom "
                             " tol=%.1e eV/atom  (cpu=%.12f, cuda=%.12f)\n",
                             delta_per_atom, kParityTolEvPerAtom, e_cpu,
                             e_cuda);
                return 1;
            }
            std::printf("PASS [parity]: |E_cuda - E_cpu|/N = %.3e eV/atom <= "
                        "%.1e eV/atom\n",
                        delta_per_atom, kParityTolEvPerAtom);
        }
    }
    return rc;
}
