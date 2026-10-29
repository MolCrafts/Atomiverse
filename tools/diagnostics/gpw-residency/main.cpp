// Public CPU/CUDA energy and every-force comparison for the resident SG path.
// These twins and ionic differences are not independent-engine qualification.
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cuda/gpw/gpw_solver.cuh"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ATV;
namespace {
// Preserve the archived p-shell water product gates. New asymmetric cases use
// the same gates; no measured value determines an acceptance threshold.
constexpr double energy_gate_ha = 5e-5;
constexpr double force_gate_ha_bohr = 5e-4;
constexpr double difference_step_a = 1e-4;
constexpr double difference_gate_ev_a = 1e-3;
void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}
struct Case {
    const char* name;
    cpu::xc::XCType xc;
    bool asymmetric;
};
constexpr Case cases[] = {{"water-PADE", cpu::xc::XCType::PADE, false},
                          {"water-asym-PBE", cpu::xc::XCType::PBE, true},
                          {"water-asym-PBEsol", cpu::xc::XCType::PBEsol, true},
                          {"water-asym-revPBE", cpu::xc::XCType::RevPBE, true},
                          {"water-asym-RPBE", cpu::xc::XCType::RPBE, true}};
std::vector<cpu::gpw::Kind> kinds() {
    return {{molrs::Element::O,
             cpu::gpw::GaussianBasis(molrs::Element::O, "SZV-GTH-PADE"),
             cpu::gpw::GthPotential(molrs::Element::O, "GTH-PADE-q6")},
            {molrs::Element::H,
             cpu::gpw::GaussianBasis(molrs::Element::H, "DZV-GTH-PADE"),
             cpu::gpw::GthPotential(molrs::Element::H, "GTH-PADE-q1")}};
}
cpu::Frame geometry(bool asymmetric, bool moved = false) {
    // Archived water geometry/cell are kept exactly, including its historical
    // bond conversion constant. The new slice breaks all reflection planes.
    constexpr double old_a0 = .52917721067;
    constexpr double a0 = units::kBohrInAngstrom;
    double positions[3][3] = {{6, 6, 6},
                              {6 + .7586 / old_a0, 6, 6 + .5043 / old_a0},
                              {6 - .7586 / old_a0, 6, 6 + .5043 / old_a0}};
    if (asymmetric) {
        positions[0][0] += .09;
        positions[0][1] -= .07;
        positions[1][1] += .23;
        positions[2][0] -= .14;
        positions[2][1] -= .19;
        positions[2][2] += .12;
    }
    if (moved)
        positions[1][2] += .017 / a0;
    cpu::Frame frame(3);
    frame.box = cpu::Box::orthorhombic(12 * a0, 12 * a0, 12 * a0);
    for (int i = 0; i < 3; ++i) {
        frame.span(kAtom_type)[i] = i ? 1 : 8;
        frame.span(kAtom_mass)[i] = i ? 1.008 : 15.999;
        frame.span(kAtom_tag)[i] = i + 1;
        frame.span(kAtom_x)[i] = positions[i][0] * a0;
        frame.span(kAtom_y)[i] = positions[i][1] * a0;
        frame.span(kAtom_z)[i] = positions[i][2] * a0;
    }
    return frame;
}
template <class Method>
void configure(Method& m) {
    m.n_occ = 4;
    m.max_iter = 80;
    m.tol = 1e-8;
    m.diis_depth = 6;
    m.mix = .4;
    m.level_shift = 0;
    m.nonperiodic = false;
    m.local_pp_on_grid = true;
    m.use_ot = false;
    m.ngrids = 1;
    m.eps_default = 1e-8;
}
std::array<double, 9> forces(const cpu::Frame& frame) {
    std::array<double, 9> result{};
    int c = 0;
    for (auto key : {kAtom_fx, kAtom_fy, kAtom_fz})
        for (int i = 0; i < 3; ++i)
            result[c++] = frame.span(key)[i];
    return result;
}
void check_finite(const std::array<double, 9>& f) {
    for (double value : f)
        require(std::isfinite(value), "nonfinite force");
}
void run(const Case& c, bool profile_only) {
    cpu::gpw::GpwDft host(kinds(), 100, c.xc);
    cuda::gpw::GpwDft device(kinds(), 100, c.xc);
    configure(host);
    configure(device);
    if (c.asymmetric) {
        host.tol = 1e-11;
        device.tol = 1e-11;
    }
    double max_de = 0, max_df = 0, max_fd = 0;
    for (int shifted = 0; shifted < (profile_only ? 1 : 2); ++shifted) {
        auto cpu_frame = geometry(c.asymmetric, shifted);
        auto gpu_frame = cpu_frame;
        host.build(cpu_frame);
        device.build(gpu_frame);
        // Exactly one public force solve on each backend for this geometry.
        const auto reference = host.forces(cpu_frame);
        const auto result = device.forces(gpu_frame);
        require(reference.converged && result.converged,
                "SCF did not converge");
        require(reference.iters <= 80 && result.iters <= 80, "SCF budget");
        const auto expected = forces(cpu_frame), got = forces(gpu_frame);
        check_finite(expected);
        check_finite(got);
        max_de = std::max(max_de, std::abs(result.e_total - reference.e_total));
        for (int lane = 0; lane < 9; ++lane) {
            max_df =
                std::max(max_df, std::abs(got[lane] - expected[lane])
                                     / units::kHartreePerBohrInEvPerAngstrom);
            std::printf(
                "FORCE %s shifted=%d lane=%d CPU_eV_A=%.15g CUDA_eV_A=%.15g\n",
                c.name, shifted, lane, expected[lane], got[lane]);
        }
        require(std::abs(gpu_frame.meta.get(kEnergy)
                         - result.e_total * units::kHartreeInEv)
                    <= 1e-10,
                "force energy publication");
        if (!profile_only && c.asymmetric && shifted == 0) {
            // An off-plane lane exercises p-shell Pulay, ion-center and grid
            // terms without a reflection-symmetry zero-force shortcut.
            double energies[2];
            for (int sign = 0; sign < 2; ++sign) {
                auto displaced = geometry(true);
                displaced.span(kAtom_y)[1] +=
                    sign ? -difference_step_a : difference_step_a;
                cpu::gpw::GpwDft fresh(kinds(), 100, c.xc);
                configure(fresh);
                fresh.tol = 1e-11;
                fresh.build(displaced);
                const auto r = fresh.run(displaced);
                require(r.converged, "displaced SCF did not converge");
                energies[sign] = r.e_total * units::kHartreeInEv;
            }
            const double fd =
                -(energies[0] - energies[1]) / (2 * difference_step_a);
            max_fd = std::abs(got[4] - fd);
            require(std::abs(got[4]) > 1e-3, "off-plane force vanished");
        }
    }
    std::printf("GPW_RESIDENCY_RESULT %s dE_Ha=%.9g max_dF_Ha_Bohr=%.9g "
                "ionic_FD_error_eV_A=%.9g\n",
                c.name, max_de, max_df, max_fd);
    require(max_de <= energy_gate_ha, "energy twin gate");
    require(max_df <= force_gate_ha_bohr, "full force twin gate");
    require(max_fd <= difference_gate_ev_a,
            "off-plane ionic force derivative gate");
    std::printf("PASS GPW_RESIDENCY %s\n", c.name);
}
} // namespace
int main(int argc, char** argv) {
    try {
        std::string selected;
        bool profile_only = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--case" && i + 1 < argc)
                selected = argv[++i];
            else if (arg == "--profile-only")
                profile_only = true;
            else
                throw std::runtime_error("unknown argument: " + arg);
        }
        bool matched = false;
        for (const auto& c : cases)
            if (selected.empty() || selected == c.name) {
                matched = true;
                run(c, profile_only);
            }
        require(matched, "unknown case");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL GPW_RESIDENCY %s\n", error.what());
        return 1;
    }
}
