#include "atomiverse/core/units.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/scf/methods.h"
#include "pins.h"
#include "scaled_slater.h"
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#ifdef ATV_HAS_MD
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#endif
#ifdef ATV_HAS_MINIMIZER
#include "atomiverse/cpu/minimize/fire.h"
#include "atomiverse/cpu/minimize/minimize.h"
#endif

using namespace ATV;

// This executable is also usable one system at a time for an installed consumer.
int main(int argc, char** argv) {
    try {
        const bool water = argc < 2 || std::string(argv[1]) == "water";
        const int n = water ? 3 : 2;
        cpu::Frame frame(n);
        const double xyz[3][3] = {
            {0, 0, 0.1173}, {0, 0.7572, -0.4692}, {0, -0.7572, -0.4692}};
        for (int i = 0; i < n; ++i) {
            frame.span(kAtom_type)[i] = water && i == 0 ? 8 : 1;
            frame.span(kAtom_mass)[i] = water && i == 0 ? 15.999 : 1.008;
            frame.span(kAtom_x)[i] = water ? xyz[i][0] : 0;
            frame.span(kAtom_y)[i] = water ? xyz[i][1] : 0;
            frame.span(kAtom_z)[i] = water ? xyz[i][2] : double(i);
        }
        frame.box = cpu::Box::cubic(20);
        auto xc = std::make_shared<const xc_example::ScaledSlater>();
        cpu::scf::RKS rks(cpu::scf::BasisSet::STO3G, xc);
        cpu::scf::UKS uks(cpu::scf::BasisSet::STO3G, 1, 0, xc, 1);
        auto check = [&](auto& method) {
            method.tol = 1e-12;
            method.max_iter = 150;
            method.grid_level = cpu::scf::GridLevel::MatchPySCF;
            const double want =
                water ? xc_example::water_energy : xc_example::h2plus_energy;
            auto verify = [&](bool with_forces = true) {
                const double energy =
                    frame.meta.get(kEnergy) / units::kHartreeInEv;
                if (!method.converged() || !std::isfinite(energy)
                    || std::abs(energy - want) > xc_example::energy_tol) {
                    std::fprintf(stderr,
                                 "FAIL E got=%.14f want=%.14f tol=%.1e Ha\n",
                                 energy, want, xc_example::energy_tol);
                    throw std::runtime_error("energy/convergence");
                }
                if (!with_forces) {
                    std::printf("PASS %s E=%.14f Ha (energy request)\n",
                                method.name().c_str(), energy);
                    return;
                }
                auto fx = frame.span(kAtom_fx), fy = frame.span(kAtom_fy),
                     fz = frame.span(kAtom_fz);
                for (int i = 0; i < n; ++i)
                    for (int d = 0; d < 3; ++d) {
                        const double grad =
                            -(d == 0   ? fx[i]
                              : d == 1 ? fy[i]
                                       : fz[i])
                            / units::kHartreePerBohrInEvPerAngstrom;
                        const double ref =
                            water ? xc_example::water_gradient[i][d]
                                  : xc_example::h2plus_gradient[i][d];
                        if (!std::isfinite(grad)
                            || std::abs(grad - ref)
                                   > xc_example::gradient_tol) {
                            std::fprintf(stderr,
                                         "FAIL gradient(%d,%d) got=%.14f "
                                         "want=%.14f tol=%.1e Ha/Bohr\n",
                                         i, d, grad, ref,
                                         xc_example::gradient_tol);
                            throw std::runtime_error("force pin");
                        }
                    }
                std::printf("PASS %s E=%.14f Ha; all %d force components\n",
                            method.name().c_str(), energy, 3 * n);
            };
            frame.span(kAtom_fx)[0] = 17;
            method.run(frame);
            verify(false);
            if (frame.span(kAtom_fx)[0] != 17)
                throw std::runtime_error("run() changed an unrequested force");
            // A const-Frame energy request preserves caller-owned results.
            frame.meta.get(kEnergy) = 123;
            frame.span(kAtom_fx)[0] = 17;
            method.run(static_cast<const cpu::Frame&>(frame));
            if (frame.meta.get(kEnergy) != 123 || frame.span(kAtom_fx)[0] != 17
                || std::abs(method.energy() - want) > xc_example::energy_tol)
                throw std::runtime_error("energy-only output preservation");
            // An unsupported request fails before changing those outputs; retry
            // on the same method uses its original immutable provider.
            method.hybrid_alpha_override = 0.25;
            bool rejected = false;
            try {
                method.run(frame);
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            if (!rejected || frame.meta.get(kEnergy) != 123
                || frame.span(kAtom_fx)[0] != 17)
                throw std::runtime_error(
                    "unsupported request did not fail atomically");
            method.hybrid_alpha_override =
                0; // explicit zero is still semi-local
            auto forces = method.forces(frame);
            verify();
            if (forces.forces.shape()[0] != std::size_t(n))
                throw std::runtime_error("force result shape");
            method.hybrid_alpha_override = -1;
            // Same concrete SCF loop, nested in the existing one-pass Driver.
            {
                cpu::Evaluate evaluator;
                evaluator.on(evaluate::Phase::evaluate).add_ref(method);
                evaluator.build();
                evaluator.run(frame);
                verify();
                evaluator.run(
                    frame); // continuation must not double accumulation
                verify();
            }
            const double z = frame.span(kAtom_z)[1];
            frame.span(kAtom_z)[1] += 0.01;
            method.run(frame);
            if (!method.converged())
                throw std::runtime_error("geometry refresh");
            frame.span(kAtom_z)[1] = z;
            frame.box = cpu::Box::cubic(22);
            method.forces(frame);
            verify();
#ifdef ATV_HAS_MD
            {
                cpu::md::MD dynamics;
                dynamics.set_integrator(cpu::md::VelocityVerlet{0.02});
                dynamics.on(md::Phase::force).add_ref(method);
                dynamics.build();
                dynamics.run(frame, 2);
                dynamics.run(frame, 1);
                if (!method.converged()
                    || !std::isfinite(frame.meta.get(kEnergy)))
                    throw std::runtime_error("MD continuation");
            }
#endif
#ifdef ATV_HAS_MINIMIZER
            {
                cpu::minimizer::Minimize optimizer;
                optimizer.set_optimizer(cpu::minimizer::FIRE{});
                optimizer.on(minimizer::Phase::force).add_ref(method);
                optimizer.build();
                optimizer.run(frame, 3);
                if (!method.converged()
                    || !std::isfinite(frame.meta.get(kEnergy)))
                    throw std::runtime_error("Minimize composition");
            }
#endif
            // Restore every coordinate after the actual MD/optimizer steps.
            for (int i = 0; i < n; ++i) {
                frame.span(kAtom_x)[i] = water ? xyz[i][0] : 0;
                frame.span(kAtom_y)[i] = water ? xyz[i][1] : 0;
                frame.span(kAtom_z)[i] = water ? xyz[i][2] : double(i);
            }
            method.forces(frame);
            verify();
            auto peer = [&] {
                if constexpr (std::is_same_v<std::decay_t<decltype(method)>,
                                             cpu::scf::RKS>)
                    return cpu::scf::RKS(cpu::scf::BasisSet::STO3G, xc);
                else
                    return cpu::scf::UKS(cpu::scf::BasisSet::STO3G, 1, 0, xc,
                                         1);
            }();
            peer.grid_level = cpu::scf::GridLevel::MatchPySCF;
            peer.tol = 1e-12;
            peer.max_iter = 150;
            peer.run(frame);
            if (!peer.converged()
                || std::abs(peer.energy() - want) > xc_example::energy_tol)
                throw std::runtime_error(
                    "shared provider contaminated an independent method");
            method.forces(frame);
            verify();
        };
        if (water)
            check(rks);
        else
            check(uks);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
