/// Native participant for the periodic E/full-F/stress benchmark.
#include "atomiverse/core/units.h"
#ifdef ATV_HAS_GPW
#include "atomiverse/cpu/gpw/gpw_solver.h"
#endif
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/planewave/qe_units.h"
#ifdef ATV_HAS_CUDA
#include "atomiverse/cuda/planewave/pw_solver.cuh"
#ifdef ATV_HAS_GPW
#include "atomiverse/cuda/gpw/gpw_solver.cuh"
#endif
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using namespace ATV;
using cpu::xc::XCType;

XCType functional(const std::string& name) {
    if (name == "PBE")
        return XCType::PBE;
    if (name == "PBEsol")
        return XCType::PBEsol;
    if (name == "RevPBE")
        return XCType::RevPBE;
    if (name == "RPBE")
        return XCType::RPBE;
    throw std::invalid_argument("Unsupported periodic benchmark functional");
}

void require_close(double value, double expected, double tolerance,
                   const char* message) {
    if (!(std::abs(value - expected) <= tolerance))
        throw std::runtime_error(message);
}

template <class Method, class Energy>
void measure(Method& method, cpu::Frame& frame, Energy energy,
             double publication_scale, int stress_size, int spin,
             int expected_electrons, const char* record = "ATV_PERIODIC_RESULT",
             int expected_kpoints = 0) {
    const auto& force_result = method.forces(frame);
    if (!force_result.converged)
        throw std::runtime_error("Force SCF did not converge");
    const double e = energy(force_result);
    require_close(frame.meta.get(kEnergy), e * publication_scale,
                  stress_size == 6 ? 1e-9 : 1e-10,
                  "Force call did not publish its energy to Frame");

    if constexpr (std::is_same_v<std::remove_cvref_t<decltype(force_result)>,
                                 cpu::pw::PWResult>) {
        require_close(force_result.n_elec_occ, expected_electrons, 1e-8,
                      "Incorrect occupied electron count");
        if (method.smearing == electronic::Smearing::none) {
            require_close(force_result.e_entropy, 0, 0,
                          "Unexpected fractional occupations");
        } else if (!std::isfinite(force_result.e_entropy)
                   || !(force_result.e_entropy < -1e-5)
                   || (spin
                       && (!std::isfinite(force_result.magnetization)
                           || !(std::abs(force_result.magnetization) > 0.1))))
            throw std::runtime_error("Expected nontrivial entropy and spin");
    } else if (expected_kpoints) {
        if (force_result.n_kpts != expected_kpoints
            || force_result.k_weights.size()
                   != static_cast<std::size_t>(expected_kpoints))
            throw std::runtime_error("GPW did not use the full k-point mesh");
        for (double weight : force_result.k_weights)
            require_close(weight, 1.0 / expected_kpoints, 1e-12,
                          "Incorrect GPW k-point weight");
        require_close(force_result.n_elec_occ, expected_electrons, 1e-8,
                      "Incorrect occupied electron count");
        if (spin) {
            const auto count = static_cast<std::size_t>(expected_kpoints);
            if (force_result.C_k.size() != count
                || force_result.C_occ_k.size() != count
                || force_result.C_occ_k_beta.size() != count)
                throw std::runtime_error("Missing UKS Bloch occupations");
            for (std::size_t k = 0; k < count; ++k) {
                const auto nb = force_result.C_k[k].shape(0);
                if (force_result.C_occ_k[k].size() != 2 * nb
                    || force_result.C_occ_k_beta[k].size() != nb)
                    throw std::runtime_error("Expected two alpha and one beta "
                                             "occupied Bloch orbitals");
            }
        }
    } else if (spin) {
        const auto& s = force_result.S;
        const auto& pa = force_result.P;
        const auto& pb = force_result.P_beta;
        if (s.size() == 0 || s.shape()[0] != s.shape()[1]
            || pa.shape() != s.shape() || pb.shape() != s.shape())
            throw std::runtime_error("Missing Gamma UKS density payload");
        double alpha = 0, beta = 0;
        for (std::size_t i = 0; i < s.shape()[0]; ++i)
            for (std::size_t j = 0; j < s.shape()[1]; ++j) {
                alpha += pa(i, j) * s(j, i);
                beta += pb(i, j) * s(j, i);
            }
        require_close(alpha, 2, 1e-8, "Incorrect alpha count");
        require_close(beta, 1, 1e-8, "Incorrect beta count");
        require_close(force_result.e_entropy, 0, 0,
                      "Unexpected fractional occupations");
        if (!force_result.mulliken_spin.empty()) {
            if (force_result.mulliken_spin.size() != frame.nlocal())
                throw std::runtime_error("Incomplete GPW Mulliken spin output");
            require_close(std::accumulate(force_result.mulliken_spin.begin(),
                                          force_result.mulliken_spin.end(), 0.),
                          alpha - beta, 1e-8,
                          "Inconsistent GPW spin population");
        }
    }

    std::vector<std::array<double, 3>> forces(frame.nlocal());
    for (std::size_t i = 0; i < forces.size(); ++i)
        forces[i] = {frame.span(kAtom_fx)[i], frame.span(kAtom_fy)[i],
                     frame.span(kAtom_fz)[i]};
    const auto sigma = method.stress(frame);
    if (!method.last_result().converged)
        throw std::runtime_error("Stress reference SCF did not converge");
    if constexpr (requires { method.last_result().n_kpts; }) {
        if (expected_kpoints && method.last_result().n_kpts != expected_kpoints)
            throw std::runtime_error(
                "GPW stress used a different k-point mesh");
    }
    require_close(energy(method.last_result()), e, 1e-8,
                  "Force and stress SCFs disagree on energy");
    for (std::size_t i = 0; i < sigma.size(); ++i) {
        require_close(frame.meta.get(kStress)[i], sigma[i] * frame.box.volume(),
                      1e-10, "Stress call did not publish its virial to Frame");
        if (i >= static_cast<std::size_t>(stress_size))
            require_close(sigma[i], 0, 0, "Unexpected GPW shear");
    }

    // A single structured record on the native solver's ordinary log stream.
    std::cout << std::setprecision(17) << record << ' ' << e;
    for (const auto& atom : forces)
        for (const double f : atom)
            std::cout << ' ' << f / units::kHartreePerBohrInEvPerAngstrom;
    for (int i = 0; i < stress_size; ++i)
        std::cout << ' '
                  << (stress_size == 6
                          ? sigma[i]
                          : sigma[i] / units::kHartreePerBohr3InEvPerAngstrom3);
    std::cout << '\n';
}

} // namespace

int main() {
    try {
        std::cin.exceptions(std::ios::badbit | std::ios::failbit);
        std::string representation, xc;
        double cutoff = 0, tol = 0;
        int max_iter = 0, spin = 0, n = 0;
        std::cin >> representation >> xc >> cutoff >> tol >> max_iter >> spin
            >> n;
        if (n <= 0)
            throw std::invalid_argument("Expected a nonempty periodic Frame");
        cpu::Frame frame(n);
        Mat3 cell{};
        for (int vector = 0; vector < 3; ++vector)
            for (int axis = 0; axis < 3; ++axis)
                std::cin >> cell(axis, vector);
        frame.box.set_cell(cell);
        for (int i = 0; i < n; ++i) {
            int atomic_number = 0;
            double mass = 0, x = 0, y = 0, z = 0;
            std::cin >> atomic_number >> mass >> x >> y >> z;
            frame.span(kAtom_tag)[i] = i + 1;
            frame.span(kAtom_atomic_number)[i] = atomic_number;
            frame.span(kAtom_mass)[i] = mass;
            frame.span(kAtom_x)[i] = x;
            frame.span(kAtom_y)[i] = y;
            frame.span(kAtom_z)[i] = z;
        }
        if (representation == "pw" || representation == "pw.cuda") {
            int expected_electrons = 0, extra_bands = 0;
            int fermi_dirac = 0, n_potentials = 0;
            double degauss = 0, mix_beta = 0;
            std::cin >> expected_electrons >> degauss >> extra_bands >> mix_beta
                >> fermi_dirac >> n_potentials;
            std::vector<
                std::shared_ptr<const electronic::NormConservingPotential>>
                potentials;
            for (int i = 0; i < n_potentials; ++i) {
                int z = 0;
                std::string kind, source;
                std::cin >> z >> kind >> source;
                if (kind == "gth") {
                    potentials.push_back(
                        std::make_shared<cpu::pw::GthPseudopotential>(
                            static_cast<molrs::Element>(z), source));
                } else if (kind == "upf") {
                    potentials.push_back(
                        std::make_shared<cpu::pw::UpfPseudopotential>(source));
                    if (potentials.back()->atomic_number() != z)
                        throw std::invalid_argument(
                            "UPF element does not match the deck");
                } else {
                    throw std::invalid_argument("Expected gth or upf provider");
                }
            }
            auto run_pw = [&](auto& method) {
                method.spin_polarized = spin != 0;
                method.multiplicity = spin + 1;
                if (fermi_dirac)
                    method.smearing = electronic::Smearing::fermi_dirac;
                method.max_iter = max_iter;
                method.tol = tol;
                method.degauss = degauss;
                method.extra_bands = extra_bands;
                method.mix_beta = mix_beta;
                measure(
                    method, frame, [](const auto& r) { return r.energy_ha; },
                    2 * cpu::pw::qe_units::rydberg_to_ev, 6, spin,
                    expected_electrons);
            };
            if (representation == "pw") {
                cpu::pw::PW method{std::move(potentials), cutoff, 1,
                                   functional(xc)};
                run_pw(method);
            } else {
#ifdef ATV_HAS_CUDA
                cuda::pw::PW method{std::move(potentials), cutoff, 1,
                                    functional(xc)};
                run_pw(method);
#else
                throw std::invalid_argument("CUDA PW support was not built");
#endif
            }
#ifdef ATV_HAS_GPW
        } else if (representation == "gpw" || representation == "gpw.cuda") {
            const auto element =
                static_cast<molrs::Element>(frame.span(kAtom_atomic_number)[0]);
            std::string potential, basis;
            double eps_default, strain_step, rel_cutoff, seed_shift_bohr;
            int diis_depth, ngrids, expected_electrons;
            int k_mesh[3];
            std::array<int, 3> first_atom_image{};
            std::cin >> potential >> basis >> eps_default >> diis_depth
                >> strain_step >> ngrids >> rel_cutoff;
            for (int& subdivision : k_mesh)
                std::cin >> subdivision;
            std::cin >> expected_electrons;
            for (int& image : first_atom_image)
                std::cin >> image;
            std::cin >> seed_shift_bohr;
            const std::vector<cpu::gpw::Kind> kinds{
                {element, cpu::gpw::GaussianBasis(element, basis),
                 cpu::gpw::GthPotential(element, potential)}};
            const auto energy = [](const auto& r) { return r.e_total; };
            auto run_gpw = [&](auto& method) {
                if (spin) {
                    method.n_alpha = 2;
                    method.n_beta = 1;
                }
                method.max_iter = max_iter;
                method.tol = tol;
                method.nonperiodic = false;
                method.local_pp_on_grid = true;
                method.use_ot = false;
                method.ngrids = ngrids;
                method.rel_cutoff = rel_cutoff;
                std::copy_n(k_mesh, 3, method.k_mesh);
                method.k_time_reversal = false;
                method.eps_default = eps_default;
                method.diis_depth = diis_depth;
                method.stress_strain_h = strain_step;
                method.stress_axes = {true, true, true};
                auto& first_x = frame.span(kAtom_x)[0];
                const auto target_x = first_x;
                first_x += seed_shift_bohr * units::kBohrInAngstrom;
                method.build(frame);
                if (seed_shift_bohr != 0) {
                    measure(method, frame, energy, units::kHartreeInEv, 3, spin,
                            0, "ATV_PERIODIC_INITIAL");
                    first_x = target_x;
                }
                const int nk = k_mesh[0] * k_mesh[1] * k_mesh[2];
                measure(method, frame, energy, units::kHartreeInEv, 3, spin,
                        expected_electrons, "ATV_PERIODIC_RESULT",
                        nk > 1 ? nk : 0);
                if (first_atom_image != std::array<int, 3>{}) {
                    frame.span(kAtom_x)[0] += first_atom_image[0] * cell(0, 0);
                    frame.span(kAtom_y)[0] += first_atom_image[1] * cell(1, 1);
                    frame.span(kAtom_z)[0] += first_atom_image[2] * cell(2, 2);
                    method.build(frame);
                    measure(method, frame, energy, units::kHartreeInEv, 3, spin,
                            0, "ATV_PERIODIC_TRANSLATED");
                }
            };
            if (representation == "gpw") {
                cpu::gpw::GpwDft method{kinds, cutoff, functional(xc)};
                run_gpw(method);
            } else {
#ifdef ATV_HAS_CUDA
                cuda::gpw::GpwDft method{kinds, cutoff, functional(xc)};
                run_gpw(method);
#else
                throw std::invalid_argument("CUDA GPW support was not built");
#endif
            }
#endif
        } else {
            throw std::invalid_argument("Unsupported periodic method");
        }
    } catch (const std::exception& error) {
        std::cerr << "periodic DFT: " << error.what() << '\n';
        return 1;
    }
}
