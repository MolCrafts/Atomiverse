// Complete public-API solves: run explicitly on a CUDA compute allocation.
#include "../../../examples/nc-extension/pins.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/planewave/qe_units.h"
#include "atomiverse/cuda/planewave/pw_solver.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

using namespace ATV;
namespace {
void require(bool ok, const std::string& message) {
    if (!ok)
        throw std::runtime_error(message);
}
void close(double a, double b, double tol, const std::string& name) {
    require(std::isfinite(a) && std::isfinite(b) && std::abs(a - b) < tol,
            name + ": got " + std::to_string(a) + " expected "
                + std::to_string(b));
}
cpu::Frame hydrogen(bool atom) {
    cpu::Frame f(atom ? 1 : 2);
    f.box = cpu::Box::cubic(4.0);
    for (int i = 0; i < f.nlocal(); ++i) {
        f.span(kAtom_type)[i] = 1;
        f.span(kAtom_mass)[i] = 1.008;
        f.span(kAtom_x)[i] = 1.3 + i * 0.85;
        f.span(kAtom_y)[i] = 1.4;
        f.span(kAtom_z)[i] = 1.7;
    }
    return f;
}
cpu::Frame silicon() {
    cpu::Frame f(2);
    constexpr double a = 5.43;
    Mat3 cell;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            cell(i, j) = (i == j) ? 0.0 : a / 2;
    f.box.set_cell(cell);
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_type)[i] = 14;
        f.span(kAtom_mass)[i] = 28.0855;
        f.span(kAtom_x)[i] = i * (a / 4 + 0.08);
        f.span(kAtom_y)[i] = i * a / 4;
        f.span(kAtom_z)[i] = i * a / 4;
    }
    return f;
}
void metadata(const cpu::pw::PWResult& got, const cpu::pw::PWResult& ref,
              const std::string& label) {
    require(got.converged == ref.converged, label + " convergence");
    require(got.iterations == ref.iterations && got.iterations > 0,
            label + " iteration count");
    require(got.n_kpts == ref.n_kpts && got.n_kpts_local == ref.n_kpts_local
                && got.n_kpts > 0,
            label + " k-points");
    close(got.energy, ref.energy, 1e-8, label + " energy/Ry");
    close(got.energy_ha, ref.energy_ha, 5e-9, label + " energy/Ha");
    close(got.magnetization, ref.magnetization, 1e-8, label + " magnetization");
    close(got.e_fermi, ref.e_fermi, 1e-8, label + " Fermi level");
    close(got.n_elec_occ, ref.n_elec_occ, 1e-8, label + " electron count");
    close(got.e_entropy, ref.e_entropy, 1e-8, label + " entropy");
}
void hydrogen_case(bool spin, bool smear) {
    auto f = hydrogen(spin);
    auto pp = std::make_shared<cpu::pw::GthPseudopotential>(molrs::Element::H,
                                                            "GTH-PBE-q1");
    cuda::pw::PW run{pp, 6.0, 1, cpu::scf::XCType::PBE};
    cuda::pw::PW deriv{pp, 6.0, 1, cpu::scf::XCType::PBE};
    for (auto* method : {&run, &deriv}) {
        method->spin_polarized = spin;
        method->multiplicity = spin ? 2 : 1;
        method->smearing =
            smear ? cpu::pw::Smearing::fermi_dirac : cpu::pw::Smearing::none;
        method->degauss = 0.02;
        method->extra_bands = 2;
        method->max_iter = 100;
        method->tol = 1e-6;
        method->mix_beta = 0.4;
    }
    const auto reference = run.run(f);
    require(reference.converged, "hydrogen reference did not converge");
    const auto force_result = deriv.forces(f);
    const std::string label = std::string("PBE H ") + (spin ? "UKS" : "RKS")
                              + (smear ? " smear" : " fixed");
    metadata(force_result, reference, label + " forces");
    close(f.meta.get(kEnergy),
          force_result.energy * cpu::pw::qe_units::rydberg_to_ev, 1e-9,
          label + " Frame energy");
    if (smear) {
        close(force_result.n_elec_occ, spin ? 1.0 : 2.0, 1e-8,
              label + " occupied electrons");
        require(force_result.e_entropy < 0, label + " nonzero entropy");
    }
    if (!(spin && smear)) {
        const auto stress = deriv.stress(f);
        metadata(deriv.last_result(), reference, label + " stress");
        for (int i = 0; i < 6; ++i) {
            require(std::isfinite(stress[i]), label + " stress finite");
            close(f.meta.get(kStress)[i], stress[i] * f.box.volume(), 1e-10,
                  label + " Frame virial");
        }
    }
    std::printf(
        "PASS %s iterations=%d E=%.12g Ry M=%.9g N=%.9g entropy=%.9g Ry\n",
        label.c_str(), reference.iterations, reference.energy,
        force_result.magnetization, force_result.n_elec_occ,
        force_result.e_entropy);
    // A previous converged result must not survive a later truncated solve.
    run.max_iter = 1;
    deriv.max_iter = 1;
    const auto truncated = run.run(f);
    require(!truncated.converged && truncated.iterations == 1,
            "truncated solve unexpectedly converged");
    metadata(deriv.forces(f), truncated, label + " truncated forces");
    if (!(spin && smear)) {
        deriv.stress(f);
        metadata(deriv.last_result(), truncated, label + " truncated stress");
    }
    std::printf("PASS %s truncated/reuse metadata\n", label.c_str());
}
void qe_silicon_case() {
    cpu::Frame frame(2);
    frame.box = cpu::Box::cubic(6.0);
    const double xyz[2][3] = {{2.0, 2.0, 2.0}, {3.8, 3.2, 3.0}};
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_type)[i] = 14;
        frame.span(kAtom_mass)[i] = 28.0855;
        frame.span(kAtom_x)[i] = xyz[i][0];
        frame.span(kAtom_y)[i] = xyz[i][1];
        frame.span(kAtom_z)[i] = xyz[i][2];
    }
    auto pp = std::make_shared<cpu::pw::GthPseudopotential>(molrs::Element::Si,
                                                            "GTH-PBE-q4");
    cuda::pw::PW gpu{pp, 12.0, 1, cpu::scf::XCType::PBE};
    gpu.tol = 1e-9;
    gpu.max_iter = 200;
    gpu.mix_beta = 0.2;
    gpu.smearing = cpu::pw::Smearing::fermi_dirac;
    gpu.degauss = 0.02;
    gpu.extra_bands = 2;
    const auto result = gpu.forces(frame);
    require(result.converged, "QE Si2 CUDA SCF did not converge");
    close(result.energy_ha, nc_example::energy_ha, nc_example::energy_tol_ha,
          "QE Si2 absolute Mermin energy/Ha");
    double error = 0;
    const auto fx = frame.span(kAtom_fx), fy = frame.span(kAtom_fy),
               fz = frame.span(kAtom_fz);
    for (int i = 0; i < 2; ++i)
        for (int d = 0; d < 3; ++d) {
            const double value = (d == 0   ? fx[i]
                                  : d == 1 ? fy[i]
                                           : fz[i])
                                 / units::kHartreePerBohrInEvPerAngstrom;
            const double target = nc_example::forces_ha_bohr[i][d];
            close(value, target, nc_example::force_tol_ha_bohr,
                  "QE Si2 full force/Ha/Bohr");
            error = std::max(error, std::abs(value - target));
        }
    std::printf("PASS actual QE7.5 Si2 CUDA Mermin E=%.12g Ha |dE|=%.9g Ha "
                "max|dF|=%.9g Ha/Bohr iterations=%d\n",
                result.energy_ha,
                std::abs(result.energy_ha - nc_example::energy_ha), error,
                result.iterations);
}

void silicon_case() {
    auto f = silicon();
    auto cpu_f = f;
    auto pp = std::make_shared<cpu::pw::GthPseudopotential>(molrs::Element::Si,
                                                            "GTH-PBE-q4");
    cpu::pw::PW cpu{pp, 8.0, 1, cpu::scf::XCType::PBE};
    cuda::pw::PW gpu{pp, 8.0, 1, cpu::scf::XCType::PBE};
    cpu.tol = gpu.tol = 1e-6;
    cpu.max_iter = gpu.max_iter = 80;
    cpu.mix_beta = gpu.mix_beta = 0.5;
    cpu.forces(cpu_f);
    const auto result = gpu.forces(f);
    require(result.converged && result.iterations > 0,
            "Si PBE force SCF did not converge");
    double error = 0, amplitude = 0;
    for (auto key : {kAtom_fx, kAtom_fy, kAtom_fz})
        for (int i = 0; i < 2; ++i) {
            const double a = cpu_f.span(key)[i], b = f.span(key)[i];
            require(std::isfinite(a) && std::isfinite(b), "Si force finite");
            error = std::max(error, std::abs(a - b));
            amplitude = std::max(amplitude, std::abs(a));
        }
    require(amplitude > 0.05, "Si force is vacuous");
    require(error < 5e-3, "Si PBE force CPU twin exceeded 5e-3 eV/A");
    close(f.meta.get(kEnergy), cpu_f.meta.get(kEnergy), 1e-4,
          "Si energy CPU twin/eV");
    std::printf("PASS PBE Si nonzero forces max|F|=%.12g max|dF|=%.12g eV/A "
                "iterations=%d\n",
                amplitude, error, result.iterations);
}
} // namespace
int main() {
    try {
        for (bool spin : {false, true})
            for (bool smear : {false, true})
                hydrogen_case(spin, smear);
        silicon_case();
        qe_silicon_case();
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
