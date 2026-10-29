// CPU preflight for the exact physical cases of gpw-driver-cuda.
// Complete-method diagnostic only; never registered with CTest.
#include "fixtures.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

using namespace ATV;
namespace fixture = ATV::diagnostics::gpw_driver;
namespace {
constexpr double ionic_step_a = 1e-4;
constexpr double ionic_fd_tolerance_eva = 1e-3;
constexpr double strain_step = .002;
constexpr double strain_fd_tolerance_eva3 = 2e-4;
void require(bool pass, const std::string& label) {
    if (!pass)
        throw std::runtime_error(label);
}
void close(double got, double expected, double tolerance,
           const std::string& label) {
    if (!std::isfinite(got) || !std::isfinite(expected)
        || std::abs(got - expected) > tolerance) {
        std::ostringstream message;
        message.precision(17);
        message << label << ": got " << got << ", expected " << expected
                << ", tolerance " << tolerance;
        throw std::runtime_error(message.str());
    }
}
void preserved(const cpu::Frame& frame, const cpu::Frame& before,
               const std::string& label) {
    require(fixture::forces(frame) == fixture::forces(before)
                && frame.meta.get(kStress) == before.meta.get(kStress)
                && frame.meta.get(kEnergy) == before.meta.get(kEnergy),
            label);
}
void run(const fixture::Case& c, int index) {
    auto frame = fixture::geometry(false);
    const auto initial = frame;
    cpu::gpw::GpwDft method(fixture::kinds(), fixture::cutoff[index],
                            fixture::xc(c));
    fixture::configure(method, c);
    method.build(frame);
    const auto reference = method.run(std::as_const(frame));
    require(reference.converged && reference.iters > 0,
            "CPU fixture SCF must converge");
    preserved(frame, initial, "const run must retain caller outputs");
    if (c.kpoints)
        require(reference.n_kpts == 2, "preflight must exercise both k points");
    const auto force_result = method.forces(frame);
    require(force_result.converged, "force reference must converge");
    close(force_result.e_total, reference.e_total, 1e-9,
          "repeated reference energy/Ha");
    const auto force = fixture::forces(frame);
    double peak = 0;
    for (double value : force) {
        require(std::isfinite(value), "every force component must be finite");
        peak = std::max(peak, std::abs(value));
    }
    require(peak > 1e-3 && std::abs(force[1]) > 1e-4,
            "ionic probe must carry nonzero force");
    const auto sigma = method.stress(frame);
    require(method.last_result().converged, "stress reference must converge");
    require(fixture::forces(frame) == force,
            "stress preserves the force vector");
    for (int k = 0; k < 6; ++k)
        close(frame.meta.get(kStress)[k], sigma[k] * frame.box.volume(), 1e-10,
              "Frame extensive virial conversion");
    // Same complete-method ionic energy derivative as the existing family
    // diagnostic, with the predeclared 1e-3 eV/A threshold retained.
    double ionic_energy[2];
    for (int sign = 0; sign < 2; ++sign) {
        auto displaced = fixture::geometry(false);
        displaced.span(kAtom_x)[1] += (sign ? -ionic_step_a : ionic_step_a);
        const auto sample = method.run(displaced);
        require(sample.converged, "ionic FD sample must converge");
        ionic_energy[sign] = displaced.meta.get(kEnergy);
    }
    const double ionic_fd =
        -(ionic_energy[0] - ionic_energy[1]) / (2 * ionic_step_a);
    close(force[1], ionic_fd, ionic_fd_tolerance_eva,
          "independent ionic energy derivative/eV/A");
    // Independent fully reconverged xx strain; production stress uses its
    // fixed-P Lagrangian. Neither calculation reads the other's derivative.
    double strain_energy[2];
    for (int sign = 0; sign < 2; ++sign) {
        auto strained = fixture::geometry(false);
        const double factor = 1 + (sign ? -strain_step : strain_step);
        auto cell = strained.box.cell();
        for (int j = 0; j < 3; ++j)
            cell(0, j) *= factor;
        strained.box.set_cell(cell);
        for (int atom = 0; atom < strained.nlocal(); ++atom)
            strained.span(kAtom_x)[atom] *= factor;
        const auto sample = method.run(strained);
        require(sample.converged, "strain FD sample must converge");
        strain_energy[sign] = strained.meta.get(kEnergy);
    }
    const double strain_fd = -(strain_energy[0] - strain_energy[1])
                             / (2 * strain_step * initial.box.volume());
    close(sigma[0], strain_fd, strain_fd_tolerance_eva3,
          "independent reconverged xx stress/eV/A3");
    method.max_iter = 1;
    frame = fixture::geometry(false);
    const auto truncated = method.run(std::as_const(frame));
    require(!truncated.converged && truncated.iters == 1,
            "budget truncation must replace earlier verdict");
    preserved(frame, initial, "const truncated run preserves caller outputs");
    for (int derivative = 0; derivative < 2; ++derivative) {
        bool failed = false;
        try {
            if (derivative)
                method.stress(frame);
            else
                method.forces(frame);
        } catch (const std::runtime_error&) {
            failed = true;
        }
        require(
            failed && !method.last_result().converged
                && method.last_result().iters == 1,
            "unconverged derivative must fail with its own exhausted status");
        preserved(frame, initial,
                  "failed derivative must preserve caller outputs");
    }
    method.max_iter = 200;
    auto shifted = fixture::geometry(true);
    const auto reused = method.run(std::as_const(shifted));
    cpu::gpw::GpwDft fresh(fixture::kinds(), fixture::cutoff[index],
                           fixture::xc(c));
    fixture::configure(fresh, c);
    fresh.build(shifted);
    const auto new_result = fresh.run(std::as_const(shifted));
    require(reused.converged && new_result.converged,
            "changed geometry must converge on reused and fresh methods");
    close(reused.e_total, new_result.e_total, 1e-9,
          "changed geometry reuse energy/Ha");
    std::printf("PASS GPW_PREFLIGHT %s-%.0f E_Ha=%.12g iterations=%d "
                "force_fd_error=%.6g xx_stress_fd_error=%.6g\n",
                c.name, fixture::cutoff[index], reference.e_total,
                reference.iters, std::abs(force[1] - ionic_fd),
                std::abs(sigma[0] - strain_fd));
    std::fflush(stdout);
}
} // namespace
int main() {
    int failures = 0;
    for (const auto& c : fixture::cases)
        for (int index = 0; index < 2; ++index) {
            try {
                run(c, index);
            } catch (const std::exception& e) {
                ++failures;
                std::fprintf(stderr, "FAIL GPW_PREFLIGHT %s-%.0f %s\n", c.name,
                             fixture::cutoff[index], e.what());
            }
        }
    return failures ? 1 : 0;
}
