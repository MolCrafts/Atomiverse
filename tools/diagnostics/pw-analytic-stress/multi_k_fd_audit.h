#pragma once
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/criterion.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "general_trace.h"
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
namespace pw_analytic_diagnostic {
using namespace ATV;
using Method = cpu::pw::PW;
using Result = cpu::pw::PWResult;
inline void require(bool condition, const char* why) {
    if (!condition)
        throw std::runtime_error(why);
}
inline void close(double a, double b, double tolerance, const char* why) {
    if (!std::isfinite(a) || !std::isfinite(b) || std::abs(a - b) > tolerance) {
        std::fprintf(stderr,
                     "GATE %s actual=%.17g expected=%.17g bound=%.17g\n", why,
                     a, b, tolerance);
        throw std::runtime_error(why);
    }
}
inline void geometry(const cpu::Frame& a, const cpu::Frame& b) {
    require(a.nlocal() == b.nlocal(), "SCF execution atom count");
    for (int i = 0; i < a.nlocal(); ++i) {
        require(a.span(kAtom_type)[i] == b.span(kAtom_type)[i], "SCF species");
        for (auto col : {kAtom_x, kAtom_y, kAtom_z})
            close(a.span(col)[i], b.span(col)[i], 0, "SCF geometry snapshot");
    }
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            close(a.box.cell()(r, c), b.box.cell()(r, c), 0,
                  "SCF cell snapshot");
}
struct Audit {
    const cpu::Frame* expected_geometry = nullptr;
    const cpu::Frame* actual = nullptr;
    cpu::Frame* parent_frame = nullptr;
    cpu::Runtime* runtime = nullptr;
    bool stage_live = false, guard_live = false;
    int builds = 0, destroys = 0, guard_builds = 0, guard_destroys = 0,
        polls = 0;
    std::vector<double> energies, residuals, energy_changes;
};
class Measure final : public cpu::Component {
    Audit& a_;
    Method& m_;

  public:
    Measure(Audit& a, Method& m) : a_(a), m_(m) {}
    std::string name() const override {
        return "pw_analytic_actual_scf_measure";
    }
    void build(const cpu::BuildContext& c) override {
        require(!a_.stage_live && a_.expected_geometry
                    && c.runtime.exec.runtime,
                "SCF stage build");
        geometry(c.frame, *a_.expected_geometry);
        if (a_.parent_frame)
            require(&c.frame == a_.parent_frame,
                    "nested SCF uses actual parent Frame");
        a_.stage_live = true;
        a_.actual = &c.frame;
        a_.runtime = c.runtime.exec.runtime;
        ++a_.builds;
    }
    void launch(const cpu::LaunchContext& c) override {
        require(a_.stage_live && a_.guard_live && a_.actual == &c.frame
                    && a_.runtime == c.runtime.exec.runtime,
                "SCF stage execution ownership");
        geometry(c.frame, *a_.expected_geometry);
        const auto* v = c.runtime.loop.metrics;
        const auto& r = m_.last_result();
        require(v && r.iterations == c.step + 1
                    && a_.energies.size() == std::size_t(c.step),
                "one collector sample per actual SCF iteration");
        require(std::isfinite(v->energy) && std::abs(v->energy) > 1e-3
                    && std::isfinite(v->residual) && v->residual >= 0,
                "actual finite SCF metrics");
        close(v->energy, r.energy_ha * units::kHartreeInEv, 1e-11,
              "measured SCF energy/eV");
        if (a_.energies.empty())
            require(std::isinf(v->energy_change) && v->energy_change > 0,
                    "first SCF delta energy is explicitly unavailable");
        else
            close(v->energy_change, std::abs(v->energy - a_.energies.back()), 0,
                  "actual SCF delta energy");
        a_.energy_changes.push_back(v->energy_change);
        a_.energies.push_back(v->energy);
        a_.residuals.push_back(v->residual);
    }
    void destroy(const cpu::DestroyContext& c) override {
        require(a_.stage_live && a_.actual == &c.frame
                    && a_.runtime == c.runtime.exec.runtime,
                "SCF stage destroy");
        a_.stage_live = false;
        ++a_.destroys;
    }
};
class Guard final : public cpu::Criterion {
    Audit& a_;

  public:
    explicit Guard(Audit& a) : a_(a) {}
    std::string name() const override { return "pw_analytic_actual_scf_guard"; }
    void launch(const cpu::LaunchContext&) override {}
    void build(const cpu::BuildContext&) override {
        require(!a_.guard_live, "guard build");
        a_.guard_live = true;
        ++a_.guard_builds;
    }
    Verdict verdict(const cpu::LoopState& s) override {
        require(a_.guard_live && a_.stage_live && a_.actual == &s.frame
                    && a_.runtime == s.runtime.exec.runtime,
                "guard uses same SCF Frame/runtime");
        require(a_.energies.size() == std::size_t(s.iteration + 1),
                "guard observes each measured iteration");
        close(s.metrics.energy, a_.energies.back(), 0, "guard energy");
        close(s.metrics.residual, a_.residuals.back(), 0, "guard residual");
        if (std::isinf(a_.energy_changes.back()))
            require(std::isinf(s.metrics.energy_change),
                    "guard first delta energy");
        else
            close(s.metrics.energy_change, a_.energy_changes.back(), 0,
                  "guard delta energy");
        ++a_.polls;
        return Verdict::Continue; // Does not replace convergence policy.
    }
    void destroy(const cpu::DestroyContext&) override {
        require(a_.guard_live, "guard destroy");
        a_.guard_live = false;
        ++a_.guard_destroys;
    }
};
inline void complete(const char* label, const Audit& a, const Result& r,
                     double tol, int nspin) {
    require(!a.stage_live && !a.guard_live && a.builds == 1 && a.destroys == 1
                && a.guard_builds == 1 && a.guard_destroys == 1,
            "exactly one completed SCF lifecycle");
    require(r.converged && r.iterations > 2 && a.polls == r.iterations
                && a.energies.size() == std::size_t(r.iterations)
                && a.residuals.size() == a.energies.size()
                && a.energy_changes.size() == a.energies.size(),
            "complete collector matches returned iterations");
    require(a.residuals.front() > tol && a.residuals.back() < tol,
            "real nonzero-to-converged residual");
    close(a.energies.back(), r.energy_ha * units::kHartreeInEv, 1e-11,
          "accepted measured energy");
    require(trace::counts.extract == 1 && trace::counts.assembly == 1
                && trace::counts.legacy_stress == 0
                && trace::counts.davidson
                       == r.n_kpts_local * nspin * (r.iterations + 1),
            "one local all-k/all-spin final sweep and no strained solve");
    std::printf("BAND_AUDIT %s local_k=%d global_k=%d Davidson=%d extract=%d "
                "assembly=%d legacy=%d\n",
                label, r.n_kpts_local, r.n_kpts, trace::counts.davidson,
                trace::counts.extract, trace::counts.assembly,
                trace::counts.legacy_stress);
    std::printf("SCF_AUDIT %s solves=1 builds=1 destroys=1 guard_builds=1 "
                "guard_destroys=1 iterations=%d polls=%d residual_first=%.16g "
                "residual_last=%.16g\n",
                label, r.iterations, a.polls, a.residuals.front(),
                a.residuals.back());
}
inline void metadata(const Result& a, const Result& b) {
    require(a.converged == b.converged && a.iterations == b.iterations
                && a.n_kpts == b.n_kpts && a.n_kpts_local == b.n_kpts_local,
            "reference verdict/iteration/k metadata preserved");
    close(a.energy, b.energy, 1e-10, "reference energy/Ry");
    close(a.energy_ha, b.energy_ha, 1e-10, "reference energy/Ha");
    close(a.magnetization, b.magnetization, 1e-12, "reference magnetization");
    close(a.e_fermi, b.e_fermi, 1e-12, "reference mu");
    close(a.n_elec_occ, b.n_elec_occ, 1e-12, "reference occupation metadata");
    close(a.e_entropy, b.e_entropy, 1e-12, "reference entropy");
}

} // namespace pw_analytic_diagnostic
