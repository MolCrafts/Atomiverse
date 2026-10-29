#pragma once
// Opt-in single-TU diagnostic only; no production sinks or solver flags.
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/planewave/qe_units.h"
#include "atomiverse/cuda/evaluate/evaluate.cuh"
#include "atomiverse/cuda/planewave/pw_solver.cuh"
#include "cuda_uks_detail.h"
#include "cuda_uks_dump.h"
#include "cuda_uks_fixture.h"
#include "cuda_uks_oracles.h"
#include "qe-gamma-uks-fixed-2026-10-28/pins.h"
#include <string>
using namespace ATV;
namespace {
int expected_quad = 0;
void check(cudaError_t e, const char* label) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(label) + ": "
                                 + cudaGetErrorString(e));
}
class ScfProbe final : public cuda::Component {
  public:
    std::string name() const override { return "private_uks_actual_scf_probe"; }
    void build(const cuda::BuildContext&) override { ++diagnostic.scf_calls; }
    void launch(const cuda::LaunchContext&) override {}
};
class RejectStage final : public cuda::Component {
  public:
    bool enabled = false;
    std::string name() const override {
        return "private_uks_once_stage_exception";
    }
    void launch(const cuda::LaunchContext&) override {
        if (enabled) {
            enabled = false;
            throw std::runtime_error(
                "intentional reference SCF stage exception");
        }
    }
};
} // namespace
#include "cuda_uks_cold.h"
#define OWNER_FINISH                                                           \
    "ZN3ATV4cuda2pw8internal25FrozenFunctionalWorkspace6finishERNS2_"          \
    "13DeviceDensityERNS2_16DeviceBandSolverERKNS_"                            \
    "3cpu2pw8PWResultEPSt6vectorISt5arrayIdLm3EESaISF_EEPSE_IdLm6EEPNS0_"      \
    "5FrameE"
dev::FrozenFinishResult real_owner(dev::FrozenFunctionalWorkspace*,
                                   dev::DeviceDensity&, dev::DeviceBandSolver&,
                                   const cpu::pw::PWResult&,
                                   std::vector<std::array<double, 3>>*,
                                   std::array<double, 6>*,
                                   cuda::Frame*) asm("__real__" OWNER_FINISH);
dev::FrozenFinishResult wrap_owner(dev::FrozenFunctionalWorkspace*,
                                   dev::DeviceDensity&, dev::DeviceBandSolver&,
                                   const cpu::pw::PWResult&,
                                   std::vector<std::array<double, 3>>*,
                                   std::array<double, 6>*,
                                   cuda::Frame*) asm("__wrap__" OWNER_FINISH);
dev::FrozenFinishResult
wrap_owner(dev::FrozenFunctionalWorkspace* self, dev::DeviceDensity& den,
           dev::DeviceBandSolver& solver, const cpu::pw::PWResult& r,
           std::vector<std::array<double, 3>>* F, std::array<double, 6>* W,
           cuda::Frame* parent) {
    ++diagnostic.owner_calls;
    const auto saved = r;
    struct Active {
        Active() { diagnostic.inside = true; }
        ~Active() { diagnostic.inside = false; }
    } active;
    dev::FrozenFinishResult route;
    try {
        route = real_owner(self, den, solver, r, F, W, parent);
    } catch (...) {
        require(same(r, saved), "exception preserves full accepted metadata");
        throw;
    }
    require(same(r, saved), "preserve full accepted metadata");
    require(route == dev::FrozenFinishResult::complete,
            "Si3 physical fixture remains inside qualified domain");
    const auto c = self->candidate();
    auto& d = diagnostic;
    d.reference = saved;
    d.outcome = c.outcome;
    require(bool(c.terms) == c.outcome.stress_requested
                && bool(c.forces) == c.outcome.forces_requested,
            "only requested tensor/force buffers allocated");
    if (c.outcome.stress_requested) {
        const auto& domain = c.outcome.spin_domain;
        require(c.outcome.spin_domain_evaluated && domain.finite
                    && !domain.violations,
                "actual native spin XC interior certificate");
        std::printf("NATIVE_UKS_DOMAIN min_a=%.17g min_b=%.17g "
                    "max_abs_zeta=%.17g flags=%u finite=%u\n",
                    domain.minimum[0], domain.minimum[1], domain.max_abs_zeta,
                    domain.violations, domain.finite);
    }
    require(c.fixed_spin && c.fixed_spin->alpha == 7 && c.fixed_spin->beta == 5,
            "typed physical7+5");
    require(c.live_bands == 7 && c.orbitals.bands == 14,
            "packed2nb exact shape");
    require(den.stream() == solver.stream(), "borrowed stream match");
    for (int spin = 0; spin < 2; ++spin) {
        const auto& a = c.outcome.spin_auxiliary[spin];
        require(a.complete && a.max_residual_ha < 1e-6
                    && a.max_normalization_error < 1e-8,
                "independent physical auxiliary status");
        std::printf("OWNER_SPIN spin=%d live_bands=%d packed_bands=%d "
                    "Hpsi_Ha=%.17g norm=%.17g fcount=%.17g orbitalNe=%.17g "
                    "rhoG0Ne=%.17g prefix_exact=%d\n",
                    spin, c.live_bands, c.orbitals.bands, a.max_residual_ha,
                    a.max_normalization_error,
                    c.outcome.fixed_device_f_count[spin],
                    c.outcome.spin_orbital_electrons[spin],
                    c.outcome.spin_rho_G0_electrons[spin],
                    c.outcome.fixed_spin_prefix_exact);
    }
    require(d.outcome.certificate.reference_unchanged,
            "all accepted total/spin fields preserved");
    if (!d.want_dump)
        return route;
    const auto& p = *c.geometry;
    d.potential = p.potential;
    d.wave = p.gv_k[0];
    d.density = p.gv0;
    d.positions = p.ws.atom_pos_bohr;
    d.volume = p.vol_bohr;
    d.occupations.resize(c.orbitals.bands);
    d.coefficients.resize(std::size_t(c.orbitals.bands) * d.wave.count());
    d.rho_ref.resize(d.density.count());
    d.rho_orb.resize(d.density.count());
    for (int s = 0; s < 2; ++s) {
        d.spin_ref[s].resize(d.density.count());
        d.spin_orb[s].resize(d.density.count());
    }
    if (c.outcome.forces_requested)
        d.forces.resize(d.positions.size());
    // All asynchronous dump destinations have executable-owned lifetime. Drain while
    // they and the live solver stream still exist on either normal or error paths.
    try {
        auto copy = [&](void* dst, const void* src, std::size_t n) {
            check(cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToHost,
                                  solver.stream()),
                  "external diagnostic same-state dump");
        };
        copy(d.occupations.data(), c.orbitals.f,
             d.occupations.size() * sizeof(double));
        copy(d.coefficients.data(), c.orbitals.C,
             d.coefficients.size() * sizeof(Complex));
        copy(d.rho_ref.data(), c.reference_density,
             d.rho_ref.size() * sizeof(Complex));
        copy(d.rho_orb.data(), c.orbital_density,
             d.rho_orb.size() * sizeof(Complex));
        for (int s = 0; s < 2; ++s) {
            copy(d.spin_ref[s].data(), c.reference_spin_density[s],
                 d.spin_ref[s].size() * sizeof(Complex));
            copy(d.spin_orb[s].data(), c.orbital_spin_density[s],
                 d.spin_orb[s].size() * sizeof(Complex));
        }
        if (c.outcome.stress_requested)
            copy(d.terms.data(), c.terms, sizeof(d.terms));
        if (c.outcome.forces_requested)
            copy(d.forces.data(), c.forces,
                 d.forces.size() * sizeof(d.forces[0]));
        check(cudaStreamSynchronize(solver.stream()),
              "external diagnostic dump fence");
    } catch (...) {
        cudaStreamSynchronize(solver.stream());
        throw;
    }
    return route;
}
#define FINAL                                                                  \
    "ZN3ATV4cuda2pw8internal16DeviceBandSolver17solve_final_bandsERNS_"        \
    "3cpu2pw8internal11PWWorkspaceENS2_20FinalBandDestinationEi"
dev::FinalBandStatus real_final(dev::DeviceBandSolver*, hp::PWWorkspace&,
                                dev::FinalBandDestination,
                                int) asm("__real__" FINAL);
dev::FinalBandStatus wrap_final(dev::DeviceBandSolver*, hp::PWWorkspace&,
                                dev::FinalBandDestination,
                                int) asm("__wrap__" FINAL);
dev::FinalBandStatus wrap_final(dev::DeviceBandSolver* self, hp::PWWorkspace& w,
                                dev::FinalBandDestination d, int budget) {
    const int actual = diagnostic.budget == 1 ? 1 : budget;
    const auto s = real_final(self, w, d, actual);
    diagnostic.statuses.push_back(s);
    std::printf("AUX_FINAL budget=%d max_residual_Ha=%.17g "
                "max_norm_error=%.17g complete=%d\n",
                actual, s.max_residual_ha, s.max_normalization_error,
                s.complete);
    return s;
}
extern "C" cudaError_t __real_cudaMemcpyAsync(void*, const void*, std::size_t,
                                              cudaMemcpyKind, cudaStream_t);
extern "C" cudaError_t __wrap_cudaMemcpyAsync(void* dst, const void* src,
                                              std::size_t n,
                                              cudaMemcpyKind kind,
                                              cudaStream_t s) {
    auto e = __real_cudaMemcpyAsync(dst, src, n, kind, s);
    if (e == cudaSuccess && diagnostic.inside && diagnostic.reject_certificate
        && kind == cudaMemcpyDeviceToHost
        && n == sizeof(dev::FrozenDerivativeCertificate)) {
        e = cudaStreamSynchronize(s);
        if (e == cudaSuccess)
            static_cast<dev::FrozenDerivativeCertificate*>(dst)->finite = 0;
    }
    return e;
}
int run_cuda_uks_total(int argc, char** argv) {
    try {
        const auto& ref = pw_cuda_uks_ref::get(argc > 1 ? argv[1] : "pbe");
        std::string mode = argc > 2 ? argv[2] : "nested";
        if (mode.rfind("--", 0) == 0)
            mode.erase(0, 2);
        expected_quad = ref.quad;
        if (mode == "offline") {
            require(argc == 4, "offline dump path");
            private_dump::read(argv[3]);
            independent::frozen_strain(ref.type);
            independent::frozen_force();
            return 0;
        }
        const bool nested = mode.rfind("nested", 0) == 0 || mode == "retry"
                            || mode == "fd" || mode == "dump";
        const bool force = mode == "force" || mode == "nested"
                           || mode == "nested-force" || mode == "retry"
                           || mode == "fd" || mode == "dump"
                           || mode == "exhausted";
        const bool stress = mode == "stress" || mode == "nested"
                            || mode == "nested-stress" || mode == "retry"
                            || mode == "fd" || mode == "dump";
        require(force || stress || mode == "energy" || mode == "nested-energy",
                "known private mode");
        diagnostic.want_dump = mode == "fd" || mode == "dump";
        auto f = pw_frozen::silicon();
        for (int i = 0; i < 3; ++i) {
            f.span(kAtom_fx)[i] = 10 + i;
            f.span(kAtom_fy)[i] = 20 + i;
            f.span(kAtom_fz)[i] = 30 + i;
        }
        f.meta.get(kEnergy) = 1234;
        f.meta.get(kStress) = {1, 2, 3, 4, 5, 6};
        const auto original = f;
        ScfProbe scf;
        RejectStage stage;
        cuda::pw::PW method(std::make_shared<cpu::pw::GthPseudopotential>(
                                molrs::Element::Si, "GTH-PBE-q4"),
                            12, 1, ref.type);
        method.spin_polarized = true;
        method.multiplicity = 3;
        method.tol = 1e-10;
        method.max_iter = mode == "exhausted" ? 1 : 200;
        method.mix_beta = .2;
        method.pulay_depth = 6;
        method.extra_bands = 0;
        method.on(ATV::pw::Phase::check).add_ref(scf, -100);
        std::array<double, 6> sigma{};
        auto unchanged = [&] {
            for (auto col : {kAtom_fx, kAtom_fy, kAtom_fz})
                for (int i = 0; i < 3; ++i)
                    require(f.span(col)[i] == original.span(col)[i],
                            "failed/unrequested force preserved");
            require(f.meta.get(kEnergy) == original.meta.get(kEnergy)
                        && f.meta.get(kStress) == original.meta.get(kStress),
                    "failed scalar channels preserved");
        };
        if (nested) {
            if (mode == "retry") {
                stage.enabled = true;
                method.on(ATV::pw::Phase::check).add_ref(stage);
            }
            cuda::Evaluate parent(force, stress);
            parent.set_evaluator_ref(method);
            parent.build();
            if (mode == "retry") {
                for (int kind = 0; kind < 3; ++kind) {
                    diagnostic.budget = kind == 1 ? 1 : 20;
                    diagnostic.reject_certificate = kind == 2;
                    const int before = diagnostic.scf_calls;
                    bool failed = false;
                    try {
                        parent.run(f);
                    } catch (const std::runtime_error& e) {
                        failed = true;
                        std::printf("EXPECTED_FAILURE kind=%d message=%s\n",
                                    kind, e.what());
                    }
                    require(failed && diagnostic.scf_calls == before + 1,
                            "single SCF failed call");
                    unchanged();
                    require(kind == 0 ? !method.last_result().converged
                                      : method.last_result().converged,
                            "failure preserves actual reference convergence "
                            "metadata");
                    if (kind == 1)
                        require(
                            !diagnostic.statuses.back().complete,
                            "real one-expansion auxiliary residual rejection");
                }
                diagnostic.budget = 20;
                diagnostic.reject_certificate = false;
            }
            const int before = diagnostic.scf_calls;
            parent.run(f);
            require(diagnostic.scf_calls == before + 1,
                    "single SCF accepted nested call");
            if (stress)
                for (int j = 0; j < 6; ++j)
                    sigma[j] = f.meta.get(kStress)[j] / f.box.volume();
        } else if (mode == "exhausted") {
            bool rejected = false;
            try {
                method.forces(f);
            } catch (const std::runtime_error& e) {
                rejected =
                    std::string(e.what())
                    == "independent spin auxiliary eigenstate incomplete";
                if (!rejected)
                    throw;
            }
            require(!method.last_result().converged
                        && method.last_result().iterations == 1,
                    "exhausted real SCF metadata false");
            require(!diagnostic.statuses.empty(),
                    "exhausted actual aux evaluated");
            if (rejected) {
                require(!diagnostic.statuses.back().complete,
                        "only actual aux failure rejects");
                unchanged();
            }
            std::printf("PASS CUDA_PW_UKS_EXHAUSTED auxiliary_rejected=%d "
                        "actual_Hpsi=%.17g policy_changed=0\n",
                        rejected, diagnostic.statuses.back().max_residual_ha);
            return 0;
        } else if (stress)
            sigma = method.stress(f);
        else if (force)
            method.forces(f);
        else
            method.run(std::as_const(f));
        const auto& accepted = method.last_result();
        require(accepted.converged, "original200SCF converged");
        close(accepted.energy_ha, ref.E, 1e-5, "actualQE energy");
        close(accepted.magnetization, 2., 1e-12, "fixed magnetization");
        const bool derivative = force || stress;
        const int failures = mode == "retry" ? 3 : 0;
        require(cold_calls == (derivative ? 1 + (mode == "retry" ? 3 : 0) : 0),
                "energy zero or one cold owner per attempted derivative SCF");
        require(diagnostic.owner_calls
                    == (derivative ? 1 + (mode == "retry" ? 2 : 0) : 0),
                "request owner call budget");
        require(diagnostic.scf_calls == 1 + failures,
                "request actual SCF budget");
        if (derivative) {
            require(same(accepted, diagnostic.reference),
                    "last_result full accepted metadata unchanged");
            require(diagnostic.outcome.published
                        && diagnostic.outcome.spin_channels == 2
                        && diagnostic.outcome.fixed_spin_prefix_exact,
                    "certified single publication");
            require(diagnostic.statuses.size()
                        == std::size_t(mode == "retry" ? 5 : 2),
                    "two accepted final solves plus genuine failed attempts");
        }
        if (nested || derivative)
            close(f.meta.get(kEnergy),
                  accepted.energy * cpu::pw::qe_units::rydberg_to_ev, 1e-10,
                  "publish accepted physical E exactly once");
        else
            require(f.meta.get(kEnergy) == 1234,
                    "const energy-only Frame energy preserved");
        double df = 0, ds = 0;
        if (force)
            for (int i = 0; i < 3; ++i)
                for (int a = 0; a < 3; ++a) {
                    const auto col = a == 0   ? kAtom_fx
                                     : a == 1 ? kAtom_fy
                                              : kAtom_fz;
                    const double actual =
                        f.span(col)[i] / units::kHartreePerBohrInEvPerAngstrom;
                    close(actual, ref.F[i][a], 1e-4, "actualQE full9 force");
                    df = std::max(df, std::abs(actual - ref.F[i][a]));
                    std::printf(
                        "FORCE atom=%d axis=%d actual_HaBohr=%.17g ref=%.17g\n",
                        i, a, actual, ref.F[i][a]);
                }
        else
            for (auto col : {kAtom_fx, kAtom_fy, kAtom_fz})
                for (int i = 0; i < 3; ++i)
                    require(f.span(col)[i] == original.span(col)[i],
                            "unrequested force sentinel preserved");
        if (stress)
            for (int j = 0; j < 6; ++j) {
                close(sigma[j], ref.sigma[j], .002, "actualQE full6 stress");
                close(f.meta.get(kStress)[j], sigma[j] * f.box.volume(), 1e-10,
                      "stress published once");
                ds = std::max(ds, std::abs(sigma[j] - ref.sigma[j]));
                std::printf("SIGMA lane=%d actual_eVA3=%.17g ref=%.17g\n", j,
                            sigma[j], ref.sigma[j]);
            }
        else
            require(f.meta.get(kStress) == original.meta.get(kStress),
                    "unrequested stress sentinel preserved");
        if (mode == "fd") {
            independent::frozen_strain(ref.type);
            independent::frozen_force();
        }
        if (mode == "dump") {
            require(argc == 4, "dump output path");
            private_dump::write(argv[3]);
        }
        std::printf(
            "PASS CUDA_PW_UKS_TOTAL model=%s mode=%s E_Ha=%.17g dE_Ha=%.17g "
            "maxF_HaBohr=%.17g maxSigma_eVA3=%.17g SCF=%d owner_calls=%d "
            "final_solves=%zu domain_evaluated=%d\n",
            ref.name, mode.c_str(), accepted.energy_ha,
            std::abs(accepted.energy_ha - ref.E), df, ds, diagnostic.scf_calls,
            diagnostic.owner_calls, diagnostic.statuses.size(),
            int(diagnostic.outcome.spin_domain_evaluated));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL CUDA_PW_UKS_TOTAL %s\n", e.what());
        return 1;
    }
}

#define PROJECT                                                                \
    "ZN3ATV4cuda2pw8internal23project_frozen_nonlocalENS2_"                    \
    "28FrozenNonlocalDeviceGeometryENS2_20FrozenBandDeviceViewENS2_"           \
    "27FrozenNonlocalDeviceScratchEP13cublasContextP11CUstream_st"
void real_project(dev::FrozenNonlocalDeviceGeometry, dev::FrozenBandDeviceView,
                  dev::FrozenNonlocalDeviceScratch, cublasHandle_t,
                  cudaStream_t) asm("__real__" PROJECT);
void wrap_project(dev::FrozenNonlocalDeviceGeometry, dev::FrozenBandDeviceView,
                  dev::FrozenNonlocalDeviceScratch, cublasHandle_t,
                  cudaStream_t) asm("__wrap__" PROJECT);
void wrap_project(dev::FrozenNonlocalDeviceGeometry g,
                  dev::FrozenBandDeviceView s,
                  dev::FrozenNonlocalDeviceScratch w, cublasHandle_t b,
                  cudaStream_t stream) {
    real_project(g, s, w, b, stream);
}
