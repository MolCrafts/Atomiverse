#pragma once
// Diagnostic only: actual finite-domain routing, candidate and old-path compatibility.
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/planewave/internal/xc_quadrature.h"
#include "atomiverse/cpu/planewave/qe_units.h"
#include "atomiverse/cuda/evaluate/evaluate.cuh"
#include "atomiverse/cuda/planewave/internal/frozen_functional_virial.cuh"
#include "atomiverse/cuda/planewave/pw_solver.cuh"
#include "cuda_uks_domain_fixture.h"
#include <bit>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
using namespace ATV;
using pw_domain_fixture::geometry;
namespace dev = cuda::pw::internal;
namespace hp = cpu::pw::internal;
namespace {
void require(bool x, const char* s) {
    if (!x)
        throw std::runtime_error(s);
}
void check(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
bool same(const cpu::pw::PWResult& a, const cpu::pw::PWResult& b) {
    auto d = [](double x, double y) {
        return std::bit_cast<std::uint64_t>(x)
               == std::bit_cast<std::uint64_t>(y);
    };
    return d(a.energy, b.energy) && d(a.energy_ha, b.energy_ha)
           && d(a.magnetization, b.magnetization) && d(a.e_fermi, b.e_fermi)
           && d(a.n_elec_occ, b.n_elec_occ) && d(a.e_entropy, b.e_entropy)
           && a.converged == b.converged && a.iterations == b.iterations
           && a.n_kpts == b.n_kpts && a.n_kpts_local == b.n_kpts_local;
}
int scfs = 0, owners = 0, commits = 0;
int final_budget = 20;
bool reject_certificate = false;
std::vector<std::array<double, 3>> legacy_force;
std::array<double, 6> legacy_sigma{};
bool inside = false;
dev::FrozenSpinDomainStatus* packet = nullptr;
cpu::pw::PWResult reference;
std::vector<dev::FinalBandStatus> auxiliary;
dev::FrozenFinishResult last_route = dev::FrozenFinishResult::complete;
dev::FrozenSpinDomainStatus domain{};
class Scf final : public cuda::Component {
  public:
    std::string name() const override { return "actual_h2_domain_scf_counter"; }
    void build(const cuda::BuildContext&) override { ++scfs; }
    void launch(const cuda::LaunchContext&) override {}
};
class RejectStage final : public cuda::Component {
  public:
    bool enabled = false;
    std::string name() const override { return "domain_once_stage_exception"; }
    void launch(const cuda::LaunchContext&) override {
        if (enabled) {
            enabled = false;
            throw std::runtime_error("intentional stage failure");
        }
    }
};

} // namespace
#define FINISH                                                                 \
    "ZN3ATV4cuda2pw8internal25FrozenFunctionalWorkspace6finishERNS2_"          \
    "13DeviceDensityERNS2_16DeviceBandSolverERKNS_"                            \
    "3cpu2pw8PWResultEPSt6vectorISt5arrayIdLm3EESaISF_EEPSE_IdLm6EEPNS0_"      \
    "5FrameE"
dev::FrozenFinishResult real_finish(dev::FrozenFunctionalWorkspace*,
                                    dev::DeviceDensity&, dev::DeviceBandSolver&,
                                    const cpu::pw::PWResult&,
                                    std::vector<std::array<double, 3>>*,
                                    std::array<double, 6>*,
                                    cuda::Frame*) asm("__real__" FINISH);
dev::FrozenFinishResult wrap_finish(dev::FrozenFunctionalWorkspace*,
                                    dev::DeviceDensity&, dev::DeviceBandSolver&,
                                    const cpu::pw::PWResult&,
                                    std::vector<std::array<double, 3>>*,
                                    std::array<double, 6>*,
                                    cuda::Frame*) asm("__wrap__" FINISH);
dev::FrozenFinishResult
wrap_finish(dev::FrozenFunctionalWorkspace* self, dev::DeviceDensity& den,
            dev::DeviceBandSolver& solver, const cpu::pw::PWResult& r,
            std::vector<std::array<double, 3>>* f, std::array<double, 6>* w,
            cuda::Frame* target) {
    ++owners;
    reference = r;
    inside = true;
    packet = nullptr;
    struct Scope {
        ~Scope() { inside = false; }
    } scope;
    dev::FrozenFinishResult result;
    try {
        result = real_finish(self, den, solver, r, f, w, target);
    } catch (...) {
        cudaStreamSynchronize(solver.stream());
        require(same(r, reference), "failed finish metadata preserved");
        throw;
    }
    require(same(r, reference), "finish metadata preserved");
    require(den.stream() == solver.stream(), "actual borrowed stream");
    require(packet, "actual32B domain status observed");
    domain = *packet;
    std::printf("ACTUAL_DOMAIN min_a=%.17g min_b=%.17g max_abs_zeta=%.17g "
                "flags=%u finite=%u route=%s\n",
                domain.minimum[0], domain.minimum[1], domain.max_abs_zeta,
                domain.violations, domain.finite,
                result == dev::FrozenFinishResult::complete
                    ? "complete"
                    : "unsupported_domain");
    last_route = result;
    if (result == dev::FrozenFinishResult::complete) {
        auto c = self->candidate();
        require(c.fixed_spin && c.fixed_spin->alpha == 1
                    && c.fixed_spin->beta == 1,
                "actual physical1+1");
        require(c.outcome.spin_domain_evaluated && c.outcome.spin_domain.finite
                    && !c.outcome.spin_domain.violations,
                "interior certified native domain");
    }
    return result;
}
#define COMMIT                                                                 \
    "ZN3ATV4cuda2pw8internal25FrozenFunctionalWorkspace15commit_"              \
    "fallbackERNS2_13DeviceDensityERKNS_"                                      \
    "3cpu2pw8PWResultERKSt6vectorISt5arrayIdLm3EESaISD_EERKSC_IdLm6EEPNS0_"    \
    "5FrameE"
void real_commit(dev::FrozenFunctionalWorkspace*, dev::DeviceDensity&,
                 const cpu::pw::PWResult&,
                 const std::vector<std::array<double, 3>>&,
                 const std::array<double, 6>&,
                 cuda::Frame*) asm("__real__" COMMIT);
void wrap_commit(dev::FrozenFunctionalWorkspace*, dev::DeviceDensity&,
                 const cpu::pw::PWResult&,
                 const std::vector<std::array<double, 3>>&,
                 const std::array<double, 6>&,
                 cuda::Frame*) asm("__wrap__" COMMIT);
void wrap_commit(dev::FrozenFunctionalWorkspace* self, dev::DeviceDensity& den,
                 const cpu::pw::PWResult& r,
                 const std::vector<std::array<double, 3>>& f,
                 const std::array<double, 6>& w, cuda::Frame* target) {
    const auto saved = r;
    require(last_route == dev::FrozenFinishResult::unsupported_domain,
            "typed finite domain miss before legacy commit");
    legacy_force = f;
    legacy_sigma = w;
    real_commit(self, den, r, f, w, target);
    require(same(r, saved), "legacy candidate commit preserves metadata");
    ++commits;
    std::printf("FALLBACK_COMMIT force_rows=%zu parent=%d input_F=eVA "
                "input_sigma=eVA3 staged_publisher=1\n",
                f.size(), target != nullptr);
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
    require(budget == 20, "original auxiliary budget unchanged");
    const int actual = final_budget == 1 ? 1 : budget;
    auto status = real_final(self, w, d, actual);
    auxiliary.push_back(status);
    std::printf("AUX budget=%d complete=%d Hpsi=%.17g norm=%.17g\n", actual,
                status.complete, status.max_residual_ha,
                status.max_normalization_error);
    return status;
}
extern "C" cudaError_t __real_cudaMemcpyAsync(void*, const void*, std::size_t,
                                              cudaMemcpyKind, cudaStream_t);
extern "C" cudaError_t __wrap_cudaMemcpyAsync(void* dst, const void* src,
                                              std::size_t n,
                                              cudaMemcpyKind kind,
                                              cudaStream_t s) {
    auto result = __real_cudaMemcpyAsync(dst, src, n, kind, s);
    if (result == cudaSuccess && inside && kind == cudaMemcpyDeviceToHost
        && n == sizeof(dev::FrozenSpinDomainStatus)) {
        require(!packet, "one actual domain packet");
        packet = static_cast<dev::FrozenSpinDomainStatus*>(dst);
    }
    if (result == cudaSuccess && inside && reject_certificate
        && kind == cudaMemcpyDeviceToHost
        && n == sizeof(dev::FrozenDerivativeCertificate)) {
        check(cudaStreamSynchronize(s));
        static_cast<dev::FrozenDerivativeCertificate*>(dst)->finite = 0;
    }
    return result;
}
int run_cuda_uks_domain(int argc, char** argv) {
    try {
        std::string mode = argc > 1 ? argv[1] : "observe";
        if (mode.rfind("--", 0) == 0)
            mode.erase(0, 2);
        const bool strict = mode == "retry", standalone = mode == "standalone";
        std::ifstream baseline;
        if (strict || standalone) {
            require(argc == 3, "legacy baseline path");
            baseline.open(argv[2]);
            require(bool(baseline), "legacy values readable");
        }
        auto pp = std::make_shared<cpu::pw::GthPseudopotential>(
            molrs::Element::H, "GTH-PBE-q1");
        for (double box : {12., 4.}) {
            auto f = geometry(box);
            hp::PWProblem p;
            hp::setup_pw_problem(f, *pp, 2., 1, p);
            std::printf("ACTUAL_QUERY box_A=%.17g Ne=%d Na=1 Nb=1 waveG=%d "
                        "base=%dx%dx%d quad=%dx%dx%d Ecut_Ry=2 budget=200 "
                        "tol=1e-10 mix=.2 SCF=0\n",
                        box, p.n_elec, p.gv_k[0].count(), p.ws.fft_nx,
                        p.ws.fft_ny, p.ws.fft_nz,
                        hp::xc_quadrature_dimension(p.ws.fft_nx, false),
                        hp::xc_quadrature_dimension(p.ws.fft_ny, false),
                        hp::xc_quadrature_dimension(p.ws.fft_nz, false));
        }
        Scf scf;
        RejectStage stage;
        cuda::pw::PW method(pp, 2., 1, cpu::xc::XCType::LSDA);
        method.spin_polarized = true;
        method.multiplicity = 1;
        method.max_iter = 200;
        method.tol = 1e-10;
        method.mix_beta = .2;
        method.pulay_depth = 6;
        method.extra_bands = 0;
        method.on(ATV::pw::Phase::check).add_ref(scf, -100);
        if (strict)
            method.on(ATV::pw::Phase::check).add_ref(stage);
        cuda::Evaluate parent(true, true);
        if (!standalone) {
            parent.set_evaluator_ref(method);
            parent.build();
        }
        if (strict) {
            auto failed = geometry(12.);
            const auto original = failed;
            for (int kind = 0; kind < 3; ++kind) {
                stage.enabled = kind == 0;
                final_budget = kind == 1 ? 1 : 20;
                reject_certificate = kind == 2;
                const int count = scfs, committed = commits;
                bool threw = false;
                try {
                    parent.run(failed);
                } catch (const std::runtime_error& e) {
                    threw = true;
                    std::printf("EXPECTED_FAILURE kind=%d message=%s\n", kind,
                                e.what());
                }
                require(threw && scfs == count + 1 && commits == committed,
                        "one failed SCF never becomes fallback commit");
                for (auto col : {kAtom_fx, kAtom_fy, kAtom_fz})
                    for (int i = 0; i < 2; ++i)
                        require(failed.span(col)[i] == original.span(col)[i],
                                "failed force unchanged");
                require(failed.meta.get(kEnergy) == original.meta.get(kEnergy)
                            && failed.meta.get(kStress)
                                   == original.meta.get(kStress),
                        "failed scalar outputs unchanged");
                require(kind == 0 ? !method.last_result().converged
                                  : method.last_result().converged,
                        "failure actual metadata verdict preserved");
                if (kind == 1)
                    require(!auxiliary.back().complete,
                            "true incomplete aux1 remains exception");
            }
            final_budget = 20;
            reject_certificate = false;
        }
        int index = 0;
        for (double box : {12., 4.}) {
            auto f = geometry(box);
            const auto original = f;
            const int before = scfs, oldcommits = commits;
            const auto aux_before = auxiliary.size();
            std::array<double, 6> returned_sigma{};
            if (standalone)
                returned_sigma = method.stress(f);
            else
                parent.run(f);
            const auto& r = method.last_result();
            require(scfs == before + 1, "one actual SCF per derivative call");
            require(same(r, reference),
                    "every accepted metadata field unchanged");
            require(auxiliary.size() == aux_before + 2,
                    "exact two original auxiliary solves per accepted call");
            for (auto j = aux_before; j < auxiliary.size(); ++j) {
                const auto& a = auxiliary[j];
                require(a.complete && a.max_residual_ha < 1e-6
                            && a.max_normalization_error < 1e-8,
                        "original auxiliary qualification");
            }
            require(std::isfinite(r.energy) && std::isfinite(r.energy_ha),
                    "finite actual reference");
            require(std::abs(f.meta.get(kEnergy)
                             - r.energy * cpu::pw::qe_units::rydberg_to_ev)
                        < 1e-10,
                    "physical E published once");
            const bool fallback =
                last_route == dev::FrozenFinishResult::unsupported_domain;
            require(commits == oldcommits + (fallback ? 1 : 0),
                    "actual typed publication route");
            if (strict || standalone)
                require(index == 0 ? fallback : !fallback,
                        "real finite total-floor then interior retry");
            if (fallback) {
                if (!standalone) {
                    require(legacy_force.size() == 2,
                            "complete F candidate rows");
                    for (int i = 0; i < 2; ++i)
                        for (int axis = 0; axis < 3; ++axis) {
                            auto col = axis == 0   ? kAtom_fx
                                       : axis == 1 ? kAtom_fy
                                                   : kAtom_fz;
                            double error = std::abs(f.span(col)[i]
                                                    - legacy_force[i][axis]);
                            require(error < 1e-12,
                                    "legacy F eVA exactly published");
                            std::printf("CANDIDATE_F atom=%d axis=%d eVA=%.17g "
                                        "published=%.17g error=%.17g\n",
                                        i, axis, legacy_force[i][axis],
                                        f.span(col)[i], error);
                        }
                }
                for (int lane = 0; lane < 6; ++lane) {
                    double expected = legacy_sigma[lane] * f.box.volume(),
                           error =
                               std::abs(f.meta.get(kStress)[lane] - expected);
                    require(error < 1e-10,
                            "legacy sigma times volume published as W eV");
                    std::printf(
                        "CANDIDATE_W lane=%d sigma_eVA3=%.17g volume_A3=%.17g "
                        "W_eV=%.17g published=%.17g error=%.17g\n",
                        lane, legacy_sigma[lane], f.box.volume(), expected,
                        f.meta.get(kStress)[lane], error);
                }
            }
            if (strict || standalone) {
                double old_box, old_energy;
                int old_converged, old_iters;
                std::array<double, 6> old_f{}, old_w{};
                baseline >> old_box >> old_energy >> old_converged >> old_iters;
                for (auto& v : old_f)
                    baseline >> v;
                for (auto& v : old_w)
                    baseline >> v;
                require(bool(baseline) && old_box == box,
                        "exact legacy frame row");
                require(std::abs(old_energy - r.energy_ha) < 1e-12
                            && bool(old_converged) == r.converged
                            && old_iters == r.iterations,
                        "old reference SCF metadata parity");
                if (fallback) {
                    if (!standalone)
                        for (int i = 0; i < 2; ++i)
                            for (int axis = 0; axis < 3; ++axis) {
                                auto col = axis == 0   ? kAtom_fx
                                           : axis == 1 ? kAtom_fy
                                                       : kAtom_fz;
                                double error = std::abs(f.span(col)[i]
                                                        - old_f[3 * i + axis]);
                                require(error < 1e-10,
                                        "complete old legacy F baseline");
                                std::printf("BASELINE_F atom=%d axis=%d "
                                            "error_eVA=%.17g\n",
                                            i, axis, error);
                            }
                    for (int lane = 0; lane < 6; ++lane) {
                        double error =
                            std::abs(f.meta.get(kStress)[lane] - old_w[lane]);
                        require(error < 1e-10,
                                "complete old legacy W baseline");
                        std::printf("BASELINE_W lane=%d error_eV=%.17g\n", lane,
                                    error);
                    }
                }
            }
            if (standalone) {
                for (auto col : {kAtom_fx, kAtom_fy, kAtom_fz})
                    for (int i = 0; i < 2; ++i)
                        require(f.span(col)[i] == original.span(col)[i],
                                "standalone stress leaves force sentinels");
                for (int lane = 0; lane < 6; ++lane)
                    require(std::abs(returned_sigma[lane] * f.box.volume()
                                     - f.meta.get(kStress)[lane])
                                < 1e-10,
                            "real standalone sigma output matches Frame W");
            }
            std::printf("OBSERVATION index=%d box_A=%.17g converged=%d "
                        "iterations=%d E_Ha=%.17g route=%s flags=%u "
                        "SCF_calls=%d owners=%d legacy_commits=%d\n",
                        index, box, r.converged, r.iterations, r.energy_ha,
                        fallback ? "unsupported_domain" : "complete",
                        domain.violations, scfs, owners, commits);
            for (int i = 0; i < 2; ++i)
                for (int a = 0; a < 3; ++a) {
                    const auto col = a == 0   ? kAtom_fx
                                     : a == 1 ? kAtom_fy
                                              : kAtom_fz;
                    require(std::isfinite(f.span(col)[i]),
                            "finite published force");
                }
            for (int lane = 0; lane < 6; ++lane)
                require(std::isfinite(f.meta.get(kStress)[lane]),
                        "finite published W");
            ++index;
            if (standalone)
                break;
        }
        std::printf(
            "PASS DOMAIN_%s actual_scf=%d final_solves=%zu owners=%d "
            "legacy_commits=%d extra_SCF=0 fallback_extra_unstrained_aux=2 "
            "symmetric_legacy_W_not_native9=1 floor_derivative_qualified=0\n",
            strict ? "ROUTE_RETRY"
                   : (standalone ? "STANDALONE_STRESS" : "OBSERVATION"),
            scfs, auxiliary.size(), owners, commits);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL DOMAIN_PROBE %s\n", e.what());
        return 1;
    }
}
