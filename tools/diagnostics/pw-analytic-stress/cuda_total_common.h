#pragma once
// Single-TU diagnostic implementation shared by LSDA and PBE-family targets.
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/planewave/internal/ewald.h"
#include "atomiverse/cpu/planewave/internal/frozen_operator_virial.h"
#include "atomiverse/cpu/planewave/internal/pw_final_state.h"
#include "atomiverse/cpu/planewave/internal/pw_forces.h"
#include "atomiverse/cpu/planewave/internal/xc_builder.h"
#include "atomiverse/cpu/planewave/qe_units.h"
#include "atomiverse/cuda/evaluate/evaluate.cuh"
#include "atomiverse/cuda/planewave/internal/frozen_functional_virial.cuh"
#include "atomiverse/cuda/planewave/pw_solver.cuh"
#include "cuda_total_detail.h"
#include "cuda_total_fixture.h"
#include "cuda_total_values.h"
#include <bit>
#include <cstdint>
#include <cstdio>
#include <map>
#include <numeric>
#include <nvtx3/nvToolsExt.h>
#include <stdexcept>
namespace {
const pw_cuda_total::Reference* active_reference = nullptr;
const pw_cuda_total::Reference& reference() { return *active_reference; }
} // namespace
using namespace ATV;
namespace hp = cpu::pw::internal;
using C = std::complex<double>;
// The only diagnostic state lives in this executable, not the SCF owner.
struct DiagnosticSink {
    pw_cuda_total::Report data;
    int calls = 0, final_budget = 20;
    std::vector<double> final_spectrum;
    bool reject_candidate = false;
    std::vector<ATV::cuda::pw::internal::FinalBandStatus> statuses;
} sink;
namespace pw_cuda_total {
void record_scf() {
    ++sink.calls;
    sink.data = Report{};
}
int diagnostic_scf_count() { return sink.calls; }
int diagnostic_final_budget() { return sink.final_budget; }
void diagnostic_aux_status(const dev::FinalBandStatus& s, int budget) {
    sink.statuses.push_back(s);
    std::printf("AUX_FINAL budget=%d max_residual_Ha=%.17g "
                "max_norm_error=%.17g complete=%d\n",
                budget, s.max_residual_ha, s.max_normalization_error,
                s.complete);
}
void diagnostic_candidate(const Report& r) {
    sink.data = r;
    if (sink.reject_candidate)
        throw std::runtime_error(
            "intentional candidate rejection before publication");
}
void diagnostic_committed() noexcept { sink.data.committed = true; }
void diagnostic_energy_result(const cpu::pw::PWResult& r) {
    sink.data.reference = r;
    sink.data.scf_calls = sink.calls;
    sink.data.committed = true;
}
} // namespace pw_cuda_total
namespace {
constexpr int aa[6] = {0, 1, 2, 0, 0, 1}, bb[6] = {0, 1, 2, 1, 2, 2};
void require(bool value, const char* label) {
    if (!value)
        throw std::runtime_error(label);
}
void close(double value, double expected, double gate, const char* label) {
    if (!std::isfinite(value) || std::abs(value - expected) > gate) {
        std::printf("FAIL %s got=%.17g expected=%.17g gate=%.17g\n", label,
                    value, expected, gate);
        throw std::runtime_error(label);
    }
}
class RealScfProbe final : public cuda::Component {
  public:
    std::string name() const override { return "formal_actual_scf_probe"; }
    void build(const cuda::BuildContext&) override {
        pw_cuda_total::record_scf();
    }
    void launch(const cuda::LaunchContext&) override {}
};
class RejectOnceStage final : public cuda::Component {
  public:
    bool enabled = false;
    std::string name() const override {
        return "pw_cuda_total_once_stage_exception";
    }
    void launch(const cuda::LaunchContext&) override {
        if (enabled) {
            enabled = false;
            throw std::runtime_error(
                "intentional reference SCF stage exception");
        }
    }
};
Mat3 identity() {
    Mat3 a;
    for (int i = 0; i < 3; ++i)
        a(i, i) = 1;
    return a;
}
hp::GVectorList transform(hp::GVectorList g, const Mat3& F) {
    auto metric = F.inverse().transpose();
    for (auto& v : g.g)
        v = metric * v;
    g.kvec = metric * g.kvec;
    for (int i = 0; i < g.count(); ++i)
        g.ekin[i] = .5 * (g.g[i] + g.kvec).norm_squared();
    return g;
}
cpu::Frame deform(cpu::Frame f, const Mat3& F) {
    f.box.set_cell(F * f.box.cell());
    for (int i = 0; i < f.nlocal(); ++i) {
        Vec3 r{f.span(kAtom_x)[i], f.span(kAtom_y)[i], f.span(kAtom_z)[i]};
        r = F * r;
        f.span(kAtom_x)[i] = r[0];
        f.span(kAtom_y)[i] = r[1];
        f.span(kAtom_z)[i] = r[2];
    }
    return f;
}
std::array<double, 6> values(const pw_cuda_total::Report& r,
                             const cpu::Frame& frame, const hp::GVectorList& g,
                             const hp::GVectorList& G,
                             std::span<const Vec3> positions,
                             std::span<const C> rho, double volume,
                             hp::XCBuilder& xc) {
    const pw_frozen::Orbitals o{r.coefficients, r.occupations, 1};
    hp::PWWorkspace ws;
    ws.gvec = &G;
    ws.n_g = G.count();
    ws.omega = volume;
    ws.fft_nx = G.nx;
    ws.fft_ny = G.ny;
    ws.fft_nz = G.nz;
    ws.fft = std::make_unique<hp::FFTPlan3D>(G.nx, G.ny, G.nz);
    ws.nr = ws.fft->nr();
    ws.nc = ws.fft->nc();
    ws.rho_G.resize({rho.size()});
    std::copy(rho.begin(), rho.end(), ws.rho_G.begin());
    xc.prepare(ws);
    xc.step(ws, reference().functional);
    std::vector<Vec3> ang;
    for (int i = 0; i < frame.nlocal(); ++i)
        ang.push_back({frame.span(kAtom_x)[i], frame.span(kAtom_y)[i],
                       frame.span(kAtom_z)[i]});
    hp::EwaldSummation ionic(
        ang, std::vector<AccReal>(frame.nlocal(), r.potential->z_val),
        frame.box);
    return {pw_frozen::value_t(g, o),
            pw_frozen::value_h(G, rho, volume),
            pw_frozen::value_local(*r.potential, G, positions, rho, volume),
            pw_frozen::value_nl(*r.potential, g, positions, o, volume),
            ws.E_xc,
            ionic.total_energy()};
}
} // namespace

bool same_accepted_result(const cpu::pw::PWResult& a,
                          const cpu::pw::PWResult& b) {
    auto same = [](double x, double y) {
        return std::bit_cast<std::uint64_t>(x)
               == std::bit_cast<std::uint64_t>(y);
    };
    return same(a.energy, b.energy) && same(a.energy_ha, b.energy_ha)
           && same(a.magnetization, b.magnetization)
           && same(a.e_fermi, b.e_fermi) && same(a.n_elec_occ, b.n_elec_occ)
           && same(a.e_entropy, b.e_entropy) && a.converged == b.converged
           && a.iterations == b.iterations && a.n_kpts == b.n_kpts
           && a.n_kpts_local == b.n_kpts_local;
}
void validate_final_filling(const pw_cuda_total::Report& r) {
    if (reference().smearing == electronic::Smearing::none)
        return;
    const double width =
        reference().degauss_ry * cpu::pw::qe_units::rydberg_to_hartree;
    require(r.final_spectrum.size() == r.occupations.size(),
            "fresh spectrum/f shape");
    double count = 0, entropy = 0;
    for (std::size_t b = 0; b < r.occupations.size(); ++b) {
        const double f = r.occupations[b],
                     expected =
                         2
                         / (1
                            + std::exp((r.final_spectrum[b] - r.final_mu_ha)
                                       / width));
        require(std::isfinite(f) && f >= 0 && f <= 2
                    && std::abs(f - expected) < 1e-12,
                "fresh physical spectrum logistic f");
        count += f;
        const double fraction = f / 2;
        if (fraction > 0 && fraction < 1)
            entropy -= 2
                       * (fraction * std::log(fraction)
                          + (1 - fraction) * std::log1p(-fraction));
    }
    require(std::abs(count - 8) < 1e-8 && std::abs(count - r.final_ne) < 1e-10,
            "final occupations exact electron count");
    require(std::abs(entropy - r.final_entropy) < 1e-11,
            "fresh entropy from same f");
    std::map<std::array<int, 3>, int> lookup;
    for (int ig = 0; ig < r.density.count(); ++ig)
        lookup[{r.density.h[ig], r.density.k[ig], r.density.l[ig]}] = ig;
    std::vector<C> expected(r.rho_orb.size());
    const int ng = r.wave.count();
    for (std::size_t b = 0; b < r.occupations.size(); ++b)
        for (int i = 0; i < ng; ++i)
            for (int j = 0; j < ng; ++j) {
                const std::array<int, 3> delta = {r.wave.h[i] - r.wave.h[j],
                                                  r.wave.k[i] - r.wave.k[j],
                                                  r.wave.l[i] - r.wave.l[j]};
                auto found = lookup.find(delta);
                require(found != lookup.end(),
                        "density support closes wave differences");
                expected[found->second] +=
                    r.occupations[b] / r.omega * r.coefficients[b * ng + i]
                    * std::conj(r.coefficients[b * ng + j]);
            }
    double gap = 0;
    for (std::size_t i = 0; i < expected.size(); ++i)
        gap = std::max(gap, std::abs(expected[i] - r.rho_orb[i]));
    require(gap < 1e-12, "same final C/f independent Fourier density");
    std::printf("FINAL_FD mu_Ha=%.17g Ne=%.17g S=%.17g entropy_Ha=%.17g "
                "rho_same_Cf_gap=%.17g aux_E_computed=%d",
                r.final_mu_ha, count, entropy, -width * entropy, gap,
                r.stress_requested);
    if (r.stress_requested)
        std::printf(" aux_internal_E_Ha=%.17g aux_F_Ha=%.17g",
                    r.terms[6].energy_ha,
                    r.terms[6].energy_ha - width * entropy);
    std::printf("\n");
}
int run_cuda_total(int argc, char** argv, const pw_cuda_total::Reference& ref) {
    active_reference = &ref;
    std::printf("CASE_REFERENCE model=%s functional=%d QE_E_Ha=%.17g\n",
                ref.model, int(ref.functional), ref.energy_ha);
    try {
        const std::string mode = argc == 2 ? argv[1] : "--nested";
        const bool nested = mode.rfind("--nested", 0) == 0 || mode == "--retry";
        const bool want_force = mode == "--force" || mode == "--nested"
                                || mode == "--nested-force" || mode == "--retry"
                                || mode == "--exhausted";
        const bool want_stress = mode == "--stress" || mode == "--nested"
                                 || mode == "--nested-stress"
                                 || mode == "--retry";
        auto f = pw_frozen::silicon();
        for (int i = 0; i < 2; ++i) {
            f.span(kAtom_fx)[i] = 10 + i;
            f.span(kAtom_fy)[i] = 20 + i;
            f.span(kAtom_fz)[i] = 30 + i;
        }
        f.meta.get(kEnergy) = 1234;
        f.meta.get(kStress) = {1, 2, 3, 4, 5, 6};
        const auto original = f;
        RealScfProbe scf_probe;
        RejectOnceStage stage;
        cuda::pw::PW method(std::make_shared<cpu::pw::GthPseudopotential>(
                                molrs::Element::Si, "GTH-PBE-q4"),
                            12, 1, reference().functional);
        method.tol = 1e-10;
        method.max_iter = 200;
        method.mix_beta = .2;
        method.smearing = reference().smearing;
        method.degauss = reference().degauss_ry;
        method.extra_bands = reference().extra_bands;
        method.on(ATV::pw::Phase::check).add_ref(scf_probe, -100);
        if (mode == "--exhausted") {
            method.max_iter = 1;
            bool aux_rejected = false;
            try {
                method.forces(f);
            } catch (const std::runtime_error& e) {
                aux_rejected =
                    std::string(e.what())
                    == "auxiliary final eigenstate residual incomplete";
                if (!aux_rejected)
                    throw;
            }
            require(!method.last_result().converged
                        && method.last_result().iterations == 1
                        && sink.calls == 1,
                    "exhaustion preserves false SCF metadata");
            require(
                !sink.statuses.empty(),
                "exhaustion independently evaluates actual auxiliary residual");
            if (aux_rejected) {
                require(!sink.statuses.back().complete && !sink.data.committed,
                        "only genuine auxiliary incompleteness rejects");
                require(f.meta.get(kEnergy) == original.meta.get(kEnergy)
                            && f.meta.get(kStress)
                                   == original.meta.get(kStress),
                        "aux failure preserves outputs");
            } else
                require(
                    sink.data.committed && sink.statuses.back().complete,
                    "best-effort derivative returns with false SCF verdict");
            std::printf("PASS CUDA_PW_EXHAUSTED SCF=1 converged=0 "
                        "auxiliary_rejected=%d actual_aux_residual_Ha=%.17g "
                        "policy_changed=false\n",
                        aux_rejected, sink.statuses.back().max_residual_ha);
            return 0;
        }
        if (nested) {
            auto unchanged = [&] {
                for (auto column : {kAtom_fx, kAtom_fy, kAtom_fz})
                    for (int i = 0; i < 2; ++i)
                        require(f.span(column)[i] == original.span(column)[i],
                                "failed call preserves force");
                require(f.meta.get(kEnergy) == original.meta.get(kEnergy)
                            && f.meta.get(kStress)
                                   == original.meta.get(kStress),
                        "failed call preserves scalar channels");
            };
            if (mode == "--retry") {
                stage.enabled = true;
                method.on(ATV::pw::Phase::check).add_ref(stage);
            }
            cuda::Evaluate parent(want_force, want_stress);
            parent.set_evaluator_ref(method);
            parent.build();
            if (mode == "--retry") {
                for (int failure = 0; failure < 3; ++failure) {
                    sink.final_budget = failure == 1 ? 1 : 20;
                    sink.reject_candidate = failure == 2;
                    const int before = sink.calls;
                    bool rejected = false;
                    try {
                        parent.run(f);
                    } catch (const std::runtime_error& e) {
                        rejected = true;
                        std::printf("EXPECTED_FAILURE kind=%d message=%s\n",
                                    failure, e.what());
                    }
                    require(rejected && sink.calls == before + 1,
                            "one real failed nested call");
                    unchanged();
                    require(!sink.data.committed,
                            "failed candidate never commits");
                    if (failure == 0)
                        require(!method.last_result().converged,
                                "interrupted SCF is not falsely converged");
                    else
                        require(method.last_result().converged,
                                "aux failure preserves accepted SCF verdict");
                    if (failure == 1)
                        require(!sink.statuses.back().complete,
                                "real final Hpsi residual failure");
                }
                sink.final_budget = 20;
                sink.reject_candidate = false;
            }
            const int before = sink.calls;
            parent.run(f);
            require(sink.calls == before + 1, "one accepted nested SCF");
        } else if (want_force)
            method.forces(f);
        else if (want_stress)
            method.stress(f);
        else
            method.run(std::as_const(f));
        if (!want_force && !want_stress)
            pw_cuda_total::diagnostic_energy_result(method.last_result());
        const auto& r = sink.data;
        if (want_force || want_stress)
            validate_final_filling(r);
        const auto& accepted = method.last_result();
        require(r.committed, "completed candidate publication");
        if (!want_force)
            for (auto col : {kAtom_fx, kAtom_fy, kAtom_fz})
                for (int i = 0; i < 2; ++i)
                    require(f.span(col)[i] == original.span(col)[i],
                            "unrequested nonzero force sentinel preserved");
        if (!want_stress)
            require(f.meta.get(kStress) == original.meta.get(kStress),
                    "unrequested nonzero virial sentinel preserved");
        if (!want_force && !want_stress) {
            require(r.final_band_solves == 0 && r.explicit_device_bytes == 0,
                    "energy-only constructs no derivative workspace or sweep");
            close(accepted.energy_ha, reference().energy_ha, 1e-5,
                  "energy-only physical QE");
            for (auto col : {kAtom_fx, kAtom_fy, kAtom_fz})
                for (int i = 0; i < 2; ++i)
                    require(f.span(col)[i] == original.span(col)[i],
                            "energy-only force unchanged");
            require(f.meta.get(kStress) == original.meta.get(kStress),
                    "energy-only stress unchanged");
            std::printf("PASS CUDA_PW_TOTAL_NEXT mode=%s SCF_calls=%d "
                        "final_sweeps=0 derivative_bytes=0\n",
                        mode.c_str(), sink.calls);
            return 0;
        }
        require(r.auxiliary_residual_ha < 1e-6 && r.auxiliary_norm_error < 1e-8
                    && r.auxiliary_budget == 20,
                "actual auxiliary Hpsi qualification");
        require(r.scf_calls == sink.calls && r.final_band_solves == 1,
                "one SCF plus one unstrained final-band sweep");
        require(r.reference_density_preserved && r.borrowed_stream_preserved,
                "reference/stream lifecycle");
        require(accepted.converged
                    && same_accepted_result(accepted, r.reference),
                "physical SCF full metadata preserved bitwise");
        close(accepted.energy_ha, reference().energy_ha, 1e-5,
              "QE accepted energy");
        close(f.meta.get(kEnergy),
              accepted.energy * cpu::pw::qe_units::rydberg_to_ev, 1e-10,
              "published physical E uses accepted SCF");
        double qe_force = 0, qe_stress = 0;
        if (want_force)
            for (int i = 0; i < 2; ++i)
                for (int a = 0; a < 3; ++a) {
                    qe_force = std::max(
                        qe_force, std::abs(r.forces_ha_bohr[i][a]
                                           - reference().forces_ha_bohr[i][a]));
                    close(r.forces_ha_bohr[i][a],
                          reference().forces_ha_bohr[i][a], 1e-4,
                          "QE full force");
                    const auto col = a == 0   ? kAtom_fx
                                     : a == 1 ? kAtom_fy
                                              : kAtom_fz;
                    close(f.span(col)[i],
                          r.forces_ha_bohr[i][a]
                              * units::kHartreePerBohrInEvPerAngstrom,
                          2e-7, "published force from same final state");
                }
        if (want_stress)
            for (int v = 0; v < 6; ++v) {
                qe_stress = std::max(
                    qe_stress,
                    std::abs(r.sigma[v] - reference().stress_ev_ang3[v]));
                close(r.sigma[v], reference().stress_ev_ang3[v], .002,
                      "QE full6 sigma");
                if (nested)
                    close(f.meta.get(kStress)[v], r.sigma[v] * f.box.volume(),
                          2e-6, "nested published virial");
            }
        hp::PWProblem p;
        p.potential = r.potential;
        p.gv0 = r.density;
        p.gv_k = {r.wave};
        p.vol_bohr = r.omega;
        p.n_elec = 8;
        p.n_bands = r.occupations.size();
        p.ws.gvec = &p.gv0;
        p.ws.n_g = p.gv0.count();
        p.ws.omega = r.omega;
        p.ws.atom_pos_bohr = r.positions_bohr;
        p.kb_proj_k.resize(1);
        hp::setup_kb_projectors(p.kb_proj_k[0], p.gv_k[0], *p.potential,
                                p.ws.atom_pos_bohr, p.vol_bohr);
        hp::FinalBandState state;
        state.reference_result = accepted;
        state.reference_density_G = r.rho_ref;
        state.orbital_density_G = r.rho_orb;
        hp::FinalBandBlock b;
        b.ik = 0;
        b.spin = 0;
        b.k_weight = 1;
        b.occupations = r.occupations;
        b.coefficients = xt::zeros<C>({r.occupations.size(), r.wave.g.size()});
        std::copy(r.coefficients.begin(), r.coefficients.end(),
                  b.coefficients.begin());
        state.local_blocks.push_back(std::move(b));
        const auto cpu_forces = hp::pw_forces_from_bands(original, p, state);
        double cpu_force_error = 0;
        if (want_force)
            for (int i = 0; i < 2; ++i)
                for (int a = 0; a < 3; ++a) {
                    const double ref = cpu_forces[i][a]
                                       / units::kHartreePerBohrInEvPerAngstrom;
                    cpu_force_error =
                        std::max(cpu_force_error,
                                 std::abs(r.forces_ha_bohr[i][a] - ref));
                    close(r.forces_ha_bohr[i][a], ref,
                          1e-8 + 1e-7 * std::abs(ref),
                          "same-state CPU full-force oracle");
                }
        const hp::FrozenBandView orbitals{r.coefficients, r.occupations, 1};
        const std::array<hp::FrozenOperatorVirial, 4> cpu = {
            hp::kinetic_frozen_virial(r.wave, orbitals),
            hp::hartree_frozen_virial(r.density, r.rho_orb, r.omega),
            hp::local_potential_frozen_virial(
                *r.potential, r.density, r.positions_bohr, r.rho_orb, r.omega),
            hp::nonlocal_frozen_virial(*r.potential, r.wave, r.positions_bohr,
                                       orbitals, r.omega)};
        if (want_stress)
            for (int t = 0; t < 4; ++t) {
                close(r.terms[t].energy_ha, cpu[t].energy_ha,
                      1e-8 + 1e-7 * std::abs(cpu[t].energy_ha),
                      "supplementary same-state CPU E");
                for (int ab = 0; ab < 9; ++ab)
                    close(r.terms[t].virial_ha[ab], cpu[t].virial_ha[ab],
                          1e-8 + 1e-7 * std::abs(cpu[t].virial_ha[ab]),
                          "supplementary same-state CPU W9");
            }
        const char* names[] = {"T", "H", "local", "NL", "XC", "Ewald", "total"};
        if (want_stress)
            for (int t = 0; t < 7; ++t) {
                std::printf("TERM %s E_Ha=%.17g W9=", names[t],
                            r.terms[t].energy_ha);
                for (auto w : r.terms[t].virial_ha)
                    std::printf(" %.17g", w);
                std::printf("\n");
            }
        double max_asym = 0;
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                max_asym = std::max(
                    max_asym, std::abs(r.terms[6].virial_ha[3 * a + b]
                                       - r.terms[6].virial_ha[3 * b + a]));
        require(max_asym < 1e-8, "unprojected total antisymmetry");
        hp::XCBuilder xc;
        double fine_fd = 0;
        if (want_stress)
            for (double h : {2e-4, 3e-5, 1e-5}) {
                double max_error = 0;
                std::array<double, 6> errors{};
                for (int v = 0; v < 6; ++v) {
                    std::array<std::array<double, 6>, 2> sample;
                    for (int sign = 0; sign < 2; ++sign) {
                        auto F = identity();
                        F(aa[v], bb[v]) += sign == 0 ? h : -h;
                        auto frame = deform(original, F);
                        auto G = transform(r.density, F),
                             g = transform(r.wave, F);
                        auto positions = r.positions_bohr;
                        for (auto& pos : positions)
                            pos = F * pos;
                        auto rho = r.rho_orb;
                        for (auto& z : rho)
                            z /= F.det();
                        sample[sign] = values(r, frame, g, G, positions, rho,
                                              r.omega * F.det(), xc);
                    }
                    double fd = 0;
                    for (int t = 0; t < 6; ++t) {
                        const double derivative =
                            -(sample[0][t] - sample[1][t]) / (2 * h);
                        const double analytic =
                            r.terms[t].virial_ha[3 * aa[v] + bb[v]];
                        errors[t] = std::max(errors[t],
                                             std::abs(derivative - analytic));
                        if (h <= 1e-5)
                            close(derivative, analytic,
                                  1e-8 + 1e-7 * std::abs(analytic),
                                  "independent value frozen-C component strain "
                                  "derivative");
                        fd += derivative;
                    }
                    max_error = std::max(
                        max_error,
                        std::abs(fd - r.terms[6].virial_ha[3 * aa[v] + bb[v]]));
                    if (h <= 1e-5)
                        close(fd, r.terms[6].virial_ha[3 * aa[v] + bb[v]],
                              1e-8
                                  + 1e-7
                                        * std::abs(
                                            r.terms[6]
                                                .virial_ha[3 * aa[v] + bb[v]]),
                              "complete frozen-C six-strain derivative");
                }
                if (h <= 1e-5)
                    fine_fd = max_error;
                std::printf("FROZEN_C6_FD h=%.17g errors_Ha=", h);
                for (double e : errors)
                    std::printf(" %.17g", e);
                std::printf(" total_Ha=%.17g\n", max_error);
                require(max_error * units::kHartreeInEv / original.box.volume()
                            < .002,
                        "unchanged frozen-C sigma FD gate");
            }
        double ionic_fd_error = 0;
        constexpr double h = 1e-5;
        if (want_force)
            for (int i = 0; i < 2; ++i)
                for (int a = 0; a < 3; ++a) {
                    double e[2];
                    for (int sign = 0; sign < 2; ++sign) {
                        auto frame = original;
                        auto positions = r.positions_bohr;
                        const double shift = sign == 0 ? h : -h;
                        positions[i][a] += shift;
                        auto col = a == 0   ? kAtom_x
                                   : a == 1 ? kAtom_y
                                            : kAtom_z;
                        frame.span(col)[i] +=
                            shift / cpu::pw::qe_units::angstrom_to_bohr;
                        auto parts = values(r, frame, r.wave, r.density,
                                            positions, r.rho_orb, r.omega, xc);
                        e[sign] =
                            std::accumulate(parts.begin(), parts.end(), 0.);
                    }
                    const double derivative = -(e[0] - e[1]) / (2 * h);
                    ionic_fd_error =
                        std::max(ionic_fd_error,
                                 std::abs(derivative - r.forces_ha_bohr[i][a]));
                    close(derivative, r.forces_ha_bohr[i][a],
                          1e-8 + 1e-7 * std::abs(r.forces_ha_bohr[i][a]),
                          "independent value complete ionic FD");
                }
        std::printf(
            "REQUEST_PROOF forces=%d stress=%d C_calls=%d aux_budget=%d "
            "actual_Hpsi_residual_Ha=%.17g explicit_bytes=%zu committed=%d\n",
            want_force, want_stress, sink.calls, r.auxiliary_budget,
            r.auxiliary_residual_ha, r.explicit_device_bytes, r.committed);
        std::printf("PASS CUDA_PW_TOTAL_NEXT mode=%s SCF_per_call=1 "
                    "unstrained_final_sweeps=1 "
                    "strained_band_solves=0 accepted_E_Ha=%.17g aux_E_Ha=%.17g "
                    "density_mismatch=%.17g max_QE_F_HaBohr=%.17g "
                    "max_QE_sigma_eVA3=%.17g max_CPU_F_HaBohr=%.17g "
                    "frozen_C6_FD_Ha=%.17g ionic_FD_HaBohr=%.17g "
                    "explicit_device_bytes=%zu accepted_density_preserved=1 "
                    "borrowed_stream_preserved=1\n",
                    mode.c_str(), accepted.energy_ha, r.terms[6].energy_ha,
                    r.density_mismatch, qe_force, qe_stress, cpu_force_error,
                    fine_fd, ionic_fd_error, r.explicit_device_bytes);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL CUDA_PW_TOTAL_NEXT %s\n", e.what());
        return 1;
    }
    return 0;
}

// Test-only link adapter. No callbacks/budget flags/dumps exist in the owner.
namespace fd = ATV::cuda::pw::internal;
namespace {
bool inside_formal_owner = false, assembly_range_open = false;
}
fd::FrozenFinishResult real_owner(
    fd::FrozenFunctionalWorkspace*, fd::DeviceDensity&, fd::DeviceBandSolver&,
    const ATV::cpu::pw::PWResult&, std::vector<std::array<double, 3>>*,
    std::array<double, 6>*,
    ATV::cuda::
        Frame*) asm("__real__"
                    "ZN3ATV4cuda2pw8internal25FrozenFunctionalWorkspace6finishE"
                    "RNS2_13DeviceDensityERNS2_16DeviceBandSolverERKNS_"
                    "3cpu2pw8PWResultEPSt6vectorISt5arrayIdLm3EESaISF_EEPSE_"
                    "IdLm6EEPNS0_5FrameE");
fd::FrozenFinishResult wrap_owner(
    fd::FrozenFunctionalWorkspace*, fd::DeviceDensity&, fd::DeviceBandSolver&,
    const ATV::cpu::pw::PWResult&, std::vector<std::array<double, 3>>*,
    std::array<double, 6>*,
    ATV::cuda::
        Frame*) asm("__wrap__"
                    "ZN3ATV4cuda2pw8internal25FrozenFunctionalWorkspace6finishE"
                    "RNS2_13DeviceDensityERNS2_16DeviceBandSolverERKNS_"
                    "3cpu2pw8PWResultEPSt6vectorISt5arrayIdLm3EESaISF_EEPSE_"
                    "IdLm6EEPNS0_5FrameE");
fd::FrozenFinishResult
wrap_owner(fd::FrozenFunctionalWorkspace* self, fd::DeviceDensity& den,
           fd::DeviceBandSolver& solver, const ATV::cpu::pw::PWResult& ref,
           std::vector<std::array<double, 3>>* F, std::array<double, 6>* W,
           ATV::cuda::Frame* parent) {
    struct Scope {
        bool live = true;
        Scope() {
            inside_formal_owner = true;
            nvtxRangePushA("ATV_FORMAL_FINISH");
        }
        void close() {
            if (assembly_range_open) {
                nvtxRangePop();
                assembly_range_open = false;
            }
            nvtxRangePop();
            inside_formal_owner = false;
            live = false;
        }
        ~Scope() {
            if (live)
                close();
        }
    } active;
    const cpu::pw::PWResult accepted_before = ref;
    fd::FrozenFinishResult actual_finish_status;
    try {
        actual_finish_status = real_owner(self, den, solver, ref, F, W, parent);
    } catch (...) {
        require(same_accepted_result(ref, accepted_before),
                "failed owner preserves every accepted metadata field");
        throw;
    }
    require(same_accepted_result(ref, accepted_before),
            "owner preserves every accepted metadata field");
    active.close();
    const auto c = self->candidate();
    auto& r = sink.data;
    const auto& p = *c.geometry;
    r.reference = accepted_before;
    r.potential = p.potential;
    r.wave = p.gv_k[0];
    r.density = p.gv0;
    r.positions_bohr = p.ws.atom_pos_bohr;
    r.omega = p.vol_bohr;
    r.occupations.resize(c.orbitals.bands);
    r.coefficients.resize(std::size_t(c.orbitals.bands) * p.gv_k[0].count());
    r.rho_ref.resize(p.gv0.count());
    r.rho_orb.resize(p.gv0.count());
    auto copy = [&](void* dst, const void* src, std::size_t bytes) {
        if (cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost,
                            solver.stream())
            != cudaSuccess)
            throw std::runtime_error("test final-state dump");
    };
    copy(r.occupations.data(), c.orbitals.f,
         r.occupations.size() * sizeof(double));
    copy(r.coefficients.data(), c.orbitals.C,
         r.coefficients.size() * sizeof(cufftDoubleComplex));
    copy(r.rho_ref.data(), c.reference_density,
         r.rho_ref.size() * sizeof(cufftDoubleComplex));
    copy(r.rho_orb.data(), c.orbital_density,
         r.rho_orb.size() * sizeof(cufftDoubleComplex));
    if (c.outcome.stress_requested)
        copy(r.terms.data(), c.terms, sizeof(r.terms));
    if (c.outcome.forces_requested) {
        r.forces_ha_bohr.resize(p.ws.atom_pos_bohr.size());
        copy(r.forces_ha_bohr.data(), c.forces,
             r.forces_ha_bohr.size() * sizeof(r.forces_ha_bohr[0]));
    }
    if (cudaStreamSynchronize(solver.stream()) != cudaSuccess)
        throw std::runtime_error("test final-state dump fence");
    r.scf_calls = sink.calls;
    r.final_band_solves = 1;
    r.committed = c.outcome.published;
    r.forces_requested = c.outcome.forces_requested;
    r.stress_requested = c.outcome.stress_requested;
    r.reference_density_preserved = c.outcome.certificate.reference_unchanged;
    r.borrowed_stream_preserved = den.stream() == solver.stream();
    r.density_mismatch =
        std::sqrt(c.outcome.certificate.density_mismatch_squared);
    r.auxiliary_budget = 20;
    r.final_mu_ha = c.outcome.auxiliary_mu_ha;
    r.final_entropy = c.outcome.auxiliary_entropy;
    r.final_ne = c.outcome.auxiliary_electron_count;
    r.final_spectrum = sink.final_spectrum;
    r.auxiliary_residual_ha = c.outcome.auxiliary.max_residual_ha;
    r.auxiliary_norm_error = c.outcome.auxiliary.max_normalization_error;
    r.explicit_device_bytes = c.outcome.explicit_device_bytes;
    const int a[6] = {0, 1, 2, 0, 0, 1}, b[6] = {0, 1, 2, 1, 2, 2};
    if (r.stress_requested)
        for (int i = 0; i < 6; ++i)
            r.sigma[i] = .5
                         * (r.terms[6].virial_ha[3 * a[i] + b[i]]
                            + r.terms[6].virial_ha[3 * b[i] + a[i]])
                         * ATV::units::kHartreeInEv
                         / pw_frozen::silicon().box.volume();
    return actual_finish_status;
}
fd::FinalBandStatus
real_final(fd::DeviceBandSolver*, ATV::cpu::pw::internal::PWWorkspace&,
           fd::FinalBandDestination,
           int) asm("__real__ZN3ATV4cuda2pw8internal16DeviceBandSolver17solve_"
                    "final_bandsERNS_3cpu2pw8internal11PWWorkspaceENS2_"
                    "20FinalBandDestinationEi");
fd::FinalBandStatus
wrap_final(fd::DeviceBandSolver*, ATV::cpu::pw::internal::PWWorkspace&,
           fd::FinalBandDestination,
           int) asm("__wrap__ZN3ATV4cuda2pw8internal16DeviceBandSolver17solve_"
                    "final_bandsERNS_3cpu2pw8internal11PWWorkspaceENS2_"
                    "20FinalBandDestinationEi");
fd::FinalBandStatus wrap_final(fd::DeviceBandSolver* self,
                               ATV::cpu::pw::internal::PWWorkspace& w,
                               fd::FinalBandDestination d, int budget) {
    const int actual = sink.final_budget == 1 ? 1 : budget;
    nvtxRangePushA("ATV_FORMAL_FINAL_BANDS");
    auto status = real_final(self, w, d, actual);
    sink.final_spectrum.assign(w.eigenvalues.begin(), w.eigenvalues.end());
    nvtxRangePop();
    pw_cuda_total::diagnostic_aux_status(status, actual);
    return status;
}
extern "C" cudaError_t __real_cudaMemcpyAsync(void*, const void*, std::size_t,
                                              cudaMemcpyKind, cudaStream_t);
extern "C" cudaError_t __wrap_cudaMemcpyAsync(void* dst, const void* src,
                                              std::size_t bytes,
                                              cudaMemcpyKind kind,
                                              cudaStream_t stream) {
    auto result = __real_cudaMemcpyAsync(dst, src, bytes, kind, stream);
    if (inside_formal_owner && assembly_range_open
        && kind == cudaMemcpyDeviceToHost
        && bytes == sizeof(fd::FrozenDerivativeCertificate)) {
        nvtxRangePop();
        assembly_range_open = false;
    }
    if (result == cudaSuccess && inside_formal_owner && sink.reject_candidate
        && kind == cudaMemcpyDeviceToHost
        && bytes == sizeof(fd::FrozenDerivativeCertificate)) {
        result = cudaStreamSynchronize(stream);
        if (result == cudaSuccess)
            static_cast<fd::FrozenDerivativeCertificate*>(dst)->finite = 0;
    }
    return result;
}

void real_project(
    fd::FrozenNonlocalDeviceGeometry, fd::FrozenBandDeviceView,
    fd::FrozenNonlocalDeviceScratch, cublasHandle_t,
    cudaStream_t) asm("__real__ZN3ATV4cuda2pw8internal23project_frozen_"
                      "nonlocalENS2_28FrozenNonlocalDeviceGeometryENS2_"
                      "20FrozenBandDeviceViewENS2_"
                      "27FrozenNonlocalDeviceScratchEP13cublasContextP11CUstrea"
                      "m_st");
void wrap_project(
    fd::FrozenNonlocalDeviceGeometry, fd::FrozenBandDeviceView,
    fd::FrozenNonlocalDeviceScratch, cublasHandle_t,
    cudaStream_t) asm("__wrap__ZN3ATV4cuda2pw8internal23project_frozen_"
                      "nonlocalENS2_28FrozenNonlocalDeviceGeometryENS2_"
                      "20FrozenBandDeviceViewENS2_"
                      "27FrozenNonlocalDeviceScratchEP13cublasContextP11CUstrea"
                      "m_st");
void wrap_project(fd::FrozenNonlocalDeviceGeometry g,
                  fd::FrozenBandDeviceView state,
                  fd::FrozenNonlocalDeviceScratch scratch, cublasHandle_t blas,
                  cudaStream_t stream) {
    if (inside_formal_owner && !assembly_range_open) {
        nvtxRangePushA("ATV_FORMAL_ASSEMBLY");
        assembly_range_open = true;
    }
    real_project(g, state, scratch, blas, stream);
}

namespace {
bool inside_cold_owner = false;
std::vector<std::array<int, 3>> cold_plans;
} // namespace
void real_cold(fd::FrozenFunctionalWorkspace*, const ATV::cpu::Frame&,
               hp::PWProblem&, int, ATV::cpu::xc::XCType,
               ATV::electronic::Smearing, double,
               std::optional<fd::FixedSpinOccupations>, cudaStream_t, bool,
               bool) asm("__real__"
                         "ZN3ATV4cuda2pw8internal25FrozenFunctionalWorkspaceC1E"
                         "RKNS_3cpu5FrameERNS4_2pw8internal9PWProblemEiNS4_"
                         "2xc6XCTypeENS_10electronic8SmearingEdSt8optionalINS2_"
                         "20FixedSpinOccupationsEEP11CUstream_stbb");
void wrap_cold(fd::FrozenFunctionalWorkspace*, const ATV::cpu::Frame&,
               hp::PWProblem&, int, ATV::cpu::xc::XCType,
               ATV::electronic::Smearing, double,
               std::optional<fd::FixedSpinOccupations>, cudaStream_t, bool,
               bool) asm("__wrap__"
                         "ZN3ATV4cuda2pw8internal25FrozenFunctionalWorkspaceC1E"
                         "RKNS_3cpu5FrameERNS4_2pw8internal9PWProblemEiNS4_"
                         "2xc6XCTypeENS_10electronic8SmearingEdSt8optionalINS2_"
                         "20FixedSpinOccupationsEEP11CUstream_stbb");
void wrap_cold(fd::FrozenFunctionalWorkspace* self, const ATV::cpu::Frame& f,
               hp::PWProblem& p, int b, ATV::cpu::xc::XCType type,
               ATV::electronic::Smearing filling, double width,
               std::optional<fd::FixedSpinOccupations> spin,
               cudaStream_t stream, bool force, bool stress) {
    struct Guard {
        Guard() {
            inside_cold_owner = true;
            cold_plans.clear();
            cold_plans.reserve(2);
        }
        ~Guard() { inside_cold_owner = false; }
    } guard;
    real_cold(self, f, p, b, type, filling, width, spin, stream, force, stress);
    try {
        require(cold_plans.size() == std::size_t(stress ? 2 : 1),
                "request-aware cold owner plan budget");
        require(cold_plans.front() == reference().base,
                "actual base FFT matches reference");
        require(
            cold_plans.front()
                == std::array<int, 3>{p.ws.fft_nx, p.ws.fft_ny, p.ws.fft_nz},
            "actual cold base FFT query");
        if (stress)
            require(cold_plans.back() == reference().quadrature,
                    "actual cold GGA quadrature matches QE");
        std::printf("NATIVE_QUAD model=%d force=%d stress=%d owner_plans=%zu "
                    "base=%dx%dx%d quad=%dx%dx%d\n",
                    int(type), force, stress, cold_plans.size(),
                    cold_plans.front()[0], cold_plans.front()[1],
                    cold_plans.front()[2], stress ? cold_plans.back()[0] : 0,
                    stress ? cold_plans.back()[1] : 0,
                    stress ? cold_plans.back()[2] : 0);
    } catch (...) {
        self->~FrozenFunctionalWorkspace();
        throw;
    }
}
void real_fft(fd::FFTPlan3D*, int, int, int, cudaStream_t) asm(
    "__real__ZN3ATV4cuda2pw8internal9FFTPlan3DC1EiiiP11CUstream_st");
void wrap_fft(fd::FFTPlan3D*, int, int, int, cudaStream_t) asm(
    "__wrap__ZN3ATV4cuda2pw8internal9FFTPlan3DC1EiiiP11CUstream_st");
void wrap_fft(fd::FFTPlan3D* self, int x, int y, int z, cudaStream_t stream) {
    real_fft(self, x, y, z, stream);
    if (inside_cold_owner) {
        try {
            cold_plans.push_back({x, y, z});
        } catch (...) {
            self->~FFTPlan3D();
            throw;
        }
    }
}
