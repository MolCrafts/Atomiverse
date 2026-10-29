#pragma once
#include "atomiverse/cpu/planewave/internal/pw_setup.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cuda/frame/frame.cuh"
#include "atomiverse/cuda/planewave/internal/device_density.cuh"
#include "atomiverse/cuda/planewave/internal/device_solver.cuh"
#include "atomiverse/cuda/planewave/internal/frozen_operator_virial.cuh"
#include <memory>
namespace pw_cuda_total {
namespace dev = ATV::cuda::pw::internal;
namespace host = ATV::cpu::pw::internal;
struct Report {
    ATV::cpu::pw::PWResult reference;
    std::shared_ptr<const host::NcPotentialData> potential;
    host::GVectorList wave, density;
    std::vector<ATV::Vec3> positions_bohr;
    std::vector<std::complex<double>> coefficients, rho_orb, rho_ref;
    std::vector<double> occupations;
    std::array<dev::FrozenOperatorVirial, 7> terms{};
    std::vector<std::array<double, 3>> forces_ha_bohr;
    std::array<double, 6> sigma{};
    double omega = 0, density_mismatch = 0;
    int scf_calls = 0, final_band_solves = 0;
    std::size_t explicit_device_bytes = 0;
    bool reference_density_preserved = false, borrowed_stream_preserved = false;
    bool forces_requested = false, stress_requested = false, committed = false;
    double auxiliary_residual_ha = 0, auxiliary_norm_error = 0;
    int auxiliary_budget = 0;
    double final_mu_ha = 0, final_entropy = 0, final_ne = 0;
    std::vector<double> final_spectrum;
};
struct Reference {
    const char* model;
    ATV::cpu::xc::XCType functional;
    double energy_ha;
    std::array<std::array<double, 3>, 2> forces_ha_bohr;
    std::array<double, 6> stress_ev_ang3;
    std::array<int, 3> base, quadrature;
    ATV::electronic::Smearing smearing = ATV::electronic::Smearing::none;
    double degauss_ry = .02;
    int extra_bands = 2;
};
} // namespace pw_cuda_total
