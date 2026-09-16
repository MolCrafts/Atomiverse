/// @file src/cpu/observe/wham.cpp
/// @brief Thin marshaling to molrs WhamCompute (no analysis math in C++).
#include "atomiverse/cpu/observe/wham.h"

#include <cstdint>
#include <stdexcept>
#include <utility>

#include "atomiverse/molrs.hpp"

namespace ATV {
namespace cpu {
namespace compute {

struct Wham::Impl {
    rust::Box<molrs::WhamCompute> obj;
    std::vector<double> last_f;
};

Wham::Wham(double beta)
    : impl_(std::make_unique<Impl>(Impl{molrs::wham_compute_new(beta), {}})) {
    if (!(beta > 0.0))
        throw std::invalid_argument("Wham: beta must be > 0");
}

Wham::~Wham() = default;
Wham::Wham(Wham&&) noexcept = default;
Wham& Wham::operator=(Wham&&) noexcept = default;

std::vector<double> Wham::free_energy(const std::vector<double>& hist,
                                      const std::vector<double>& n_samples,
                                      const std::vector<double>& k,
                                      const std::vector<double>& centers,
                                      double s_min, double s_max, int n_bins,
                                      int max_iter, double tol) {
    rust::Vec<double> out = impl_->obj->solve(
        {hist.data(), hist.size()}, {n_samples.data(), n_samples.size()},
        {k.data(), k.size()}, {centers.data(), centers.size()}, s_min, s_max,
        static_cast<std::int64_t>(n_bins), static_cast<std::int64_t>(max_iter),
        tol);
    rust::Vec<double> fw = impl_->obj->last_f_windows();
    impl_->last_f.assign(fw.begin(), fw.end());
    return std::vector<double>(out.begin(), out.end());
}

bool Wham::last_converged() const { return impl_->obj->last_converged() != 0; }

int Wham::last_iterations() const {
    return static_cast<int>(impl_->obj->last_iterations());
}

const std::vector<double>& Wham::last_f_windows() const {
    return impl_->last_f;
}

} // namespace compute
} // namespace cpu
} // namespace ATV
