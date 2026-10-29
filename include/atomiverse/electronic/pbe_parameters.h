/// @file electronic/pbe_parameters.h
/// @brief Immutable PBE-family model parameters shared by host and device XC.
#pragma once

namespace ATV::electronic {

/// @brief Dimensionless parameters of PBE exchange and PW92-based correlation.
/// These are physical data, independent of grids, solvers and backend state.
/// Device kernels receive the selected parameters by value.
struct PbeParameters {
    double mu;        ///< Exchange gradient coefficient.
    double kappa;     ///< Exchange enhancement bound.
    double beta;      ///< Correlation gradient coefficient.
    bool exponential; ///< RPBE exponential enhancement instead of PBE rational.
};

/// @brief Original PBE (Perdew, Burke and Ernzerhof, 1996).
inline constexpr PbeParameters kPbeParameters = {0.2195149727645171, 0.804,
                                                 0.06672455060314922, false};
/// @brief PBEsol (Perdew et al., 2008), with gradient-expansion exchange.
inline constexpr PbeParameters kPbeSolParameters = {10.0 / 81.0, 0.804, 0.046,
                                                    false};
/// @brief revPBE (Zhang and Yang, 1998), with the revised exchange bound.
inline constexpr PbeParameters kRevPbeParameters = {kPbeParameters.mu, 1.245,
                                                    kPbeParameters.beta, false};
/// @brief RPBE (Hammer, Hansen and Norskov, 1999), exponential exchange.
inline constexpr PbeParameters kRpbeParameters = {
    kPbeParameters.mu, kPbeParameters.kappa, kPbeParameters.beta, true};

} // namespace ATV::electronic
