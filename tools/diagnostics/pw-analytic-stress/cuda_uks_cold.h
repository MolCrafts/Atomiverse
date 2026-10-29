#pragma once
// Diagnostic-only constructor/FFT wrapping. No production registration or hooks.
inline bool inside_cold_owner = false;
inline int cold_calls = 0;
inline std::vector<std::array<int, 3>> cold_plans;
#define COLD_OWNER                                                             \
    "ZN3ATV4cuda2pw8internal25FrozenFunctionalWorkspaceC1ERKNS_"               \
    "3cpu5FrameERNS4_2pw8internal9PWProblemEiNS4_2xc6XCTypeENS_"               \
    "10electronic8SmearingEdSt8optionalINS2_"                                  \
    "20FixedSpinOccupationsEEP11CUstream_stbb"
void real_cold(dev::FrozenFunctionalWorkspace*, const ATV::cpu::Frame&,
               hp::PWProblem&, int, ATV::cpu::xc::XCType,
               ATV::electronic::Smearing, double,
               std::optional<dev::FixedSpinOccupations>, cudaStream_t, bool,
               bool) asm("__real__" COLD_OWNER);
void wrap_cold(dev::FrozenFunctionalWorkspace*, const ATV::cpu::Frame&,
               hp::PWProblem&, int, ATV::cpu::xc::XCType,
               ATV::electronic::Smearing, double,
               std::optional<dev::FixedSpinOccupations>, cudaStream_t, bool,
               bool) asm("__wrap__" COLD_OWNER);
void wrap_cold(dev::FrozenFunctionalWorkspace* self, const ATV::cpu::Frame& f,
               hp::PWProblem& p, int b, ATV::cpu::xc::XCType t,
               ATV::electronic::Smearing filling, double width,
               std::optional<dev::FixedSpinOccupations> spin,
               cudaStream_t stream, bool force, bool stress) {
    struct Scope {
        Scope() {
            inside_cold_owner = true;
            cold_plans.clear();
            cold_plans.reserve(2);
        }
        ~Scope() { inside_cold_owner = false; }
    } scope;
    ++cold_calls;
    require(spin && spin->alpha == 7 && spin->beta == 5 && b == 7,
            "physical typed cold spin shape");
    real_cold(self, f, p, b, t, filling, width, spin, stream, force, stress);
    try {
        require(cold_plans.size() == std::size_t(stress ? 2 : 1),
                "request-aware cold FFT budget");
        require(cold_plans.front() == std::array<int, 3>{20, 20, 20},
                "model actual base query grid");
        if (stress)
            require(cold_plans.back()
                        == std::array<int, 3>{expected_quad, expected_quad,
                                              expected_quad},
                    "model actual tensor quadrature");
        std::printf("COLD_REQUEST force=%d stress=%d plans=%zu base=20x20x20 "
                    "quad=%d packed_bands=14\n",
                    force, stress, cold_plans.size(),
                    stress ? expected_quad : 0);
    } catch (...) {
        self->~FrozenFunctionalWorkspace();
        throw;
    }
}
#define FFT_CTOR "ZN3ATV4cuda2pw8internal9FFTPlan3DC1EiiiP11CUstream_st"
void real_fft(dev::FFTPlan3D*, int, int, int,
              cudaStream_t) asm("__real__" FFT_CTOR);
void wrap_fft(dev::FFTPlan3D*, int, int, int,
              cudaStream_t) asm("__wrap__" FFT_CTOR);
void wrap_fft(dev::FFTPlan3D* self, int x, int y, int z, cudaStream_t stream) {
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
