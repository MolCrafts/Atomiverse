/// @file nlist_error.cuh
/// @brief CUDA error checking for neighborlist internals.
///
/// Thin wrapper: pulls in the project-wide ATV_CUDA_CHECK macro and
/// re-exports it into the nlist namespace for backward compatibility.
#pragma once

#include "atomiverse/cuda/device/error.cuh"
