/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifndef HL_FUNCTIONS_H_
#define HL_FUNCTIONS_H_

#include "hl_base.h"

/**
 * sigmoid threshold maximum
 */
#define SIGMOID_THRESHOLD_MIN -40.0

/**
 * sigmoid threshold minimum
 */
#define SIGMOID_THRESHOLD_MAX 13.0

#ifndef __HIPCC__
namespace hppl {
/*
 * forward activation
 */
real relu(const real a);
real sigmoid(const real a);
real tanh(const real a);
real linear(const real a);

/*
 * backward activation
 */
real relu(const real a, const real b);
real sigmoid(const real a, const real b);
real tanh(const real a, const real b);
real linear(const real a, const real b);
}  // namespace hppl

#ifdef __AVX__
#include "hl_avx_functions.h"
#endif

#else
#include "hl_gpu_functions.cuh"
#endif

#include <type_traits>
// To expand function pointer array since HCC doesn't support it.
template<class F>
void visit_activation(hl_activation_mode_t a, F f)
{
  switch(a)
  {
    case HL_ACTIVATION_SIGMOID:
      f(std::integral_constant<hl_activation_mode_t, HL_ACTIVATION_SIGMOID>{});
      break;
    case HL_ACTIVATION_RELU:
      f(std::integral_constant<hl_activation_mode_t, HL_ACTIVATION_RELU>{});
      break;
    case HL_ACTIVATION_TANH:
      f(std::integral_constant<hl_activation_mode_t, HL_ACTIVATION_TANH>{});
      break;
    case HL_ACTIVATION_LINEAR:
      f(std::integral_constant<hl_activation_mode_t, HL_ACTIVATION_LINEAR>{});
      break;
    default:
      break;
  };
}

#endif  // HL_FUNCTIONS_H_
