// Auto-generated file. Do not edit!
//   Template: src/f32-vbinary/vopc-sse.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xmmintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/vbinary.h"


void xnn_f32_vsubc_minmax_ukernel__sse_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m128 voutput_min = _mm_set1_ps(params->scalar.min);
  const __m128 voutput_max = _mm_set1_ps(params->scalar.max);
  XNN_FORCE_REALIZATION(voutput_min);
  XNN_FORCE_REALIZATION(voutput_max);
  const __m128 vb = _mm_load1_ps(input_b);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m128 va0 = _mm_loadu_ps(input_a);
    const __m128 va1 = _mm_loadu_ps(input_a + 4);
    input_a += 8;

    __m128 vacc0 = _mm_sub_ps(va0, vb);
    __m128 vacc1 = _mm_sub_ps(va1, vb);


    vacc0 = _mm_max_ps(vacc0, voutput_min);
    vacc1 = _mm_max_ps(vacc1, voutput_min);

    vacc0 = _mm_min_ps(vacc0, voutput_max);
    vacc1 = _mm_min_ps(vacc1, voutput_max);

    _mm_storeu_ps(output, vacc0);
    _mm_storeu_ps(output + 4, vacc1);
    output += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 va = _mm_loadu_ps(input_a);
    input_a += 4;

    __m128 vacc = _mm_sub_ps(va, vb);
    vacc = _mm_max_ps(vacc, voutput_min);
    vacc = _mm_min_ps(vacc, voutput_max);

    _mm_storeu_ps(output, vacc);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 va = _mm_loadu_ps(input_a);

    __m128 vacc = _mm_sub_ps(va, vb);
    vacc = _mm_max_ps(vacc, voutput_min);
    vacc = _mm_min_ps(vacc, voutput_max);
    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vacc);
      vacc = _mm_movehl_ps(vacc, vacc);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vacc);
    }
  }
}
