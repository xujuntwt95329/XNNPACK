// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/avx-broadcast-opt.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/gemm.h"
#include "xnnpack/microparams.h"


void xnn_f32_gemm_minmax_ukernel_9x8__fma3_broadcast_opt(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 9);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const float* a4 = (const float*) ((uintptr_t) a3 + a_stride);
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const float* a5 = (const float*) ((uintptr_t) a4 + a_stride);
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    a5 = a4;
    c5 = c4;
  }
  const float* a6 = (const float*) ((uintptr_t) a5 + a_stride);
  float* c6 = (float*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    a6 = a5;
    c6 = c5;
  }
  const float* a7 = (const float*) ((uintptr_t) a6 + a_stride);
  float* c7 = (float*) ((uintptr_t) c6 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 8) {
    a7 = a6;
    c7 = c6;
  }
  const float* a8 = (const float*) ((uintptr_t) a7 + a_stride);
  float* c8 = (float*) ((uintptr_t) c7 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 8) {
    a8 = a7;
    c8 = c7;
  }

  const __m256 vmin = _mm256_set1_ps(params->scalar.min);
  const __m256 vmax = _mm256_set1_ps(params->scalar.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    __m256 vacc0x01234567 = _mm256_load_ps(w + 0);
    __m256 vacc1x01234567 = vacc0x01234567;
    __m256 vacc2x01234567 = vacc0x01234567;
    __m256 vacc3x01234567 = vacc0x01234567;
    __m256 vacc4x01234567 = vacc0x01234567;
    __m256 vacc5x01234567 = vacc0x01234567;
    __m256 vacc6x01234567 = vacc0x01234567;
    __m256 vacc7x01234567 = vacc0x01234567;
    __m256 vacc8x01234567 = vacc0x01234567;
    w += 8;

    size_t k = kc;
    do {
      __m256 va;

      const __m256 vb01234567 = _mm256_load_ps(w);
      w += 8;

      va = _mm256_broadcast_ss(a0);
      vacc0x01234567 = _mm256_fmadd_ps(va, vb01234567, vacc0x01234567);
      va = _mm256_broadcast_ss(a1);
      vacc1x01234567 = _mm256_fmadd_ps(va, vb01234567, vacc1x01234567);
      va = _mm256_broadcast_ss(a2);
      vacc2x01234567 = _mm256_fmadd_ps(va, vb01234567, vacc2x01234567);
      va = _mm256_broadcast_ss(a3);
      vacc3x01234567 = _mm256_fmadd_ps(va, vb01234567, vacc3x01234567);
      va = _mm256_broadcast_ss(a4);
      vacc4x01234567 = _mm256_fmadd_ps(va, vb01234567, vacc4x01234567);
      va = _mm256_broadcast_ss(a5);
      vacc5x01234567 = _mm256_fmadd_ps(va, vb01234567, vacc5x01234567);
      va = _mm256_broadcast_ss(a6);
      vacc6x01234567 = _mm256_fmadd_ps(va, vb01234567, vacc6x01234567);
      va = _mm256_broadcast_ss(a7);
      vacc7x01234567 = _mm256_fmadd_ps(va, vb01234567, vacc7x01234567);
      va = _mm256_broadcast_ss(a8);
      vacc8x01234567 = _mm256_fmadd_ps(va, vb01234567, vacc8x01234567);

      a0 += 1;
      a1 += 1;
      a2 += 1;
      a3 += 1;
      a4 += 1;
      a5 += 1;
      a6 += 1;
      a7 += 1;
      a8 += 1;

      k -= sizeof(float);
    } while (k != 0);

    vacc0x01234567 = _mm256_max_ps(vmin, vacc0x01234567);
    vacc1x01234567 = _mm256_max_ps(vmin, vacc1x01234567);
    vacc2x01234567 = _mm256_max_ps(vmin, vacc2x01234567);
    vacc3x01234567 = _mm256_max_ps(vmin, vacc3x01234567);
    vacc4x01234567 = _mm256_max_ps(vmin, vacc4x01234567);
    vacc5x01234567 = _mm256_max_ps(vmin, vacc5x01234567);
    vacc6x01234567 = _mm256_max_ps(vmin, vacc6x01234567);
    vacc7x01234567 = _mm256_max_ps(vmin, vacc7x01234567);
    vacc8x01234567 = _mm256_max_ps(vmin, vacc8x01234567);

    vacc0x01234567 = _mm256_min_ps(vmax, vacc0x01234567);
    vacc1x01234567 = _mm256_min_ps(vmax, vacc1x01234567);
    vacc2x01234567 = _mm256_min_ps(vmax, vacc2x01234567);
    vacc3x01234567 = _mm256_min_ps(vmax, vacc3x01234567);
    vacc4x01234567 = _mm256_min_ps(vmax, vacc4x01234567);
    vacc5x01234567 = _mm256_min_ps(vmax, vacc5x01234567);
    vacc6x01234567 = _mm256_min_ps(vmax, vacc6x01234567);
    vacc7x01234567 = _mm256_min_ps(vmax, vacc7x01234567);
    vacc8x01234567 = _mm256_min_ps(vmax, vacc8x01234567);

    if XNN_LIKELY(nc >= 8) {
      _mm256_storeu_ps(c0, vacc0x01234567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      _mm256_storeu_ps(c1, vacc1x01234567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm256_storeu_ps(c2, vacc2x01234567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      _mm256_storeu_ps(c3, vacc3x01234567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      _mm256_storeu_ps(c4, vacc4x01234567);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      _mm256_storeu_ps(c5, vacc5x01234567);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      _mm256_storeu_ps(c6, vacc6x01234567);
      c6 = (float*) ((uintptr_t) c6 + cn_stride);
      _mm256_storeu_ps(c7, vacc7x01234567);
      c7 = (float*) ((uintptr_t) c7 + cn_stride);
      _mm256_storeu_ps(c8, vacc8x01234567);
      c8 = (float*) ((uintptr_t) c8 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);
      a4 = (const float*) ((uintptr_t) a4 - kc);
      a5 = (const float*) ((uintptr_t) a5 - kc);
      a6 = (const float*) ((uintptr_t) a6 - kc);
      a7 = (const float*) ((uintptr_t) a7 - kc);
      a8 = (const float*) ((uintptr_t) a8 - kc);

      nc -= 8;
    } else {
      __m128 vacc0x0123 = _mm256_castps256_ps128(vacc0x01234567);
      __m128 vacc1x0123 = _mm256_castps256_ps128(vacc1x01234567);
      __m128 vacc2x0123 = _mm256_castps256_ps128(vacc2x01234567);
      __m128 vacc3x0123 = _mm256_castps256_ps128(vacc3x01234567);
      __m128 vacc4x0123 = _mm256_castps256_ps128(vacc4x01234567);
      __m128 vacc5x0123 = _mm256_castps256_ps128(vacc5x01234567);
      __m128 vacc6x0123 = _mm256_castps256_ps128(vacc6x01234567);
      __m128 vacc7x0123 = _mm256_castps256_ps128(vacc7x01234567);
      __m128 vacc8x0123 = _mm256_castps256_ps128(vacc8x01234567);
      if (nc & 4) {
        _mm_storeu_ps(c0, vacc0x0123);
        _mm_storeu_ps(c1, vacc1x0123);
        _mm_storeu_ps(c2, vacc2x0123);
        _mm_storeu_ps(c3, vacc3x0123);
        _mm_storeu_ps(c4, vacc4x0123);
        _mm_storeu_ps(c5, vacc5x0123);
        _mm_storeu_ps(c6, vacc6x0123);
        _mm_storeu_ps(c7, vacc7x0123);
        _mm_storeu_ps(c8, vacc8x0123);

        vacc0x0123 = _mm256_extractf128_ps(vacc0x01234567, 1);
        vacc1x0123 = _mm256_extractf128_ps(vacc1x01234567, 1);
        vacc2x0123 = _mm256_extractf128_ps(vacc2x01234567, 1);
        vacc3x0123 = _mm256_extractf128_ps(vacc3x01234567, 1);
        vacc4x0123 = _mm256_extractf128_ps(vacc4x01234567, 1);
        vacc5x0123 = _mm256_extractf128_ps(vacc5x01234567, 1);
        vacc6x0123 = _mm256_extractf128_ps(vacc6x01234567, 1);
        vacc7x0123 = _mm256_extractf128_ps(vacc7x01234567, 1);
        vacc8x0123 = _mm256_extractf128_ps(vacc8x01234567, 1);

        c0 += 4;
        c1 += 4;
        c2 += 4;
        c3 += 4;
        c4 += 4;
        c5 += 4;
        c6 += 4;
        c7 += 4;
        c8 += 4;
      }
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vacc0x0123);
        _mm_storel_pi((__m64*) c1, vacc1x0123);
        _mm_storel_pi((__m64*) c2, vacc2x0123);
        _mm_storel_pi((__m64*) c3, vacc3x0123);
        _mm_storel_pi((__m64*) c4, vacc4x0123);
        _mm_storel_pi((__m64*) c5, vacc5x0123);
        _mm_storel_pi((__m64*) c6, vacc6x0123);
        _mm_storel_pi((__m64*) c7, vacc7x0123);
        _mm_storel_pi((__m64*) c8, vacc8x0123);

        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);
        vacc1x0123 = _mm_movehl_ps(vacc1x0123, vacc1x0123);
        vacc2x0123 = _mm_movehl_ps(vacc2x0123, vacc2x0123);
        vacc3x0123 = _mm_movehl_ps(vacc3x0123, vacc3x0123);
        vacc4x0123 = _mm_movehl_ps(vacc4x0123, vacc4x0123);
        vacc5x0123 = _mm_movehl_ps(vacc5x0123, vacc5x0123);
        vacc6x0123 = _mm_movehl_ps(vacc6x0123, vacc6x0123);
        vacc7x0123 = _mm_movehl_ps(vacc7x0123, vacc7x0123);
        vacc8x0123 = _mm_movehl_ps(vacc8x0123, vacc8x0123);

        c0 += 2;
        c1 += 2;
        c2 += 2;
        c3 += 2;
        c4 += 2;
        c5 += 2;
        c6 += 2;
        c7 += 2;
        c8 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c0, vacc0x0123);
        _mm_store_ss(c1, vacc1x0123);
        _mm_store_ss(c2, vacc2x0123);
        _mm_store_ss(c3, vacc3x0123);
        _mm_store_ss(c4, vacc4x0123);
        _mm_store_ss(c5, vacc5x0123);
        _mm_store_ss(c6, vacc6x0123);
        _mm_store_ss(c7, vacc7x0123);
        _mm_store_ss(c8, vacc8x0123);
      }

      nc = 0;
    }
  } while (nc != 0);
}
