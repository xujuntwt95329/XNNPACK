// Auto-generated file. Do not edit!
//   Template: src/f32-igemm/avx-broadcast.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/igemm.h"


void xnn_f32_igemm_minmax_ukernel_14x8__fma3_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 14);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (14 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    c3 = c2;
  }
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    c4 = c3;
  }
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    c5 = c4;
  }
  float* c6 = (float*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    c6 = c5;
  }
  float* c7 = (float*) ((uintptr_t) c6 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 8) {
    c7 = c6;
  }
  float* c8 = (float*) ((uintptr_t) c7 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 8) {
    c8 = c7;
  }
  float* c9 = (float*) ((uintptr_t) c8 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 10) {
    c9 = c8;
  }
  float* c10 = (float*) ((uintptr_t) c9 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 10) {
    c10 = c9;
  }
  float* c11 = (float*) ((uintptr_t) c10 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 12) {
    c11 = c10;
  }
  float* c12 = (float*) ((uintptr_t) c11 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 12) {
    c12 = c11;
  }
  float* c13 = (float*) ((uintptr_t) c12 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 14) {
    c13 = c12;
  }

  const __m256 vmin = _mm256_set1_ps(params->scalar.min);
  const __m256 vmax = _mm256_set1_ps(params->scalar.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    __m256 vacc0x01234567 = _mm256_load_ps(w);
    __m256 vacc1x01234567 = vacc0x01234567;
    __m256 vacc2x01234567 = vacc0x01234567;
    __m256 vacc3x01234567 = vacc0x01234567;
    __m256 vacc4x01234567 = vacc0x01234567;
    __m256 vacc5x01234567 = vacc0x01234567;
    __m256 vacc6x01234567 = vacc0x01234567;
    __m256 vacc7x01234567 = vacc0x01234567;
    __m256 vacc8x01234567 = vacc0x01234567;
    __m256 vacc9x01234567 = vacc0x01234567;
    __m256 vacc10x01234567 = vacc0x01234567;
    __m256 vacc11x01234567 = vacc0x01234567;
    __m256 vacc12x01234567 = vacc0x01234567;
    __m256 vacc13x01234567 = vacc0x01234567;
    w += 8;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      const float* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const float*) ((uintptr_t) a1 + a_offset);
      }
      const float* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const float*) ((uintptr_t) a2 + a_offset);
      }
      const float* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const float*) ((uintptr_t) a3 + a_offset);
      }
      const float* restrict a4 = a[4];
      assert(a4 != NULL);
      if XNN_UNPREDICTABLE(a4 != zero) {
        a4 = (const float*) ((uintptr_t) a4 + a_offset);
      }
      const float* restrict a5 = a[5];
      assert(a5 != NULL);
      if XNN_UNPREDICTABLE(a5 != zero) {
        a5 = (const float*) ((uintptr_t) a5 + a_offset);
      }
      const float* restrict a6 = a[6];
      assert(a6 != NULL);
      if XNN_UNPREDICTABLE(a6 != zero) {
        a6 = (const float*) ((uintptr_t) a6 + a_offset);
      }
      const float* restrict a7 = a[7];
      assert(a7 != NULL);
      if XNN_UNPREDICTABLE(a7 != zero) {
        a7 = (const float*) ((uintptr_t) a7 + a_offset);
      }
      const float* restrict a8 = a[8];
      assert(a8 != NULL);
      if XNN_UNPREDICTABLE(a8 != zero) {
        a8 = (const float*) ((uintptr_t) a8 + a_offset);
      }
      const float* restrict a9 = a[9];
      assert(a9 != NULL);
      if XNN_UNPREDICTABLE(a9 != zero) {
        a9 = (const float*) ((uintptr_t) a9 + a_offset);
      }
      const float* restrict a10 = a[10];
      assert(a10 != NULL);
      if XNN_UNPREDICTABLE(a10 != zero) {
        a10 = (const float*) ((uintptr_t) a10 + a_offset);
      }
      const float* restrict a11 = a[11];
      assert(a11 != NULL);
      if XNN_UNPREDICTABLE(a11 != zero) {
        a11 = (const float*) ((uintptr_t) a11 + a_offset);
      }
      const float* restrict a12 = a[12];
      assert(a12 != NULL);
      if XNN_UNPREDICTABLE(a12 != zero) {
        a12 = (const float*) ((uintptr_t) a12 + a_offset);
      }
      const float* restrict a13 = a[13];
      assert(a13 != NULL);
      if XNN_UNPREDICTABLE(a13 != zero) {
        a13 = (const float*) ((uintptr_t) a13 + a_offset);
      }
      a += 14;

      size_t k = kc;
      do {
        const __m256 vb01234567 = _mm256_load_ps(w);
        w += 8;

        const __m256 va0 = _mm256_broadcast_ss(a0);
        a0 += 1;
        const __m256 va1 = _mm256_broadcast_ss(a1);
        a1 += 1;
        const __m256 va2 = _mm256_broadcast_ss(a2);
        a2 += 1;
        const __m256 va3 = _mm256_broadcast_ss(a3);
        a3 += 1;
        const __m256 va4 = _mm256_broadcast_ss(a4);
        a4 += 1;
        const __m256 va5 = _mm256_broadcast_ss(a5);
        a5 += 1;
        const __m256 va6 = _mm256_broadcast_ss(a6);
        a6 += 1;
        const __m256 va7 = _mm256_broadcast_ss(a7);
        a7 += 1;
        const __m256 va8 = _mm256_broadcast_ss(a8);
        a8 += 1;
        const __m256 va9 = _mm256_broadcast_ss(a9);
        a9 += 1;
        const __m256 va10 = _mm256_broadcast_ss(a10);
        a10 += 1;
        const __m256 va11 = _mm256_broadcast_ss(a11);
        a11 += 1;
        const __m256 va12 = _mm256_broadcast_ss(a12);
        a12 += 1;
        const __m256 va13 = _mm256_broadcast_ss(a13);
        a13 += 1;

        vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567, vacc0x01234567);
        vacc1x01234567 = _mm256_fmadd_ps(va1, vb01234567, vacc1x01234567);
        vacc2x01234567 = _mm256_fmadd_ps(va2, vb01234567, vacc2x01234567);
        vacc3x01234567 = _mm256_fmadd_ps(va3, vb01234567, vacc3x01234567);
        vacc4x01234567 = _mm256_fmadd_ps(va4, vb01234567, vacc4x01234567);
        vacc5x01234567 = _mm256_fmadd_ps(va5, vb01234567, vacc5x01234567);
        vacc6x01234567 = _mm256_fmadd_ps(va6, vb01234567, vacc6x01234567);
        vacc7x01234567 = _mm256_fmadd_ps(va7, vb01234567, vacc7x01234567);
        vacc8x01234567 = _mm256_fmadd_ps(va8, vb01234567, vacc8x01234567);
        vacc9x01234567 = _mm256_fmadd_ps(va9, vb01234567, vacc9x01234567);
        vacc10x01234567 = _mm256_fmadd_ps(va10, vb01234567, vacc10x01234567);
        vacc11x01234567 = _mm256_fmadd_ps(va11, vb01234567, vacc11x01234567);
        vacc12x01234567 = _mm256_fmadd_ps(va12, vb01234567, vacc12x01234567);
        vacc13x01234567 = _mm256_fmadd_ps(va13, vb01234567, vacc13x01234567);
        k -= sizeof(float);
      } while (k != 0);
      p -= 14 * sizeof(void*);
    } while (p != 0);

    vacc0x01234567 = _mm256_max_ps(vmin, vacc0x01234567);
    vacc1x01234567 = _mm256_max_ps(vmin, vacc1x01234567);
    vacc2x01234567 = _mm256_max_ps(vmin, vacc2x01234567);
    vacc3x01234567 = _mm256_max_ps(vmin, vacc3x01234567);
    vacc4x01234567 = _mm256_max_ps(vmin, vacc4x01234567);
    vacc5x01234567 = _mm256_max_ps(vmin, vacc5x01234567);
    vacc6x01234567 = _mm256_max_ps(vmin, vacc6x01234567);
    vacc7x01234567 = _mm256_max_ps(vmin, vacc7x01234567);
    vacc8x01234567 = _mm256_max_ps(vmin, vacc8x01234567);
    vacc9x01234567 = _mm256_max_ps(vmin, vacc9x01234567);
    vacc10x01234567 = _mm256_max_ps(vmin, vacc10x01234567);
    vacc11x01234567 = _mm256_max_ps(vmin, vacc11x01234567);
    vacc12x01234567 = _mm256_max_ps(vmin, vacc12x01234567);
    vacc13x01234567 = _mm256_max_ps(vmin, vacc13x01234567);

    vacc0x01234567 = _mm256_min_ps(vmax, vacc0x01234567);
    vacc1x01234567 = _mm256_min_ps(vmax, vacc1x01234567);
    vacc2x01234567 = _mm256_min_ps(vmax, vacc2x01234567);
    vacc3x01234567 = _mm256_min_ps(vmax, vacc3x01234567);
    vacc4x01234567 = _mm256_min_ps(vmax, vacc4x01234567);
    vacc5x01234567 = _mm256_min_ps(vmax, vacc5x01234567);
    vacc6x01234567 = _mm256_min_ps(vmax, vacc6x01234567);
    vacc7x01234567 = _mm256_min_ps(vmax, vacc7x01234567);
    vacc8x01234567 = _mm256_min_ps(vmax, vacc8x01234567);
    vacc9x01234567 = _mm256_min_ps(vmax, vacc9x01234567);
    vacc10x01234567 = _mm256_min_ps(vmax, vacc10x01234567);
    vacc11x01234567 = _mm256_min_ps(vmax, vacc11x01234567);
    vacc12x01234567 = _mm256_min_ps(vmax, vacc12x01234567);
    vacc13x01234567 = _mm256_min_ps(vmax, vacc13x01234567);

    if XNN_LIKELY(nc >= 8) {
      _mm256_storeu_ps(c13, vacc13x01234567);
      c13 = (float*) ((uintptr_t) c13 + cn_stride);
      _mm256_storeu_ps(c12, vacc12x01234567);
      c12 = (float*) ((uintptr_t) c12 + cn_stride);
      _mm256_storeu_ps(c11, vacc11x01234567);
      c11 = (float*) ((uintptr_t) c11 + cn_stride);
      _mm256_storeu_ps(c10, vacc10x01234567);
      c10 = (float*) ((uintptr_t) c10 + cn_stride);
      _mm256_storeu_ps(c9, vacc9x01234567);
      c9 = (float*) ((uintptr_t) c9 + cn_stride);
      _mm256_storeu_ps(c8, vacc8x01234567);
      c8 = (float*) ((uintptr_t) c8 + cn_stride);
      _mm256_storeu_ps(c7, vacc7x01234567);
      c7 = (float*) ((uintptr_t) c7 + cn_stride);
      _mm256_storeu_ps(c6, vacc6x01234567);
      c6 = (float*) ((uintptr_t) c6 + cn_stride);
      _mm256_storeu_ps(c5, vacc5x01234567);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      _mm256_storeu_ps(c4, vacc4x01234567);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      _mm256_storeu_ps(c3, vacc3x01234567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      _mm256_storeu_ps(c2, vacc2x01234567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      _mm256_storeu_ps(c1, vacc1x01234567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm256_storeu_ps(c0, vacc0x01234567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
      __m128 vacc13x0123 = _mm256_castps256_ps128(vacc13x01234567);
      __m128 vacc12x0123 = _mm256_castps256_ps128(vacc12x01234567);
      __m128 vacc11x0123 = _mm256_castps256_ps128(vacc11x01234567);
      __m128 vacc10x0123 = _mm256_castps256_ps128(vacc10x01234567);
      __m128 vacc9x0123 = _mm256_castps256_ps128(vacc9x01234567);
      __m128 vacc8x0123 = _mm256_castps256_ps128(vacc8x01234567);
      __m128 vacc7x0123 = _mm256_castps256_ps128(vacc7x01234567);
      __m128 vacc6x0123 = _mm256_castps256_ps128(vacc6x01234567);
      __m128 vacc5x0123 = _mm256_castps256_ps128(vacc5x01234567);
      __m128 vacc4x0123 = _mm256_castps256_ps128(vacc4x01234567);
      __m128 vacc3x0123 = _mm256_castps256_ps128(vacc3x01234567);
      __m128 vacc2x0123 = _mm256_castps256_ps128(vacc2x01234567);
      __m128 vacc1x0123 = _mm256_castps256_ps128(vacc1x01234567);
      __m128 vacc0x0123 = _mm256_castps256_ps128(vacc0x01234567);
      if (nc & 4) {
        _mm_storeu_ps(c13, vacc13x0123);
        _mm_storeu_ps(c12, vacc12x0123);
        _mm_storeu_ps(c11, vacc11x0123);
        _mm_storeu_ps(c10, vacc10x0123);
        _mm_storeu_ps(c9, vacc9x0123);
        _mm_storeu_ps(c8, vacc8x0123);
        _mm_storeu_ps(c7, vacc7x0123);
        _mm_storeu_ps(c6, vacc6x0123);
        _mm_storeu_ps(c5, vacc5x0123);
        _mm_storeu_ps(c4, vacc4x0123);
        _mm_storeu_ps(c3, vacc3x0123);
        _mm_storeu_ps(c2, vacc2x0123);
        _mm_storeu_ps(c1, vacc1x0123);
        _mm_storeu_ps(c0, vacc0x0123);

        vacc13x0123 = _mm256_extractf128_ps(vacc13x01234567, 1);
        vacc12x0123 = _mm256_extractf128_ps(vacc12x01234567, 1);
        vacc11x0123 = _mm256_extractf128_ps(vacc11x01234567, 1);
        vacc10x0123 = _mm256_extractf128_ps(vacc10x01234567, 1);
        vacc9x0123 = _mm256_extractf128_ps(vacc9x01234567, 1);
        vacc8x0123 = _mm256_extractf128_ps(vacc8x01234567, 1);
        vacc7x0123 = _mm256_extractf128_ps(vacc7x01234567, 1);
        vacc6x0123 = _mm256_extractf128_ps(vacc6x01234567, 1);
        vacc5x0123 = _mm256_extractf128_ps(vacc5x01234567, 1);
        vacc4x0123 = _mm256_extractf128_ps(vacc4x01234567, 1);
        vacc3x0123 = _mm256_extractf128_ps(vacc3x01234567, 1);
        vacc2x0123 = _mm256_extractf128_ps(vacc2x01234567, 1);
        vacc1x0123 = _mm256_extractf128_ps(vacc1x01234567, 1);
        vacc0x0123 = _mm256_extractf128_ps(vacc0x01234567, 1);

        c13 += 4;
        c12 += 4;
        c11 += 4;
        c10 += 4;
        c9 += 4;
        c8 += 4;
        c7 += 4;
        c6 += 4;
        c5 += 4;
        c4 += 4;
        c3 += 4;
        c2 += 4;
        c1 += 4;
        c0 += 4;
      }
      if (nc & 2) {
        _mm_storel_pi((__m64*) c13, vacc13x0123);
        _mm_storel_pi((__m64*) c12, vacc12x0123);
        _mm_storel_pi((__m64*) c11, vacc11x0123);
        _mm_storel_pi((__m64*) c10, vacc10x0123);
        _mm_storel_pi((__m64*) c9, vacc9x0123);
        _mm_storel_pi((__m64*) c8, vacc8x0123);
        _mm_storel_pi((__m64*) c7, vacc7x0123);
        _mm_storel_pi((__m64*) c6, vacc6x0123);
        _mm_storel_pi((__m64*) c5, vacc5x0123);
        _mm_storel_pi((__m64*) c4, vacc4x0123);
        _mm_storel_pi((__m64*) c3, vacc3x0123);
        _mm_storel_pi((__m64*) c2, vacc2x0123);
        _mm_storel_pi((__m64*) c1, vacc1x0123);
        _mm_storel_pi((__m64*) c0, vacc0x0123);

        vacc13x0123 = _mm_movehl_ps(vacc13x0123, vacc13x0123);
        vacc12x0123 = _mm_movehl_ps(vacc12x0123, vacc12x0123);
        vacc11x0123 = _mm_movehl_ps(vacc11x0123, vacc11x0123);
        vacc10x0123 = _mm_movehl_ps(vacc10x0123, vacc10x0123);
        vacc9x0123 = _mm_movehl_ps(vacc9x0123, vacc9x0123);
        vacc8x0123 = _mm_movehl_ps(vacc8x0123, vacc8x0123);
        vacc7x0123 = _mm_movehl_ps(vacc7x0123, vacc7x0123);
        vacc6x0123 = _mm_movehl_ps(vacc6x0123, vacc6x0123);
        vacc5x0123 = _mm_movehl_ps(vacc5x0123, vacc5x0123);
        vacc4x0123 = _mm_movehl_ps(vacc4x0123, vacc4x0123);
        vacc3x0123 = _mm_movehl_ps(vacc3x0123, vacc3x0123);
        vacc2x0123 = _mm_movehl_ps(vacc2x0123, vacc2x0123);
        vacc1x0123 = _mm_movehl_ps(vacc1x0123, vacc1x0123);
        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);

        c13 += 2;
        c12 += 2;
        c11 += 2;
        c10 += 2;
        c9 += 2;
        c8 += 2;
        c7 += 2;
        c6 += 2;
        c5 += 2;
        c4 += 2;
        c3 += 2;
        c2 += 2;
        c1 += 2;
        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c13, vacc13x0123);
        _mm_store_ss(c12, vacc12x0123);
        _mm_store_ss(c11, vacc11x0123);
        _mm_store_ss(c10, vacc10x0123);
        _mm_store_ss(c9, vacc9x0123);
        _mm_store_ss(c8, vacc8x0123);
        _mm_store_ss(c7, vacc7x0123);
        _mm_store_ss(c6, vacc6x0123);
        _mm_store_ss(c5, vacc5x0123);
        _mm_store_ss(c4, vacc4x0123);
        _mm_store_ss(c3, vacc3x0123);
        _mm_store_ss(c2, vacc2x0123);
        _mm_store_ss(c1, vacc1x0123);
        _mm_store_ss(c0, vacc0x0123);
      }

      nc = 0;
    }
  } while (nc != 0);
}
