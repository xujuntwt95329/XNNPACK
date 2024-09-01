// Auto-generated file. Do not edit!
//   Template: src/f32-igemm/avx-shuffle4.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/igemm.h"


void xnn_f32_igemm_minmax_ukernel_3x16s4__fma3_broadcast(
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
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 3);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (3 * sizeof(void*)) == 0);
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

  const __m256 vmax = _mm256_set1_ps(params->scalar.max);
  const __m256 vmin = _mm256_set1_ps(params->scalar.min);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    __m256 vacc0x01234567 = _mm256_load_ps(w);
    __m256 vacc0x89ABCDEF = _mm256_load_ps(w + 8);
    __m256 vacc1x01234567 = vacc0x01234567;
    __m256 vacc1x89ABCDEF = vacc0x89ABCDEF;
    __m256 vacc2x01234567 = vacc0x01234567;
    __m256 vacc2x89ABCDEF = vacc0x89ABCDEF;
    w += 16;

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
      a += 3;

      size_t k = kc;
      while (k >= 4 * sizeof(float)) {
        __m256 va0 = _mm256_broadcast_ps((const __m128*) a0);
        a0 += 4;
        __m256 va1 = _mm256_broadcast_ps((const __m128*) a1);
        a1 += 4;
        __m256 va2 = _mm256_broadcast_ps((const __m128*) a2);
        a2 += 4;


        const __m256 vb01234567c0 = _mm256_load_ps(w + 0);
        const __m256 vb89ABCDEFc0 = _mm256_load_ps(w + 8);

        vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c0, vacc0x01234567);
        vacc1x01234567 = _mm256_fmadd_ps(va1, vb01234567c0, vacc1x01234567);
        vacc2x01234567 = _mm256_fmadd_ps(va2, vb01234567c0, vacc2x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc0, vacc0x89ABCDEF);
        vacc1x89ABCDEF = _mm256_fmadd_ps(va1, vb89ABCDEFc0, vacc1x89ABCDEF);
        vacc2x89ABCDEF = _mm256_fmadd_ps(va2, vb89ABCDEFc0, vacc2x89ABCDEF);

        va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
        va1 = _mm256_permute_ps(va1, _MM_SHUFFLE(0, 3, 2, 1));
        va2 = _mm256_permute_ps(va2, _MM_SHUFFLE(0, 3, 2, 1));

        const __m256 vb01234567c1 = _mm256_load_ps(w + 16);
        const __m256 vb89ABCDEFc1 = _mm256_load_ps(w + 24);

        vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c1, vacc0x01234567);
        vacc1x01234567 = _mm256_fmadd_ps(va1, vb01234567c1, vacc1x01234567);
        vacc2x01234567 = _mm256_fmadd_ps(va2, vb01234567c1, vacc2x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc1, vacc0x89ABCDEF);
        vacc1x89ABCDEF = _mm256_fmadd_ps(va1, vb89ABCDEFc1, vacc1x89ABCDEF);
        vacc2x89ABCDEF = _mm256_fmadd_ps(va2, vb89ABCDEFc1, vacc2x89ABCDEF);

        va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
        va1 = _mm256_permute_ps(va1, _MM_SHUFFLE(0, 3, 2, 1));
        va2 = _mm256_permute_ps(va2, _MM_SHUFFLE(0, 3, 2, 1));

        const __m256 vb01234567c2 = _mm256_load_ps(w + 32);
        const __m256 vb89ABCDEFc2 = _mm256_load_ps(w + 40);

        vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c2, vacc0x01234567);
        vacc1x01234567 = _mm256_fmadd_ps(va1, vb01234567c2, vacc1x01234567);
        vacc2x01234567 = _mm256_fmadd_ps(va2, vb01234567c2, vacc2x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc2, vacc0x89ABCDEF);
        vacc1x89ABCDEF = _mm256_fmadd_ps(va1, vb89ABCDEFc2, vacc1x89ABCDEF);
        vacc2x89ABCDEF = _mm256_fmadd_ps(va2, vb89ABCDEFc2, vacc2x89ABCDEF);

        va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
        va1 = _mm256_permute_ps(va1, _MM_SHUFFLE(0, 3, 2, 1));
        va2 = _mm256_permute_ps(va2, _MM_SHUFFLE(0, 3, 2, 1));

        const __m256 vb01234567c3 = _mm256_load_ps(w + 48);
        const __m256 vb89ABCDEFc3 = _mm256_load_ps(w + 56);

        vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c3, vacc0x01234567);
        vacc1x01234567 = _mm256_fmadd_ps(va1, vb01234567c3, vacc1x01234567);
        vacc2x01234567 = _mm256_fmadd_ps(va2, vb01234567c3, vacc2x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc3, vacc0x89ABCDEF);
        vacc1x89ABCDEF = _mm256_fmadd_ps(va1, vb89ABCDEFc3, vacc1x89ABCDEF);
        vacc2x89ABCDEF = _mm256_fmadd_ps(va2, vb89ABCDEFc3, vacc2x89ABCDEF);


        w += 64;
        k -= 4 * sizeof(float);
      }
      if XNN_UNLIKELY(k != 0) {
        __m256 va0 = _mm256_broadcast_ps((const __m128*) a0);
        a0 = (const float*) ((uintptr_t) a0 + k);
        __m256 va1 = _mm256_broadcast_ps((const __m128*) a1);
        a1 = (const float*) ((uintptr_t) a1 + k);
        __m256 va2 = _mm256_broadcast_ps((const __m128*) a2);
        a2 = (const float*) ((uintptr_t) a2 + k);

        const __m256 vzero = _mm256_setzero_ps();

        const __m256 vb01234567c0 = _mm256_load_ps(w + 0);
        const __m256 vb89ABCDEFc0 = _mm256_load_ps(w + 8);

        vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c0, vzero, _CMP_NEQ_OQ)), vb01234567c0, vacc0x01234567);
        vacc1x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb01234567c0, vzero, _CMP_NEQ_OQ)), vb01234567c0, vacc1x01234567);
        vacc2x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb01234567c0, vzero, _CMP_NEQ_OQ)), vb01234567c0, vacc2x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc0, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc0, vacc0x89ABCDEF);
        vacc1x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb89ABCDEFc0, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc0, vacc1x89ABCDEF);
        vacc2x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb89ABCDEFc0, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc0, vacc2x89ABCDEF);

        va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
        va1 = _mm256_permute_ps(va1, _MM_SHUFFLE(0, 3, 2, 1));
        va2 = _mm256_permute_ps(va2, _MM_SHUFFLE(0, 3, 2, 1));

        const __m256 vb01234567c1 = _mm256_load_ps(w + 16);
        const __m256 vb89ABCDEFc1 = _mm256_load_ps(w + 24);

        vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c1, vzero, _CMP_NEQ_OQ)), vb01234567c1, vacc0x01234567);
        vacc1x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb01234567c1, vzero, _CMP_NEQ_OQ)), vb01234567c1, vacc1x01234567);
        vacc2x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb01234567c1, vzero, _CMP_NEQ_OQ)), vb01234567c1, vacc2x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc1, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc1, vacc0x89ABCDEF);
        vacc1x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb89ABCDEFc1, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc1, vacc1x89ABCDEF);
        vacc2x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb89ABCDEFc1, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc1, vacc2x89ABCDEF);

        va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
        va1 = _mm256_permute_ps(va1, _MM_SHUFFLE(0, 3, 2, 1));
        va2 = _mm256_permute_ps(va2, _MM_SHUFFLE(0, 3, 2, 1));

        const __m256 vb01234567c2 = _mm256_load_ps(w + 32);
        const __m256 vb89ABCDEFc2 = _mm256_load_ps(w + 40);

        vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c2, vzero, _CMP_NEQ_OQ)), vb01234567c2, vacc0x01234567);
        vacc1x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb01234567c2, vzero, _CMP_NEQ_OQ)), vb01234567c2, vacc1x01234567);
        vacc2x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb01234567c2, vzero, _CMP_NEQ_OQ)), vb01234567c2, vacc2x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc2, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc2, vacc0x89ABCDEF);
        vacc1x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb89ABCDEFc2, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc2, vacc1x89ABCDEF);
        vacc2x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb89ABCDEFc2, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc2, vacc2x89ABCDEF);

        va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
        va1 = _mm256_permute_ps(va1, _MM_SHUFFLE(0, 3, 2, 1));
        va2 = _mm256_permute_ps(va2, _MM_SHUFFLE(0, 3, 2, 1));

        const __m256 vb01234567c3 = _mm256_load_ps(w + 48);
        const __m256 vb89ABCDEFc3 = _mm256_load_ps(w + 56);

        vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c3, vzero, _CMP_NEQ_OQ)), vb01234567c3, vacc0x01234567);
        vacc1x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb01234567c3, vzero, _CMP_NEQ_OQ)), vb01234567c3, vacc1x01234567);
        vacc2x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb01234567c3, vzero, _CMP_NEQ_OQ)), vb01234567c3, vacc2x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc3, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc3, vacc0x89ABCDEF);
        vacc1x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb89ABCDEFc3, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc3, vacc1x89ABCDEF);
        vacc2x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb89ABCDEFc3, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc3, vacc2x89ABCDEF);


        w += 64;
      }
      p -= 3 * sizeof(void*);
    } while (p != 0);

    vacc0x01234567 = _mm256_max_ps(vmin, vacc0x01234567);
    vacc1x01234567 = _mm256_max_ps(vmin, vacc1x01234567);
    vacc2x01234567 = _mm256_max_ps(vmin, vacc2x01234567);
    vacc0x89ABCDEF = _mm256_max_ps(vmin, vacc0x89ABCDEF);
    vacc1x89ABCDEF = _mm256_max_ps(vmin, vacc1x89ABCDEF);
    vacc2x89ABCDEF = _mm256_max_ps(vmin, vacc2x89ABCDEF);

    vacc0x01234567 = _mm256_min_ps(vmax, vacc0x01234567);
    vacc1x01234567 = _mm256_min_ps(vmax, vacc1x01234567);
    vacc2x01234567 = _mm256_min_ps(vmax, vacc2x01234567);
    vacc0x89ABCDEF = _mm256_min_ps(vmax, vacc0x89ABCDEF);
    vacc1x89ABCDEF = _mm256_min_ps(vmax, vacc1x89ABCDEF);
    vacc2x89ABCDEF = _mm256_min_ps(vmax, vacc2x89ABCDEF);

    if XNN_LIKELY(nc >= 16) {
      _mm256_storeu_ps(c2, vacc2x01234567);
      _mm256_storeu_ps(c2 + 8, vacc2x89ABCDEF);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      _mm256_storeu_ps(c1, vacc1x01234567);
      _mm256_storeu_ps(c1 + 8, vacc1x89ABCDEF);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm256_storeu_ps(c0, vacc0x01234567);
      _mm256_storeu_ps(c0 + 8, vacc0x89ABCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 16;
    } else {
      if (nc & 8) {
        _mm256_storeu_ps(c2, vacc2x01234567);
        _mm256_storeu_ps(c1, vacc1x01234567);
        _mm256_storeu_ps(c0, vacc0x01234567);

        vacc2x01234567 = vacc2x89ABCDEF;
        vacc1x01234567 = vacc1x89ABCDEF;
        vacc0x01234567 = vacc0x89ABCDEF;

        c2 += 8;
        c1 += 8;
        c0 += 8;
      }
      __m128 vacc2x0123 = _mm256_castps256_ps128(vacc2x01234567);
      __m128 vacc1x0123 = _mm256_castps256_ps128(vacc1x01234567);
      __m128 vacc0x0123 = _mm256_castps256_ps128(vacc0x01234567);
      if (nc & 4) {
        _mm_storeu_ps(c2, vacc2x0123);
        _mm_storeu_ps(c1, vacc1x0123);
        _mm_storeu_ps(c0, vacc0x0123);

        vacc2x0123 = _mm256_extractf128_ps(vacc2x01234567, 1);
        vacc1x0123 = _mm256_extractf128_ps(vacc1x01234567, 1);
        vacc0x0123 = _mm256_extractf128_ps(vacc0x01234567, 1);

        c2 += 4;
        c1 += 4;
        c0 += 4;
      }
      if (nc & 2) {
        _mm_storel_pi((__m64*) c2, vacc2x0123);
        _mm_storel_pi((__m64*) c1, vacc1x0123);
        _mm_storel_pi((__m64*) c0, vacc0x0123);

        vacc2x0123 = _mm_movehl_ps(vacc2x0123, vacc2x0123);
        vacc1x0123 = _mm_movehl_ps(vacc1x0123, vacc1x0123);
        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);

        c2 += 2;
        c1 += 2;
        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c2, vacc2x0123);
        _mm_store_ss(c1, vacc1x0123);
        _mm_store_ss(c0, vacc0x0123);
      }

      nc = 0;
    }
  } while (nc != 0);
}
