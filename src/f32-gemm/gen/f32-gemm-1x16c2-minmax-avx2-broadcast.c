// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/c2-avx-broadcast.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>
#include "xnnpack/math.h"

#include "xnnpack/gemm.h"


void xnn_f32_gemm_minmax_ukernel_1x16c2__avx2_broadcast(
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
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 2 * sizeof(float));
  const float* a0 = a;
  float* c0 = c;

  const __m256 vmin = _mm256_set1_ps(params->scalar.min);
  const __m256 vmax = _mm256_set1_ps(params->scalar.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    __m256 vacc0x0123c2 = _mm256_castsi256_ps(_mm256_cvtepu32_epi64(_mm_castps_si128(_mm_load_ps(w + 0))));
    __m256 vacc0x4567c2 = _mm256_castsi256_ps(_mm256_cvtepu32_epi64(_mm_castps_si128(_mm_load_ps(w + 4))));
    __m256 vacc0x89ABc2 = _mm256_castsi256_ps(_mm256_cvtepu32_epi64(_mm_castps_si128(_mm_load_ps(w + 8))));
    __m256 vacc0xCDEFc2 = _mm256_castsi256_ps(_mm256_cvtepu32_epi64(_mm_castps_si128(_mm_load_ps(w + 12))));
    w += 16;

    size_t k = kc;
    while (k >= 2 * sizeof(float)) {
      const __m256 va0 = _mm256_castsi256_ps(_mm256_set1_epi64x(*(int64_t *)a0));
      a0 += 2;

      const __m256 vb0x0123c2 = _mm256_load_ps(w);
      const __m256 vb0x4567c2 = _mm256_load_ps(w + 8);
      w += 16;
      const __m256 vb1x0123c2 = _mm256_load_ps(w);
      const __m256 vb1x4567c2 = _mm256_load_ps(w + 8);
      w += 16;

      vacc0x0123c2 = _mm256_fmadd_ps(va0, vb0x0123c2, vacc0x0123c2);
      vacc0x4567c2 = _mm256_fmadd_ps(va0, vb0x4567c2, vacc0x4567c2);
      vacc0x89ABc2 = _mm256_fmadd_ps(va0, vb1x0123c2, vacc0x89ABc2);
      vacc0xCDEFc2 = _mm256_fmadd_ps(va0, vb1x4567c2, vacc0xCDEFc2);
      
      k -= 2 * sizeof(float);
    }

    __m256 vsum0x01452367 = _mm256_hadd_ps(vacc0x0123c2, vacc0x4567c2);
    __m256 vsum0x89CDABEF = _mm256_hadd_ps(vacc0x89ABc2, vacc0xCDEFc2);

    __m256 vacc0x01234567 = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(vsum0x01452367), _MM_SHUFFLE(3, 1, 2, 0)));
    __m256 vacc0x89ABCDEF = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(vsum0x89CDABEF), _MM_SHUFFLE(3, 1, 2, 0)));

    vacc0x01234567 = _mm256_max_ps(vmin, vacc0x01234567);
    vacc0x89ABCDEF = _mm256_max_ps(vmin, vacc0x89ABCDEF);

    vacc0x01234567 = _mm256_min_ps(vmax, vacc0x01234567);
    vacc0x89ABCDEF = _mm256_min_ps(vmax, vacc0x89ABCDEF);

    if XNN_LIKELY(nc >= 16) {
      _mm256_storeu_ps(c0, vacc0x01234567);
      _mm256_storeu_ps(c0 + 8, vacc0x89ABCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 16;
    } else {
      if (nc & 8) {
        _mm256_storeu_ps(c0, vacc0x01234567);

        vacc0x01234567 = vacc0x89ABCDEF;

        c0 += 8;
      }
      __m128 vacc0x0123 = _mm256_castps256_ps128(vacc0x01234567);
      if (nc & 4) {
        _mm_storeu_ps(c0, vacc0x0123);

        vacc0x0123 = _mm256_extractf128_ps(vacc0x01234567, 1);

        c0 += 4;
      }
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vacc0x0123);

        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);

        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c0, vacc0x0123);
      }

      nc = 0;
    }
  } while (nc != 0);
}
