# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: microkernel filename lists for avx512fp16
#
# Auto-generated file. Do not edit!
#   Generator: tools/update-microkernels.py


SET(PROD_AVX512FP16_MICROKERNEL_SRCS
  src/f16-gemm/gen/f16-gemm-1x64-minmax-avx512fp16-broadcast.c
  src/f16-gemm/gen/f16-gemm-7x64-minmax-avx512fp16-broadcast.c
  src/f16-igemm/gen/f16-igemm-1x64-minmax-avx512fp16-broadcast.c
  src/f16-igemm/gen/f16-igemm-7x64-minmax-avx512fp16-broadcast.c
  src/f16-rminmax/gen/f16-rmax-avx512fp16-u128-acc4.c
  src/f16-rminmax/gen/f16-rminmax-avx512fp16-u128-acc4.c
  src/f16-vbinary/gen/f16-vadd-minmax-avx512fp16-u64.c
  src/f16-vbinary/gen/f16-vaddc-minmax-avx512fp16-u64.c
  src/f16-vbinary/gen/f16-vdiv-minmax-avx512fp16-u64.c
  src/f16-vbinary/gen/f16-vdivc-minmax-avx512fp16-u64.c
  src/f16-vbinary/gen/f16-vmax-avx512fp16-u64.c
  src/f16-vbinary/gen/f16-vmaxc-avx512fp16-u64.c
  src/f16-vbinary/gen/f16-vmin-avx512fp16-u64.c
  src/f16-vbinary/gen/f16-vminc-avx512fp16-u64.c
  src/f16-vbinary/gen/f16-vmul-minmax-avx512fp16-u64.c
  src/f16-vbinary/gen/f16-vmulc-minmax-avx512fp16-u64.c
  src/f16-vbinary/gen/f16-vrdivc-minmax-avx512fp16-u64.c
  src/f16-vbinary/gen/f16-vrsubc-minmax-avx512fp16-u64.c
  src/f16-vbinary/gen/f16-vsqrdiff-avx512fp16-u64.c
  src/f16-vbinary/gen/f16-vsqrdiffc-avx512fp16-u64.c
  src/f16-vbinary/gen/f16-vsub-minmax-avx512fp16-u64.c
  src/f16-vbinary/gen/f16-vsubc-minmax-avx512fp16-u64.c)

SET(NON_PROD_AVX512FP16_MICROKERNEL_SRCS
  src/f16-gemm/gen/f16-gemm-1x32-minmax-avx512fp16-broadcast.c
  src/f16-gemm/gen/f16-gemm-4x32-minmax-avx512fp16-broadcast.c
  src/f16-gemm/gen/f16-gemm-4x64-minmax-avx512fp16-broadcast.c
  src/f16-gemm/gen/f16-gemm-5x32-minmax-avx512fp16-broadcast.c
  src/f16-gemm/gen/f16-gemm-5x64-minmax-avx512fp16-broadcast.c
  src/f16-gemm/gen/f16-gemm-6x32-minmax-avx512fp16-broadcast.c
  src/f16-gemm/gen/f16-gemm-6x64-minmax-avx512fp16-broadcast.c
  src/f16-gemm/gen/f16-gemm-7x32-minmax-avx512fp16-broadcast.c
  src/f16-gemm/gen/f16-gemm-8x32-minmax-avx512fp16-broadcast.c
  src/f16-gemm/gen/f16-gemm-8x64-minmax-avx512fp16-broadcast.c
  src/f16-igemm/gen/f16-igemm-1x32-minmax-avx512fp16-broadcast.c
  src/f16-igemm/gen/f16-igemm-4x32-minmax-avx512fp16-broadcast.c
  src/f16-igemm/gen/f16-igemm-4x64-minmax-avx512fp16-broadcast.c
  src/f16-igemm/gen/f16-igemm-5x32-minmax-avx512fp16-broadcast.c
  src/f16-igemm/gen/f16-igemm-5x64-minmax-avx512fp16-broadcast.c
  src/f16-igemm/gen/f16-igemm-6x32-minmax-avx512fp16-broadcast.c
  src/f16-igemm/gen/f16-igemm-6x64-minmax-avx512fp16-broadcast.c
  src/f16-igemm/gen/f16-igemm-7x32-minmax-avx512fp16-broadcast.c
  src/f16-igemm/gen/f16-igemm-8x32-minmax-avx512fp16-broadcast.c
  src/f16-igemm/gen/f16-igemm-8x64-minmax-avx512fp16-broadcast.c
  src/f16-rminmax/gen/f16-rmax-avx512fp16-u32.c
  src/f16-rminmax/gen/f16-rmax-avx512fp16-u64-acc2.c
  src/f16-rminmax/gen/f16-rmax-avx512fp16-u96-acc3.c
  src/f16-rminmax/gen/f16-rmax-avx512fp16-u128-acc2.c
  src/f16-rminmax/gen/f16-rmin-avx512fp16-u32.c
  src/f16-rminmax/gen/f16-rmin-avx512fp16-u64-acc2.c
  src/f16-rminmax/gen/f16-rmin-avx512fp16-u96-acc3.c
  src/f16-rminmax/gen/f16-rmin-avx512fp16-u128-acc2.c
  src/f16-rminmax/gen/f16-rmin-avx512fp16-u128-acc4.c
  src/f16-rminmax/gen/f16-rminmax-avx512fp16-u32.c
  src/f16-rminmax/gen/f16-rminmax-avx512fp16-u64-acc2.c
  src/f16-rminmax/gen/f16-rminmax-avx512fp16-u96-acc3.c
  src/f16-rminmax/gen/f16-rminmax-avx512fp16-u128-acc2.c
  src/f16-rsum/gen/f16-rsum-avx512fp16-u32.c
  src/f16-rsum/gen/f16-rsum-avx512fp16-u64-acc2.c
  src/f16-rsum/gen/f16-rsum-avx512fp16-u96-acc3.c
  src/f16-rsum/gen/f16-rsum-avx512fp16-u128-acc2.c
  src/f16-rsum/gen/f16-rsum-avx512fp16-u128-acc4.c
  src/f16-vbinary/gen/f16-vadd-minmax-avx512fp16-u32.c
  src/f16-vbinary/gen/f16-vaddc-minmax-avx512fp16-u32.c
  src/f16-vbinary/gen/f16-vdiv-minmax-avx512fp16-u32.c
  src/f16-vbinary/gen/f16-vdivc-minmax-avx512fp16-u32.c
  src/f16-vbinary/gen/f16-vmax-avx512fp16-u32.c
  src/f16-vbinary/gen/f16-vmaxc-avx512fp16-u32.c
  src/f16-vbinary/gen/f16-vmin-avx512fp16-u32.c
  src/f16-vbinary/gen/f16-vminc-avx512fp16-u32.c
  src/f16-vbinary/gen/f16-vmul-minmax-avx512fp16-u32.c
  src/f16-vbinary/gen/f16-vmulc-minmax-avx512fp16-u32.c
  src/f16-vbinary/gen/f16-vrdivc-minmax-avx512fp16-u32.c
  src/f16-vbinary/gen/f16-vrsubc-minmax-avx512fp16-u32.c
  src/f16-vbinary/gen/f16-vsqrdiff-avx512fp16-u32.c
  src/f16-vbinary/gen/f16-vsqrdiffc-avx512fp16-u32.c
  src/f16-vbinary/gen/f16-vsub-minmax-avx512fp16-u32.c
  src/f16-vbinary/gen/f16-vsubc-minmax-avx512fp16-u32.c
  src/f16-vsqrt/gen/f16-vsqrt-avx512fp16-sqrt-u32.c
  src/f16-vsqrt/gen/f16-vsqrt-avx512fp16-sqrt-u64.c
  src/f16-vsqrt/gen/f16-vsqrt-avx512fp16-sqrt-u128.c)

SET(ALL_AVX512FP16_MICROKERNEL_SRCS ${PROD_AVX512FP16_MICROKERNEL_SRCS} + ${NON_PROD_AVX512FP16_MICROKERNEL_SRCS})
