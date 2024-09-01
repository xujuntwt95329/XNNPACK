// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Microkernel: f16-vtanh
//   Generator: tools/generate-vunary-benchmark.py

#include <stddef.h>
#include <stdint.h>

#include <benchmark/benchmark.h>
#include "bench/f16-vunary-benchmark.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams.h"

void f16_vtanh(benchmark::State& state, uint64_t arch_flags, xnn_f16_vtanh_ukernel_fn ukernel,
              xnn_init_f16_tanh_params_fn init_params = nullptr) {
  f16_vunary_benchmark<xnn_f16_tanh_params>(
      state, ukernel,
      init_params,
      arch_flags,
      /*range_min=*/-5.0,
      /*range_max=*/5.0);
}

#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile,              \
                                datatype, params_type, init_params)                        \
BENCHMARK_CAPTURE(f16_vtanh, ukernel, arch_flags, ukernel, init_params)                    \
  ->Apply(benchmark::utils::UnaryElementwiseParameters<datatype, datatype>)                \
  ->UseRealTime();
#include "src/f16-vtanh/f16-vtanh.h"
#undef XNN_UKERNEL_WITH_PARAMS


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
