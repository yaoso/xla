#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/lowering_context.h"

namespace torch_xla {

class ShardingUtil {
 public:
  // Annotate HLO instructions in the lowered compuation by the embedded XLA
  // builder. For this call to be effective, this needs to be called after the
  // lowering and before building the computation; otherwise, this is a no-op.
  static void SetHloSharding(LoweringContext* lowering_ctx);

  // This is called separately before xrt compilation. This is also useful
  // for debugging partitioned HLO computation and sharding propation.
  static xla::HloModuleProto SpmdPartitioningPass(
      const xla::HloModuleProto& hlo_proto,
      bool conv_halo_exchange_always_on_lhs = true,
      bool choose_faster_windowed_einsum_over_mem = false,
      bool unroll_windowed_einsum = false,
      bool bidirectional_windowed_einsum = false);
};

}  // namespace torch_xla
