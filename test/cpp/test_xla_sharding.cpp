#include <ATen/ATen.h>
#include <gtest/gtest.h>

#include <iostream>

#include "cpp_test_util.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/multi_wait.h"
#include "tensorflow/compiler/xla/xla_client/thread_pool.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/torch_util.h"
#include "torch_xla_test.h"

namespace torch_xla {
namespace cpp_test {

// A placeholder for future SPMD sharding tests.

class XLAShardingTest : public AtenXlaTensorTestBase {};

TEST_F(XLAShardingTest, TestSPMDPartitioner) {
<<<<<<< HEAD
  // For debugging purposes only.
  const int64_t replica_count_ = 1;
  const int64_t num_partitions_ = 8;

  absl::string_view hlo_text = R"
  HloModule IrToHlo.10

  ENTRY %IrToHlo.10 (p0.6: f32[64,10]) -> (f32[64,10]) {
    p0.6 = f32[64,10]{0,1} parameter(0)
    constant.1 = f32[] constant(1)
    reshape.2 = f32[1,1]{1,0} reshape(f32[] constant.1)
    broadcast.3 = f32[1,1]{1,0} broadcast(f32[1,1]{1,0} %reshape.2), dimensions={0,1}
    reshape.4 = f32[] reshape(f32[1,1]{1,0} %broadcast.3)
    broadcast.5 = f32[64,10]{1,0} broadcast(f32[] %reshape.4), dimensions={}
    multiply.7 = f32[64,10]{0,1} multiply(f32[64,10]{0,1} p0.6, f32[64,10]{1,0} %broadcast.5)
    add.8 = f32[64,10]{0,1} add(f32[64,10]{0,1} p0.6, f32[64,10]{0,1} %multiply.7)
    ROOT tuple.9 = (f32[64,10]{0,1}) tuple(f32[64,10]{0,1} add.8)
  }
  ";

  HloModuleConfig config;
  config.set_replica_count(replica_count_);
  config.set_num_partitions(num_partitions_);

  // Parse and return a verified module
  // TODO: might need to update sharding with the given replica_count_
  // and re-run SpmdPartitioner.
  auto module = absl::make_unique<HloModule>("TestSPMDPartitioner", config);
  auto parser = HloParser::CreateHloParserForTests(hlo_text);
  parser->run(module);
<<<<<<< HEAD
=======
  const int64_t replica_count_ = 8;

  absl::string_view hlo_string =
      R"(
      HloModule module

      ENTRY entry {
        constant = f32[3,3]{1,0} constant({{1,3,7},{5,1,4},{1,2,8}}),
          sharding={replicated}
        constant.1 = f32[3,3]{1,0} constant({{2,7,2},{2,9,2},{2,6,2}}),
          sharding={replicated}
        multiply = f32[3,3]{1,0} multiply(constant, constant.1),
          sharding={devices=[2,1,2]0,1,2,3 last_tile_dim_replicate}
        add = f32[3,3]{1,0} add(multiply, constant.1),
          sharding={devices=[4,1]0,1,2,3}
        ROOT copy = f32[3,3]{1,0} copy(add),
          sharding={replicated}
      }
      )";
  // Run SPMDPartitioner
>>>>>>> e211ec20 (Update sharding spec to support full replication & mesh sharding)
=======
>>>>>>> 4ee1ece0 (CreateXrtSpmdComputation only if spmd is enalbed in HloModule)
}

}  // namespace cpp_test
}  // namespace torch_xla
