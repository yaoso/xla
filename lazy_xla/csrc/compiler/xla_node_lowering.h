#pragma once

#include "torch_xla/csrc/compiler/node_lowering.h"

namespace torch_lazy_tensors {
namespace compiler {

NodeLowering* GetXlaNodeLowering();
std::unique_ptr<NodeLowering> CreateXlaNodeLowering(ir::LoweringContext* loctx);

}  // namespace compiler
}  // namespace torch_lazy_tensors