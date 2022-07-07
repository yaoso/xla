#include "tensorflow/compiler/xla/xla_client/pjrt_computation_client.h"

#include <algorithm>

#include "absl/strings/ascii.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/pjrt/cpu_device.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/tpu_client.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/env_vars.h"
#include "tensorflow/compiler/xla/xla_client/tf_logging.h"

namespace xla {

namespace {

std::string PjRtDeviceToString(PjRtDevice* const device) {
  std::string platform =
      absl::AsciiStrToUpper(device->client()->platform_name());
  std::string str = absl::StrFormat("%s:%d", platform, device->id());
  return str;
}

std::vector<std::string> PjRtDevicesToString(
    absl::Span<PjRtDevice* const> devices) {
  std::vector<std::string> strs;
  strs.reserve(devices.size());

  for (auto* device : devices) {
    strs.push_back(PjRtDeviceToString(device));
  }

  return strs;
}

}  // namespace

PjRtComputationClient::PjRtComputationClient() {
  std::string device_type = sys_util::GetEnvString(env::kEnvPjRtDevice, "");
  if (device_type == "CPU") {
    TF_VLOG(1) << "Initializing PjRt CPU client...";
    client_ = std::move(xla::GetCpuClient(/*asynchronous=*/false).ValueOrDie());
  } else if (device_type == "TPU") {
    TF_VLOG(1) << "Initializing PjRt TPU client...";
    client_ = xla::GetTpuClient(/*max_inflight_computations=*/1).ValueOrDie();
  } else {
    XLA_ERROR() << absl::StrFormat("Unknown %s '%s'", env::kEnvPjRtDevice,
                                   device_type);
  }

  XLA_CHECK(client_.get() != nullptr);

  for (auto* device : client_->devices()) {
    std::string device_str = PjRtDeviceToString(device);
    string_to_device_.emplace(device_str, device);
  }
}

void PjRtComputationClient::PjRtData::Assign(const Data& data) {
  const PjRtData& pjrt_data = dynamic_cast<const PjRtData&>(data);
  if (&pjrt_data != this) {
    buffer = pjrt_data.buffer;
  }
}

ComputationClient::DataPtr PjRtComputationClient::CreateDataPlaceholder(
    std::string device, Shape shape) {
  return std::make_shared<PjRtData>(device, shape);
}

std::vector<ComputationClient::DataPtr> PjRtComputationClient::TransferToServer(
    absl::Span<const TensorSource> tensors) {
  std::vector<ComputationClient::DataPtr> datas;
  datas.reserve(tensors.size());
  for (auto& tensor : tensors) {
    xla::Literal literal(tensor.shape);
    tensor.populate_fn(tensor, literal.untyped_data(), literal.size_bytes());

    PjRtDevice* pjrt_device = StringToPjRtDevice(tensor.device);
    std::shared_ptr<xla::PjRtBuffer> buffer =
        client_->BufferFromHostLiteral(literal, pjrt_device).ValueOrDie();
    buffer->GetReadyFuture().Await();
    ComputationClient::DataPtr data =
        std::make_shared<PjRtData>(tensor.device, tensor.shape, buffer);
    datas.push_back(data);
  }

  return datas;
}

std::vector<xla::Literal> PjRtComputationClient::TransferFromServer(
    absl::Span<const DataPtr> handles) {
  std::vector<xla::Literal> literals;
  literals.reserve(handles.size());

  for (auto handle : handles) {
    const PjRtData& pjrt_data = dynamic_cast<const PjRtData&>(*handle);

    std::shared_ptr<xla::Literal> literal =
        pjrt_data.buffer->ToLiteralSync().ValueOrDie();
    literals.push_back(std::move(*literal));
  }

  return literals;
}

std::vector<ComputationClient::ComputationPtr> PjRtComputationClient::Compile(
    std::vector<ComputationClient::CompileInstance> instances) {
  std::vector<ComputationClient::ComputationPtr> computations;

  for (auto& instance : instances) {
    // TODO(yeounoh) we could consider having a separate Compile fxn for SPMD,
    // instead of maintaining a different set of compile_options within.
    xla::CompileOptions compile_options;

    // TODO(yeounoh) temporary test flag; replace with a cheker.
    if (sys_util::GetEnvString(env::kEnvSpmdTest, "0") == "1") {
      // If SPMD is enabled, then we assign multi-cores to a single replica;
      // otherwise, all cores participate in replication.
      compile_options.executable_build_options.set_use_spmd_partitioning(true);
      compile_options.executable_build_options.set_num_partitions(
          client_->device_count());
      compile_options.executable_build_options.set_num_replicas(1);
      std::cout << "Testing SPMD flag in PJRT Compile()" << std::endl;
    } else {
      // TODO(wcromar): set compile_options.argument_layouts, enable strict
      // shapes
      compile_options.executable_build_options.set_num_partitions(1);
      compile_options.executable_build_options.set_num_replicas(
          client_->device_count());
    }

    PjRtDevice* pjrt_device = StringToPjRtDevice(instance.compilation_device);

    std::unique_ptr<xla::PjRtExecutable> executable =
        ConsumeValue(client_->Compile(instance.computation, compile_options));

    const auto& hlo_modules = ConsumeValue(executable->GetHloModules());
    HloComputation* hlo_computation = hlo_modules[0]->entry_computation();
    xla::ProgramShape program_shape =
        xla::ProgramShape(hlo_computation->ToProto().program_shape());
    std::cout << "new program shape: " << program_shape.ToString() << std::endl;
    std::cout << "layout: "
              << hlo_modules[0]->compute_computation_layout().ToString()
              << std::endl;

    // TODO(yeounoh) create a new CompileInstance
    std::shared_ptr<PjRtComputation> pjrt_computation =
        std::make_shared<PjRtComputation>(
            std::move(xla::XlaComputation(hlo_modules[0]->ToProto())),
            program_shape, instance.devices, std::move(executable));

    computations.push_back(pjrt_computation);
  }

  return computations;
}

std::vector<ComputationClient::DataPtr>
PjRtComputationClient::ExecuteComputation(
    const ComputationClient::Computation& computation,
    absl::Span<const ComputationClient::DataPtr> arguments,
    const std::string& device, const ExecuteComputationOptions& options) {
  xla::ExecuteOptions execute_options;
  execute_options.untuple_result = options.explode_tuple;
  execute_options.strict_shape_checking = false;

  const PjRtComputation& pjrt_computation =
      dynamic_cast<const PjRtComputation&>(computation);

  // TODO(yeounoh) temporary test flag; replace with a cheker.
  if (sys_util::GetEnvString(env::kEnvSpmdTest, "0") == "1") {
    TF_VLOG(1)
        << "Executing PjRt computation on all devices in replication mode.";

    execute_options.strict_shape_checking = true;
    execute_options.multi_slice_config = nullptr;  // run on single-slice

    std::vector<std::string> devices = GetAllDevices();

    // TODO(yeounoh):
    // - Avoid the data pre-loading in case of SPMD
    // - Shard input (input handler)
    // - Execute replicated mode
    // - Return outputs (output handler)
    std::vector<std::vector<PjRtBuffer*>> argument_handles;
    for (auto& device : devices) {
      xla::PjRtDevice* pjrt_device = StringToPjRtDevice(device);
      XLA_CHECK(pjrt_device->IsAddressable()) << pjrt_device->DebugString();

      std::vector<PjRtBuffer*> buffers;
      for (auto& argument : arguments) {
        const PjRtData* pjrt_data = dynamic_cast<PjRtData*>(argument.get());

        XLA_CHECK(pjrt_device == pjrt_data->buffer->device())
            << pjrt_device->DebugString() << " vs "
            << pjrt_data->buffer->device()->DebugString();
        buffers.push_back(pjrt_data->buffer.get());
      }
      argument_handles.push_back(buffers);
    }

    std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> results =
        pjrt_computation.executable->Execute(argument_handles, execute_options)
            .ValueOrDie();

    std::vector<std::vector<DataPtr>> data_handles;
    for (auto& result : results) {
      std::vector<DataPtr> datas;
      for (int32_t i = 0; i < result.size(); ++i) {
        std::unique_ptr<xla::PjRtBuffer> buffer = std::move(result[i]);

        std::shared_ptr<PjRtData> data = std::make_shared<PjRtData>(
            device, buffer->logical_on_device_shape().ValueOrDie(),
            std::move(buffer));

        datas.push_back(data);
      }
      data_handles.push_back(datas);
    }

    TF_VLOG(1) << "Returning " << data_handles.size() << " sets of results";
    return data_handles[0];
  } else {
    TF_VLOG(1) << "Executing PjRt computation on " << device;

    xla::PjRtDevice* pjrt_device = StringToPjRtDevice(device);
    XLA_CHECK(pjrt_device->IsAddressable()) << pjrt_device->DebugString();

    std::vector<xla::PjRtBuffer*> buffers;
    buffers.reserve(arguments.size());
    for (auto& argument : arguments) {
      const PjRtData* pjrt_data = dynamic_cast<PjRtData*>(argument.get());

      XLA_CHECK(pjrt_device == pjrt_data->buffer->device())
          << pjrt_device->DebugString() << " vs "
          << pjrt_data->buffer->device()->DebugString();
      buffers.push_back(pjrt_data->buffer.get());
    }

    std::vector<std::unique_ptr<xla::PjRtBuffer>> results =
        pjrt_computation.executable
            ->ExecuteSharded(buffers, pjrt_device, execute_options)
            .ValueOrDie();

    std::vector<DataPtr> datas;
    datas.reserve(results.size());
    for (auto& result : results) {
      std::unique_ptr<xla::PjRtBuffer> buffer = std::move(result);

      std::shared_ptr<PjRtData> data = std::make_shared<PjRtData>(
          device, buffer->logical_on_device_shape().ValueOrDie(),
          std::move(buffer));

      datas.push_back(data);
    }

    TF_VLOG(1) << "Returning " << datas.size() << " results";
    return datas;
  }
}

size_t PjRtComputationClient::GetNumDevices() const {
  return client_->addressable_device_count();
}

std::string PjRtComputationClient::GetDefaultDevice() const {
  return PjRtDeviceToString(client_->addressable_devices()[0]);
}

std::vector<std::string> PjRtComputationClient::GetLocalDevices() const {
  return PjRtDevicesToString(client_->addressable_devices());
}

std::vector<std::string> PjRtComputationClient::GetAllDevices() const {
  return PjRtDevicesToString(client_->devices());
}

void PjRtComputationClient::SetReplicationDevices(
    std::shared_ptr<std::vector<std::string>> devices) {
  replication_devices_ = std::move(devices);
}

std::shared_ptr<std::vector<std::string>>
PjRtComputationClient::GetReplicationDevices() {
  return replication_devices_;
}

xla::PjRtDevice* PjRtComputationClient::StringToPjRtDevice(
    const std::string& device) {
  XLA_CHECK(string_to_device_.find(device) != string_to_device_.end())
      << "Unknown device " << device;
  xla::PjRtDevice* pjrt_device = string_to_device_[device];
  return pjrt_device;
}

}  // namespace xla
