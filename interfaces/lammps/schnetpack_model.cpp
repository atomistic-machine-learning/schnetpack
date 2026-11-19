#include "schnetpack_model.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>

#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/torch.h>
#include <torch/version.h>

#if TORCH_VERSION_MAJOR < 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR < 14)
#error "pair_style schnetpack needs libtorch >= 2.14"
#endif

namespace schnetpack {

// the arguments of a package, in order (schnetpack.interfaces.lammps_interface)
static const char *const INPUTS = "positions,atomic_numbers,idx_i,idx_j,offsets,local";

struct Model::Torch {
  std::unique_ptr<torch::inductor::AOTIModelPackageLoader> loader;
  torch::Device device = torch::kCPU;
  torch::Dtype dtype = torch::kFloat32;
};

Model::Model(const std::string &path, int local_rank) : torch_(new Torch)
{
  auto metadata =
      torch::inductor::AOTIModelPackageLoader::load_metadata_from_package(path, "model");
  for (const char *key :
       {"cutoff", "receptive_field", "dtype", "device", "torch_version", "inputs"})
    if (metadata.count(key) == 0)
      throw std::runtime_error(path + " has no '" + key +
                               "' metadata; deploy the model with spkdeploy");

  if (metadata["inputs"] != INPUTS)
    throw std::runtime_error(path + " takes inputs (" + metadata["inputs"] +
                             "), this pair style passes (" + INPUTS +
                             "); deploy the model with the matching SchNetPack version");

  // a package runs only with the torch version that compiled it
  torch_version = metadata["torch_version"];
  const std::string expected =
      std::to_string(TORCH_VERSION_MAJOR) + "." + std::to_string(TORCH_VERSION_MINOR) + ".";
  if (torch_version.compare(0, expected.size(), expected) != 0)
    throw std::runtime_error(path + " was compiled with torch " + torch_version +
                             ", but this is libtorch " + TORCH_VERSION +
                             "; run spkdeploy with the torch version LAMMPS is built with");

  cutoff = std::stod(metadata["cutoff"]);
  receptive_field = std::stod(metadata["receptive_field"]);

  dtype = metadata["dtype"];
  if (dtype == "float32")
    torch_->dtype = torch::kFloat32;
  else if (dtype == "float64")
    torch_->dtype = torch::kFloat64;
  else
    throw std::runtime_error(path + " has unknown dtype " + dtype);

  device = metadata["device"];
  c10::DeviceIndex index = -1;
  if (device == "cuda") {
    int n_gpus = static_cast<int>(torch::cuda::device_count());
    if (n_gpus == 0)
      throw std::runtime_error(path + " was compiled for CUDA, but no GPU is visible");
    index = static_cast<c10::DeviceIndex>(local_rank % n_gpus);
    torch_->device = torch::Device(torch::kCUDA, index);
  } else if (device != "cpu") {
    throw std::runtime_error(path + " was compiled for unsupported device " + device);
  }

  torch_->loader =
      std::make_unique<torch::inductor::AOTIModelPackageLoader>(path, "model", false, 1, index);
}

Model::~Model() = default;

void Model::compute(int64_t n_nodes, const double *positions, const int64_t *atomic_numbers,
                    const double *local, int64_t n_edges, const int64_t *idx_i,
                    const int64_t *idx_j, const double *offsets, Result &result)
{
  const Torch &t = *torch_;
  auto f64 = torch::TensorOptions().dtype(torch::kFloat64);
  auto i64 = torch::TensorOptions().dtype(torch::kInt64);
  // copies, as from_blob does not own the caller's memory
  auto real = [&](const double *data, std::vector<int64_t> shape) {
    return torch::from_blob(const_cast<double *>(data), shape, f64)
        .to(t.device, t.dtype, false, true);
  };
  auto index = [&](const int64_t *data, int64_t n) {
    return torch::from_blob(const_cast<int64_t *>(data), {n}, i64)
        .to(t.device, torch::kInt64, false, true);
  };

  std::vector<at::Tensor> inputs = {
      real(positions, {n_nodes, 3}),
      index(atomic_numbers, n_nodes),
      index(idx_i, n_edges),
      index(idx_j, n_edges),
      offsets ? real(offsets, {n_edges, 3})
              : torch::zeros({n_edges, 3}, f64.dtype(t.dtype).device(t.device)),
      real(local, {n_nodes}),
  };

  std::vector<at::Tensor> outputs;
  {
    torch::NoGradGuard no_grad;
    outputs = t.loader->run(inputs);
  }

  // energy, per-atom energies, forces, virial
  auto host = [](const at::Tensor &x) { return x.to(torch::kCPU, torch::kFloat64).contiguous(); };
  result.energy = host(outputs[0]).item<double>();
  at::Tensor energy_atom = host(outputs[1]);
  at::Tensor forces = host(outputs[2]);
  at::Tensor virial = host(outputs[3]);
  result.energy_atom.assign(energy_atom.data_ptr<double>(),
                            energy_atom.data_ptr<double>() + n_nodes);
  result.forces.assign(forces.data_ptr<double>(), forces.data_ptr<double>() + 3 * n_nodes);
  std::copy(virial.data_ptr<double>(), virial.data_ptr<double>() + 9, result.virial);
}

}    // namespace schnetpack
