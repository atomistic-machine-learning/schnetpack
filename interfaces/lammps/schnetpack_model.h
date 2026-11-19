/* ----------------------------------------------------------------------
   A potential that spkdeploy compiled with AOTInductor
   (schnetpack.interfaces.lammps_interface), run on flat arrays.

   Independent of LAMMPS, so that pair_style schnetpack and the C++ test in
   tests/ run the same code. No torch header is included here: LAMMPS pulls
   pair-style headers into style_pair.h, where torch's unqualified "Device"
   would clash with LAMMPS_NS::Device.
------------------------------------------------------------------------- */

#ifndef SCHNETPACK_MODEL_H
#define SCHNETPACK_MODEL_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace schnetpack {

struct Result {
  double energy = 0.0;
  // [n_nodes], zero for nodes that are not local
  std::vector<double> energy_atom;
  // [n_nodes * 3], on every node, ghost atoms included
  std::vector<double> forces;
  // sum_i R_i (x) F_i, row-major; LAMMPS' sign convention
  double virial[9] = {0.0};
};

class Model {
 public:
  // Reads and checks the metadata of the package at `path` and loads it; a
  // CUDA package goes to GPU `local_rank % n_gpus`. Throws std::runtime_error.
  explicit Model(const std::string &path, int local_rank = 0);
  ~Model();

  // Inputs for N nodes and E edges: positions [N*3], atomic numbers [N],
  // local [N] (1 for atoms whose energy counts, 0 for ghost atoms), edges
  // idx_i, idx_j [E] and Cartesian shifts of atom j [E*3], or nullptr for
  // none.
  void compute(int64_t n_nodes, const double *positions, const int64_t *atomic_numbers,
               const double *local, int64_t n_edges, const int64_t *idx_i, const int64_t *idx_j,
               const double *offsets, Result &result);

  double cutoff = 0.0;
  // how far the energy of an atom reaches: ghost atoms must cover it
  double receptive_field = 0.0;
  std::string device, dtype, torch_version;

 private:
  struct Torch;
  std::unique_ptr<Torch> torch_;
};

}    // namespace schnetpack

#endif
