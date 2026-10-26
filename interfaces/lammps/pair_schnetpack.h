/* ----------------------------------------------------------------------
   SchNetPack pair style: runs a potential that spkdeploy compiled with
   AOTInductor (schnetpack.interfaces.lammps_interface).

   References:

   .. [#pair_nequip] https://github.com/mir-group/pair_nequip
   .. [#lammps] https://github.com/lammps/lammps

------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(schnetpack,PairSCHNETPACK);
// clang-format on
#else

#ifndef LMP_PAIR_SCHNETPACK_H
#define LMP_PAIR_SCHNETPACK_H

#include "pair.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace schnetpack {
class Model;
struct Result;
}    // namespace schnetpack

namespace LAMMPS_NS {

class PairSCHNETPACK : public Pair {
 public:
  PairSCHNETPACK(class LAMMPS *);
  ~PairSCHNETPACK() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  double init_one(int, int) override;
  void init_style() override;

 protected:
  void allocate();

  // the graph covers local and ghost atoms (any number of MPI ranks), or
  // only the local atoms with periodic shifts (one rank)
  bool domain_decomposition = true;
  // atomic number of each LAMMPS atom type, indexed from 1
  std::vector<int64_t> type_to_z;

  // libtorch stays behind schnetpack::Model, so no torch header reaches
  // style_pair.h (see schnetpack_model.h)
  std::unique_ptr<schnetpack::Model> model;
  std::unique_ptr<schnetpack::Result> result;

  // inputs of the model, kept across steps to avoid reallocation
  std::vector<double> positions, offsets, local;
  std::vector<int64_t> atomic_numbers, idx_i, idx_j;
  // local index of each atom ID, for no_domain_decomposition
  std::vector<int> owner;
};

}    // namespace LAMMPS_NS

#endif
#endif
