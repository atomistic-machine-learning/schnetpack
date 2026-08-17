/* ----------------------------------------------------------------------
References:

   .. [#pair_nequip] https://github.com/mir-group/pair_nequip
   .. [#lammps] https://github.com/lammps/lammps

------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(schnetpack,PairSCHNETPACK)

#else

#ifndef LMP_PAIR_SCHNETPACK_H
#define LMP_PAIR_SCHNETPACK_H

#include "pair.h"

#include <memory>

namespace LAMMPS_NS {
    
class PairSCHNETPACK : public Pair {
 public:
  PairSCHNETPACK(class LAMMPS *);
  virtual ~PairSCHNETPACK();
  virtual void compute(int, int);
  void settings(int, char **);
  virtual void coeff(int, char **);
  virtual double init_one(int, int);
  virtual void init_style();
  void allocate();

  double cutoff;

  // The libtorch state is held behind an opaque pointer so that this header
  // does not include <torch/torch.h>. LAMMPS pulls every pair-style header
  // into the generated style_pair.h, which force.cpp and lammps.cpp include
  // after their own "using namespace LAMMPS_NS;" -- and there the unqualified
  // "Device" that ATen uses matches both c10::Device and the LAMMPS_NS::Device
  // enumerator of "enum ExecutionSpace" in src/pointers.h.
  struct Impl;
  std::unique_ptr<Impl> impl;

 protected:
  int * type_mapper;
  int debug_mode = 0;

};

}

#endif
#endif
