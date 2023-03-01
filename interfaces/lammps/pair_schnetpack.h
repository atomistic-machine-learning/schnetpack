/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov
   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.
   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(schnetpack,PairSCHNETPACK)

#else

#ifndef LMP_PAIR_SCHNETPACK_H
#define LMP_PAIR_SCHNETPACK_H

#include "pair.h"

#include <torch/torch.h>

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
  torch::jit::script::Module model;
  torch::Device device = torch::kCPU;

 protected:
  int * type_mapper;
  int debug_mode = 0;

};

}

#endif
#endif
