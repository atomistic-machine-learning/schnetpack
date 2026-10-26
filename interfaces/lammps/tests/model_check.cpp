/* ----------------------------------------------------------------------
   Runs schnetpack::Model, the part of pair_style schnetpack that loads and
   runs a package, on inputs from a file and prints its results, so that
   tests/interfaces/test_lammps_interface.py compares them to Python.

     model_check model.pt2 inputs.txt

   inputs.txt: N E S, then positions [N*3], atomic numbers [N], local [N],
   idx_i [E], idx_j [E] and, if S is 1, offsets [E*3].
   Prints the energy, then energy_atom [N], forces [N*3] and virial [9].
------------------------------------------------------------------------- */

#include "schnetpack_model.h"

#include <cstdio>
#include <exception>
#include <fstream>
#include <vector>

int main(int argc, char **argv)
{
  if (argc != 3) {
    std::fprintf(stderr, "usage: model_check model.pt2 inputs.txt\n");
    return 2;
  }
  std::ifstream in(argv[2]);
  int64_t n, e, with_offsets;
  in >> n >> e >> with_offsets;
  std::vector<double> positions(3 * n), local(n), offsets(3 * e);
  std::vector<int64_t> z(n), idx_i(e), idx_j(e);
  for (auto &v : positions) in >> v;
  for (auto &v : z) in >> v;
  for (auto &v : local) in >> v;
  for (auto &v : idx_i) in >> v;
  for (auto &v : idx_j) in >> v;
  if (with_offsets)
    for (auto &v : offsets) in >> v;

  try {
    schnetpack::Model model(argv[1]);
    schnetpack::Result result;
    model.compute(n, positions.data(), z.data(), local.data(), e, idx_i.data(), idx_j.data(),
                  with_offsets ? offsets.data() : nullptr, result);
    std::printf("%.17g\n", result.energy);
    for (double v : result.energy_atom) std::printf("%.17g\n", v);
    for (double v : result.forces) std::printf("%.17g\n", v);
    for (double v : result.virial) std::printf("%.17g\n", v);
  } catch (const std::exception &err) {
    std::fprintf(stderr, "%s\n", err.what());
    return 1;
  }
  return 0;
}
