/* ----------------------------------------------------------------------
   SchNetPack pair style: runs a potential that spkdeploy compiled with
   AOTInductor (schnetpack.interfaces.lammps_interface).

   The package maps flat tensors -- positions, atomic numbers, edges, edge
   shifts and a mask of the atoms whose energy counts -- to the energy, the
   per-atom energies, the forces and the virial. Forces and virial are part
   of the compiled graph, so no autograd runs here.

     pair_style schnetpack [no_domain_decomposition]
     pair_coeff * * model.pt2 Z_1 ... Z_ntypes

   Domain decomposition (default): local and ghost atoms are the nodes of
   the graph; only local atoms count towards the energy, and the forces on
   ghost atoms are communicated back to their owners (newton pair on). The
   ghost cutoff is raised to the receptive field of the model, so the
   energy is exact on any number of MPI ranks.

   no_domain_decomposition: one MPI rank; only local atoms are nodes, and an
   edge to a ghost atom goes to the atom it images, with the periodic shift.
   Cheaper for small periodic boxes, where the ghost shell would be large.

   References:

   .. [#pair_nequip] https://github.com/mir-group/pair_nequip
   .. [#lammps] https://github.com/lammps/lammps

------------------------------------------------------------------------- */

#include "pair_schnetpack.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "schnetpack_model.h"

#include <mpi.h>

#include <algorithm>
#include <cstring>
#include <exception>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairSCHNETPACK::PairSCHNETPACK(LAMMPS *lmp) : Pair(lmp), result(new schnetpack::Result)
{
  restartinfo = 0;
  single_enable = 0;
  one_coeff = 1;
  manybody_flag = 1;
  // the virial comes from the model
  no_virial_fdotr_compute = 1;
  centroidstressflag = CENTROID_NOTAVAIL;
}

PairSCHNETPACK::~PairSCHNETPACK()
{
  if (copymode) return;
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cutghost);
  }
}

/* ---------------------------------------------------------------------- */

void PairSCHNETPACK::allocate()
{
  allocated = 1;
  int n = atom->ntypes + 1;
  memory->create(setflag, n, n, "pair:setflag");
  memory->create(cutsq, n, n, "pair:cutsq");
  memory->create(cutghost, n, n, "pair:cutghost");
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) setflag[i][j] = 0;
}

void PairSCHNETPACK::settings(int narg, char **arg)
{
  domain_decomposition = true;
  for (int iarg = 0; iarg < narg; iarg++) {
    if (strcmp(arg[iarg], "no_domain_decomposition") == 0)
      domain_decomposition = false;
    else
      error->all(FLERR, "Unknown pair_style schnetpack keyword: {}", arg[iarg]);
  }
  // ghost atoms are nodes of the graph and need their own neighbors
  ghostneigh = domain_decomposition ? 1 : 0;
}

void PairSCHNETPACK::coeff(int narg, char **arg)
{
  if (!allocated) allocate();
  int ntypes = atom->ntypes;
  if (narg != 3 + ntypes)
    error->all(FLERR, "pair_coeff * * <model.pt2> needs one atomic number per atom type ({} types)",
               ntypes);
  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "pair_coeff for pair_style schnetpack must be * *");

  type_to_z.assign(ntypes + 1, 0);
  for (int t = 1; t <= ntypes; t++) {
    type_to_z[t] = utils::inumeric(FLERR, arg[2 + t], false, lmp);
    if (type_to_z[t] < 1)
      error->all(FLERR, "Atom type {} has invalid atomic number {}", t, arg[2 + t]);
  }

  // a CUDA package goes to the GPU of this rank's index on its node
  int local_rank = 0;
#ifndef MPI_STUBS
  MPI_Comm node;
  MPI_Comm_split_type(world, MPI_COMM_TYPE_SHARED, comm->me, MPI_INFO_NULL, &node);
  MPI_Comm_rank(node, &local_rank);
  MPI_Comm_free(&node);
#endif
  try {
    model = std::make_unique<schnetpack::Model>(arg[2], local_rank);
  } catch (const std::exception &e) {
    error->one(FLERR, "Cannot load SchNetPack model {}: {}", arg[2], e.what());
  }
  if (comm->me == 0)
    utils::logmesg(lmp, "SchNetPack: {} on {} in {}, cutoff {}, receptive field {}, torch {}{}\n",
                   arg[2], model->device, model->dtype, model->cutoff, model->receptive_field,
                   model->torch_version, domain_decomposition ? "" : ", no domain decomposition");

  for (int i = 1; i <= ntypes; i++)
    for (int j = i; j <= ntypes; j++) setflag[i][j] = 1;
}

/* ---------------------------------------------------------------------- */

void PairSCHNETPACK::init_style()
{
  if (atom->tag_enable == 0) error->all(FLERR, "Pair style schnetpack requires atom IDs");

  if (domain_decomposition) {
    if (force->newton_pair == 0)
      error->all(FLERR, "Pair style schnetpack requires newton pair on, as the forces on ghost "
                 "atoms are communicated back to their owners");
    neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);
    // the ghost atoms must cover the receptive field of every local atom until
    // the next reneighboring
    double cutghost_min = model->receptive_field + neighbor->skin;
    if (comm->cutghostuser < cutghost_min) {
      comm->cutghostuser = cutghost_min;
      if (comm->me == 0)
        utils::logmesg(lmp, "SchNetPack: ghost cutoff set to {} (receptive field + skin)\n",
                       cutghost_min);
    }
  } else {
    if (comm->nprocs > 1)
      error->all(FLERR, "pair_style schnetpack no_domain_decomposition runs on one MPI rank");
    neighbor->add_request(this, NeighConst::REQ_FULL);
  }
}

double PairSCHNETPACK::init_one(int i, int j)
{
  cutghost[i][j] = cutghost[j][i] = model->cutoff;
  return model->cutoff;
}

/* ---------------------------------------------------------------------- */

void PairSCHNETPACK::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);
  if (vflag_atom) error->all(FLERR, "Pair style schnetpack does not compute per-atom virials");

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  tagint *tag = atom->tag;
  const int nlocal = atom->nlocal;
  if (nlocal == 0) return;

  // nodes: local and ghost atoms, or only local atoms
  const int n_nodes = domain_decomposition ? nlocal + atom->nghost : nlocal;
  const int n_lists = domain_decomposition ? list->inum + list->gnum : list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  positions.resize(3 * n_nodes);
  atomic_numbers.resize(n_nodes);
  local.resize(n_nodes);
  for (int i = 0; i < n_nodes; i++) {
    for (int k = 0; k < 3; k++) positions[3 * i + k] = x[i][k];
    atomic_numbers[i] = type_to_z[type[i]];
    local[i] = i < nlocal ? 1.0 : 0.0;
  }

  if (!domain_decomposition) {
    tagint max_tag = 0;
    for (int i = 0; i < nlocal; i++) max_tag = std::max(max_tag, tag[i]);
    owner.assign(max_tag + 1, -1);
    for (int i = 0; i < nlocal; i++) owner[tag[i]] = i;
  }

  idx_i.clear();
  idx_j.clear();
  offsets.clear();
  const double cutoff_sq = model->cutoff * model->cutoff;
  for (int ii = 0; ii < n_lists; ii++) {
    const int i = ilist[ii];
    int *jlist = firstneigh[i];
    for (int jj = 0; jj < numneigh[i]; jj++) {
      const int j = jlist[jj] & NEIGHMASK;
      const double dx = x[j][0] - x[i][0];
      const double dy = x[j][1] - x[i][1];
      const double dz = x[j][2] - x[i][2];
      if (dx * dx + dy * dy + dz * dz >= cutoff_sq) continue;
      int node = j;
      if (!domain_decomposition) {
        // j may be a ghost image: the edge goes to its owner, shifted
        node = tag[j] < static_cast<tagint>(owner.size()) ? owner[tag[j]] : -1;
        if (node < 0) error->one(FLERR, "Atom {} has no local owner", tag[j]);
        for (int k = 0; k < 3; k++) offsets.push_back(x[j][k] - x[node][k]);
      }
      idx_i.push_back(i);
      idx_j.push_back(node);
    }
  }

  try {
    model->compute(n_nodes, positions.data(), atomic_numbers.data(), local.data(),
                   static_cast<int64_t>(idx_i.size()), idx_i.data(), idx_j.data(),
                   domain_decomposition ? nullptr : offsets.data(), *result);
  } catch (const std::exception &e) {
    error->one(FLERR, "SchNetPack model failed: {}", e.what());
  }

  for (int i = 0; i < n_nodes; i++)
    for (int k = 0; k < 3; k++) f[i][k] += result->forces[3 * i + k];
  if (eflag_global) eng_vdwl += result->energy;
  if (eflag_atom)
    for (int i = 0; i < nlocal; i++) eatom[i] += result->energy_atom[i];
  if (vflag_global) {
    // xx, yy, zz, xy, xz, yz
    const double *w = result->virial;
    virial[0] += w[0];
    virial[1] += w[4];
    virial[2] += w[8];
    virial[3] += w[1];
    virial[4] += w[2];
    virial[5] += w[5];
  }
}
