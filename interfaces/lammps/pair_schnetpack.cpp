/* ----------------------------------------------------------------------
References:

   .. [#pair_nequip] https://github.com/mir-group/pair_nequip
   .. [#lammps] https://github.com/lammps/lammps

------------------------------------------------------------------------- */

#include <pair_schnetpack.h>
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "potential_file_reader.h"
#include "tokenizer.h"

#include <cmath>
#include <cstring>
#include <numeric>
#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/csrc/jit/runtime/graph_executor.h>


using namespace LAMMPS_NS;

PairSCHNETPACK::PairSCHNETPACK(LAMMPS *lmp) : Pair(lmp) {
  restartinfo = 0;
  manybody_flag = 1;

  if(torch::cuda::is_available()){
    device = torch::kCUDA;
  }
  else {
    device = torch::kCPU;
  }
  std::cout << "SCHNETPACK is using device " << device << "\n";

  if(const char* env_p = std::getenv("SCHNETPACK_DEBUG")){
    std::cout << "PairSCHNETPACK is in DEBUG mode, since SCHNETPACK_DEBUG is in env\n";
    debug_mode = 1;
  }
}

PairSCHNETPACK::~PairSCHNETPACK(){
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(type_mapper);
  }
}

void PairSCHNETPACK::init_style(){
  if (atom->tag_enable == 0)
    error->all(FLERR,"Pair style SchNetPack requires atom IDs");

  neighbor->add_request(this, NeighConst::REQ_FULL);

  // TODO: I think Newton should be off, enforce this.
  // The network should just directly compute the total forces
  // on the "real" atoms, with no need for reverse "communication".
  // May not matter, since f[j] will be 0 for the ghost atoms anyways.
  if (force->newton_pair == 1)
    error->all(FLERR,"Pair style SchNetPack requires newton pair off");
}

double PairSCHNETPACK::init_one(int i, int j)
{
  return cutoff;
}

void PairSCHNETPACK::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(type_mapper, n+1, "pair:type_mapper");

}

void PairSCHNETPACK::settings(int narg, char ** /*arg*/) {
  // "flare" should be the only word after "pair_style" in the input file.
  if (narg > 0)
    error->all(FLERR, "Illegal pair_style command");
}

void PairSCHNETPACK::coeff(int narg, char **arg) {

  if (!allocated)
    allocate();

  int ntypes = atom->ntypes;

  // Should be exactly 3 arguments following "pair_coeff" in the input file.
  if (narg != (3+ntypes))
    error->all(FLERR, "Incorrect args for pair coefficients");

  // Ensure I,J args are "* *".
  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "Incorrect args for pair coefficients");

  for (int i = 1; i <= ntypes; i++)
    for (int j = i; j <= ntypes; j++)
      setflag[i][j] = 0;

  // Initiate type mapper
  for (int i = 1; i<= ntypes; i++){
      type_mapper[i] = -1;
  }

  std::cout << "Loading model from " << arg[2] << "\n";

  
  std::unordered_map<std::string, std::string> metadata = {
    {"cutoff", ""},
  };
  model = torch::jit::load(std::string(arg[2]), device, metadata);
  model.eval();


  cutoff = std::stod(metadata["cutoff"]);

  // match lammps types to atomic numbers
  int counter = 1;
  for (int i = 3; i < narg; i++){
      type_mapper[counter] = std::stoi(arg[i]);
      counter++;
  }
  
  if(debug_mode){
    std::cout << "cutoff" << cutoff << "\n";
    for (int i = 0; i <= ntypes+1; i++){
        std::cout << type_mapper[i] << "\n";
    }
  }

  // set setflag i,j for type pairs where both are mapped to elements
  for (int i = 1; i <= ntypes; i++)
    for (int j = i; j <= ntypes; j++)
        if ((type_mapper[i] >= 0) && (type_mapper[j] >= 0))
            setflag[i][j] = 1;

}

// Force and energy computation
void PairSCHNETPACK::compute(int eflag, int vflag){
  ev_init(eflag, vflag);

  // Get info from lammps:

  // Atom positions, including ghost atoms
  double **x = atom->x;
  // Atom forces
  double **f = atom->f;
  // Atom IDs, unique, reproducible, the "real" indices
  // Probably 1-based
  tagint *tag = atom->tag;
  // Atom types, 1-based
  int *type = atom->type;
  // Number of local/real atoms
  int nlocal = atom->nlocal;
  // Whether Newton is on (i.e. reverse "communication" of forces on ghost atoms).
  int newton_pair = force->newton_pair;
  // Should probably be off.
  if (newton_pair==1)
    error->all(FLERR, "Pair style SCHNETPACK requires 'newton off'");

  // Number of local/real atoms
  int inum = list->inum;
  assert(inum==nlocal); // This should be true, if my understanding is correct
  // Number of ghost atoms
  int nghost = list->gnum;
  // Total number of atoms
  int ntotal = inum + nghost;
  // Mapping from neigh list ordering to x/f ordering
  int *ilist = list->ilist;
  // Number of neighbors per atom
  int *numneigh = list->numneigh;
  // Neighbor list per atom
  int **firstneigh = list->firstneigh;

  // Total number of bonds (sum of number of neighbors)
  int nedges = std::accumulate(numneigh, numneigh+ntotal, 0);

  torch::Tensor pos_tensor = torch::zeros({nlocal, 3});
  torch::Tensor tag2type_tensor = torch::zeros({nlocal}, torch::TensorOptions().dtype(torch::kInt64));
  torch::Tensor periodic_shift_tensor = torch::zeros({3});
  torch::Tensor cell_tensor = torch::zeros({3,3});

  auto pos = pos_tensor.accessor<float, 2>();
  long edges[2*nedges];
  float edge_cell_shifts[3*nedges];
  auto tag2type = tag2type_tensor.accessor<long, 1>();
  auto periodic_shift = periodic_shift_tensor.accessor<float, 1>();
  auto cell = cell_tensor.accessor<float,2>();

  // Inverse mapping from tag to "real" atom index
  std::vector<int> tag2i(inum);

  // Loop over real atoms to store tags, types and positions
  for(int ii = 0; ii < inum; ii++){
    int i = ilist[ii];
    int itag = tag[i];
    int itype = type[i];

    // Inverse mapping from tag to x/f atom index
    tag2i[itag-1] = i; // tag is probably 1-based
    tag2type[itag-1] = type_mapper[itype];
    pos[itag-1][0] = x[i][0];
    pos[itag-1][1] = x[i][1];
    pos[itag-1][2] = x[i][2];
  }

  // Get cell
  cell[0][0] = domain->boxhi[0] - domain->boxlo[0];

  cell[1][0] = domain->xy;
  cell[1][1] = domain->boxhi[1] - domain->boxlo[1];

  cell[2][0] = domain->xz;
  cell[2][1] = domain->yz;
  cell[2][2] = domain->boxhi[2] - domain->boxlo[2];


  auto cell_inv = cell_tensor.inverse().transpose(0,1);

  // Loop over atoms and neighbors,
  // store edges and _cell_shifts
  // ii follows the order of the neighbor lists,
  // i follows the order of x, f, etc.
  int edge_counter = 0;
  if (debug_mode) printf("SchNetPack edges: i j xi[:] xj[:] cell_shift[:] rij\n");
  for(int ii = 0; ii < nlocal; ii++){
    int i = ilist[ii];
    int itag = tag[i];
    int itype = type[i];

    int jnum = numneigh[i];
    int *jlist = firstneigh[i];
    for(int jj = 0; jj < jnum; jj++){
      int j = jlist[jj];
      j &= NEIGHMASK;
      int jtag = tag[j];
      int jtype = type[j];

      // TODO: check sign
      periodic_shift[0] = x[j][0] - pos[jtag-1][0];
      periodic_shift[1] = x[j][1] - pos[jtag-1][1];
      periodic_shift[2] = x[j][2] - pos[jtag-1][2];

      double dx = x[i][0] - x[j][0];
      double dy = x[i][1] - x[j][1];
      double dz = x[i][2] - x[j][2];

      double rsq = dx*dx + dy*dy + dz*dz;
      if (rsq < cutoff*cutoff){
          torch::Tensor cell_shift_tensor = cell_inv.matmul(periodic_shift_tensor);
          auto cell_shift = cell_shift_tensor.accessor<float, 1>();
          float * e_vec = &edge_cell_shifts[edge_counter*3];
          e_vec[0] = std::round(cell_shift[0]);
          e_vec[1] = std::round(cell_shift[1]);
          e_vec[2] = std::round(cell_shift[2]);
          //std::cout << "cell shift: " << cell_shift_tensor << "\n";

          // TODO: double check order
          edges[edge_counter*2] = itag - 1; // tag is probably 1-based
          edges[edge_counter*2+1] = jtag - 1; // tag is probably 1-based
          edge_counter++;

          if (debug_mode){
              printf("%d %d %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g\n", itag-1, jtag-1,
                pos[itag-1][0],pos[itag-1][1],pos[itag-1][2],pos[jtag-1][0],pos[jtag-1][1],pos[jtag-1][2],
                e_vec[0],e_vec[1],e_vec[2],sqrt(rsq));
          }

      }
    }
  }
  if (debug_mode) printf("end SchNetPack edges\n");

  // shorten the list before sending to nequip
  torch::Tensor edges_tensor = torch::zeros({2,edge_counter}, torch::TensorOptions().dtype(torch::kInt64));
  torch::Tensor edge_cell_shifts_tensor = torch::zeros({edge_counter,3});
  auto new_edges = edges_tensor.accessor<long, 2>();
  auto new_edge_cell_shifts = edge_cell_shifts_tensor.accessor<float, 2>();
  for (int i=0; i<edge_counter; i++){

      long *e=&edges[i*2];
      new_edges[0][i] = e[0];
      new_edges[1][i] = e[1];

      float *ev = &edge_cell_shifts[i*3];
      new_edge_cell_shifts[i][0] = ev[0];
      new_edge_cell_shifts[i][1] = ev[1];
      new_edge_cell_shifts[i][2] = ev[2];
  }
  
  // define SchNetPack specific inputs
  torch::Tensor idx_m = torch::zeros({nlocal}, torch::TensorOptions().dtype(torch::kInt64));
  
  // define SchNetPack n_atoms input
  torch::Tensor n_atoms_tensor = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt64));
  n_atoms_tensor[0] = nlocal;


  c10::Dict<std::string, torch::Tensor> input;
  input.insert("_positions", pos_tensor.to(device));
  input.insert("_idx_i", edges_tensor[0].to(device));
  input.insert("_idx_j", edges_tensor[1].to(device));
  input.insert("_idx_m", idx_m.to(device));
  input.insert("_offsets", edge_cell_shifts_tensor.to(device));
  input.insert("_cell", cell_tensor.to(device));
  input.insert("_n_atoms", n_atoms_tensor.to(device));
  input.insert("_atomic_numbers", tag2type_tensor.to(device));
  std::vector<torch::IValue> input_vector(1, input);

  if(debug_mode){
    std::cout << "SchNetPack model input:\n";
    std::cout << "_positions:\n" << pos_tensor << "\n";
    std::cout << "_idx_i:\n" << edges_tensor[0] << "\n";
    std::cout << "_idx_j:\n" << edges_tensor[1] << "\n";
    std::cout << "_idx_m:\n" << idx_m << "\n";
    std::cout << "_offsets:\n" << edge_cell_shifts_tensor << "\n";
    std::cout << "_cell:\n" << cell_tensor << "\n";
    std::cout << "_atomic_numbers:\n" << tag2type_tensor << "\n";
  }
  
  auto output = model.forward(input_vector).toGenericDict();
  
  torch::Tensor forces_tensor = output.at("forces").toTensor().cpu();
  auto forces = forces_tensor.accessor<float, 2>();

  torch::Tensor total_energy_tensor = output.at("energy").toTensor().cpu();

  // store the total energy where LAMMPS wants it
  eng_vdwl = total_energy_tensor.data_ptr<float>()[0];

  if(debug_mode){
    std::cout << "SchNetPack model output:\n";
    std::cout << "forces: " << forces_tensor << "\n";
    std::cout << "energy: " << total_energy_tensor << "\n";
  }
  
  // Write forces and per-atom energies (0-based tags here)
  for(int itag = 0; itag < inum; itag++){
    int i = tag2i[itag];
    f[i][0] = forces[itag][0];
    f[i][1] = forces[itag][1];
    f[i][2] = forces[itag][2];
  }
  
}
