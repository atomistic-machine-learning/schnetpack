/* ----------------------------------------------------------------------
   Registers pair_style schnetpack with LAMMPS' plugin loader:

     plugin load /path/to/schnetpackplugin.so
------------------------------------------------------------------------- */

#include "lammpsplugin.h"

#include "version.h"

#include "pair_schnetpack.h"

using namespace LAMMPS_NS;

static Pair *schnetpackcreator(LAMMPS *lmp)
{
  return new PairSCHNETPACK(lmp);
}

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
  lammpsplugin_t plugin;
  lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc) regfunc;

  plugin.version = LAMMPS_VERSION;
  plugin.style = "pair";
  plugin.name = "schnetpack";
  plugin.info = "SchNetPack potentials compiled with AOTInductor (spkdeploy)";
  plugin.author = "SchNetPack developers";
  plugin.creator.v1 = (lammpsplugin_factory1 *) &schnetpackcreator;
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);
}
