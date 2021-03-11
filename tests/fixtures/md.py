import pytest
from ase import units

__all__ = [
    "md_config",
    "md_thermostats",
    "rpmd_thermostats",
    "md_integrator",
    "rpmd_integrator",
    "unit_conversion",
]


@pytest.fixture
def md_config():
    config = {
        "device": "cpu",
        "simulation_dir": "SIMULATION_DIR",
        "seed": 662524648,
        "overwrite": True,
        "calculator": {
            "type": "schnet",
            "model_file": "MODEL_PATH",
            "required_properties": ["energy", "forces"],
            "force_handle": "forces",
        },
        "system": {
            "molecule_file": "MOLECULE_PATH",
            "n_replicas": 1,
            "initializer": {
                "type": "maxwell-boltzmann",
                "temperature": 300,
                "remove_translation": True,
                "remove_rotation": False,
            },
        },
        "dynamics": {
            "n_steps": 2,
            "integrator": {"type": "verlet", "time_step": 0.5},
            "remove_com_motion": {"every_n_steps": 100, "remove_rotation": True},
        },
        "logging": {
            "file_logger": {
                "buffer_size": 50,
                "streams": ["molecules", "properties", "dynamic"],
            },
            "temperature_logger": 50,
            "write_checkpoints": 100,
        },
    }
    return config


@pytest.fixture(params=[None, "berendsen", "langevin", "nhc", "nhc-massive", "gle"])
def md_thermostats(request):
    thermostat = request.param
    return thermostat


@pytest.fixture(
    params=[None, "pile-l", "pile-g", "trpmd", "piglet", "pi-nhc-l", "pi-nhc-g"]
)
def rpmd_thermostats(request):
    thermostat = request.param
    return thermostat


@pytest.fixture(scope="module")
def md_integrator():
    return {"type": "verlet", "time_step": 0.5}


@pytest.fixture(scope="module")
def rpmd_integrator():
    return {"type": "ring_polymer", "time_step": 0.2, "temperature": 300.0}


@pytest.fixture
def unit_conversion():
    conversions = {
        "kcal / mol": units.kcal / units.Hartree / units.mol,
        "kcal/mol": units.kcal / units.Hartree / units.mol,
        "kcal / mol / Angstrom": units.kcal / units.Hartree / units.mol * units.Bohr,
        "kcal / mol / Angs": units.kcal / units.Hartree / units.mol * units.Bohr,
        "kcal / mol / A": units.kcal / units.Hartree / units.mol * units.Bohr,
        "kcal / mol / Bohr": units.kcal / units.Hartree / units.mol * units.Angstrom,
        "eV": units.eV / units.Ha,
        "Ha": 1.0,
        "Hartree": 1.0,
        0.57667: 0.57667,
    }
    return conversions
