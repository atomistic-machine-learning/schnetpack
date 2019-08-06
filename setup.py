import os
import io

from setuptools import setup, find_packages


def read(fname):
    with io.open(os.path.join(os.path.dirname(__file__), fname), encoding="utf-8") as f:
        return f.read()


setup(
    name="schnetpack",
    version="0.3",
    author="Kristof T. Schuett, Michael Gastegger, Pan Kessel, Kim Nicoli",
    email="kristof.schuett@tu-berlin.de",
    url="https://github.com/atomistic-machine-learning/schnetpack",
    packages=find_packages("src"),
    scripts=[
        "src/scripts/spk_ase.py",
        "src/scripts/spk_load.py",
        "src/scripts/spk_md.py",
        "src/scripts/spk_parse.py",
        "src/scripts/spk_run.py",
    ],
    package_dir={"": "src"},
    python_requires=">=3.6",
    install_requires=[
        "torch>=0.4.1",
        "numpy",
        "ase>=3.18",
        "h5py",
        "tensorboardX",
        "tqdm",
        "pyyaml",
        "pytest-console-scripts",
    ],
    extras_require={"test": ["pytest", "sacred", "pytest-console-scripts"]},
    license="MIT",
    description="SchNetPack - Deep Neural Networks for Atomistic Systems",
    long_description=read("README.md"),
)
