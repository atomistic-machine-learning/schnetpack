import os
import io

from setuptools import setup, find_packages


def read(fname):
    with io.open(os.path.join(os.path.dirname(__file__), fname), encoding="utf-8") as f:
        return f.read()


setup(
    name="schnetpack",
    version="2.0.4",
    author="Kristof T. Schuett, Michael Gastegger, Stefaan Hessmann, Niklas Gebauer, Jonas Lederer",
    url="https://github.com/atomistic-machine-learning/schnetpack",
    packages=find_packages("src"),
    scripts=[
        "src/scripts/spkconvert",
        "src/scripts/spktrain",
        "src/scripts/spkpredict",
        "src/scripts/spkmd",
        "src/scripts/spkdeploy",
    ],
    package_dir={"": "src"},
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "sympy",
        "ase>=3.21",
        "h5py",
        "pyyaml",
        "hydra-core>=1.1.0",
        "torch>=1.9",
        "pytorch_lightning>=2.0.0",
        "torchmetrics==1.0.1",
        "hydra-colorlog>=1.1.0",
        "rich",
        "fasteners",
        "dirsync",
        "torch-ema",
        "matscipy",
        "tensorboard",
    ],
    include_package_data=True,
    extras_require={"test": ["pytest", "pytest-datadir", "pytest-benchmark"]},
    license="MIT",
    description="SchNetPack - Deep Neural Networks for Atomistic Systems",
    long_description="""
        SchNetPack aims to provide accessible atomistic neural networks that can be
        trained and applied out-of-the-box, while still being extensible to custom 
        atomistic architectures""",
)
