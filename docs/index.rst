.. SchNetPack documentation master file, created by
   sphinx-quickstart on Mon Jul 30 18:07:50 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SchNetPack documentation
========================

SchNetPack is a toolbox for the development and application of deep neural networks to the prediction of
potential energy surfaces and other quantum-chemical properties of molecules and materials. It contains
basic building blocks of atomistic neural networks, manages their training and provides simple access
to common benchmark datasets. This allows for an easy implementation and evaluation of new models.

Contents
========

.. toctree::
   :glob:
   :caption: Getting Started
   :maxdepth: 1

   getstarted/firststeps
   getstarted/train_cli
   getstarted/md

.. toctree::
   :glob:
   :caption: SchNetPack from Python
   :maxdepth: 1

   tutorials/tutorial_01_preparing_data
   tutorials/tutorial_02_qm9
   tutorials/tutorial_03_force_models
   tutorials/tutorial_04_molecular_dynamics

.. toctree::
   :glob:
   :caption: Reference
   :maxdepth: 1

   api/schnetpack
   api/atomistic
   api/data
   api/datasets
   api/representation
   api/nn
   api/train
   api/transform
