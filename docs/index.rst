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
   :caption: Get Started
   :maxdepth: 1

   getstarted

.. toctree::
   :glob:
   :caption: User guide
   :maxdepth: 1

   userguide/overview
   userguide/configs
   userguide/md

.. toctree::
   :glob:
   :caption: Tutorials
   :maxdepth: 1

   tutorials/tutorial_01_preparing_data
   tutorials/tutorial_02_qm9
   tutorials/tutorial_03_force_models
   tutorials/tutorial_04_molecular_dynamics

.. toctree::
   :glob:
   :caption: How-To
   :maxdepth: 1

   howtos/howto_batchwise_relaxations

.. toctree::
   :glob:
   :caption: Reference
   :maxdepth: 1

   api/schnetpack
   api/atomistic
   api/data
   api/datasets
   api/task
   api/model
   api/representation
   api/nn
   api/train
   api/transform
