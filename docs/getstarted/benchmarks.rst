.. _benchmark:


=============================
Benchmarks and Trained Models
=============================

QM9
---
The QM9 benchmarks use a SchNet model with 128 features, 6 interaction blocks, 50
gaussians and a cosine cutoff-function with a cutoff at 10. The model is trained with
a batch size of 100 and an initial learning rate of 1e-4. The learning rate is
scheduled using spk.train.ReduceOnPlateauHook with a learning rate patience of 25, a
learning rate decay of 0.8 and a minimum learning rate of 1e-6. 100000 data-points
have been used for training and 10000 data-points are used for validation. The
remaining dataset is used for evaluating the test-set.

=========================  =====  ========
Property                     MAE  Unit
=========================  =====  ========
heat_capacity              0.034  Kcal/mol
zpve                       1.616  meV
gap                        0.074  eV
energy_U0                  0.012  eV
enthalpy_H                 0.012  eV
homo                       0.047  eV
electronic_spatial_extent  0.158  Bohr**2
energy_U                   0.012  eV
free_energy                0.013  eV
isotropic_polarizability   0.124  Bohr**3
lumo                       0.039  eV
dipole_moment              0.021  Debye
=========================  =====  ========


MD17
----
The MD17 benchmarks use a SchNet model with 64 features, 6 interaction blocks, 25
gaussians and a cosine cutoff-function with a cutoff at 5. The model is trained with
a batch size of 100 and an initial learning rate of 1e-3. The learning rate is
scheduled using spk.train.ReduceOnPlateauHook with a learning rate patience of 150, a
learning rate decay of 0.8 and a minimum learning rate of 1e-6. The loss tradeoff is
set to multiply the energy loss with 0.01 and the forces loss value with 0.99. 950
data-points have been used for training and 50 data-points are used for validation.
The remaining dataset is used for evaluating the test-set.

==========  =====  ======
Property      MAE  Unit
==========  =====  ======
energy      0.069  eV
==========  =====  ======


Trained Models
--------------
The trained benchmark-models can be downloaded
`here <http:www.quantum-machine.org/datasets/trained_schnet_models.zip>`_.