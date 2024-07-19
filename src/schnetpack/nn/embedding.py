import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import schnetpack.properties as properties
from schnetpack.nn.activations import shifted_softplus
from schnetpack.nn.blocks import ResidualMLP
from typing import Callable, Union


__all__ = ["NuclearEmbedding", "ElectronicEmbedding"]


"""
The usage of the electron configuration is to provide a shorthand descriptor. This descriptor encode
information about the groundstate information of an atom, the nuclear charge and the number of electrons in the 
valence shell.
The table is read as follows:
The specific descriptor e.g for Oxygen would be: Z = 8, ground-state configuration [1s 2s 2p4] and valence shell vs = 2, vp = 4

Background:
Electrons are arranged around an atom's nucleus in energy levels (shells ranging from K,L,M ..., holding 2n^2 electrons)., 
[K : 2, L : 8, M : 18, N : 32, O : 50, P: 72 ]
and these shells contain subshells designated as 
s: (sharp, orbital angular momentum 0), 
p (principal, orbital angular momentum 1), 
d: (diffuse,orbital angular momentum 2), 
f (fundamental, orbital angular momentum 3).

The arrangement follows the Pauli Exclusion Principle and Hund's Rule ensuring the Aufbau Principle.
This provides the basis for the periodic table's structure and the periodicity of the elements' chemical behavior.

When invoking the complex nuclear embedding method a linear mapping 
from the electron configuration descriptor to a (num_features)-dimensional vector will be learned
Applying the complex nuclear embedding encourages to capture similiarities between different elements based on the electron configuration
This is justified by the fact that the chemistry of an element is mainly dominated by the valence shell.
E.g Bromine and Chlorine tend both to form -1 ions (uptake of one electron for fullfilling the ocette rule)
(Indicated by the same pattern in the electron configuration)


"""

# fmt: off
# up until Z = 100; vs = valence s, vp = valence p, vd = valence d, vf = valence f. 
# electron configuration follows the Aufbauprinzip. Exceptions are in the Lanthanides and Actinides (5f and 6d subshells are energetically very close).
electron_config = np.array([            
  #  Z 1s 2s 2p 3s 3p 4s  3d 4p 5s  4d 5p 6s  4f  5d 6p 7s 5f 6d   vs vp  vd  vf
  [  0, 0, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  0, 0,  0,  0], # n
  [  1, 1, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  1, 0,  0,  0], # H
  [  2, 2, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 0,  0,  0], # He
  [  3, 2, 1, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  1, 0,  0,  0], # Li
  [  4, 2, 2, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 0,  0,  0], # Be
  [  5, 2, 2, 1, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 1,  0,  0], # B
  [  6, 2, 2, 2, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 2,  0,  0], # C
  [  7, 2, 2, 3, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 3,  0,  0], # N
  [  8, 2, 2, 4, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 4,  0,  0], # O
  [  9, 2, 2, 5, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 5,  0,  0], # F
  [ 10, 2, 2, 6, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 6,  0,  0], # Ne
  [ 11, 2, 2, 6, 1, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  1, 0,  0,  0], # Na
  [ 12, 2, 2, 6, 2, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 0,  0,  0], # Mg
  [ 13, 2, 2, 6, 2, 1, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 1,  0,  0], # Al
  [ 14, 2, 2, 6, 2, 2, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 2,  0,  0], # Si
  [ 15, 2, 2, 6, 2, 3, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 3,  0,  0], # P
  [ 16, 2, 2, 6, 2, 4, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 4,  0,  0], # S
  [ 17, 2, 2, 6, 2, 5, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 5,  0,  0], # Cl
  [ 18, 2, 2, 6, 2, 6, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 6,  0,  0], # Ar
  [ 19, 2, 2, 6, 2, 6, 1,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  1, 0,  0,  0], # K
  [ 20, 2, 2, 6, 2, 6, 2,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 0,  0,  0], # Ca
  [ 21, 2, 2, 6, 2, 6, 2,  1, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 0,  1,  0], # Sc
  [ 22, 2, 2, 6, 2, 6, 2,  2, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 0,  2,  0], # Ti
  [ 23, 2, 2, 6, 2, 6, 2,  3, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 0,  3,  0], # V
  [ 24, 2, 2, 6, 2, 6, 1,  5, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  1, 0,  5,  0], # Cr
  [ 25, 2, 2, 6, 2, 6, 2,  5, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 0,  5,  0], # Mn
  [ 26, 2, 2, 6, 2, 6, 2,  6, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 0,  6,  0], # Fe
  [ 27, 2, 2, 6, 2, 6, 2,  7, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 0,  7,  0], # Co
  [ 28, 2, 2, 6, 2, 6, 2,  8, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 0,  8,  0], # Ni
  [ 29, 2, 2, 6, 2, 6, 1, 10, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  1, 0, 10,  0], # Cu
  [ 30, 2, 2, 6, 2, 6, 2, 10, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 0, 10,  0], # Zn
  [ 31, 2, 2, 6, 2, 6, 2, 10, 1, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 1, 10,  0], # Ga
  [ 32, 2, 2, 6, 2, 6, 2, 10, 2, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 2, 10,  0], # Ge
  [ 33, 2, 2, 6, 2, 6, 2, 10, 3, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 3, 10,  0], # As
  [ 34, 2, 2, 6, 2, 6, 2, 10, 4, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 4, 10,  0], # Se
  [ 35, 2, 2, 6, 2, 6, 2, 10, 5, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 5, 10,  0], # Br
  [ 36, 2, 2, 6, 2, 6, 2, 10, 6, 0,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 6, 10,  0], # Kr
  [ 37, 2, 2, 6, 2, 6, 2, 10, 6, 1,  0, 0, 0,  0,  0, 0, 0, 0, 0,  1, 0,  0,  0], # Rb
  [ 38, 2, 2, 6, 2, 6, 2, 10, 6, 2,  0, 0, 0,  0,  0, 0, 0, 0, 0,  2, 0,  0,  0], # Sr
  [ 39, 2, 2, 6, 2, 6, 2, 10, 6, 2,  1, 0, 0,  0,  0, 0, 0, 0, 0,  2, 0,  1,  0], # Y
  [ 40, 2, 2, 6, 2, 6, 2, 10, 6, 2,  2, 0, 0,  0,  0, 0, 0, 0, 0,  2, 0,  2,  0], # Zr
  [ 41, 2, 2, 6, 2, 6, 2, 10, 6, 1,  4, 0, 0,  0,  0, 0, 0, 0, 0,  1, 0,  4,  0], # Nb
  [ 42, 2, 2, 6, 2, 6, 2, 10, 6, 1,  5, 0, 0,  0,  0, 0, 0, 0, 0,  1, 0,  5,  0], # Mo
  [ 43, 2, 2, 6, 2, 6, 2, 10, 6, 2,  5, 0, 0,  0,  0, 0, 0, 0, 0,  2, 0,  5,  0], # Tc
  [ 44, 2, 2, 6, 2, 6, 2, 10, 6, 1,  7, 0, 0,  0,  0, 0, 0, 0, 0,  1, 0,  7,  0], # Ru
  [ 45, 2, 2, 6, 2, 6, 2, 10, 6, 1,  8, 0, 0,  0,  0, 0, 0, 0, 0,  1, 0,  8,  0], # Rh
  [ 46, 2, 2, 6, 2, 6, 2, 10, 6, 0, 10, 0, 0,  0,  0, 0, 0, 0, 0,  0, 0, 10,  0], # Pd
  [ 47, 2, 2, 6, 2, 6, 2, 10, 6, 1, 10, 0, 0,  0,  0, 0, 0, 0, 0,  1, 0, 10,  0], # Ag
  [ 48, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 0, 0,  0,  0, 0, 0, 0, 0,  2, 0, 10,  0], # Cd
  [ 49, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 1, 0,  0,  0, 0, 0, 0, 0,  2, 1, 10,  0], # In
  [ 50, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 2, 0,  0,  0, 0, 0, 0, 0,  2, 2, 10,  0], # Sn
  [ 51, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 3, 0,  0,  0, 0, 0, 0, 0,  2, 3, 10,  0], # Sb
  [ 52, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 4, 0,  0,  0, 0, 0, 0, 0,  2, 4, 10,  0], # Te
  [ 53, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 5, 0,  0,  0, 0, 0, 0, 0,  2, 5, 10,  0], # I
  [ 54, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0,  0,  0, 0, 0, 0, 0,  2, 6, 10,  0], # Xe
  [ 55, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 1,  0,  0, 0, 0, 0, 0,  1, 0,  0,  0], # Cs
  [ 56, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2,  0,  0, 0, 0, 0, 0,  2, 0,  0,  0], # Ba
  [ 57, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2,  0,  1, 0, 0, 0, 0,  2, 0,  1,  0], # La
  [ 58, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2,  1,  1, 0, 0, 0, 0,  2, 0,  1,  1], # Ce
  [ 59, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2,  3,  0, 0, 0, 0, 0,  2, 0,  0,  3], # Pr
  [ 60, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2,  4,  0, 0, 0, 0, 0,  2, 0,  0,  4], # Nd
  [ 61, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2,  5,  0, 0, 0, 0, 0,  2, 0,  0,  5], # Pm
  [ 62, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2,  6,  0, 0, 0, 0, 0,  2, 0,  0,  6], # Sm
  [ 63, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2,  7,  0, 0, 0, 0, 0,  2, 0,  0,  7], # Eu
  [ 64, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2,  7,  1, 0, 0, 0, 0,  2, 0,  1,  7], # Gd
  [ 65, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2,  9,  0, 0, 0, 0, 0,  2, 0,  0,  9], # Tb
  [ 66, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 10,  0, 0, 0, 0, 0,  2, 0,  0, 10], # Dy
  [ 67, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 11,  0, 0, 0, 0, 0,  2, 0,  0, 11], # Ho
  [ 68, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 12,  0, 0, 0, 0, 0,  2, 0,  0, 12], # Er
  [ 69, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 13,  0, 0, 0, 0, 0,  2, 0,  0, 13], # Tm
  [ 70, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14,  0, 0, 0, 0, 0,  2, 0,  0, 14], # Yb
  [ 71, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14,  1, 0, 0, 0, 0,  2, 0,  1, 14], # Lu
  [ 72, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14,  2, 0, 0, 0, 0,  2, 0,  2, 14], # Hf
  [ 73, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14,  3, 0, 0, 0, 0,  2, 0,  3, 14], # Ta
  [ 74, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14,  4, 0, 0, 0, 0,  2, 0,  4, 14], # W
  [ 75, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14,  5, 0, 0, 0, 0,  2, 0,  5, 14], # Re
  [ 76, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14,  6, 0, 0, 0, 0,  2, 0,  6, 14], # Os
  [ 77, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14,  7, 0, 0, 0, 0,  2, 0,  7, 14], # Ir
  [ 78, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 1, 14,  9, 0, 0, 0, 0,  1, 0,  9, 14], # Pt
  [ 79, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 1, 14, 10, 0, 0, 0, 0,  1, 0, 10, 14], # Au
  [ 80, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 0, 0, 0, 0,  2, 0, 10, 14], # Hg
  [ 81, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 1, 0, 0, 0,  2, 1, 10, 14], # Tl
  [ 82, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 2, 0, 0, 0,  2, 2, 10, 14], # Pb
  [ 83, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 3, 0, 0, 0,  2, 3, 10, 14], # Bi
  [ 84, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 4, 0, 0, 0,  2, 4, 10, 14], # Po
  [ 85, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 5, 0, 0, 0,  2, 5, 10, 14], # At
  [ 86, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 0, 0, 0,  2, 6, 10, 14], # Rn
  [ 87, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 1, 0, 0,  1, 0,  0,  0], # Fr
  [ 88, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 0, 0,  2, 0,  0,  0], # Ra
  [ 89, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 0, 1,  2, 0,  1,  0], # Ac
  [ 90, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 0, 2,  2, 0,  2,  0], # Th
  [ 91, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 2, 1,  2, 0,  1,  2], # Pa
  [ 92, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 3, 1,  2, 0,  3,  1], # U
  [ 93, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 4, 1,  2, 0,  1,  4], # Np
  [ 94, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 6, 0,  2, 0,  0,  6], # Pu
  [ 95, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 7, 0,  2, 0,  0,  7], # Am
  [ 96, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 7, 1,  2, 0,  1,  7], # Cm  
  [ 97, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 9, 0,  2, 0,  0,  9], # Bk
  [ 98, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 10,0,  2, 0,  0, 10], # Cf
  [ 99, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 11,0,  2, 0,  0, 11], # Es
  [100, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 12,0,  2, 0,  0, 12]  # Fm
            
], dtype=np.float32)            
# fmt: on
# normalize entries (between 0.0 and 1.0)
# normalization just for numerical reasons
electron_config = electron_config / np.max(electron_config, axis=0)


class NuclearEmbedding(nn.Module):
    """
    Embedding which maps scalar nuclear charges Z to vectors in a
    (num_features)-dimensional feature space. The embedding consists of a freely
    learnable parameter matrix [Zmax, num_features] and a learned linear mapping
    from the electron configuration to a (num_features)-dimensional vector. The
    latter part encourages alchemically meaningful representations without
    restricting the expressivity of learned embeddings.
    Using complexe nuclear embedding can have negative impact on the model
    performance, when spin charge embedding is activated
    Negative performance in regard of the duration until the model converges.
    The model will converge to a lower value, but the duration is longer.
    """

    def __init__(self, max_z: int, num_features: int, zero_init: bool = True):
        """
        Args:
        num_features: Dimensions of feature space.
        Zmax: Maximum nuclear charge of atoms. The default is 100, so all
            elements up to Fermium (Fe) (Z=100) are supported.
            Can be kept at the default value (has minimal memory impact).
        zero_init: If True, initialize the embedding with zeros. Otherwise, use
            uniform initialization.
        """
        super(NuclearEmbedding, self).__init__()
        self.num_features = num_features
        self.register_buffer("electron_config", torch.tensor(electron_config))
        self.register_parameter(
            "element_embedding", nn.Parameter(torch.Tensor(max_z, self.num_features))
        )
        self.register_buffer(
            "embedding", torch.Tensor(max_z, self.num_features), persistent=False
        )
        self.config_linear = nn.Linear(
            self.electron_config.size(1), self.num_features, bias=False
        )
        self.reset_parameters(zero_init)

    def reset_parameters(self, zero_init: bool = True) -> None:
        """Initialize parameters."""
        if zero_init:
            nn.init.zeros_(self.element_embedding)
            nn.init.zeros_(self.config_linear.weight)
        else:
            nn.init.uniform_(self.element_embedding, -math.sqrt(3), math.sqrt(3))
            nn.init.orthogonal_(self.config_linear.weight)

    def train(self, mode: bool = True) -> None:
        """Switch between training and evaluation mode."""
        super(NuclearEmbedding, self).train(mode=mode)
        if not self.training:
            with torch.no_grad():
                self.embedding = self.element_embedding + self.config_linear(
                    self.electron_config
                )

    def forward(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        """
        Assign corresponding embeddings to nuclear charges.

        Args:
            atomic_numbers: nuclear charges

        Returns:
            Embeddings of all atoms.

        """
        if self.training:  # during training, the embedding needs to be recomputed
            self.embedding = self.element_embedding + self.config_linear(
                self.electron_config
            )
        if self.embedding.device.type == "cpu":  # indexing is faster on CPUs
            return self.embedding[atomic_numbers]
        else:  # gathering is faster on GPUs
            return torch.gather(
                self.embedding,
                0,
                atomic_numbers.view(-1, 1).expand(-1, self.num_features),
            )


class ElectronicEmbedding(nn.Module):
    """
    Single Head self attention like block for updating atomic features through nonlocal interactions with the
    electrons.
    The embeddings are used to map the total molecular charge or molecular spin to a feature vector.
    Since those properties are not localized on a specific atom they have to be delocalized over the whole molecule.
    The delocalization is achieved by using a self attention like mechanism.
    """

    def __init__(
        self,
        property_key: str,
        num_features: int,
        is_charged: bool,
        num_residual: int = 1,
        activation: Union[Callable, nn.Module] = shifted_softplus,
        epsilon: float = 1e-8,
    ):
        """
        Args:
            property_key: key of electronic property in the spk 'inputs' dictionary
            num_features: Dimensions of feature space aka the number of features to describe atomic environments.
                This determines the size of each embedding vector
            num_residual: Number of residual blocks applied to atomic features
            activation: activation function.
            is_charged: True corresponds to building embedding for molecular charge and
                separate weights are used for positive and negative charges.
                False corresponds to building embedding for spin values,
                no seperate weights are used
            epsilon: numerical stability parameter
        """
        super(ElectronicEmbedding, self).__init__()
        self.property_key = property_key
        self.is_charged = is_charged
        self.linear_q = nn.Linear(num_features, num_features)
        if is_charged:  # charges are duplicated to use separate weights for +/-
            self.linear_k = nn.Linear(2, num_features, bias=False)
            self.linear_v = nn.Linear(2, num_features, bias=False)
        else:
            self.linear_k = nn.Linear(1, num_features, bias=False)
            self.linear_v = nn.Linear(1, num_features, bias=False)
        self.resblock = ResidualMLP(
            num_features,
            num_residual,
            activation=activation,
            zero_init=True,
            bias=False,
        )
        self.epsilon = epsilon
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters."""
        nn.init.orthogonal_(self.linear_k.weight)
        nn.init.orthogonal_(self.linear_v.weight)
        nn.init.orthogonal_(self.linear_q.weight)
        nn.init.zeros_(self.linear_q.bias)

    def forward(
        self,
        input_embedding,
        inputs,
    ) -> torch.Tensor:
        """
        Evaluate interaction block.

        Args:
            input_embedding: embedding of nuclear charges (and other electronic embeddings)
            inputs: spk style input dictionary

        """

        num_batch = len(inputs[properties.idx])
        idx_m = inputs[properties.idx_m]
        electronic_feature = inputs[self.property_key]

        # queries (Batchsize x N_atoms, n_atom_basis)
        q = self.linear_q(input_embedding)

        # to account for negative and positive charge
        if self.is_charged:
            e = F.relu(torch.stack([electronic_feature, -electronic_feature], dim=-1))
        # +/- spin is the same => abs
        else:
            e = torch.abs(electronic_feature).unsqueeze(-1)
        enorm = torch.maximum(e, torch.ones_like(e))

        # keys (Batchsize x N_atoms, n_atom_basis), the idx_m ensures that the key is the same for all atoms belonging to the same graph
        k = self.linear_k(e / enorm)[idx_m]

        # values (Batchsize x N_atoms, n_atom_basis) the idx_m ensures that the value is the same for all atoms belonging to the same graph
        v = self.linear_v(e)[idx_m]

        # unnormalized, scaled attention weights, obtained by dot product of queries and keys (are logits)
        # scaling by square root of attention dimension
        weights = torch.sum(k * q, dim=-1) / k.shape[-1] ** 0.5

        # probability distribution of scaled unnormalized attention weights, by applying softmax function
        a = nn.functional.softmax(
            weights, dim=0
        )  # nn.functional.softplus(weights) seems to function to but softmax might be more stable
        # normalization factor for every molecular graph, by adding up attention weights of every atom in the graph
        anorm = a.new_zeros(num_batch).index_add_(0, idx_m, a)
        # make tensor filled with anorm value at the position of the corresponding molecular graph,
        # indexing faster on CPU, gather faster on GPU
        if a.device.type == "cpu":
            anorm = anorm[idx_m]
        else:
            anorm = torch.gather(anorm, 0, idx_m)
        # return probability distribution of scaled normalized attention weights, eps is added for numerical stability (sum / batchsize equals 1)
        return self.resblock((a / (anorm + self.epsilon)).unsqueeze(-1) * v)
