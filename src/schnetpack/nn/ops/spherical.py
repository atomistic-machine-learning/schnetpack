import torch
import numpy as np
import logging
import pkg_resources
import torch
import itertools as it
import math
import schnetpack.nn as snn
'''
calculation of real spherical harmonics
x,y,z are normalized unit vector (aka x/r, y/r, z/r, with r = sqrt(x^2 + y^2 + z^2)
'''

# TODO: probabily contains stuff already present 
# in so3Net, check if both can be combined, but for now keep
# needs the provided cgmatrix.npz file
# first part is the normalization constant
# N_l^m = sqrt((2l+1)/(4pi) * (l-m)!/(l+m)!)
# l = 0
_Y00 = lambda x, y, z: 1/2 * math.sqrt(1/math.pi)  # in shape: (...) / out shape: (...)

# l = 1
_Y1_1 = lambda x, y, z: math.sqrt(3/(4*math.pi)) * y  # in shape: (...) / out shape: (...) 
_Y10 = lambda x, y, z: math.sqrt(3/(4*math.pi)) * z  # in shape: (...) / out_shape: (...)
_Y11 = lambda x, y, z: math.sqrt(3/(4*math.pi)) * x  # in shape: (...) / out_shape: (...)

# l = 2
_Y2_2 = lambda x, y, z: 1/2 * math.sqrt(15/math.pi) * x * y
_Y2_1 = lambda x, y, z: 1/2 * math.sqrt(15/math.pi) * y * z
_Y20 = lambda x, y, z: 1/4 * math.sqrt(5/math.pi) * (3*z**2 - 1)
_Y21 = lambda x, y, z: 1/2 * math.sqrt(15/math.pi) * x * z
_Y22 = lambda x, y, z: 1/4 * math.sqrt(15/math.pi) * (x**2 - y**2)

# l = 3
_Y3_3 = lambda x, y, z: 1/4 * math.sqrt(35 / (2*math.pi)) * y * (3*x**2 - y**2)
_Y3_2 = lambda x, y, z: 1/2 * math.sqrt(105 / math.pi) * x * y * z
_Y3_1 = lambda x, y, z: 1/4 * math.sqrt(21 / (2*math.pi)) * y * (5*z**2 - 1)
_Y30 = lambda x, y, z: 1/4 * math.sqrt(7/math.pi) * (5*z**3 - 3*z)
_Y31 = lambda x, y, z: 1/4 * math.sqrt(21 / (2*math.pi)) * x * (5*z**2 - 1)
_Y32 = lambda x, y, z: 1/4 * math.sqrt(105 / math.pi) * (x**2 - y**2) * z
_Y33 = lambda x, y, z: 1/4 * math.sqrt(35 / (2*math.pi)) * x * (x**2 - 3*y**2)

# l = 4
_Y4_4 = lambda x, y, z: 3/4 * math.sqrt(35 / math.pi) * x * y * (x**2 - y**2)
_Y4_3 = lambda x, y, z: 3/4 * math.sqrt(35 / (2*math.pi)) * y * (3*x**2 - y**2) * z
_Y4_2 = lambda x, y, z: 3/4 * math.sqrt(5 / math.pi) * x * y * (7*z**2 - 1)
_Y4_1 = lambda x, y, z: 3/4 * math.sqrt(5 / (2*math.pi)) * y * (7*z**3 - 3*z)
_Y40 = lambda x, y, z: 3/16 * math.sqrt(1 / math.pi) * (35*z**4 - 30*z**2 + 3)
_Y41 = lambda x, y, z: 3/4 * math.sqrt(5 / (2*math.pi)) * x * (7*z**3 - 3*z)
_Y42 = lambda x, y, z: 3/8 * math.sqrt(5 / math.pi) * (x**2 - y**2) * (7*z**2 - 1)
_Y43 = lambda x, y, z: 3/4 * math.sqrt(35 / (2*math.pi)) * x * (x**2 - 3*y**2) * z
_Y44 = lambda x, y, z: 3/16 * math.sqrt(35 / torch.pi) * (x**2 * (x**2 - 3*y**2) - y**2 * (3*x**2 - y**2))



def fn_Y0(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Spherical harmonics expansion of order l=0. Distance vector is assumed to be normalized to 1.

    Args:
        x (): X-coordinate, shape: (...,1)
        y (): Y-coordinate, shape: (...,1)
        z (): Z-coordinate, shape: (...,1)

    Returns: Expansion coefficients, shape: (...,2*l+1) = (...,1)

    """
    return torch.ones_like(x)*_Y00(x, y, z)



def fn_Y1(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Spherical harmonics expansion of order l=1. Distance vector is assumed to be normalized to 1.

    Args:
        x (): X-coordinate, shape: (...,1)
        y (): Y-coordinate, shape: (...,1)
        z (): Z-coordinate, shape: (...,1)

    Returns: Expansion coefficients, shape: (...,2*l + 1) = (...,3)

    """
    return torch.concatenate([_Y1_1(x, y, z), _Y10(x, y, z), _Y11(x, y, z)], dim=-1)



def fn_Y2(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Spherical harmonics expansion of order l=2. Distance vector is assumed to be normalized to 1.

    Args:
        x (): X-coordinate, shape: (...,1)
        y (): Y-coordinate, shape: (...,1)
        z (): Z-coordinate, shape: (...,1)

    Returns: Expansion coefficients, shape: (...,2*l + 1) = (...,5)

    """
    return torch.concatenate([_Y2_2(x, y, z),
                            _Y2_1(x, y, z),
                            _Y20(x, y, z),
                            _Y21(x, y, z),
                            _Y22(x, y, z)], dim=-1)



def fn_Y3(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Spherical harmonics expansion of order l=3. Distance vector is assumed to be normalized to 1.

    Args:
        x (): X-coordinate, shape: (...,1)
        y (): Y-coordinate, shape: (...,1)
        z (): Z-coordinate, shape: (...,1)

    Returns: Expansion coefficients, shape: (...,2*l + 1) = (...,7)

    """
    return torch.concatenate([_Y3_3(x, y, z),
                            _Y3_2(x, y, z),
                            _Y3_1(x, y, z),
                            _Y30(x, y, z),
                            _Y31(x, y, z),
                            _Y32(x, y, z),
                            _Y33(x, y, z)], dim=-1)



def fn_Y4(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Spherical harmonics expansion of order l=4. Distance vector is assumed to be normalized to 1.

    Args:
        x (): X-coordinate, shape: (...,1)
        y (): Y-coordinate, shape: (...,1)
        z (): Z-coordinate, shape: (...,1)

    Returns: Expansion coefficients, shape: (...,2*l + 1) = (...,9)

    """
    return torch.concatenate([_Y4_4(x, y, z),
                            _Y4_3(x, y, z),
                            _Y4_2(x, y, z),
                            _Y4_1(x, y, z),
                            _Y40(x, y, z),
                            _Y41(x, y, z),
                            _Y42(x, y, z),
                            _Y43(x, y, z),
                            _Y44(x, y, z)], dim=-1)

# wrapper defined because lambda functions are not serializable and
# therefore make saving the model not possible

def fn_Y0_wrapper(rij):
    return fn_Y0(*torch.split(rij, split_size_or_sections=1, dim=-1))

def fn_Y1_wrapper(rij):
    return fn_Y1(*torch.split(rij, split_size_or_sections=1, dim=-1))

def fn_Y2_wrapper(rij):
    return fn_Y2(*torch.split(rij, split_size_or_sections=1, dim=-1))

def fn_Y3_wrapper(rij):
    return fn_Y3(*torch.split(rij, split_size_or_sections=1, dim=-1))

def fn_Y4_wrapper(rij):
    return fn_Y4(*torch.split(rij, split_size_or_sections=1, dim=-1))

def init_sph_fn(l: int):
    if l == 0:
        return fn_Y0_wrapper
    elif l == 1:
        return fn_Y1_wrapper
    elif l == 2:
        return fn_Y2_wrapper
    elif l == 3:
        return fn_Y3_wrapper
    elif l == 4:
        return fn_Y4_wrapper

# def init_sph_fn(l: int):
#     if l == 0:
#         return lambda rij: fn_Y0(*torch.split(rij, split_size_or_sections=1, dim=-1))
#     elif l == 1:
#         return lambda rij: fn_Y1(*torch.split(rij, split_size_or_sections=1, dim=-1))
#     elif l == 2:
#         return lambda rij: fn_Y2(*torch.split(rij, split_size_or_sections=1, dim=-1))
#     elif l == 3:
#         return lambda rij: fn_Y3(*torch.split(rij, split_size_or_sections=1, dim=-1))
#     elif l == 4:
#         return lambda rij: fn_Y4(*torch.split(rij, split_size_or_sections=1, dim=-1))
#     else:
#         logging.error('Spherical harmonics are only defined up to order l = 4.')
#         raise NotImplementedError

indx_fn = lambda x: int((x+1)**2) if x >= 0 else 0

def load_cgmatrix(degrees):
    stream = pkg_resources.resource_stream(__name__, 'cgmatrix.npz')
    return torch.tensor(np.load(stream)['cg'], dtype=torch.float32,device=degrees.device)


def init_clebsch_gordan_matrix(degrees, l_out_max=None):
    """
    Initialize the Clebsch-Gordan matrix (coefficients for the Clebsch-Gordan expansion of spherical basis functions)
    for given ``degrees`` and a maximal output order ``l_out_max`` up to which the given all_degrees shall be
    expanded. Minimal output order is ``min(degrees)``.

    Args:
        degrees (List): Sequence of degrees l. The lowest order can be chosen freely. However, it should
            be noted that all following all_degrees must be exactly one order larger than the following one. E.g.
            [0,1,2,3] or [1,2,3] are valid but [0,1,3] or [0,2] are not.
        l_out_max (int): Maximal output order. Can be both, smaller or larger than maximal order in degrees.
            Defaults to the maximum value of the passed degrees.

    Returns: Clebsch-Gordan matrix,
        shape: (``(l_out_max+1)**2, (l_out_max+1)**2 - (l_in_min)**2, (l_out_max+1)**2 - (l_in_min)**2``)

    """
    if l_out_max is None:
        _l_out_max = max(degrees)
    else:
        _l_out_max = l_out_max

    l_in_max = max(degrees)
    l_in_min = min(degrees)

    offset_corr = indx_fn(l_in_min - 1)
    _cg = load_cgmatrix(degrees)
    # 0:1, 0:9, 0:9
    return _cg[offset_corr:indx_fn(_l_out_max), offset_corr:indx_fn(l_in_max), offset_corr:indx_fn(l_in_max)]


def make_l0_contraction_fn(degrees):
    # get CG coefficients
    cg = torch.diagonal(init_clebsch_gordan_matrix(degrees=torch.tensor(list({0, *degrees}),device=degrees.device), l_out_max=0), dim1=1, dim2=2)[0]
    # shape: (m_tot**2)
    # if 0 not in degrees:
    #     cg = cg[1:]  # remove degree zero if not in degrees

    cg_rep = []
    #reps = [(d,degrees.count(d)) for d in set(degrees)]

    for d, r in zip(*torch.unique(degrees, return_counts=True)):
        cg_rep += [torch.tile(cg[indx_fn(d - 1): indx_fn(d)], (r,))]

    cg_rep = torch.concatenate(cg_rep)  # shape: (m_tot), m_tot = \sum_l 2l+1 for l in degrees
    #cg_rep = torch.tensor(cg_rep,device=degrees.device)#device=idx_i.device)  # shape: (m_tot)

    segment_ids = torch.tensor(
        [y for y in it.chain(*[[n] * int(2 * degrees[n] + 1) for n in range(len(degrees))])], dtype=torch.long, device=degrees.device)  # shape: (m_tot
    num_segments = len(degrees)

    def contraction_fn(sphc):
        """
        Args:
            sphc (Tensor): Spherical harmonic coordinates, shape: (n, m_tot)
        Returns: Contraction on degree l=0 for each degree up to l_max, shape: (n, |l|)
        """
        # Element-wise multiplication and squaring
        weighted_sphc = sphc * sphc * cg_rep[None, :]  # shape: (n, m_tot)
        
        # Using torch_scatter to perform segment sum
        result = snn.scatter_add(weighted_sphc, segment_ids, dim=1, dim_size=num_segments)

        return result  # shape: (n, len(degrees))

    return contraction_fn


def make_degree_norm(degrees):

    segment_ids = torch.tensor(
        [y for y in it.chain(*[[n] * int(2 * degrees[n] + 1) for n in range(len(degrees))])], dtype=torch.long, device=degrees.device)  # shape: (m_tot
    num_segments = len(degrees)

    def fn(sphc):

        norm_result = snn.scatter_add(sphc**2, segment_ids, dim=1, dim_size=num_segments)
        per_degree_norm = torch.where(
                norm_result > 0,
                torch.sqrt(norm_result + 1e-8), 0)
        return per_degree_norm
    
    return fn




def wrapper_make_degree_norm(chi,idx_j,idx_i,degrees):
    chi_ij = chi[idx_j] - chi[idx_i]
    degree_norm_fn = make_degree_norm(degrees)
    return degree_norm_fn(chi_ij)

def order_contraction(chi,idx_j,idx_i,degrees):

    chi_ij = chi[idx_j] - chi[idx_i]
    contraction_fn = make_l0_contraction_fn(degrees)
    return contraction_fn(chi_ij)

def interaction_order_contraction(chi,degrees):

    contraction_fn = make_l0_contraction_fn(degrees)
    return contraction_fn(chi)