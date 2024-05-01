import numpy as np
import logging

'''
calculation of real spherical harmonics
x,y,z are normalized unit vector (aka x/r, y/r, z/r, with r = sqrt(x^2 + y^2 + z^2)
'''

# TODO: checken in so3Net ob beide Sachen kombiniert werden können

# l = 0
_Y00 = lambda x, y, z: 1/2 * np.sqrt(1/np.pi)  # in shape: (...) / out shape: (...)

# l = 1
_Y1_1 = lambda x, y, z: np.sqrt(3/(4*np.pi)) * y  # in shape: (...) / out shape: (...) 
_Y10 = lambda x, y, z: np.sqrt(3/(4*np.pi)) * z  # in shape: (...) / out_shape: (...)
_Y11 = lambda x, y, z: np.sqrt(3/(4*np.pi)) * x  # in shape: (...) / out_shape: (...)

# l = 2
_Y2_2 = lambda x, y, z: 1/2 * np.sqrt(15/np.pi) * x * y
_Y2_1 = lambda x, y, z: 1/2 * np.sqrt(15/np.pi) * y * z
_Y20 = lambda x, y, z: 1/4 * np.sqrt(5/np.pi) * (3*z**2 - 1)
_Y21 = lambda x, y, z: 1/2 * np.sqrt(15/np.pi) * x * z
_Y22 = lambda x, y, z: 1/4 * np.sqrt(15/np.pi) * (x**2 - y**2)

# l = 3
_Y3_3 = lambda x, y, z: 1/4 * np.sqrt(35 / (2*np.pi)) * y * (3*x**2 - y**2)
_Y3_2 = lambda x, y, z: 1/2 * np.sqrt(105 / np.pi) * x * y * z
_Y3_1 = lambda x, y, z: 1/4 * np.sqrt(21 / (2*np.pi)) * y * (5*z**2 - 1)
_Y30 = lambda x, y, z: 1/4 * np.sqrt(7/np.pi) * (5*z**3 - 3*z)
_Y31 = lambda x, y, z: 1/4 * np.sqrt(21 / (2*np.pi)) * x * (5*z**2 - 1)
_Y32 = lambda x, y, z: 1/4 * np.sqrt(105 / np.pi) * (x**2 - y**2) * z
_Y33 = lambda x, y, z: 1/4 * np.sqrt(35 / (2*np.pi)) * x * (x**2 - 3*y**2)

# l = 4
_Y4_4 = lambda x, y, z: 3/4 * np.sqrt(35 / np.pi) * x * y * (x**2 - y**2)
_Y4_3 = lambda x, y, z: 3/4 * np.sqrt(35 / (2*np.pi)) * y * (3*x**2 - y**2) * z
_Y4_2 = lambda x, y, z: 3/4 * np.sqrt(5 / np.pi) * x * y * (7*z**2 - 1)
_Y4_1 = lambda x, y, z: 3/4 * np.sqrt(5 / (2*np.pi)) * y * (7*z**3 - 3*z)
_Y40 = lambda x, y, z: 3/16 * np.sqrt(1 / np.pi) * (35*z**4 - 30*z**2 + 3)
_Y41 = lambda x, y, z: 3/4 * np.sqrt(5 / (2*np.pi)) * x * (7*z**3 - 3*z)
_Y42 = lambda x, y, z: 3/8 * np.sqrt(5 / np.pi) * (x**2 - y**2) * (7*z**2 - 1)
_Y43 = lambda x, y, z: 3/4 * np.sqrt(35 / (2*np.pi)) * x * (x**2 - 3*y**2) * z
_Y44 = lambda x, y, z: 3/16 * np.sqrt(35 / np.pi) * (x**2 * (x**2 - 3*y**2) - y**2 * (3*x**2 - y**2))



def fn_Y0(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Spherical harmonics expansion of order l=0. Distance vector is assumed to be normalized to 1.

    Args:
        x (): X-coordinate, shape: (...,1)
        y (): Y-coordinate, shape: (...,1)
        z (): Z-coordinate, shape: (...,1)

    Returns: Expansion coefficients, shape: (...,2*l+1) = (...,1)

    """
    return np.ones_like(x)*_Y00(x, y, z)



def fn_Y1(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Spherical harmonics expansion of order l=1. Distance vector is assumed to be normalized to 1.

    Args:
        x (): X-coordinate, shape: (...,1)
        y (): Y-coordinate, shape: (...,1)
        z (): Z-coordinate, shape: (...,1)

    Returns: Expansion coefficients, shape: (...,2*l + 1) = (...,3)

    """
    return np.concatenate([_Y1_1(x, y, z), _Y10(x, y, z), _Y11(x, y, z)], axis=-1)



def fn_Y2(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Spherical harmonics expansion of order l=2. Distance vector is assumed to be normalized to 1.

    Args:
        x (): X-coordinate, shape: (...,1)
        y (): Y-coordinate, shape: (...,1)
        z (): Z-coordinate, shape: (...,1)

    Returns: Expansion coefficients, shape: (...,2*l + 1) = (...,5)

    """
    return np.concatenate([_Y2_2(x, y, z),
                            _Y2_1(x, y, z),
                            _Y20(x, y, z),
                            _Y21(x, y, z),
                            _Y22(x, y, z)], axis=-1)



def fn_Y3(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Spherical harmonics expansion of order l=3. Distance vector is assumed to be normalized to 1.

    Args:
        x (): X-coordinate, shape: (...,1)
        y (): Y-coordinate, shape: (...,1)
        z (): Z-coordinate, shape: (...,1)

    Returns: Expansion coefficients, shape: (...,2*l + 1) = (...,7)

    """
    return np.concatenate([_Y3_3(x, y, z),
                            _Y3_2(x, y, z),
                            _Y3_1(x, y, z),
                            _Y30(x, y, z),
                            _Y31(x, y, z),
                            _Y32(x, y, z),
                            _Y33(x, y, z)], axis=-1)



def fn_Y4(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Spherical harmonics expansion of order l=4. Distance vector is assumed to be normalized to 1.

    Args:
        x (): X-coordinate, shape: (...,1)
        y (): Y-coordinate, shape: (...,1)
        z (): Z-coordinate, shape: (...,1)

    Returns: Expansion coefficients, shape: (...,2*l + 1) = (...,9)

    """
    return np.concatenate([_Y4_4(x, y, z),
                            _Y4_3(x, y, z),
                            _Y4_2(x, y, z),
                            _Y4_1(x, y, z),
                            _Y40(x, y, z),
                            _Y41(x, y, z),
                            _Y42(x, y, z),
                            _Y43(x, y, z),
                            _Y44(x, y, z)], axis=-1)


def init_sph_fn(l: int):
    if l == 0:
        return lambda rij: fn_Y0(*np.split(rij, indices_or_sections=3, axis=-1))
    elif l == 1:
        return lambda rij: fn_Y1(*np.split(rij, indices_or_sections=3, axis=-1))
    elif l == 2:
        return lambda rij: fn_Y2(*np.split(rij, indices_or_sections=3, axis=-1))
    elif l == 3:
        return lambda rij: fn_Y3(*np.split(rij, indices_or_sections=3, axis=-1))
    elif l == 4:
        return lambda rij: fn_Y4(*np.split(rij, indices_or_sections=3, axis=-1))
    else:
        logging.error('Spherical harmonics are only defined up to order l = 4.')
        raise NotImplementedError