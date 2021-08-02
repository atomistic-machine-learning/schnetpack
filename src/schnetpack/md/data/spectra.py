"""
Collection of utilities for computing autocorrelation functions and molecular spectra
from HDF5 files generated during molecular dynamics. For a good overview on how to compute
spectra from molecular dynamics simulations and details on the techniques used, we recommend [#spectra1]_ .

References
----------
.. [#spectra1] Martin, Brehm, Fligg, Vöhringer, Kirchner:
               Computing vibrational spectra from ab initio molecular dynamics.
               Phys. Chem. Chem. Phys., 15 (18), 6608--6622. 2013.
"""
import numpy as np
from ase import units as ase_units
from schnetpack.md.data import HDF5Loader
import logging

from schnetpack import properties
from schnetpack import units as spk_units

__all__ = ["VibrationalSpectrum", "PowerSpectrum", "IRSpectrum", "RamanSpectrum"]


def cosine_sq_window(n_points: int):
    """
    Squared cosine window function for spectra.

    Args:
        n_points (int): Number of points in spectrum

    Returns:
        numpy.array: Squared cosine window function.
    """
    points = np.arange(n_points)
    window = np.cos(np.pi * points / (n_points - 1) / 2) ** 2
    return window


def fft_autocorrelation(data: np.array, n_lags: int):
    """
    Routine for fast computation of autocorrelation using FFT and Wiener--Kinchie theorem.

    Args:
        data (numpy.array): Array containing data for which autocorrelation should be computed.
        n_lags (int): Number of time lags used for extracting the autocorrelation.

    Returns:
        numpy.array: Autocorrelation function of the input array
    """
    # Normalize data to get autocorrelation[0] = 1
    data = (data - np.mean(data)) / np.std(data)
    n_points = data.shape[0]
    fft_forward = np.fft.fft(data, n=2 * n_points)
    fft_autocorr = fft_forward * np.conjugate(fft_forward)
    fft_backward = np.fft.ifft(fft_autocorr)[:n_points] / n_points
    autocorrelation = np.real(fft_backward[: n_lags + 1])
    return autocorrelation


class VibrationalSpectrum:
    """
    Base class for computing vibrational spectra from HDF5 datasets using autocorrelation functions and
    fast fourier transforms.

    Args:
        data (schnetpack.md.utils.HDF5Loader): Loaded dataset.
        resolution (int): Resolution used when computing the spectrum. Indicates how many time lags are considered
                          in the autocorrelation function is used.
        window (function, optional): Window function used for computing the spectrum.
    """

    def __init__(
        self,
        data: HDF5Loader,
        resolution: int = 4096,
        window: callable = cosine_sq_window,
    ):
        self.data = data
        self.timestep = data.time_step / spk_units.fs  # Convert to fs
        self.resolution = resolution
        self.window = window

        spectral_range = 0.5 / self.timestep / (ase_units._c / 1e13)
        spectral_resolution = spectral_range / (4 * resolution)
        logging.info(
            "Spectral resolutions: {:12.3f} [cm^-1]".format(spectral_resolution)
        )
        logging.info("Spectral range:       {:12.3f} [cm^-1]".format(spectral_range))

        self.res = spectral_resolution
        self.frequencies = []
        self.intensities = []

    def compute_spectrum(self, molecule_idx: int = 0):
        """
        Main routine for computing spectra. First the relavant data is read,
        then autocorrelations are computed and processed. Based on the
        processed autocorrelations, spectra are computed and, if requested,
        subjected to additional postprocessing.

        Args:
            molecule_idx (int): Index of the molecule for which the spectrum should be computed.
                                Uses the same conventions as schnetpack.md.System.
        """
        # Get appropriate data
        relevant_data = self._get_data(molecule_idx)

        # Compute autocorrelation function
        autocorrelation = self._compute_autocorrelations(relevant_data)
        # Process the autocorrelation function (e.g. weighting, Raman, ...)
        autocorrelation = self._process_autocorrelation(autocorrelation)

        self.frequencies = []
        self.intensities = []
        # Compute spectrum
        for autocorr in autocorrelation:
            frequencies, intensities = self._compute_spectrum(autocorr)
            self.frequencies.append(frequencies)
            self.intensities.append(intensities)

        self._process_spectrum()

    def _compute_spectrum(self, autocorrelation: np.array):
        """
        Compute the spectrum from the autocorrelation function.

        Args:
            autocorrelation (numpy.array): Autorcorrelation function.

        Returns:
            (numpy.array,numpy.array):
                frequencies:
                    Vibrational frequencies in inverse centimeters.
                intensities:
                    Intensities of the vibrational bands.
        """
        data = autocorrelation[: self.resolution]

        # Various tricks for nicer spectra
        # 1) Apply window function
        n_unpadded = data.shape[0]
        if self.window is not None:
            data *= self.window(n_unpadded)
        # 2) Zero padding
        data_padded = np.zeros(4 * n_unpadded)
        data_padded[:n_unpadded] = data
        # 3) Mirror around origin
        data_mirrored = np.hstack((np.flipud(data_padded), data_padded))
        # Compute the spectrum
        n_fourier = 8 * n_unpadded
        intensities = np.abs(
            self.timestep * np.fft.fft(data_mirrored, n=n_fourier)[: n_fourier // 2]
        )
        frequencies = np.arange(n_fourier // 2) / (n_fourier * self.timestep)
        # Conversion to inverse cm
        frequencies /= ase_units._c / 1e13
        return frequencies, intensities

    @staticmethod
    def _compute_autocorrelations(data: np.array):
        """
        Compute the autocorrelation function of the data. A separate autocorrelation is computred
        for every array dimension except the first axis.

        Args:
            data (numpy.array): Function array.

        Returns:
            numpy.array: Autocorrelation of the inputs.
        """
        n_data = data.shape[0]
        data_dim = data.shape[1:]
        n_lags = n_data - 2

        # Flatten data for easier iteration
        reshaped_data = data.reshape((n_data, -1))
        n_fields = reshaped_data.shape[1]

        # Compute all individual autocorrelations
        autocorrelations = np.zeros((n_fields, n_lags + 1))
        for field in range(n_fields):
            autocorrelations[field, ...] = fft_autocorrelation(
                reshaped_data[..., field], n_lags
            )

        # Reconstruct shape of original property
        autocorrelations = autocorrelations.reshape((*data_dim, -1))
        return autocorrelations

    def _get_data(self, molecule_idx: int):
        """
        Placeholder for extracting teh required data from the HDF5 dataset.

        Args:
            molecule_idx (int): Index of the molecule for which the spectrum should be computed.
                                Uses the same conventions as schnetpack.md.System.
        """
        raise NotImplementedError

    def _process_autocorrelation(self, autocorrelation: np.array):
        """
        Placeholder for postprocessing the autocorrelation functions (e.g. weighting).

        Args:
            autocorrelation (numpy.array): Autorcorrelation function.
        """
        raise NotImplementedError

    def _process_spectrum(self):
        """
        Placeholder function if postprocessing should be applied to the spectrum (e.g. quantum coorections).
        """
        pass

    def get_spectrum(self):
        """
        Returns all computed spectra in the form of a list of tuples of frequencies and intensities.

        Returns:
            list: List of tuples of frequencies and intensities of all computed spectra.
        """
        spectrum = list(zip(self.frequencies, self.intensities))
        if len(spectrum) == 1:
            return spectrum[0]
        else:
            return spectrum


class PowerSpectrum(VibrationalSpectrum):
    """
    Compute power spectra from a molecular dynamics HDF5 dataset.

    Args:
        data (schnetpack.md.utils.HDF5Loader): Loaded dataset.
        resolution (int, optional): Resolution used when computing the spectrum. Indicates how many time lags
                                    are considered in the autocorrelation function is used.
    """

    def __init__(self, data: HDF5Loader, resolution: int = 4096):
        super(PowerSpectrum, self).__init__(data, resolution=resolution)

    def _get_data(self, molecule_idx: int):
        """
        Extract the molecular velocities for computing the spectrum.

        Args:
            molecule_idx (int): Index of the molecule for which the spectrum should be computed.
                                Uses the same conventions as schnetpack.md.System.

        Returns:
            numpy.array: Array holding molecular velocities.
        """
        relevant_data = self.data.get_velocities(molecule_idx)
        return relevant_data

    def _process_autocorrelation(self, autocorrelation: np.array):
        """
        Sum over number of atoms and the three Cartesian components.

        Args:
            autocorrelation (numpy.array): Autorcorrelation function.

        Returns:
            numpy.array: Updated autocorrelation.
        """
        vdos_autocorrelation = np.sum(autocorrelation, axis=1)
        vdos_autocorrelation = np.mean(vdos_autocorrelation, axis=0)
        return [vdos_autocorrelation]


class IRSpectrum(VibrationalSpectrum):
    """
    Compute infrared spectra from a molecular dynamics HDF5 dataset. This class requires the dipole moments
    to be present in the HDF5 dataset.

    Args:
        data (schnetpack.md.utils.HDF5Loader): Loaded dataset.
        resolution (int, optional): Resolution used when computing the spectrum. Indicates how many time lags
                                    are considered in the autocorrelation function is used.
        dipole_moment_handle (str, optional): Indentifier used for extracting dipole data.
    """

    def __init__(
        self,
        data: HDF5Loader,
        resolution: int = 4096,
        dipole_moment_handle: str = properties.dipole_moment,
    ):
        super(IRSpectrum, self).__init__(data, resolution=resolution)
        self.dipole_moment_handle = dipole_moment_handle

    def _get_data(self, molecule_idx: int):
        """
        Extract the molecular dipole moments and compute their time derivative via central
        difference.

        Args:
            molecule_idx (int): Index of the molecule for which the spectrum should be computed.
                                Uses the same conventions as schnetpack.md.System.

        Returns:
            numpy.array: Array holding the dipole moment derivatives.
        """
        relevant_data = self.data.get_property(
            self.dipole_moment_handle, False, mol_idx=molecule_idx
        )
        # Compute numerical derivative via central differences
        relevant_data = (relevant_data[2:, ...] - relevant_data[:-2, ...]) / (
            2 * self.timestep
        )
        return relevant_data

    def _process_autocorrelation(self, autocorrelation: np.array):
        """
        Sum over the three Cartesian components.

        Args:
            autocorrelation (numpy.array): Dipole moment flux autorcorrelation functions.

        Returns:
            numpy.array: Updated autocorrelation.
        """
        dipole_autocorrelation = np.sum(autocorrelation, axis=0)
        return [dipole_autocorrelation]


class RamanSpectrum(VibrationalSpectrum):
    """
    Compute Raman spectra from a molecular dynamics HDF5 dataset. This class requires the polarizabilities
    to be present in the HDF5 dataset.

    Args:
        data (schnetpack.md.utils.HDF5Loader): Loaded dataset.
        incident_frequency (float): laser frequency used for spectrum (in cm$^{-1}$).
                                    One typical value would be 19455.25 cm^-1 (514 nm)
        temperature (float): temperature used for spectrum (in K).
        polarizability_handle (str, optional): Identifier used for extracting polarizability data.
        resolution (int, optional): Resolution used when computing the spectrum. Indicates how many time lags
                                    are considered in the autocorrelation function is used.
        averaged (bool): compute rotationally averaged Raman spectrum.
    """

    def __init__(
        self,
        data: HDF5Loader,
        incident_frequency: float,
        temperature: float,
        polarizability_handle: str = properties.polarizability,
        resolution: int = 4096,
        averaged: bool = False,
    ):
        super(RamanSpectrum, self).__init__(data, resolution=resolution)
        self.incident_frequency = incident_frequency
        self.temperature = temperature
        self.averaged = averaged
        self.polarizability_handle = polarizability_handle

    def _get_data(self, molecule_idx: int):
        """
        Extract the molecular dipole moments and compute their time derivative via central
        difference.

        Args:
            molecule_idx (int): Index of the molecule for which the spectrum should be computed.
                                Uses the same conventions as schnetpack.md.System.

        Returns:
            numpy.array: Array holding the dipole moment derivatives.
        """
        relevant_data = self.data.get_property(
            self.polarizability_handle, False, mol_idx=molecule_idx
        )

        # Compute numerical derivative via central differences
        relevant_data = (relevant_data[2:, ...] - relevant_data[:-2, ...]) / (
            2 * self.timestep
        )

        # Compute isotropic and anisotropic part
        if self.averaged:
            # Setup for random orientations of the molecule
            polar_data = np.zeros((relevant_data.shape[0], 7))
            # Isotropic contribution:
            polar_data[:, 0] = np.trace(relevant_data, axis1=1, axis2=2) / 3
            # Anisotropic contributions
            polar_data[:, 1] = relevant_data[..., 0, 0] - relevant_data[..., 1, 1]
            polar_data[:, 2] = relevant_data[..., 1, 1] - relevant_data[..., 2, 2]
            polar_data[:, 3] = relevant_data[..., 2, 2] - relevant_data[..., 0, 0]
            polar_data[:, 4] = relevant_data[..., 0, 1]
            polar_data[:, 5] = relevant_data[..., 0, 2]
            polar_data[:, 6] = relevant_data[..., 1, 2]
        else:
            polar_data = np.zeros((relevant_data.shape[0], 2))
            # Typical experimental setup
            # xx
            polar_data[:, 0] = relevant_data[..., 0, 0]
            # xy
            polar_data[:, 1] = relevant_data[..., 0, 1]

        return polar_data

    def _process_autocorrelation(self, autocorrelation):
        """
        Compute isotropic and anisotropic components.

        Args:
            autocorrelation (numpy.array): Dipole moment flux autorcorrelation functions.

        Returns:
            numpy.array: Updated autocorrelation.
        """
        if self.averaged:
            isotropic = autocorrelation[0, :]
            anisotropic = (
                0.5 * autocorrelation[1, :]
                + 0.5 * autocorrelation[2, :]
                + 0.5 * autocorrelation[3, :]
                + 3.0 * autocorrelation[4, :]
                + 3.0 * autocorrelation[5, :]
                + 3.0 * autocorrelation[6, :]
            )
        else:
            isotropic = autocorrelation[0, :]
            anisotropic = autocorrelation[1, :]

        autocorrelation = [isotropic, anisotropic]

        return autocorrelation

    def _process_spectrum(self):
        """
        Apply temperature and frequency dependent cross section.
        """
        frequencies = self.frequencies[0]
        cross_section = (
            (self.incident_frequency - frequencies) ** 4
            / frequencies
            / (
                1
                - np.exp(
                    -(spk_units.hbar2icm * frequencies)
                    / (spk_units.kB * self.temperature)
                )
            )
        )
        cross_section[0] = 0

        for i in range(len(self.intensities)):
            self.intensities[i] *= cross_section
            self.intensities[i] *= 4.160440e-18  # Where does this come from?
            self.intensities[i][0] = 0.0

        if self.averaged:
            isotropic, anisotropic = self.intensities
            parallel = isotropic + 4 / 45 * anisotropic
            orthogonal = anisotropic / 15

            self.intensities = [parallel, orthogonal]
